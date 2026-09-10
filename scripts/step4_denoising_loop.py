#!/usr/bin/env python3
"""
Step 4 - the denoising loop, with entropy printed every round  [STAGE 0 GATE]
=============================================================================

    python scripts/step4_denoising_loop.py

This is the hard gate. If entropy does not evolve sensibly across denoising
rounds here, nothing downstream in the project is measurable and the approach
has to be reconsidered before any GPU time is spent in Phase C.

Why this file implements its own sampler
----------------------------------------
Step 3 established that LLaDA-family checkpoints ship **no sampler**. Their
remote code (`modeling_lladamoe.py`) is the architecture only - its `topk`
calls are MoE expert routing, not denoising. The sampling loop lives separately
in ML-GSAI/LLaDA's `generate.py`.

So there is nothing to monkeypatch. We write the loop ourselves, following the
reference implementation, with instrumentation built in from the start. That is
better than patching third-party code: nothing breaks when a model repository
updates, and the result is a self-contained file fit for publication.

This script is deliberately written to be *read*. Step 5 refactors it into
`src/logging_patch.py` with efficient arrays; here everything is explicit.

The algorithm
-------------
Reference: ML-GSAI/LLaDA `generate.py` lines 101-204.

    x = [prompt tokens] + [MASK] * gen_length

    for each round:
        logits     = model(x)                      # full bidirectional pass
        x0         = argmax(logits)                # best guess at EVERY position
        x0_p       = P(x0)                         # confidence in that guess
        confidence = where(still_masked, x0_p, -inf)     # reference line 198
        reveal the k highest-confidence masked positions
        everything else stays masked and is re-predicted next round

`-inf` (negative, not positive - the plan had this backwards) is what stops
`topk` ever re-selecting a revealed position. A revealed token is frozen for
the rest of generation.

Three charter warnings implemented here
---------------------------------------
**#1 Never log full logits.** 32 rounds x 256 positions x 126,464 vocab x 4
bytes is 4.1 GB per question. We keep four scalars per position per round -
argmax id, its probability, entropy, mask state - about 98 KB per question.

**#3 Exclude prompt positions.** The prompt is never masked, so it has no
trajectory. This script slices logits to the generation region *before*
computing anything, which enforces the rule and incidentally halves peak
memory - a full-sequence fp32 softmax over a 157k vocabulary does not fit
comfortably in the ~0.5 GiB of headroom left on a 6 GB card.

**#4 Flips live at masked positions.** A revealed token cannot change, so a
"flip" must mean the argmax *prediction at a still-masked position* differing
between consecutive rounds. This script reports flips accordingly.

Finding B, demonstrated live
----------------------------
Reference line 197 (`x0 = torch.where(mask_index, x0, x)`) overwrites the
model's prediction at revealed positions with the committed token, so anything
logged afterwards cannot see what the model *would now say* about a token it
already fixed. We capture the raw argmax before that point and report
**regret**: revealed positions where the model's current best guess differs
from what it is stuck with. No published trajectory method measures this.

Expected result
---------------
    entropy at still-masked positions falls as rounds progress
    the text assembles progressively rather than appearing at once
    flips are non-zero in early rounds and fall towards zero
    regret is reported (it may legitimately be zero)

A flat entropy profile would be the failure case: it would mean there is no
per-round signal for `features.py` to read.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step4_denoising_loop.py")
    sys.exit(1)

# A question needing a real multi-token answer. A one-word trivia answer would
# fill two positions and pad the other 62 with EOS, leaving almost no trajectory
# to look at - which is exactly the artefact charter Warning #2 is about.
QUESTION = "In two sentences, explain why the sky appears blue."

REPORT_PATH = config.OUT_DIR / "step4_denoising_report.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save_report(code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(code)


def get_num_transfer_tokens(mask_index, steps: int):
    """How many positions to reveal on each round.

    Faithful to the reference implementation (generate.py line 28). The masked
    count is divided evenly across rounds, with the remainder handed to the
    earliest rounds so the totals add up exactly.

    Note this schedule is **fixed in advance and not adaptive** - the model's
    confidence decides *which* positions are revealed, never *how many*. Any
    settle-round feature in Step 19 has to be interpreted against this schedule
    rather than treated as a free measurement.
    """
    import torch
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps,
                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, : remainder[i]] += 1
    return out


def build_prompt_ids(tokenizer):
    """Format the question with the model's chat template.

    Step 2c showed this template injects `<role>SYSTEM</role>detailed thinking
    off<|role_end|>` by default. That is recorded in the report because it
    affects generation and has to be pinned explicitly in Phase C.
    """
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids), "chat template"
    except Exception:
        return tokenizer(f"Question: {QUESTION}\nAnswer:")["input_ids"], "plain text"


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 4: denoising loop with per-round entropy   [STAGE 0 GATE]")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    import torch

    for line in config.describe().splitlines():
        say(line)
    if not model_utils.preflight_memory_check(model_size_gb=15.0, log=say):
        save_report(1)

    try:
        model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(log=say)
    except RuntimeError as exc:
        say("\nLOAD FAILED")
        say(str(exc))
        save_report(1)
    say(f"  config: {cfg_label}")

    # `mask_id` is taken from the tokenizer, never hard-coded. The reference
    # implementation defaults to 126336 (LLaDA-8B); LLaDA-MoE uses 156895.
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    gen_length = config.GEN_LENGTH        # 64 - TraceDet found this best
    steps = config.DENOISING_STEPS        # 32
    block_length = config.BLOCK_LENGTH    # 64, i.e. a single block

    prompt_ids, how = build_prompt_ids(tokenizer)
    prompt_len = len(prompt_ids)

    say("-" * 78)
    say(f"QUESTION      : {QUESTION!r}")
    say(f"prompt via    : {how}")
    say(f"prompt tokens : {prompt_len}")
    say(f"gen_length    : {gen_length}   steps: {steps}   block_length: {block_length}")
    say(f"mask_id       : {mask_id}      eos_id: {eos_id}")
    say(f"temperature   : 0.0 (pure argmax - deterministic, reproducible)")
    say(f"prompt decoded: {tokenizer.decode(prompt_ids)!r}")

    # ---- initial state: prompt followed by a block of masks ---------------
    device = model.device
    x = torch.full((1, prompt_len + gen_length), mask_id,
                   dtype=torch.long, device=device)
    x[0, :prompt_len] = torch.tensor(prompt_ids, device=device)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    say("-" * 78)
    say(f"DENOISING - {num_blocks} block(s), {steps_per_block} rounds each")
    say("")
    say(" round  reveal   H(masked)   H(no-EOS)   flips  regret   text so far")
    say(" -----  ------   ---------   ---------   -----  ------   -----------")

    prev_pred = None          # argmax at masked positions from the previous round
    round_log = []            # (round, mean_H, mean_H_noeos, flips, regret)
    t_start = time.time()

    try:
        for block in range(num_blocks):
            b0 = prompt_len + block * block_length
            b1 = prompt_len + (block + 1) * block_length

            block_mask = (x[:, b0:b1] == mask_id)
            n_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                # Which positions are still masked, in generation-region space.
                mask_index_gen = (x[:, prompt_len:] == mask_id)   # (1, gen_length)

                with torch.no_grad():
                    logits = model(x).logits                      # (1, L, V)

                    # Slice to the generation region BEFORE any softmax.
                    # Warning #3 (prompt positions have no trajectory) and a
                    # memory necessity: a full-sequence fp32 softmax over 157k
                    # vocab does not fit in the headroom on a 6 GB card.
                    gen_logits = logits[:, prompt_len:, :].float()

                    # fp32 throughout. The laptop supports bf16 and Kaggle's T4
                    # does not; computing in fp32 on both is what keeps the two
                    # sets of numbers comparable.
                    probs = torch.softmax(gen_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)  # (1,G)

                    x0_gen = gen_logits.argmax(-1)                         # (1,G)
                    x0_p = probs.gather(-1, x0_gen.unsqueeze(-1)).squeeze(-1)

                    del gen_logits, probs, logits

                # ---- FINDING B: regret, measured before the overwrite ------
                # Reference line 197 replaces x0 with the committed token at
                # revealed positions. We still hold the raw argmax, so we can
                # ask: at positions already fixed, would the model now choose
                # something different? Invisible to every published method.
                revealed = ~mask_index_gen
                if revealed.any():
                    committed = x[:, prompt_len:][revealed]
                    would_say = x0_gen[revealed]
                    regret = int((committed != would_say).sum().item())
                else:
                    regret = 0

                # ---- flips: predictions at STILL-MASKED positions (Warning #4)
                if prev_pred is None:
                    flips = 0
                else:
                    both = mask_index_gen[0]
                    flips = int((x0_gen[0][both] != prev_pred[both]).sum().item())
                prev_pred = x0_gen[0].clone()

                # ---- entropy summaries over still-masked positions only ----
                m = mask_index_gen[0]
                if m.any():
                    h_masked = entropy[0][m].mean().item()
                    # Warning #2: EOS positions settle early because of an SFT
                    # padding artefact. Reported separately so the difference is
                    # visible rather than silently folded into the average.
                    not_eos = m & (x0_gen[0] != eos_id)
                    h_noeos = (entropy[0][not_eos].mean().item()
                               if not_eos.any() else float("nan"))
                else:
                    h_masked = h_noeos = float("nan")

                # ---- the reveal step, following the reference --------------
                x0_p_sched = x0_p.clone()
                x0_p_sched[:, (b1 - prompt_len):] = -float("inf")   # future blocks
                confidence = torch.where(mask_index_gen, x0_p_sched,
                                         torch.tensor(-float("inf"), device=device))

                k = int(n_transfer[0, step].item())
                if k > 0:
                    _, sel = torch.topk(confidence[0], k=k)
                    gen_slice = x[:, prompt_len:]
                    gen_slice[0, sel] = x0_gen[0, sel]
                    x[:, prompt_len:] = gen_slice

                # ---- report -------------------------------------------------
                shown = tokenizer.decode(
                    [t for t in x[0, prompt_len:].tolist() if t != mask_id]
                ).replace("\n", " ")[:34]

                say(f" {step + 1:5d}  {k:6d}   {h_masked:9.4f}   {h_noeos:9.4f}   "
                    f"{flips:5d}  {regret:6d}   {shown!r}")
                round_log.append((step + 1, h_masked, h_noeos, flips, regret))

        elapsed = time.time() - t_start

        # ---- final answer --------------------------------------------------
        answer_ids = x[0, prompt_len:].tolist()
        answer = tokenizer.decode(answer_ids)
        say("")
        say("-" * 78)
        say("FINAL ANSWER")
        say(f"  {answer!r}")
        say("")
        say(f"  {steps} rounds in {elapsed:.0f}s "
            f"({elapsed / steps:.2f}s per forward pass)")
        say(f"  peak VRAM: "
            f"{model_utils.gib(torch.cuda.max_memory_allocated(0)):.2f} GiB")

        # ---- the gate -------------------------------------------------------
        # Two conditions. Entropy must FALL as the model commits - that is the
        # signal every trajectory detector reads. And flips must occur early -
        # if predictions never change between rounds there is no trajectory,
        # only a single decision replayed 32 times.
        valid = [h for _, h, _, _, _ in round_log if h == h]     # drop NaN
        first_h = valid[0] if valid else float("nan")
        last_h = valid[-1] if valid else float("nan")
        total_flips = sum(f for _, _, _, f, _ in round_log)
        total_regret = sum(r for _, _, _, _, r in round_log)

        say("")
        say("-" * 78)
        say("GATE")
        say(f"  entropy first round : {first_h:.4f}")
        say(f"  entropy last round  : {last_h:.4f}")
        say(f"  fell?               : {last_h < first_h}")
        say(f"  total flips         : {total_flips}")
        say(f"  total regret        : {total_regret}   (Finding B)")
        say("")
        if last_h < first_h and total_flips > 0:
            say("  PASS - entropy falls as the model commits, and predictions")
            say("         change between rounds. There is a trajectory to")
            say("         measure. Stage 0 is cleared; proceed to Step 5.")
        else:
            say("  FAIL - no usable trajectory signal. Do NOT proceed to")
            say("         Phase C. Send this report.")

    except torch.cuda.OutOfMemoryError:
        say("")
        say("  OUT OF MEMORY. Lower GEN_LENGTH in src/config.py from "
            f"{gen_length} to 32 and retry.")
    except Exception:
        import traceback
        tb = traceback.format_exc()
        say("  FAILED:")
        print(tb)
        _lines.append(tb)

    say("=" * 78)
    save_report(0)


if __name__ == "__main__":
    main()
