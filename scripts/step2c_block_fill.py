#!/usr/bin/env python3
"""
Step 2c - single-pass block fill, the way LLaDA is actually prompted
====================================================================

    python scripts/step2c_block_fill.py

Purpose
-------
Confirm that the model produces correct, content-aware predictions at masked
positions when prompted the way it was trained. This is the last check before
instrumenting the real denoising loop in Steps 3-6.

What went wrong in Steps 2 and 2b, and why it was informative
-------------------------------------------------------------
**Step 2** read logits at the last position of a plain prompt with no mask
anywhere. Result: normalised entropy 0.85, essentially uniform. A masked
diffusion model is trained only on sequences that *contain* masks, so a
mask-free sequence is off-distribution and the output head produces noise.

**Step 2b** appended one `<|mask|>` and compared it against an unmasked
position. Two things came out of that:

  * The unmasked position went from 0.85 to 0.118 normalised entropy purely
    because a mask now existed elsewhere in the sequence. That is strong
    evidence the model behaves as a denoiser: the presence of masks is what
    puts it in-distribution.

  * The comparison itself was ill-posed. An unmasked position is a *copy*
    task - bidirectional attention lets the model see the token it is being
    asked to predict - so it will always be more confident than a genuine
    prediction at a masked position. Expecting the mask to be more confident
    was backwards.

  * Top-1 at the mask was `' the'`, not `' Paris'`. With one hole and no
    end-of-sequence signal, "The capital city of France is the ..." is a
    reasonable continuation. The model was continuing a fragment, not
    answering a question.

This script fixes all three: chat template, a block of masks, and a check on
the decoded answer rather than on a single token.

How LLaDA generation actually works
-----------------------------------
    1. Format the question with the model's chat template.
    2. Append `gen_length` mask tokens - the answer region.
    3. Denoise over `steps` rounds. Each round predicts every masked position,
       keeps the most confident predictions, and re-masks the rest.

This script runs **round 1 only** and reveals every position at once. That is a
deliberately weakened version of generation - one pass instead of 32 - so the
text may be rough. What matters is whether the correct answer is present and
whether entropy varies sensibly across positions. Steps 3-6 add the loop.

Expected result
---------------
    "Paris" appears in the greedy fill
    entropy varies across masked positions rather than being flat
    positions near the prompt are more confident than distant ones

That last pattern matters to the project: it is the raw material of the
trajectory. If every masked position had identical entropy there would be no
per-position signal for `features.py` to read.
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# `src.config` sets HF_HOME and must be imported before torch or transformers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root:  python scripts/step2c_block_fill.py")
    sys.exit(1)

QUESTION = "What is the capital city of France? Answer in one short sentence."
EXPECTED = "paris"          # lower-cased substring we look for in the fill

# 16 masks, not the 64 used in production. Enough to hold a short answer while
# keeping the logits tensor small - VRAM headroom is thin on a 6 GB card.
N_MASKS = 16

# How many masked positions to print in full. All 16 would bury the verdict.
N_DETAIL = 6

REPORT_PATH = config.OUT_DIR / "step2c_block_fill_report.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save_report(exit_code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(exit_code)


def build_prompt_ids(tokenizer):
    """Format the question the way the instruct model expects.

    Returns `(ids, how)` where `how` records which path was taken, because an
    instruct checkpoint prompted without its chat template is off-distribution
    and that would need to appear in the report rather than pass silently.
    """
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}],
            add_generation_prompt=True,     # emit the assistant-turn header
            tokenize=True,
        )
        # Some templates return a tensor or a BatchEncoding rather than a list.
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids), "chat template"
    except Exception as exc:
        say(f"  chat template unavailable ({type(exc).__name__}), "
            f"falling back to plain text")
        text = f"Question: {QUESTION}\nAnswer:"
        return tokenizer(text, add_special_tokens=True)["input_ids"], "plain text"


def main() -> None:
    say("=" * 72)
    say("  TRIAGE - Step 2c: single-pass block fill")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 72)

    import torch

    for line in config.describe().splitlines():
        say(line)

    if not model_utils.preflight_memory_check(model_size_gb=15.0, log=say):
        save_report(1)

    t0 = time.time()
    try:
        model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(log=say)
    except RuntimeError as exc:
        say("")
        say("LOAD FAILED")
        say(str(exc))
        save_report(1)
    say(f"  loaded in {time.time() - t0:.0f}s using config {cfg_label}")

    mask_id = tokenizer.mask_token_id

    # ---- build: <chat-formatted prompt> + N_MASKS mask tokens -------------
    say("-" * 72)
    prompt_ids, how = build_prompt_ids(tokenizer)
    ids = list(prompt_ids) + [mask_id] * N_MASKS

    prompt_len = len(prompt_ids)
    mask_positions = list(range(prompt_len, prompt_len + N_MASKS))
    input_ids = torch.tensor([ids], device=model.device)

    say(f"PROMPT VIA    : {how}")
    say(f"QUESTION      : {QUESTION!r}")
    say(f"  prompt tokens : {prompt_len}")
    say(f"  mask tokens   : {N_MASKS}  (positions {mask_positions[0]}"
        f"-{mask_positions[-1]})")
    say(f"  total length  : {len(ids)}")
    say("")
    say("  prompt decoded:")
    say(f"    {tokenizer.decode(prompt_ids)!r}")

    # ---- one forward pass over the whole sequence ------------------------
    #
    # No KV cache: a masked diffusion model attends bidirectionally and
    # recomputes the entire sequence every round. 32 rounds means 32 complete
    # forward passes per question - the reason Phase C costs ~55 GPU-hours.
    try:
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        say("")
        say(f"  logits shape  : {tuple(logits.shape)}  "
            f"forward {time.time() - t0:.2f}s")

        # ---- per-position entropy across the answer block ----------------
        say("-" * 72)
        say(f"MASKED POSITIONS (first {N_DETAIL} of {N_MASKS})")
        say("")
        say("   pos  offset   entropy   norm    p(top1)  top-1 token")
        say("   ---  ------   -------   -----   -------  -----------")

        entropies, norms, fill_ids = [], [], []
        for offset, pos in enumerate(mask_positions):
            entropy, norm, probs = model_utils.position_entropy(logits, pos)
            top_p, top_i = probs.topk(1)
            tok_id = top_i[0].item()

            entropies.append(entropy)
            norms.append(norm)
            fill_ids.append(tok_id)

            if offset < N_DETAIL:
                say(f"   {pos:3d}  {offset:6d}   {entropy:7.3f}   {norm:5.3f}   "
                    f"{top_p[0].item():7.4f}  {tokenizer.decode([tok_id])!r}")

        # ---- the greedy fill ---------------------------------------------
        #
        # Taking the argmax at every masked position simultaneously is one
        # denoising round with nothing re-masked. Real generation reveals only
        # the most confident positions each round and re-predicts the rest,
        # which is what makes the text coherent - and what creates the
        # trajectory this project measures.
        fill_text = tokenizer.decode(fill_ids)
        say("")
        say("GREEDY FILL (all positions revealed at once, 1 round)")
        say(f"  {fill_text!r}")

        # ---- summary statistics ------------------------------------------
        import statistics
        say("")
        say("ENTROPY ACROSS THE BLOCK")
        say(f"  mean   : {statistics.mean(norms):.4f} normalised")
        say(f"  min    : {min(norms):.4f}  (position "
            f"{mask_positions[norms.index(min(norms))]})")
        say(f"  max    : {max(norms):.4f}  (position "
            f"{mask_positions[norms.index(max(norms))]})")
        say(f"  spread : {max(norms) - min(norms):.4f}")

        # ---- verdict ------------------------------------------------------
        #
        # Two things must hold before Step 3:
        #   1. the correct answer is present - the model knows the fact and
        #      quantisation has not destroyed it
        #   2. entropy varies across positions - there is a per-position signal
        #      for features.py to read. A flat profile would mean no trajectory.
        found = EXPECTED in fill_text.lower()
        varies = (max(norms) - min(norms)) > 0.05

        say("")
        say("-" * 72)
        say("VERDICT")
        say(f"  '{EXPECTED}' present in fill : {found}")
        say(f"  entropy varies by position   : {varies} "
            f"(spread {max(norms) - min(norms):.4f}, need > 0.05)")
        say("")
        if found and varies:
            say("  PASS - the model answers correctly at masked positions and")
            say("         entropy carries per-position structure. Step 2 is")
            say("         complete; proceed to Step 3.")
        elif varies and not found:
            say("  PARTIAL - entropy structure is there but the answer is not.")
            say("         One denoising round is weak, so this can happen.")
            say("         Send the report; the greedy fill above decides")
            say("         whether this is a prompt-format issue or something")
            say("         worse.")
        else:
            say("  UNEXPECTED - send this report before continuing.")

        say("")
        say(f"  peak VRAM : "
            f"{model_utils.gib(torch.cuda.max_memory_allocated(0)):.2f} GiB")

    except torch.cuda.OutOfMemoryError:
        say("  OUT OF MEMORY during the forward pass.")
        say(f"  Try lowering N_MASKS from {N_MASKS} to 8 at the top of this file.")
        _lines.append(traceback.format_exc())
    except Exception:
        say("  FAILED:")
        tb = traceback.format_exc()
        print(tb)
        _lines.append(tb)

    say("=" * 72)
    save_report(0)


if __name__ == "__main__":
    main()
