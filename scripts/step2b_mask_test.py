#!/usr/bin/env python3
"""
Step 2b - entropy at a masked position
======================================

    python scripts/step2b_mask_test.py

Purpose
-------
Establish that a masked diffusion language model produces a usable confidence
signal, and that the signal lives *only* at masked positions. Everything in
`src/features.py` later depends on this being true, so it is worth measuring
once, explicitly, before building anything on top of it.

Background: why the obvious test fails
--------------------------------------
The first version of this check asked the model "what token comes next?" by
reading the logits at the final position of an ordinary prompt. It returned
near-uniform noise - normalised entropy 0.85, top-1 probability 0.028 - and
the top prediction was simply the token already sitting at that position.

That is correct behaviour, not a broken model. LLaDA is trained with a masked
diffusion objective: the loss is computed **only over masked positions**. The
output head at an unmasked position is never optimised, so what it emits there
is meaningless. There is no "next token" in a bidirectional denoiser; the only
well-posed question is "what belongs in this hole?"

So this script punches a hole with the `<|mask|>` token and reads the logits
there instead, and measures an unmasked position alongside it as a control.

Why the contrast matters to the project
---------------------------------------
This is charter Warning #4 made concrete. In vanilla LLaDA a position, once
revealed, stays revealed - its confidence is set to infinity so the remasking
step can never touch it again. Trajectory "flips" therefore have to be measured
on the argmax *prediction at still-masked positions* across denoising rounds,
never on revealed tokens appearing to change. If the two entropies below are
not clearly different, that premise is wrong and the feature design has to be
reconsidered before Step 5.

Expected result
---------------
    masked position    normalised entropy < 0.55, ' Paris' at or near rank 1
    unmasked position  normalised entropy around 0.85, arbitrary tokens

The script prints its own PASS / UNEXPECTED verdict and writes a copy of
everything to `outputs/step2b_mask_report.txt`.

Model note
----------
Uses LLaDA-MoE (7B total, ~1.4B active) purely because it fits in 6 GB of VRAM
for local development. It is a debugging model and never appears in reported
results - those come from LLaDA-8B-Instruct and Dream-7B-Instruct, run on
Kaggle in Phase C.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# `src.config` must be imported before torch or transformers: it sets HF_HOME,
# and the HuggingFace libraries read that variable once, at import time.
# Inserting the repo root on sys.path lets this run from anywhere.
# --------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root:  python scripts/step2b_mask_test.py")
    print("and check that src/config.py and src/model_utils.py both exist.")
    sys.exit(1)

# The prompt is deliberately a fact the model should be certain about. A
# confident answer here is evidence the weights loaded correctly; an uncertain
# one would suggest quantisation damage rather than a genuine lack of knowledge.
PROMPT = "The capital city of France is"

REPORT_PATH = config.OUT_DIR / "step2b_mask_report.txt"

# Every line printed is also collected so the report file matches the screen.
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save_report(exit_code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(exit_code)


def describe_position(logits, index: int, tokenizer, label: str):
    """Print entropy and the top-5 tokens at one position.

    Returns `(normalised_entropy, top1_token)` so the caller can compare the
    masked and unmasked positions and decide pass/fail.
    """
    entropy, normalised, probs = model_utils.position_entropy(logits, index)
    top_p, top_i = probs.topk(5)

    say("")
    say(f"  {label}")
    say(f"    entropy    : {entropy:8.4f} nats")
    say(f"    normalised : {normalised:8.4f}    (0 = certain, 1 = uniform)")
    say(f"    top-1 prob : {top_p[0].item():8.4f}")
    say("    top 5:")
    for rank, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()), start=1):
        say(f"      {rank}. {tokenizer.decode([i])!r:<18} p = {p:.4f}  (id {i})")

    return normalised, tokenizer.decode([top_i[0].item()])


def main() -> None:
    say("=" * 70)
    say("  TRIAGE - Step 2b: entropy at a masked position")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 70)

    import torch
    import transformers

    say(f"transformers  : {transformers.__version__}")
    say(f"torch         : {torch.__version__}")
    for line in config.describe().splitlines():
        say(line)

    # Check memory before committing to a load that can take many minutes.
    if not model_utils.preflight_memory_check(model_size_gb=15.0, log=say):
        save_report(1)

    # ---- load -----------------------------------------------------------
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
    if mask_id is None:
        say(">>> STOP: this tokenizer reports no mask_token_id.")
        save_report(1)

    # ---- build the input:  <prompt tokens> <MASK> ------------------------
    #
    # `add_special_tokens=False` keeps the sequence minimal and predictable.
    # A BOS token would shift every index and make the report harder to read;
    # for a single-position probe it adds nothing.
    say("-" * 70)
    ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"] + [mask_id]
    mask_pos = len(ids) - 1        # the hole we punched: signal expected here
    real_pos = len(ids) - 2        # ordinary token: control, noise expected

    input_ids = torch.tensor([ids], device=model.device)

    say(f"INPUT         : {PROMPT!r} + <|mask|>")
    say(f"  token ids   : {ids}")
    say(f"  decoded     : {[tokenizer.decode([i]) for i in ids]}")
    say(f"  mask index {mask_pos}, control index {real_pos}")

    # ---- one forward pass ------------------------------------------------
    #
    # No KV cache and no incremental decoding: a masked diffusion model attends
    # bidirectionally over the whole sequence and recomputes everything each
    # time. That is why Phase C costs ~55 GPU-hours - 32 denoising rounds means
    # 32 complete forward passes per question, not 32 cheap cached steps.
    try:
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        say("")
        say(f"  logits shape: {tuple(logits.shape)}  "
            f"forward {time.time() - t0:.2f}s")

        # Checkpoints often pad the vocabulary up to a multiple of 64 or 128 for
        # kernel alignment. Those extra rows are real model outputs, so we leave
        # them in the entropy calculation and simply note them - excluding them
        # would make our numbers incomparable with published baselines.
        if logits.shape[-1] != len(tokenizer):
            say(f"  note: {logits.shape[-1] - len(tokenizer)} padded vocab "
                f"slots (logits {logits.shape[-1]} vs tokenizer "
                f"{len(tokenizer)}) - normal, kept in the entropy")

        say("-" * 70)
        say("THE COMPARISON")
        norm_mask, top_mask = describe_position(
            logits, mask_pos, tokenizer, "AT THE <|mask|> POSITION")
        norm_real, top_real = describe_position(
            logits, real_pos, tokenizer, "AT AN UNMASKED TOKEN (control)")

        # ---- verdict -----------------------------------------------------
        say("")
        say("-" * 70)
        say("VERDICT")
        say(f"  masked   : normalised entropy {norm_mask:.4f}  top-1 {top_mask!r}")
        say(f"  unmasked : normalised entropy {norm_real:.4f}  top-1 {top_real!r}")
        say(f"  gap      : {norm_real - norm_mask:+.4f}")
        say("")
        if norm_mask < 0.55 and norm_mask < norm_real:
            say("  PASS - the masked position is markedly more confident than")
            say("         the unmasked one. The model loaded correctly and the")
            say("         signal lives where the feature design assumes.")
        else:
            say("  UNEXPECTED - the masked position is not clearly more")
            say("         confident. Do not continue to Step 3; send this report.")

        snap = model_utils.vram_snapshot()
        say("")
        say(f"  peak VRAM : {model_utils.gib(torch.cuda.max_memory_allocated(0)):.2f} GiB")
        if snap:
            say(f"  free VRAM : {snap['free']:.2f} of {snap['total']:.2f} GiB")

    except torch.cuda.OutOfMemoryError:
        say("  OUT OF MEMORY during the forward pass. The weights fit but the")
        say("  activations did not. Report this - the fix is to slice hidden")
        say("  states to masked positions before the output projection.")
        _lines.append(traceback.format_exc())
    except Exception:
        say("  FAILED:")
        tb = traceback.format_exc()
        print(tb)
        _lines.append(tb)

    say("=" * 70)
    save_report(0)


if __name__ == "__main__":
    main()
