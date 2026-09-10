#!/usr/bin/env python3
"""
Step 8 - LLaDA-8B timing benchmark on Kaggle
============================================

    python scripts/step8_kaggle_benchmark.py

RUNS ON KAGGLE, not the laptop. LLaDA-8B needs ~10 GB in 4-bit; the RTX 4050 has
6 GB. Use a T4 accelerator.

Why this must run before Phase C
--------------------------------
The Phase C budget rests on one unmeasured number. Local timings come from
LLaDA-MoE, which activates ~1.4B of 7B parameters; LLaDA-8B is dense. The
arithmetic:

    LLaDA-MoE on RTX 4050   0.69 s/forward  ->  22 s/question
    9,000 questions                          ->   55 GPU-hours

    If LLaDA-8B on T4 is 3-5x slower        ->  100-190 s/question
    9,000 questions                          -> 250-475 GPU-hours

At Kaggle's 30 GPU-hours per week that is the difference between a fortnight and
four months. The plan restructures on the second number - batching, fewer
datasets, or a smaller sample - and those are decisions to make in October, not
January.

What it does
------------
Loads LLaDA-8B-Instruct, runs the full 32-round loop on 5 real TriviaQA
questions, and reports seconds per forward pass, seconds per question, and the
projected Phase C budget. It also verifies reconstruction on those 5, which is
the Step 6 gate carried over to the production model - the sampler has only ever
been validated on LLaDA-MoE.

Two things this is the first real test of
-----------------------------------------
**float16 instead of bfloat16.** The T4 is Turing (compute 7.5) and has no
bfloat16. `model_utils.preferred_dtype()` now selects fp16 there and bf16 on the
laptop. Entropy stays float32 on both, which is what keeps the numbers
comparable - a fp16 softmax over a 126k vocabulary loses enough tail precision to
shift entropy measurably.

**LLaDA-8B's mask token.** Its id, 126336, is a RESERVED token that the tokenizer
does not declare, so `tokenizer.mask_token_id` returns None. A None mask id does
not crash - no position compares equal to it, nothing is ever masked, and the run
produces an empty trajectory while every check appears to pass.
`model_utils.resolve_mask_id` consults the tokenizer, then a table of known
models, and raises rather than guessing.

Both were caught by reading rather than by running, so treat this script's output
as the actual evidence they work.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step8_kaggle_benchmark.py")
    sys.exit(1)

MODEL_ID = config.MODEL_LLADA          # GSAI-ML/LLaDA-8B-Instruct
N_QUESTIONS = 5
PHASE_C_QUESTIONS = 9000               # 1500 x 3 datasets x 2 models

REPORT_PATH = config.OUT_DIR / "step8_benchmark_report.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save(code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(code)


def main() -> None:
    import numpy as np
    import torch

    say("=" * 78)
    say("  TRIAGE - Step 8: LLaDA-8B timing benchmark")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    for line in config.describe().splitlines():
        say(line)
    say(f"model         : {MODEL_ID}")
    say(f"compute dtype : {model_utils.describe_dtype()}")

    # LLaDA-8B in 4-bit is ~10 GB of weights (untied head, 126,464 vocab).
    if not model_utils.preflight_memory_check(model_size_gb=16.0, log=say):
        save(1)

    free_b, total_b = torch.cuda.mem_get_info(0)
    if model_utils.gib(total_b) < 12:
        say("")
        say(f"  STOP - {model_utils.gib(total_b):.1f} GiB of VRAM. LLaDA-8B in")
        say("  4-bit needs ~10 GB of weights plus activations. This script is")
        say("  meant for a Kaggle T4 (16 GB), not the laptop.")
        save(1)

    # ---- load ------------------------------------------------------------
    say("")
    say("-" * 78)
    t0 = time.time()
    try:
        model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
            MODEL_ID, log=say)
    except RuntimeError as exc:
        say("\nLOAD FAILED")
        say(str(exc))
        save(1)
    load_s = time.time() - t0
    say(f"  loaded in {load_s:.0f}s using config {cfg_label}")

    # The critical resolution. On LLaDA-8B the tokenizer returns None here and
    # the table supplies 126336.
    mask_id = model_utils.resolve_mask_id(tokenizer, MODEL_ID)
    declared = getattr(tokenizer, "mask_token_id", None)
    say("")
    say("MASK TOKEN RESOLUTION")
    say(f"  tokenizer declares : {declared}")
    say(f"  resolved to        : {mask_id}")
    if declared is None:
        say("  -> came from KNOWN_MASK_IDS, as expected for LLaDA-8B.")
        say("     Had this returned None, the run would have produced an empty")
        say("     trajectory without raising anything.")

    # ---- questions -------------------------------------------------------
    say("")
    say("-" * 78)
    recs = data.load_records("triviaqa", n=N_QUESTIONS * 4, seed=config.SEED)
    kept, _excluded, _rep = data_quality.clean_records(recs)
    recs = kept[:N_QUESTIONS]
    say(f"{len(recs)} TriviaQA questions (real prompts, evidence withheld)")

    # ---- run -------------------------------------------------------------
    say("")
    say("  #  prompt  fwd(s)  total(s)  recon  answer")
    say("  -  ------  ------  --------  -----  ------")

    per_question, per_forward, recon_ok = [], [], []

    for i, rec in enumerate(recs, 1):
        prompt_text = data.build_prompt_text(rec.question, rec.choices)
        data.assert_evidence_withheld(prompt_text, rec)
        prompt_ids = data.build_prompt_ids(tokenizer, rec)

        final_text, traj = logging_patch.generate_with_logging(
            model, tokenizer, prompt_ids,
            gen_length=config.GEN_LENGTH,
            steps=config.DENOISING_STEPS,
            block_length=config.BLOCK_LENGTH,
            temperature=0.0,
            question=rec.question, question_id=rec.qid,
            quant_config=cfg_label, seed=config.SEED,
            mask_id=mask_id,                      # never rely on the tokenizer
        )

        ok = bool(np.array_equal(traj.reconstruct_final(), traj.final_ids))
        recon_ok.append(ok)
        per_question.append(traj.elapsed_s)
        per_forward.append(traj.elapsed_s / traj.steps)

        say(f"  {i}  {len(prompt_ids):6d}  {traj.elapsed_s/traj.steps:6.2f}  "
            f"{traj.elapsed_s:8.1f}  {'OK ' if ok else 'FAIL':<5}  "
            f"{final_text.strip()[:40]!r}")

    # ---- results ---------------------------------------------------------
    mean_q = sum(per_question) / len(per_question)
    mean_f = sum(per_forward) / len(per_forward)
    hours = mean_q * PHASE_C_QUESTIONS / 3600

    say("")
    say("=" * 78)
    say("TIMING")
    say(f"  per forward pass : {mean_f:.2f}s")
    say(f"  per question     : {mean_q:.1f}s  ({config.DENOISING_STEPS} rounds)")
    say(f"  model load       : {load_s:.0f}s  (once per Kaggle session)")
    say("")
    say(f"  PHASE C ({PHASE_C_QUESTIONS:,} questions): {hours:.0f} GPU-hours")
    say(f"  at 30 GPU-hours/week          : {hours/30:.1f} weeks")

    say("")
    say("  reference: LLaDA-MoE on the laptop was 0.69 s/forward, 22 s/question,")
    say(f"  projecting 55 GPU-hours. This run is {mean_f/0.69:.1f}x that per pass.")

    say("")
    say("VERDICT")
    if hours <= 80:
        say("  BUDGET HOLDS. Phase C is roughly as planned. Proceed to Step 9.")
    elif hours <= 160:
        say("  BUDGET TIGHT. 3-5 weeks of Kaggle quota. Workable, but consider")
        say("  reducing the sample from 1500 to ~1000 per dataset once Step 10")
        say("  measures the base rate - the failure-mode split needs enough")
        say("  wrong answers, not enough questions.")
    else:
        say("  BUDGET BROKEN. Do not start Phase C on this plan. Options, best")
        say("  first:")
        say("    1. Batch questions - the forward pass is memory-bound at batch")
        say("       size 1, so batching 4-8 may cost little extra time each.")
        say("    2. Drop to one diffusion model for the main result and use the")
        say("       second only for a smaller validation slice.")
        say("    3. Cut the sample per dataset, guided by the Step 10 base rate.")
        say("    4. Reduce denoising steps from 32 to 16 - but this changes the")
        say("       trajectory itself and breaks comparability with TraceDet.")
        say("       Last resort.")

    say("")
    say("RECONSTRUCTION (the Step 6 gate, on the production model)")
    say(f"  {sum(recon_ok)} / {len(recon_ok)} exact")
    if not all(recon_ok):
        say("  FAIL - the sampler was only ever validated on LLaDA-MoE. Do not")
        say("  run Phase C until this is 5/5.")
    else:
        say("  The sampler works on LLaDA-8B as well as on LLaDA-MoE.")

    say("=" * 78)
    save(0 if all(recon_ok) else 1)


if __name__ == "__main__":
    main()
