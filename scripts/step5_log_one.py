#!/usr/bin/env python3
"""
Step 5 - exercise the trajectory logger on one real question
============================================================

    python scripts/step5_log_one.py

Runs `src/logging_patch.generate_with_logging` end to end, saves the trajectory
to disk, reloads it, and reports the derived statistics. Confirms three things
before Step 6 scales the check to 50 questions:

  1. The logger produces a trajectory whose reconstruction matches the model's
     actual output exactly. (Step 6 is this test at scale; a failure here means
     stopping now.)
  2. The file is around 26 KB, not gigabytes. Charter Warning #1.
  3. Save and reload round-trips without loss, so Phase C can checkpoint.

Step 4 printed a live table; this one produces the artefact Phase C will
actually generate 9,000 of.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step5_log_one.py")
    sys.exit(1)

QUESTION = "In two sentences, explain why the sky appears blue."
QUESTION_ID = "demo-0001"

REPORT_PATH = config.OUT_DIR / "step5_log_report.txt"
TRAJ_PATH = config.TRAJ_DIR / f"{QUESTION_ID}.npz"

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save_report(code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(code)


def build_prompt_ids(tokenizer):
    """Chat-template the question. Same path Phase C will use."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def main() -> None:
    import numpy as np

    say("=" * 76)
    say("  TRIAGE - Step 5: trajectory logger, one question")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 76)

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

    prompt_ids = build_prompt_ids(tokenizer)
    say("-" * 76)
    say(f"QUESTION : {QUESTION!r}")
    say(f"prompt   : {len(prompt_ids)} tokens")
    say(f"settings : gen_length={config.GEN_LENGTH} steps={config.DENOISING_STEPS} "
        f"block={config.BLOCK_LENGTH} temperature=0.0")
    say("")
    say("Generating (32 forward passes)...")

    t0 = time.time()
    final_text, traj = logging_patch.generate_with_logging(
        model, tokenizer, prompt_ids,
        gen_length=config.GEN_LENGTH,
        steps=config.DENOISING_STEPS,
        block_length=config.BLOCK_LENGTH,
        temperature=0.0,
        question=QUESTION,
        question_id=QUESTION_ID,
        quant_config=cfg_label,
        seed=config.SEED,
    )
    say(f"  done in {time.time() - t0:.0f}s "
        f"({traj.elapsed_s / traj.steps:.2f}s per forward pass)")

    say("")
    say("ANSWER")
    say(f"  {final_text!r}")

    # ---- 1. reconstruction ------------------------------------------------
    #
    # Rebuild the output using ONLY pred_ids and mask_state. If this differs
    # from what the model actually emitted, the log does not describe the run.
    say("")
    say("-" * 76)
    say("CHECK 1 - RECONSTRUCTION FROM THE LOG")
    rebuilt = traj.reconstruct_final()
    ids_match = bool(np.array_equal(rebuilt, traj.final_ids))
    text_match = tokenizer.decode(rebuilt.tolist()) == final_text
    say(f"  token ids identical : {ids_match}")
    say(f"  decoded text identical (character for character) : {text_match}")
    if not ids_match:
        bad = np.flatnonzero(rebuilt != traj.final_ids)
        say(f"  MISMATCH at positions {bad.tolist()[:10]}")

    # ---- 2. size ----------------------------------------------------------
    say("")
    say("-" * 76)
    say("CHECK 2 - SIZE ON DISK (Warning #1)")
    traj.save(TRAJ_PATH)
    kb = TRAJ_PATH.stat().st_size / 1024
    full_logits_gb = (traj.steps * traj.gen_length * traj.vocab_size * 4) / 1024 ** 3
    say(f"  saved to {TRAJ_PATH}")
    say(f"  file size          : {kb:.1f} KB")
    say(f"  full logits would be: {full_logits_gb:.2f} GB per question")
    say(f"  9,000 questions     : {kb * 9000 / 1024 / 1024:.2f} GB total")

    # ---- 3. round trip ----------------------------------------------------
    say("")
    say("-" * 76)
    say("CHECK 3 - SAVE / RELOAD ROUND TRIP")
    back = logging_patch.Trajectory.load(TRAJ_PATH)
    ok = all([
        np.array_equal(back.pred_ids, traj.pred_ids),
        np.allclose(back.pred_probs, traj.pred_probs),
        np.allclose(back.entropy, traj.entropy),
        np.array_equal(back.mask_state, traj.mask_state),
        np.array_equal(back.final_ids, traj.final_ids),
        back.question == traj.question,
        back.mask_id == traj.mask_id,
    ])
    say(f"  arrays and metadata survive the round trip : {ok}")

    # ---- derived statistics ------------------------------------------------
    say("")
    say("-" * 76)
    say("DERIVED SIGNALS")

    flips = traj.flips_per_round()
    regret = traj.regret_per_round()
    settle = traj.revealed_at()
    content = traj.content_mask()

    say(f"  flips  : total {flips.sum()}, "
        f"first half {flips[:len(flips)//2].sum()}, "
        f"second half {flips[len(flips)//2:].sum()}")
    say(f"  regret : total {regret.sum()}, final round {regret[-1]}")
    say(f"  settle round: min {settle.min()}, median "
        f"{int(np.median(settle))}, max {settle.max()}")

    # Warning #2 made quantitative: how much of the block is EOS padding, and
    # what it does to the headline entropy number.
    masked = traj.mask_state == 1
    is_content = content & masked
    if masked.any():
        h_all = float(traj.entropy[masked].mean())
        h_content = (float(traj.entropy[is_content].mean())
                     if is_content.any() else float("nan"))
        pad_frac = 1.0 - (is_content.sum() / masked.sum())
        say("")
        say("  Warning #2 - EOS/padding contamination:")
        say(f"    masked positions that are EOS or mask : {pad_frac:.1%}")
        say(f"    mean entropy, all masked              : {h_all:.4f}")
        say(f"    mean entropy, content only            : {h_content:.4f}")
        say(f"    difference                            : {h_all - h_content:+.4f}")

    # ---- verdict -----------------------------------------------------------
    say("")
    say("=" * 76)
    if ids_match and text_match and ok:
        say("  PASS - the log fully describes the run, round-trips cleanly, and")
        say("         is small enough to store at scale. Proceed to Step 6,")
        say("         which repeats check 1 across 50 questions.")
    else:
        say("  FAIL - do not proceed. Send this report.")
    say("=" * 76)
    save_report(0)


if __name__ == "__main__":
    main()
