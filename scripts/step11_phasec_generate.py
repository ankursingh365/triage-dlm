#!/usr/bin/env python3
"""
Phase C - trajectory generation
===============================

    modal run --detach modal_app.py::run --script step11_phasec_generate.py

A100, about 6.3 GPU-hours and $13.19, in a single run.

**The --detach is not optional.** Without it Modal creates an ephemeral App
that it stops the moment the client disconnects - and the client is the process
on your laptop. A sleeping machine, a closed lid or dropped wifi kills the GPU
job. That already happened once to this exact script:

    socket.gaierror: [Errno 11001] getaddrinfo failed

which is the laptop's DNS dying in sleep mode, not a fault in the container.
With --detach the job runs to completion regardless; watch it at modal.com/apps
or with `modal app logs <app-id>`.

WHAT THIS PRODUCES
==================
The trajectories every remaining step reads:

    triviaqa    3,750 questions
    hotpotqa    7,500 questions
    ----------------------------
                11,250

CommonsenseQA is deliberately NOT here. At 11% error with 73% of its wrong
answers degenerate, it cannot currently serve as the negative control; the
open-ended-prompt fix has to be tested first. Adding it later costs one more
run, because this script resumes rather than restarting.

WHERE THE SAMPLE SIZE COMES FROM
================================
Step 10d, applying rules fitted to 59 hand-labelled trajectories (two blind
annotators, Cohen's kappa 0.882):

    dataset     usable%   rarest mode      share   questions
    triviaqa       39%    interleaving     10.3%      3,750
    hotpotqa       41%    interleaving      4.9%      7,500

"Usable" excludes echo, degenerate and untestable - 43% of wrong answers turn
out not to be hallucinations at all, but the model restating its input or
running out of budget. Sizing on the raw error rate would have over-counted
the available data by nearly half.

**This is a conservative estimate**, and deliberately so. Step 10d's echo
detector over-fires on comparison questions ("which of A or B..."), where the
correct answer names something the question already listed and therefore
overlaps it completely:

    'The Rescuers was released earlier.'   overlap 1.00, 9 tokens   -> real answer
    'Roger O. Egeberg was Assistant...'    overlap 1.00, 40 tokens  -> real echo

It ate 3 locked-in and 1 interleaving case out of 59. The error runs in one
direction only: it UNDER-counts the modes Phase C needs, so the true
requirement is smaller than 11,250, not larger. Generating this many is safe.

Trajectories are classifier-independent. Whatever Step 17 decides the rules
should be, these files are re-classified for free - which is why generating
before the taxonomy is final costs nothing.

WHY IT STOPS EARLY AND ASKS TO BE RE-RUN
========================================
It should not need to any more. Modal's real ceiling is 24 hours, `modal_app.py`
now requests it, and 6.3 hours fits inside one run. The wall-clock check that
remains is a backstop at 20 hours, not a planned split.

The resume machinery stays regardless, because it is what makes an interrupted
run cheap rather than catastrophic - and one run has already been interrupted.
`modal_app.py` also commits the Volumes every five minutes now, so at most a
few minutes of generation can be lost to a crash instead of the whole run.

Resume is by file existence. Each question writes one `.npz` named for its id,
so a re-run skips what already exists and continues. That also makes a crash
cost at most one question.

The cache path includes `config.RUN_TAG`, so changing any decoder setting
starts a fresh directory instead of silently mixing configurations - a mistake
this project has already made once (see `claude/decoder-config-decision.md`).
Every loaded trajectory is additionally checked against config.py, because a
path is a convention and a check is a check.

ON BATCHING
===========
Generation runs one question at a time. Batching would cut the wall clock
substantially, but `generate_with_logging` assumes a batch of one throughout -
the mask index, the confidence freeze and the per-round logging all index
`[0]`. Rewriting that for batches is a real change to the most correctness-
critical file in the project, to save roughly $8. Not worth it here. If Phase D
needs 50,000 questions, revisit it then, and re-run the Step 6 reconstruction
check afterwards.
"""

import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

TRAJ_ROOT = config.TRAJ_DIR / "phasec" / config.RUN_TAG
REPORT_PATH = config.OUT_DIR / "step11_phasec_progress.txt"
CSV_PATH = config.TAB_DIR / "step11_phasec_answers.csv"

# (dataset, questions wanted). CommonsenseQA held back - see the docstring.
TARGETS = (("triviaqa", 3750), ("hotpotqa", 7500))

# Safety stop. Modal's hard ceiling on one Function call is 24 hours, and
# modal_app.py now asks for the full 24, so this budget is a backstop rather
# than the thing that shapes the run: 11,250 questions at ~2 s each is 6.3
# hours and finishes in ONE invocation.
#
# The earlier value here was 3.4 hours, chosen against a 4-hour timeout that
# was itself a guess. That mistake split a job that never needed splitting.
TIME_BUDGET_S = 20 * 3600
SEC_PER_Q = 2.01          # measured, A100, g64s64b32
USD_PER_HOUR = 2.10
PROGRESS_EVERY = 100

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def check_config(traj, path) -> None:
    """Refuse a cached trajectory generated under different decoder settings."""
    want = (config.GEN_LENGTH, config.DENOISING_STEPS, config.BLOCK_LENGTH)
    got = (int(traj.gen_length), int(traj.steps), int(traj.block_length))
    if got != want:
        raise SystemExit(
            f"\nSTOPPED - {path}\n"
            f"  config.py : gen {want[0]} steps {want[1]} block {want[2]}\n"
            f"  this file : gen {got[0]} steps {got[1]} block {got[2]}\n"
            f"  Delete that directory or restore the old settings.\n")


def main() -> None:
    t_start = time.time()
    say("=" * 78)
    say("  TRIAGE - Phase C generation")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  config   : gen {config.GEN_LENGTH}  steps {config.DENOISING_STEPS}"
        f"  block {config.BLOCK_LENGTH}   (tag {config.RUN_TAG})")
    say(f"  target   : " + ", ".join(f"{d} {n:,}" for d, n in TARGETS))
    say(f"  writing  : {TRAJ_ROOT}")

    # ---- what is already on disk -----------------------------------------
    say("")
    say("  Selecting questions and checking what already exists...")
    todo, done_before = [], 0
    for dataset, want in TARGETS:
        recs = data.load_records(dataset, n=want * 2, seed=config.SEED)
        kept, _exc, _rep = data_quality.clean_records(recs)
        chosen = kept[:want]
        if len(chosen) < want:
            say(f"  WARNING {dataset}: only {len(chosen):,} survived quality "
                f"filtering, wanted {want:,}")
        out_dir = TRAJ_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in chosen:
            path = out_dir / f"{rec.qid}.npz"
            if path.exists():
                done_before += 1
            else:
                todo.append((dataset, rec, path))

    total = done_before + len(todo)
    say(f"  {total:,} questions selected")
    say(f"  {done_before:,} already generated, {len(todo):,} remaining")

    if not todo:
        say("")
        say("=" * 78)
        say("  DONE - every trajectory exists. Nothing to generate.")
        say(f"  {TRAJ_ROOT}")
        say("=" * 78)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        return

    est_h = len(todo) * SEC_PER_Q / 3600
    say(f"  estimate : {est_h:.1f} GPU-hours, ${est_h * USD_PER_HOUR:.2f} "
        f"at {SEC_PER_Q:.2f} s/question")
    if est_h * 3600 > TIME_BUDGET_S:
        say(f"  This exceeds one run's {TIME_BUDGET_S/3600:.1f}h budget, so it "
            f"will stop partway and ask to be re-run.")

    # ---- model ------------------------------------------------------------
    say("")
    model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
        config.MODEL_LLADA, log=say)
    mask_id = model_utils.resolve_mask_id(tokenizer, config.MODEL_LLADA)
    say(f"  loaded   : {cfg_label}   mask_id {mask_id}")

    # ---- generate ---------------------------------------------------------
    say("")
    say("=" * 78)
    say("  GENERATING")
    say("")
    rows, n_done, stopped_early = [], 0, False
    t_gen = time.time()

    for dataset, rec, path in todo:
        if time.time() - t_start > TIME_BUDGET_S:
            stopped_early = True
            break

        prompt_ids = data.build_prompt_ids(tokenizer, rec)
        answer, traj = logging_patch.generate_with_logging(
            model, tokenizer, prompt_ids,
            gen_length=config.GEN_LENGTH,
            steps=config.DENOISING_STEPS,
            block_length=config.BLOCK_LENGTH,
            temperature=0.0,
            question=rec.question, question_id=rec.qid,
            quant_config=cfg_label, seed=config.SEED, mask_id=mask_id,
        )
        check_config(traj, path)
        traj.save(path)
        rows.append(dict(dataset=dataset, qid=rec.qid,
                         question=rec.question[:200], answer=answer[:300],
                         gold=" | ".join(rec.gold_answers[:5])[:200]))
        n_done += 1

        if n_done % PROGRESS_EVERY == 0:
            elapsed = time.time() - t_gen
            rate = elapsed / n_done
            left = len(todo) - n_done
            eta = timedelta(seconds=int(rate * left))
            pct = (done_before + n_done) / total
            say(f"    {done_before + n_done:6,} / {total:,}  ({pct:5.1%})   "
                f"{rate:.2f} s/q   this run ${elapsed/3600*USD_PER_HOUR:5.2f}   "
                f"eta {eta}")

    # ---- answers CSV, appended so re-runs accumulate ----------------------
    if rows:
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not CSV_PATH.exists()
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    # ---- summary ----------------------------------------------------------
    elapsed = time.time() - t_gen
    total_done = done_before + n_done
    say("")
    say("=" * 78)
    say("  THIS RUN")
    say(f"    generated   : {n_done:,}")
    say(f"    wall clock  : {elapsed/3600:.2f} h")
    if n_done:
        say(f"    rate        : {elapsed/n_done:.2f} s/question")
    say(f"    cost        : ${elapsed/3600*USD_PER_HOUR:.2f}")
    say("")
    say("  OVERALL")
    say(f"    complete    : {total_done:,} / {total:,}  ({total_done/total:.1%})")
    say("")

    if stopped_early or total_done < total:
        left = total - total_done
        say("=" * 78)
        say(f"  NOT FINISHED - {left:,} questions remain "
            f"(about {left*SEC_PER_Q/3600:.1f}h, ${left*SEC_PER_Q/3600*USD_PER_HOUR:.2f})")
        say("")
        say("  Stopped before the 4-hour Modal timeout so the Volume commit")
        say("  would run. Nothing is lost. Run exactly the same command again:")
        say("")
        say("      modal run modal_app.py::run --script step11_phasec_generate.py")
        say("")
        say("  It skips everything already on disk and continues.")
        say("=" * 78)
    else:
        say("=" * 78)
        say("  DONE - Phase C generation complete.")
        say(f"    trajectories : {TRAJ_ROOT}")
        say(f"    answers CSV  : {CSV_PATH}")
        say("")
        say("  Next: re-score with the Qwen3 judge, then apply the Step 10d")
        say("  rules to get the real per-failure-mode split on 11,250 questions")
        say("  instead of 150.")
        say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()