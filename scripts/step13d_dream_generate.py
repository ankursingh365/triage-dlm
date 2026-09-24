#!/usr/bin/env python3
"""
Step 13d - Dream-7B at scale. Completes plan step 13, and Phase C.
===================================================================

    modal run --detach modal_app.py::run --script step13d_dream_generate.py

A100. About 6 hours and $12.65 at the measured 1.76 s/question.
**--detach is not optional** - a sleeping laptop kills an attached run.

WHAT THIS RESTS ON
==================
Step 13c's reconstruction LOCK passed 50 of 50 on Dream's own trajectories, so
the port describes what Dream actually did. That gate is the only reason this
script is allowed to spend six GPU-hours.

    rounds recorded    64 of 64 per question
    adaptive schedule  0-5 positions per round, mean 1.00, 37% of rounds
                       reveal nothing at all
    confidence freeze  yes, as in LLaDA
    regret             mean 0.97/round, 19% of rounds at zero, max 4 -
                       it VARIES, so Finding B transfers

THE SAME QUESTIONS AS LLaDA, DELIBERATELY
=========================================
A replication arm that asks different questions is not a replication. The
record selection here reproduces step 11's and step 12b's exactly - same
loader, same `n`, same seed, same quality filter, same slice:

    triviaqa        load(n=7500,  seed) -> clean -> [:3750]
    hotpotqa        load(n=15000, seed) -> clean -> [:7500]   (7,370 survive)
    commonsenseqa   load(n=None,  seed) -> clean -> all       (1,197 survive)

CommonsenseQA uses the `brief` prompt adopted in step 12b, imported from
step12a rather than retyped. A different prompt would make the two models'
CommonsenseQA incomparable, which is the one thing the negative control cannot
afford.

THE CSV BUG THIS FIXES
======================
Step 11 accumulated its answers in memory and wrote the CSV once, at the end.
Its first detached run was killed at 9,410 questions and every one of those
rows was lost - step 12 had to decode answers back out of the .npz files
instead. Here the CSV is flushed every `CSV_EVERY` questions, so a killed
container costs at most that many rows.

RESUME
======
By file existence, as in step 11: one .npz per question, re-running skips what
exists. Every loaded trajectory is additionally checked against config, because
a path is a convention and a check is a check.
"""

import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
DREAM_TAG = f"dream_g{config.GEN_LENGTH}s{config.DENOISING_STEPS}_origin"
TRAJ_ROOT = config.TRAJ_DIR / "phasec_dream" / DREAM_TAG
LOCK_DIR = config.TRAJ_DIR / "dream_lock" / DREAM_TAG
REPORT_PATH = config.OUT_DIR / "step13d_dream_progress.txt"
CSV_PATH = config.TAB_DIR / "step13d_dream_answers.csv"

ALG = "origin"
CSQA_FMT = "brief"
TARGETS = (("triviaqa", 3750), ("hotpotqa", 7500), ("commonsenseqa", None))

SEC_PER_Q = 1.76          # measured in step 13c, 50 questions, after warm-up
USD_PER_HOUR = 2.10
TIME_BUDGET_S = 20 * 3600
PROGRESS_EVERY = 250
CSV_EVERY = 250

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def check_config(traj, path) -> None:
    """Refuse a cached trajectory generated under different settings.

    Dream has no blocks, so `block_length` is `gen_length` by convention (see
    step13c). Only gen and steps are meaningful here.
    """
    want = (config.GEN_LENGTH, config.DENOISING_STEPS)
    got = (int(traj.gen_length), int(traj.steps))
    if got != want:
        raise SystemExit(
            f"\nSTOPPED - {path}\n"
            f"  config.py : gen {want[0]} steps {want[1]}\n"
            f"  this file : gen {got[0]} steps {got[1]}\n")


def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 13d: Dream-7B at scale  (completes Phase C)")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  model : {config.MODEL_DREAM}")
    say(f"  alg   : {ALG}   tag {DREAM_TAG}")
    say(f"  csqa  : prompt format {CSQA_FMT!r}, as adopted in step 12b")
    say(f"  write : {TRAJ_ROOT}")
    if LIMIT:
        say(f"  LIMIT {LIMIT} per dataset - a partial run, not plan step 13")

    if not LOCK_DIR.exists():
        say("")
        say(f"STOPPED - {LOCK_DIR} is missing, so step 13c's reconstruction")
        say("LOCK has not been run against this configuration. No Dream data")
        say("is generated at scale until that gate passes.")
        sys.exit(1)
    say(f"  gate  : {len(list(LOCK_DIR.glob('*.npz')))} locked trajectories "
        f"from step 13c")

    # ---- the port, and the CommonsenseQA prompt, from their own scripts ----
    import importlib.util

    def _load(name, filename):
        spec = importlib.util.spec_from_file_location(name, HERE / filename)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    port = _load("dream_port", "step13c_dream_port.py")
    sweep = _load("csqa_sweep", "step12a_csqa_prompt_sweep.py")
    say(f"  port  : step13c.generate_with_logging_dream")
    say(f"  csqa prompt: {sweep.FORMATS[CSQA_FMT]!r}")

    # ---- selection, reproducing step 11 and step 12b exactly --------------
    say("")
    say("  Selecting questions and checking what already exists...")
    todo, done_before, total = [], 0, 0
    for dataset, want in TARGETS:
        n_load = None if want is None else want * 2
        recs = data.load_records(dataset, n=n_load, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        chosen = kept if want is None else kept[:want]
        if want is not None and len(chosen) < want:
            say(f"  WARNING {dataset}: {len(chosen):,} survived filtering, "
                f"wanted {want:,}")
        if LIMIT:
            chosen = chosen[:LIMIT]
        out_dir = TRAJ_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in chosen:
            path = out_dir / f"{rec.qid}.npz"
            total += 1
            if path.exists():
                done_before += 1
            else:
                todo.append((dataset, rec, path))
        say(f"    {dataset:<14} {len(chosen):6,} selected")

    say(f"  {total:,} total, {done_before:,} already on disk, "
        f"{len(todo):,} to generate")
    if not todo:
        say("")
        say("=" * 78)
        say("  DONE - every Dream trajectory exists.")
        say("=" * 78)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        return
    est = len(todo) * SEC_PER_Q / 3600
    say(f"  estimate {est:.1f} GPU-hours, ${est*USD_PER_HOUR:.2f} "
        f"at {SEC_PER_Q} s/question")

    # ---- model ------------------------------------------------------------
    say("")
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.MODEL_DREAM, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda").eval()
    MASK = int(tok.mask_token_id)
    say(f"  loaded  mask_id {MASK}  eos_id {tok.eos_token_id}")

    def build_ids(rec, dataset):
        if dataset == "commonsenseqa":
            text = sweep.build_text(CSQA_FMT, rec)
        else:
            text = data.build_prompt_text(rec.question, rec.choices)
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = tok(text, return_tensors="pt")["input_ids"]
        return ids.to(model.device)

    # ---- generate ---------------------------------------------------------
    say("")
    say("=" * 78)
    say("  GENERATING")
    say("")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_csv = not CSV_PATH.exists()
    buf, n_done, stopped, hook_errs = [], 0, False, 0

    def flush():
        nonlocal buf, new_csv
        if not buf:
            return
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(buf[0].keys()))
            if new_csv:
                w.writeheader()
                new_csv = False
            w.writerows(buf)
        buf = []

    for dataset, rec, path in todo:
        if time.time() - t0 > TIME_BUDGET_S:
            stopped = True
            break
        answer, traj, seen, errs = port.generate_with_logging_dream(
            model, tok, build_ids(rec, dataset),
            gen_length=config.GEN_LENGTH, steps=config.DENOISING_STEPS,
            alg=ALG, mask_id=MASK, question=rec.question,
            question_id=rec.qid, seed=config.SEED)
        hook_errs += len(errs)
        check_config(traj, path)
        traj.save(path)
        buf.append(dict(dataset=dataset, qid=rec.qid,
                        question=rec.question[:200], answer=answer[:300],
                        gold=" | ".join(rec.gold_answers[:5])[:200],
                        rounds=len(seen)))
        n_done += 1
        if n_done % CSV_EVERY == 0:
            flush()
        if n_done % PROGRESS_EVERY == 0:
            el = time.time() - t0
            rate = el / n_done
            eta = timedelta(seconds=int(rate * (len(todo) - n_done)))
            say(f"    {done_before+n_done:6,} / {total:,}  "
                f"({(done_before+n_done)/total:5.1%})   {rate:.2f} s/q   "
                f"${el/3600*USD_PER_HOUR:5.2f}   eta {eta}")
    flush()

    # ---- summary ----------------------------------------------------------
    el = time.time() - t0
    done = done_before + n_done
    say("")
    say("=" * 78)
    say("  THIS RUN")
    say(f"    generated  : {n_done:,}")
    say(f"    wall clock : {el/3600:.2f} h")
    if n_done:
        say(f"    rate       : {el/n_done:.2f} s/question")
    say(f"    cost       : ${el/3600*USD_PER_HOUR:.2f}")
    if hook_errs:
        say(f"    HOOK ERRORS: {hook_errs}  <- our bug, not Dream's. The")
        say("                 affected trajectories are incomplete.")
    say("")
    say("  OVERALL")
    say(f"    complete   : {done:,} / {total:,}  ({done/total:.1%})")
    say("")
    if stopped or done < total:
        left = total - done
        say("=" * 78)
        say(f"  NOT FINISHED - {left:,} remain "
            f"(~{left*SEC_PER_Q/3600:.1f}h, ${left*SEC_PER_Q/3600*USD_PER_HOUR:.2f})")
        say("  Run the same command again; it skips what is on disk.")
        say("=" * 78)
    else:
        say("=" * 78)
        say("  DONE - Dream-7B generated on all three datasets.")
        say(f"    trajectories : {TRAJ_ROOT}")
        say(f"    answers CSV  : {CSV_PATH}")
        say("")
        say("  PLAN STEP 13 COMPLETE.  ***PHASE C IS NOW COMPLETE.***")
        say("    11 generation at scale        done")
        say("    12 LLaDA-8B x 3 datasets      done")
        say("    13 Dream-7B x 3 datasets      done")
        say("    14 LOCK reconstruction        passed, 11,120/11,120")
        say("")
        say("  Next: Phase E, step 19 - src/features.py. Two constraints are")
        say("  already fixed by measurement and must be honoured there:")
        say("    - padding is 82% of masked positions, so content_mask() is")
        say("      load-bearing on EVERY positional feature (step 14 part B)")
        say("    - LLaDA's reveal schedule is fixed and Dream's is adaptive,")
        say("      so settle-round features cannot be shared between them")
        say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
