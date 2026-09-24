#!/usr/bin/env python3
"""
Step 14 - LOCK: reconstruction check on the real Phase C data
==============================================================

    modal run modal_app.py::run_cpu --script step14_reconstruction_lock.py

CPU only. About 25-40 minutes, free. No model, no GPU - only the tokenizer,
which is already on the cache Volume.

    --limit 500     smoke run, 500 trajectories per dataset

WHY THIS IS RE-RUN
==================
Step 6 passed this gate on 50 purpose-built questions and the project moved on.
Then the decoder configuration changed:

    steps  32 -> 64        block_length  64 -> 32

That change rewrote the reveal schedule, which is precisely what
`reconstruct_final()` reads. The gate has not been run since, so it is
currently **open on every trajectory the project actually owns**. The charter
is blunt about what that means:

    "If this fails, every downstream number in the project is garbage."

WHY IT RUNS ON ALL 11,120 AND NOT 50
====================================
`reconstruct_final()` needs only `pred_ids` and `mask_state`. Both are stored
in the .npz. Nothing has to be regenerated and no model has to be loaded, so
the sampled version of this gate is strictly worse than the exhaustive one for
the same money - which is none.

    rebuilt[i] = pred_ids[ revealed_at()[i], i ]        must equal final_ids

`revealed_at()` is derived from `mask_state`, and `final_ids` was captured from
the model's actual output tensor, so the two sides come from different places.
That is what makes this a test rather than a tautology. `committed_ids()` is
derived, never stored, for the same reason.

Both halves of Step 6's check are kept: array equality on token ids, and
character-for-character equality of the decoded strings. Id equality is the
stricter of the two - two different id sequences can decode to the same text,
but not the reverse - and the charter names the text one, so both are reported.

THE TWO THINGS THIS PRODUCES BESIDES PASS/FAIL
==============================================
Step 6's docstring lists them, and both are needed by the next step.

**1. EOS contamination, on real data.** `src/features.py` (step 19) excludes
EOS and padding positions. That exclusion is currently justified by a rule in
`content_mask()` and a single observation from step 2c. Part B measures the gap
between mean entropy over all masked positions and over content positions only,
across 11,120 questions. If the gap is small the exclusion is cosmetic; if it
is large, every feature averaged over all positions is measuring the padding
schedule.

**2. A regret and flip baseline, per failure mode.** Part C. Descriptive only -
it is NOT an AUROC and NOT the experiment, which is steps 19-23. But it is the
first look at whether the thesis has a pulse: a locked-in error is defined by
the absence of hesitation, so its flip and regret counts should sit below the
other modes. If they do not, that is worth knowing before writing 25 features
to measure it.
"""

import csv
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

PHASEC_ROOT = config.TRAJ_DIR / "phasec" / config.RUN_TAG
MODES_CSV = config.TAB_DIR / "step12c_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step14_reconstruction_lock.txt"
CSV_PATH = config.TAB_DIR / "step14_reconstruction.csv"

DATASETS = ("triviaqa", "hotpotqa")
MODES = ("locked_in", "interleaving", "inconsistent", "echo", "degenerate",
         "untestable")
PROGRESS_EVERY = 1000

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def check_config(traj, path) -> None:
    """Refuse a trajectory generated under different decoder settings."""
    want = (config.GEN_LENGTH, config.DENOISING_STEPS, config.BLOCK_LENGTH)
    got = (int(traj.gen_length), int(traj.steps), int(traj.block_length))
    if got != want:
        raise SystemExit(
            f"\nSTOPPED - {path}\n"
            f"  config.py : gen {want[0]} steps {want[1]} block {want[2]}\n"
            f"  this file : gen {got[0]} steps {got[1]} block {got[2]}\n")


def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 14: reconstruction LOCK on all of Phase C")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  config : gen {config.GEN_LENGTH}  steps {config.DENOISING_STEPS}"
        f"  block {config.BLOCK_LENGTH}   (tag {config.RUN_TAG})")
    say(f"  data   : {PHASEC_ROOT}")
    if LIMIT:
        say(f"  SMOKE RUN - {LIMIT} per dataset. A smoke run does NOT close")
        say("  the gate; the gate is exhaustive or it is not a gate.")

    if not PHASEC_ROOT.exists():
        say(f"\nMissing {PHASEC_ROOT}. Run step11 first.")
        sys.exit(1)

    modes = {}
    if MODES_CSV.exists():
        with open(MODES_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                modes[(r["dataset"], r["qid"])] = (
                    "correct" if str(r["correct"]).lower() == "true"
                    else r["mode"])
        say(f"  modes  : {len(modes):,} labels joined from step12c")
    else:
        say("  modes  : step12c CSV absent - Part C will be skipped")

    say("")
    say("  Loading tokenizer (no model, no GPU)...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_LLADA,
                                        trust_remote_code=True)
    say(f"  vocab {len(tok):,}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("A. THE GATE")
    say("")
    rows, failures, n = [], [], 0
    for ds in DATASETS:
        sub = PHASEC_ROOT / ds
        if not sub.exists():
            say(f"  WARNING {ds}: {sub} missing, skipping")
            continue
        paths = sorted(sub.glob("*.npz"))
        if LIMIT:
            paths = paths[:LIMIT]
        say(f"  {ds:<10} {len(paths):,} trajectories")
        for path in paths:
            traj = logging_patch.Trajectory.load(path)
            check_config(traj, path)

            rebuilt = traj.reconstruct_final()
            ids_ok = bool(np.array_equal(rebuilt, traj.final_ids))
            text_ok = (tok.decode(rebuilt.tolist())
                       == tok.decode(traj.final_ids.tolist()))
            ok = ids_ok and text_ok
            if not ok:
                failures.append((ds, path.stem, ids_ok, text_ok))

            masked = traj.mask_state == 1
            content = traj.content_mask() & masked
            eos_frac = (1.0 - content.sum() / masked.sum()) if masked.any() \
                else float("nan")
            h_all = float(traj.entropy[masked].mean()) if masked.any() \
                else float("nan")
            h_cont = float(traj.entropy[content].mean()) if content.any() \
                else float("nan")
            ans_tokens = int(sum(1 for t in traj.final_ids.tolist()
                                 if t not in (traj.eos_id, traj.mask_id)))

            rows.append(dict(
                dataset=ds, qid=path.stem, recon_ok=ok, ids_ok=ids_ok,
                text_ok=text_ok, ans_tokens=ans_tokens, eos_frac=eos_frac,
                h_all=h_all, h_content=h_cont,
                flips=int(traj.flips_per_round().sum()),
                regret=int(traj.regret_per_round().sum()),
                settle_median=int(np.median(traj.revealed_at())),
                mode=modes.get((ds, path.stem), "")))

            n += 1
            if n % PROGRESS_EVERY == 0:
                el = time.time() - t0
                eta = timedelta(seconds=int(el / n * (11120 - n)))
                say(f"    {n:6,} checked   {len(failures)} failures   "
                    f"[{timedelta(seconds=int(el))} elapsed, eta {eta}]")

    n_ok = sum(1 for r in rows if r["recon_ok"])
    say("")
    say(f"  checked      {len(rows):,}")
    say(f"  reconstruct  {n_ok:,}")
    say(f"  failures     {len(failures):,}")
    say("")
    if failures:
        say("  FAILURES (first 20):")
        for ds, qid, i_ok, t_ok in failures[:20]:
            say(f"    {ds}/{qid}   ids_ok={i_ok}  text_ok={t_ok}")
        say("")
        say("=" * 78)
        say("  GATE FAILED. STOP.")
        say("")
        say("  The log does not describe what the model did, so every number")
        say("  computed from these trajectories is garbage - the Phase C split,")
        say("  the failure modes, the power calculation, all of it. Do not")
        say("  write features.py against this data. The trajectories must be")
        say("  regenerated once logging_patch is fixed.")
        say("=" * 78)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  GATE PASSED - every trajectory rebuilds exactly, ids and text.")
    if LIMIT:
        say("  ...on a SMOKE SUBSET. Re-run without --limit to close the gate.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. EOS CONTAMINATION - the number features.py needs")
    say("")

    def stat(key):
        v = np.array([r[key] for r in rows if r[key] == r[key]])
        return v.mean(), np.median(v), v.min(), v.max()

    say("  quantity                        mean   median      min      max")
    say("  ----------------------------  ------  -------  -------  -------")
    for key, label in (("eos_frac", "padding fraction"),
                       ("h_all", "entropy, ALL masked posns"),
                       ("h_content", "entropy, CONTENT only"),
                       ("ans_tokens", "answer length, tokens")):
        m, md, lo, hi = stat(key)
        fmt = "{:6.1%}  {:7.1%}  {:7.1%}  {:7.1%}" if key == "eos_frac" \
            else "{:6.3f}  {:7.3f}  {:7.3f}  {:7.3f}"
        say(f"  {label:<28}  " + fmt.format(m, md, lo, hi))

    ha, _b, _c, _d = stat("h_all")
    hc, *_ = stat("h_content")
    say("")
    say(f"  Averaging entropy over ALL masked positions gives {ha:.3f}.")
    say(f"  Averaging over CONTENT positions only gives      {hc:.3f}.")
    ratio = hc / ha if ha else float("nan")
    say(f"  Content entropy is {ratio:.2f}x the all-positions figure.")
    say("")
    if abs(ratio - 1.0) < 0.10:
        say("  The two are within 10% of each other. The EOS exclusion in")
        say("  content_mask() is defensible but not load-bearing - say so in")
        say("  the paper rather than implying it rescued the features.")
    else:
        say("  These are materially different numbers. Any feature averaged")
        say("  over all masked positions is largely measuring the padding")
        say("  schedule, not the model. The exclusion in content_mask() is")
        say("  load-bearing and step 19 must apply it to EVERY positional")
        say("  feature, not only entropy.")

    # =======================================================================
    if modes:
        say("")
        say("=" * 78)
        say("C. FLIPS AND REGRET BY FAILURE MODE - descriptive, NOT the result")
        say("")
        say("  A locked-in error is DEFINED by the absence of hesitation, so its")
        say("  flip and regret counts should sit below the other wrong modes.")
        say("  This is a first look, not an AUROC. Steps 19-23 are the test.")
        say("")
        by = defaultdict(list)
        for r in rows:
            if r["mode"]:
                by[r["mode"]].append(r)
        say("  mode               n     flips    regret   settle   ans_toks")
        say("  --------------  ------  -------  -------  -------  ---------")
        for m in ("correct",) + MODES:
            rs = by.get(m, [])
            if not rs:
                continue
            say(f"  {m:<14}  {len(rs):6,}  "
                f"{np.mean([r['flips'] for r in rs]):7.1f}  "
                f"{np.mean([r['regret'] for r in rs]):7.1f}  "
                f"{np.mean([r['settle_median'] for r in rs]):7.1f}  "
                f"{np.mean([r['ans_tokens'] for r in rs]):9.1f}")
        li = by.get("locked_in", [])
        ic = by.get("inconsistent", [])
        if li and ic:
            fl_li = np.mean([r["flips"] for r in li])
            fl_ic = np.mean([r["flips"] for r in ic])
            say("")
            say(f"  locked_in flips {fl_li:.1f} vs inconsistent {fl_ic:.1f}"
                f"   ({fl_li - fl_ic:+.1f})")
            if fl_li < fl_ic:
                say("  Direction matches the hypothesis. Not evidence yet -")
                say("  flips correlate with answer length, and the two modes")
                say("  differ in length above. Step 19 must control for it.")
            else:
                say("  Direction is AGAINST the hypothesis. Worth understanding")
                say("  before step 19 rather than after step 23.")

    # ---- CSV --------------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    say("")
    say("=" * 78)
    say(f"  rows       : {len(rows):,}")
    say(f"  CSV        : {CSV_PATH}")
    say(f"  wall clock : {timedelta(seconds=int(time.time() - t0))}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
