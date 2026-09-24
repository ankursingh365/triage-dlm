#!/usr/bin/env python3
"""
Step 12c - the float-boundary bug, and the echo rule's final operating point
=============================================================================

    modal run modal_app.py::run_cpu --script step12c_apply_operating_point.py

CPU only, about two minutes, free. Reads two CSVs and no trajectories.
Supersedes step12b entirely - delete that file once this runs clean.

THE BUG STEP 12b's CHECKSUM CAUGHT
==================================
Step 12b re-derived Step 12's split from the stored features and got different
numbers. Every category moved except `degenerate`:

    locked_in     1,010 -> 931        echo   2,673 -> 3,013
    interleaving    216 -> 192        untestable 134 -> 121
    inconsistent  3,834 -> 3,610

The cause is one line. Step 12 fitted its thresholds by grid search over

    np.arange(0.40, 1.01, 0.01)

and the element that prints as `0.75` is **not** 0.75:

    repr(value)          0.7500000000000003
    formatted "%.2f"     0.75
    value > 0.75         True

`question_overlap` is a ratio of small integers, so it lands on exactly 3/4
constantly - any answer with 4, 8, 12, 16 ... content words can. For those rows

    step12   0.75 >= 0.7500000000000003   ->  False   (not an echo)
    step12b  0.75 >= 0.75                 ->  True    (echo)

340 of 8,222 wrong answers sit exactly on that boundary. Floating-point drift
of 3.3e-16 moved 4% of the dataset between categories.

**This is not a rounding curiosity, it is a reproducibility defect.** A
threshold reported in a paper as 0.75 that is actually 0.7500000000000003
cannot be reproduced by anyone reading the paper, including its authors. Part A
proves the diagnosis by reclassifying with the drifted value and showing it
reproduces Step 12 exactly; everything after that uses exact decimal grids
(`i/100`), where the value that prints as 0.75 *is* 0.75.

It also invalidates a smaller claim. Step 12's report listed only one
interleaving case eaten by the echo rule. Two more were sitting exactly on the
boundary and escaped by 3.3e-16 rather than by any property of the data, so the
real count was three of six.

THE OPERATING POINT
===================
Step 12b's curve, computed at a clean 0.75, is correct as printed:

    run cut   X caught   signal lost (L or I)
    >= 1         18            6
    >= 6         12            1
    >= 7         11            0

The declared criterion - maximise X recall subject to destroying at most one
hand-labelled locked-in or interleaving case - selects `run >= 6`.

**What that costs, stated plainly.** Seven of nineteen hand-labelled X survive
the filter, and five of them land in `locked_in` - contamination inside the
paper's headline mode. That is a real price and Part D prints it rather than
burying it.

It is still the right trade, for one reason that is checkable rather than
rhetorical: the surviving X are mostly the kind the judge removes.

    'The Leonberger is considered a giant dog breed.'   run 5

is not an echo at all. It is a CORRECT answer that whole-word matching scored
wrong, and Step 13 will score it correct and drop it from the wrong-set
entirely. A locked-in case destroyed by an over-firing echo rule has no such
second chance - it is gone from every table in the paper.

So the residual splits into a part with an owner and a part without, and Part D
reports both. What is left after the judge - genuine copies like "Billie Jean
King's maiden name was Billie Jean King" that survive at run >= 6 - is reported
as a contamination rate in the data section. That is the honest end state.

WHY THE TUNING STOPS HERE
=========================
Three thresholds are already being fitted on 59 hand labels. A fourth feature
would be fitted on the same 59 and could not be validated on anything else.
Every count below also moves when the judge changes the wrong-set, so tuning
further against these particular numbers is fitting to a target that is about
to shift. Step 13 first; revisit the taxonomy afterwards, with a clean
wrong-set and more labels if the residual justifies them.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

CAL_CSV = config.TAB_DIR / "step10d_final_modes.csv"
FULL_CSV = config.TAB_DIR / "step12_phasec_modes.csv"
REPORT_PATH = config.OUT_DIR / "step12c_final_modes.txt"
OUT_CSV = config.TAB_DIR / "step12c_final_modes.csv"

# Exact decimal grids. `i/100` is the same double as the literal `0.75`;
# `np.arange(0.40, 1.01, 0.01)` is not. Every threshold in this project is
# fitted from grids built this way from here on.
OVERLAP_GRID = [i / 100 for i in range(40, 101)]
STABLE_GRID = [i / 100 for i in range(5, 90)]
CANDS_GRID = [i / 10 for i in range(15, 55)]
RUN_GRID = list(range(1, 13))

# The value Step 12 actually compared against, reconstructed rather than
# pasted, so the demonstration cannot drift from the thing it demonstrates.
_ar = np.arange(0.40, 1.01, 0.01)
STEP12_E_CUT = float(_ar[int(np.argmin(np.abs(_ar - 0.75)))])
STEP12_CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=STEP12_E_CUT, r_cut=1)
STEP12_ORDER = "A"
STEP12_SPLIT = {"locked_in": 1010, "interleaving": 216, "inconsistent": 3834,
                "echo": 2673, "degenerate": 355, "untestable": 134}
STEP12_N_ROWS, STEP12_N_WRONG = 11120, 8222

MAX_SIGNAL_DESTROYED = 1
TARGET_PER_CELL = 150
DATASETS = ("triviaqa", "hotpotqa")

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def max_copied_run(answer: str, question: str) -> int:
    """Longest run of consecutive answer words running consecutively in the
    question. Stopwords kept - a copied span carries its function words."""
    a, q = normalise(answer).split(), normalise(question).split()
    if not a or not q:
        return 0
    prev, best = [0] * (len(q) + 1), 0
    for i in range(1, len(a) + 1):
        cur, ai = [0] * (len(q) + 1), a[i - 1]
        for j in range(1, len(q) + 1):
            if ai == q[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def as_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify(r, cuts, order="A") -> str:
    if r["truncated"]:
        return "degenerate"
    echo = r["q_overlap"] >= cuts["e_cut"] and r["copy_run"] >= cuts["r_cut"]
    if order == "A":
        if echo:
            return "echo"
        if not r["gold_testable"]:
            return "untestable"
        if r["gold_ever"]:
            return "interleaving"
    else:
        if r["gold_testable"] and r["gold_ever"]:
            return "interleaving"
        if echo:
            return "echo"
        if not r["gold_testable"]:
            return "untestable"
    if r["stable_top3"] >= cuts["s_cut"] and r["cands_top3"] <= cuts["c_cut"]:
        return "locked_in"
    return "inconsistent"


MAP = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
       "echo": "X", "degenerate": "X", "untestable": None}


def fit_locked(rows) -> tuple:
    """Grid-search (stable_top3, cands_top3) on hand-labelled L vs C.

    Same objective as Step 10d - balanced accuracy is defensible here, because
    L and C are both hallucination modes and confusing one for the other does
    not destroy a category. Only the echo rule needed a different criterion.
    """
    LC = [r for r in rows if r["hand"] in ("L", "C")]
    n_l = sum(1 for r in LC if r["hand"] == "L")
    n_c = len(LC) - n_l
    best = None
    for sc in STABLE_GRID:
        for cn in CANDS_GRID:
            tp = sum(1 for r in LC if r["hand"] == "L"
                     and r["stable_top3"] >= sc and r["cands_top3"] <= cn)
            fp = sum(1 for r in LC if r["hand"] == "C"
                     and r["stable_top3"] >= sc and r["cands_top3"] <= cn)
            if not n_l or not n_c:
                continue
            bal = 0.5 * (tp / n_l + (n_c - fp) / n_c)
            if best is None or bal > best[0]:
                best = (bal, sc, cn, tp, n_l - tp, fp, n_c - fp)
    return best


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 12c: float fix, and the final operating point")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    for p in (CAL_CSV, FULL_CSV):
        if not p.exists():
            say(f"\nMissing {p}. Run step10d and step12 first.")
            sys.exit(1)

    full = []
    with open(FULL_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            full.append(dict(
                dataset=r["dataset"], qid=r["qid"],
                correct=as_bool(r.get("correct")),
                q_overlap=as_float(r.get("q_overlap")),
                copy_run=int(as_float(r.get("copy_run"))),
                stable_top3=as_float(r.get("stable_top3")),
                cands_top3=as_float(r.get("cands_top3")),
                truncated=as_bool(r.get("truncated")),
                gold_ever=as_bool(r.get("gold_ever")),
                gold_testable=as_bool(r.get("gold_testable")),
                old_mode=r.get("mode", "")))
    wrong = [r for r in full if not r["correct"]]

    # =======================================================================
    # A - prove the diagnosis
    # =======================================================================
    say("")
    say("=" * 78)
    say("A. THE FLOAT BOUNDARY, PROVED")
    say("")
    say(f"  Step 12 fitted its echo cut from np.arange(0.40, 1.01, 0.01).")
    say(f"  The element that prints as 0.75 is actually {STEP12_E_CUT!r},")
    say(f"  which is larger than 0.75 by {STEP12_E_CUT - 0.75:.3e}.")
    say("")
    on_edge = sum(1 for r in wrong if abs(r["q_overlap"] - 0.75) < 1e-12)
    say(f"  {on_edge:,} of {len(wrong):,} wrong answers have question_overlap")
    say("  exactly 0.75 - it is a ratio of small integers, so 4, 8, 12, 16 ...")
    say("  content words all land on it. Those rows fall on opposite sides of")
    say("  the two thresholds.")
    say("")
    say("  Reclassifying with the DRIFTED value must reproduce Step 12 exactly.")
    say("  If it does, the CSV is lossless and the float is the sole cause:")
    say("")
    repro = Counter(classify(r, STEP12_CUTS, STEP12_ORDER) for r in wrong)
    all_ok = (len(full) == STEP12_N_ROWS and len(wrong) == STEP12_N_WRONG
              and all(repro[k] == v for k, v in STEP12_SPLIT.items()))
    say(f"    {'rows':<14} {len(full):6,}  "
        f"{'ok' if len(full) == STEP12_N_ROWS else 'MISMATCH'}")
    say(f"    {'wrong':<14} {len(wrong):6,}  "
        f"{'ok' if len(wrong) == STEP12_N_WRONG else 'MISMATCH'}")
    for k, v in STEP12_SPLIT.items():
        say(f"    {k:<14} {repro[k]:6,}  "
            f"{'ok' if repro[k] == v else f'MISMATCH (expected {v})'}")
    say("")
    if not all_ok:
        say("  STOPPED - the drifted value does NOT reproduce Step 12, so the")
        say("  float is not the whole story and something else is wrong too.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  All reproduce. The CSV is lossless; the float was the sole cause.")
    say("")
    clean = Counter(classify(r, dict(STEP12_CUTS, e_cut=0.75), STEP12_ORDER)
                    for r in wrong)
    say("  Same rule at an exact 0.75, which is what Step 12 meant to apply:")
    for k in STEP12_SPLIT:
        say(f"    {k:<14} {clean[k]:6,}   ({clean[k] - STEP12_SPLIT[k]:+,})")

    # =======================================================================
    # B - the calibration set, refitted on exact grids
    # =======================================================================
    say("")
    say("=" * 78)
    say("B. OPERATING CURVE, EXACT GRIDS")
    say("")
    cal = []
    with open(CAL_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cal.append(dict(
                dataset=r["dataset"], qid=r["qid"],
                hand=(r.get("hand") or "").strip() or None,
                question=r.get("question", ""), answer=r.get("answer", ""),
                q_overlap=as_float(r.get("q_overlap")),
                copy_run=max_copied_run(r.get("answer", ""), r.get("question", "")),
                stable_top3=as_float(r.get("stable_top3")),
                cands_top3=as_float(r.get("cands_top3")),
                n_content=int(as_float(r.get("n_content"))),
                truncated=as_bool(r.get("truncated")),
                gold_ever=as_bool(r.get("gold_ever")),
                gold_testable=as_bool(r.get("gold_testable"))))
    lab = [r for r in cal if r["hand"] in ("L", "I", "C", "X")]
    n_x = sum(1 for r in lab if r["hand"] == "X")
    say(f"  {len(cal)} calibration rows, {len(lab)} hand-labelled, {n_x} of them X")
    say("")
    say("   run cut   X caught   X missed   signal lost   C lost")
    say("   -------   --------   --------   -----------   ------")
    curve = []
    for run in RUN_GRID:
        fires = [r for r in lab if r["q_overlap"] >= 0.75 and r["copy_run"] >= run]
        caught = sum(1 for r in fires if r["hand"] == "X")
        sig = sum(1 for r in fires if r["hand"] in ("L", "I"))
        clost = sum(1 for r in fires if r["hand"] == "C")
        curve.append((run, caught, sig, clost))
        say(f"   >= {run:<5d}  {caught:8d}   {n_x - caught:8d}   {sig:11d}   "
            f"{clost:6d}")
        if caught == 0:
            break

    ok_pts = [c for c in curve if c[2] <= MAX_SIGNAL_DESTROYED]
    if not ok_pts:
        say("")
        say("  STOPPED - no span cut destroys at most "
            f"{MAX_SIGNAL_DESTROYED} signal case. The echo")
        say("  rule needs a different feature, not a different threshold.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    r_cut = max(ok_pts, key=lambda c: c[1])[0]
    chosen_caught = max(ok_pts, key=lambda c: c[1])[1]
    say("")
    say(f"  Criterion: maximise X recall subject to at most "
        f"{MAX_SIGNAL_DESTROYED} L/I destroyed.")
    say(f"  Selected: question_overlap >= 0.75 AND max_copied_run >= {r_cut}")

    fit = fit_locked(lab)
    bal, s_cut, c_cut, tp, fn, fp, tn = fit
    say("")
    say(f"  locked_in refitted on exact grids: stable_top3 >= {s_cut}  AND  "
        f"cands_top3 <= {c_cut}")
    say(f"  bal.acc {bal:.0%}  recall {tp/(tp+fn):.0%}  "
        f"specificity {tn/(fp+tn):.0%}")

    cuts = dict(s_cut=s_cut, c_cut=c_cut, e_cut=0.75, r_cut=r_cut)
    say("")
    for order in ("A", "B"):
        n_ok = sum(1 for r in lab if MAP.get(classify(r, cuts, order)) == r["hand"])
        say(f"  order {order} reproduces {n_ok}/{len(lab)} = {n_ok/len(lab):.0%}")
    order = max("AB", key=lambda o: sum(
        1 for r in lab if MAP.get(classify(r, cuts, o)) == r["hand"]))
    say(f"  Using order {order}.")
    say("")
    say("              hand L   hand I   hand C   hand X")
    for code in ("locked_in", "interleaving", "inconsistent", "echo",
                 "degenerate", "untestable"):
        cnts = [sum(1 for r in lab if classify(r, cuts, order) == code
                    and r["hand"] == h) for h in ("L", "I", "C", "X")]
        if sum(cnts):
            say(f"  {code:<13}" + "".join(f"{c:>9}" for c in cnts))

    # =======================================================================
    # C - what the residual contamination is, and who owns it
    # =======================================================================
    say("")
    say("=" * 78)
    say("C. RESIDUAL CONTAMINATION")
    say("")
    say(f"  {n_x - chosen_caught} of {n_x} hand-labelled X survive the filter. "
        "Where they land, and")
    say("  whether Step 13 can remove them:")
    say("")
    say("   lands in       run  toks  ovlap  answer")
    say("   -------------  ---  ----  -----  " + "-" * 40)
    survivors = [r for r in lab if r["hand"] == "X"
                 and classify(r, cuts, order) != "echo"]
    for r in sorted(survivors, key=lambda r: classify(r, cuts, order)):
        say(f"   {classify(r, cuts, order):<13}  {r['copy_run']:3d}  "
            f"{r['n_content']:4d}  {r['q_overlap']:5.2f}  {r['answer'][:40]!r}")
    say("")
    say("  A surviving X that is actually a CORRECT answer leaves the wrong-set")
    say("  entirely when the judge rescores it. A surviving X that is a genuine")
    say("  copy stays, and is reported as a contamination rate in the paper's")
    say("  data section. Step 13 tells us which is which - it cannot be decided")
    say("  from strings, which is the whole reason the judge exists.")

    # =======================================================================
    # D - apply
    # =======================================================================
    say("")
    say("=" * 78)
    say("D. FINAL SPLIT")
    say("")
    for r in wrong:
        r["mode"] = classify(r, cuts, order)
    say("  PROVISIONAL - right/wrong is still string matching.")
    say("")
    say("  dataset          wrong  locked-in  interleav  inconsist   echo  degen"
        "  untest")
    say("  --------------  ------  ---------  ---------  ---------  -----  -----"
        "  ------")
    by = defaultdict(list)
    for r in wrong:
        by[r["dataset"]].append(r)
    for ds in DATASETS:
        c = Counter(r["mode"] for r in by.get(ds, []))
        say(f"  {ds:<14}  {len(by.get(ds, [])):6,}  {c['locked_in']:9,}  "
            f"{c['interleaving']:9,}  {c['inconsistent']:9,}  {c['echo']:5,}  "
            f"{c['degenerate']:5,}  {c['untestable']:6,}")
    tot = Counter(r["mode"] for r in wrong)
    say(f"  {'ALL':<14}  {len(wrong):6,}  {tot['locked_in']:9,}  "
        f"{tot['interleaving']:9,}  {tot['inconsistent']:9,}  {tot['echo']:5,}  "
        f"{tot['degenerate']:5,}  {tot['untestable']:6,}")
    say("")
    say("  against Step 12 as reported (drifted cut):")
    for m in ("locked_in", "interleaving", "inconsistent", "echo"):
        say(f"    {m:<14} {STEP12_SPLIT[m]:6,} -> {tot[m]:6,}   "
            f"{tot[m] - STEP12_SPLIT[m]:+,}")

    # =======================================================================
    # E - supply
    # =======================================================================
    say("")
    say("=" * 78)
    say("E. SUPPLY AGAINST THE 150-PER-CELL TARGET")
    say("")
    say("  dataset         mode              have   target   shortfall")
    say("  --------------  -------------  -------  -------  ----------")
    short = []
    for ds in DATASETS:
        c = Counter(r["mode"] for r in by.get(ds, []))
        for m in ("locked_in", "interleaving", "inconsistent"):
            gap = max(0, TARGET_PER_CELL - c[m])
            if gap:
                short.append((ds, m, c[m], gap))
            say(f"  {ds:<14}  {m:<13}  {c[m]:7,}  {TARGET_PER_CELL:7,}  "
                f"{('-' if not gap else f'{gap:,} short'):>10}")
    say("")
    if short:
        say("  Short cells exist. DO NOT size another generation run on this.")
        say("  Every count above rests on a string-matched wrong-set that the")
        say("  judge moves in both directions. Re-read this table after Step 13.")
    else:
        say("  Every cell meets the target on the string-matched wrong-set.")
        say("  Re-read after Step 13 regardless - the judge moves these.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, extrasaction="ignore", fieldnames=[
            "dataset", "qid", "correct", "mode", "old_mode", "stable_top3",
            "cands_top3", "q_overlap", "copy_run", "truncated", "gold_ever",
            "gold_testable"])
        w.writeheader()
        for r in full:
            r.setdefault("mode", "correct")
        w.writerows(full)
    say("")
    say("=" * 78)
    say(f"  rule : overlap >= 0.75 AND copied_run >= {r_cut}, "
        f"locked_in {s_cut}/{c_cut}, order {order}")
    say(f"  CSV  : {OUT_CSV}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
