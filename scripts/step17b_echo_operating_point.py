#!/usr/bin/env python3
"""
Step 12b - choose the echo rule's operating point by looking at the curve
=========================================================================

    modal run modal_app.py::run_cpu --script step12b_echo_operating_point.py

CPU only. Under two minutes, free. Reads two CSVs and no trajectories - every
number it needs was already measured by Step 10d and Step 12, which is the
whole point of keeping measurement and classification apart.

WHAT WENT WRONG IN STEP 12
==========================
Step 12 added `max_copied_run` to the echo rule and let a grid search fit it
against the 59 hand labels, maximising balanced accuracy. The search chose
`max_copied_run >= 1` - which is the old one-dimensional rule exactly - and all
five wrongly-eaten cases stayed eaten.

That result is arithmetically forced, and the arithmetic is worth writing down
because it is the actual lesson:

    one-dimensional      X recall 18/19   specificity 35/40   bal.acc 0.911
    with run >= 4        X recall  k/19   specificity 39/40   bal.acc 0.5(k/19 + 0.975)

    beats 0.911 only when k >= 17

So the span term loses unless at least 17 of the 19 true X cases have a long
copied run. It lost. **At least three genuine echo/copy cases reuse the
question without reproducing a long span of it**, and the feature cannot see
them. Part A prints all nineteen so that claim is visible rather than inferred.

THE DEEPER MISTAKE
==================
Balanced accuracy says that failing to catch one X costs exactly as much as
destroying one locked-in case. For this paper that is false, and it is false in
a way that matters:

    contamination left in    measurable. It is reported as a rate in the data
                             section, and a reader can check it.

    signal destroyed         invisible. An interleaving case classified as echo
                             never appears in any count, in any table, ever.
                             Nothing downstream can recover it.

And the two are not symmetric in supply either. Step 12 found interleaving at
216 of 8,222 wrong answers - 2.6%, the rarest mode, and already short of the
150-per-cell target the Phase C sample size was built on. The echo rule ate one
of the six hand-labelled interleaving cases: 17%. That is not affordable.

So this step stops optimising balanced accuracy and instead reports the
**operating curve** - for every candidate cut, how much contamination is removed
and how much signal is destroyed - and picks the point under a stated
constraint rather than a single blended score. The constraint is declared here,
before the numbers are seen, so it cannot be tuned to flatter a result:

    maximise X recall subject to destroying at most one hand-labelled
    locked-in or interleaving case.

`inconsistent` is deliberately not protected: it is 47% of wrong answers and
losing one costs nothing. Only the two rare modes the paper is about are.

WHAT THIS DOES NOT DECIDE
=========================
Right/wrong is still string matching. Every count below moves when the judge
runs. Nothing here should be used to size another generation run - see Part E.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

CAL_CSV = config.TAB_DIR / "step10d_final_modes.csv"      # the 150, with hand labels
FULL_CSV = config.TAB_DIR / "step12_phasec_modes.csv"     # the 11,120, all features
REPORT_PATH = config.OUT_DIR / "step12b_operating_point.txt"
OUT_CSV = config.TAB_DIR / "step12b_phasec_modes.csv"

# Step 12's cuts, recorded from its report. Not thresholds being asserted - a
# checksum. Part D0 re-derives Step 12's split from the stored features using
# these, and it must match, or the CSV round-trip has lost something and every
# number after it is suspect.
STEP12_CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=1)
STEP12_ORDER = "A"
STEP12_EXPECTED = {"locked_in": 1010, "interleaving": 216, "inconsistent": 3834,
                   "echo": 2673, "degenerate": 355, "untestable": 134}

# The constraint, declared before the curve is read.
MAX_SIGNAL_DESTROYED = 1

TARGET_PER_CELL = 150
DATASETS = ("triviaqa", "hotpotqa")

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Text handling - identical to Step 12.
# ===========================================================================

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
    """Identical logic to Step 12; cuts passed in so several can be compared."""
    if r["truncated"]:
        return "degenerate"
    echo = r["q_overlap"] >= cuts["e_cut"] and r["copy_run"] >= cuts["r_cut"]
    gold, testable = r["gold_ever"], r["gold_testable"]
    if order == "A":
        if echo:
            return "echo"
        if not testable:
            return "untestable"
        if gold:
            return "interleaving"
    else:
        if testable and gold:
            return "interleaving"
        if echo:
            return "echo"
        if not testable:
            return "untestable"
    if r["stable_top3"] >= cuts["s_cut"] and r["cands_top3"] <= cuts["c_cut"]:
        return "locked_in"
    return "inconsistent"


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 12b: the echo rule's operating point")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    for p in (CAL_CSV, FULL_CSV):
        if not p.exists():
            say(f"\nMissing {p}. Run step10d and step12 first.")
            sys.exit(1)

    cal = []
    with open(CAL_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            hand = (r.get("hand") or "").strip()
            cal.append(dict(
                dataset=r["dataset"], qid=r["qid"], hand=hand or None,
                question=r.get("question", ""), answer=r.get("answer", ""),
                q_overlap=as_float(r.get("q_overlap")),
                copy_run=max_copied_run(r.get("answer", ""), r.get("question", "")),
                stable_top3=as_float(r.get("stable_top3")),
                cands_top3=as_float(r.get("cands_top3")),
                n_content=int(as_float(r.get("n_content"))),
                truncated=as_bool(r.get("truncated")),
                gold_ever=as_bool(r.get("gold_ever")),
                gold_testable=as_bool(r.get("gold_testable")) or
                              (r.get("gold_testable", "") == "")))
    lab = [r for r in cal if r["hand"] in ("L", "I", "C", "X")]
    say(f"  {len(cal)} calibration rows, {len(lab)} hand-labelled")

    # =======================================================================
    # A - look at the nineteen X cases
    # =======================================================================
    say("")
    say("=" * 78)
    say("A. EVERY HAND-LABELLED X, SORTED BY COPIED-SPAN LENGTH")
    say("")
    say("  The span hypothesis says these should all have long runs. Step 12's")
    say("  fit implies at least three do not. This is that claim, checkable.")
    say("")
    say("   run  toks  ovlap  answer")
    say("   ---  ----  -----  " + "-" * 50)
    xs = sorted([r for r in lab if r["hand"] == "X"], key=lambda r: r["copy_run"])
    for r in xs:
        say(f"   {r['copy_run']:3d}  {r['n_content']:4d}  {r['q_overlap']:5.2f}  "
            f"{r['answer'][:50]!r}")
    short = [r for r in xs if r["copy_run"] <= 3]
    say("")
    say(f"  {len(short)} of {len(xs)} true X have a copied run of 3 or less.")
    say("  Those are the ones the span term cannot separate from a selection,")
    say("  and they are why balanced accuracy rejected it.")

    say("")
    say("  And the cases it wrongly eats, for comparison:")
    say("")
    say("   hand  run  toks  ovlap  answer")
    say("   ----  ---  ----  -----  " + "-" * 48)
    for r in sorted([x for x in lab if x["hand"] in ("L", "I")],
                    key=lambda r: r["copy_run"]):
        if r["q_overlap"] >= 0.75:
            say(f"   {r['hand']:>4}  {r['copy_run']:3d}  {r['n_content']:4d}  "
                f"{r['q_overlap']:5.2f}  {r['answer'][:48]!r}")

    # =======================================================================
    # B - the operating curve
    # =======================================================================
    say("")
    say("=" * 78)
    say("B. OPERATING CURVE")
    say("")
    say("  At question_overlap >= 0.75, sweeping the span cut. 'signal lost' is")
    say("  hand-labelled L or I wrongly called echo - the quantity that cannot")
    say("  be recovered downstream.")
    say("")
    say("   run cut   X caught   X missed   signal lost   C lost")
    say("   -------   --------   --------   -----------   ------")
    curve = []
    n_x = sum(1 for r in lab if r["hand"] == "X")
    for run in range(1, 13):
        fires = [r for r in lab if r["q_overlap"] >= 0.75 and r["copy_run"] >= run]
        caught = sum(1 for r in fires if r["hand"] == "X")
        sig = sum(1 for r in fires if r["hand"] in ("L", "I"))
        clost = sum(1 for r in fires if r["hand"] == "C")
        curve.append((run, caught, sig, clost))
        say(f"   >= {run:<5d}  {caught:8d}   {n_x - caught:8d}   "
            f"{sig:11d}   {clost:6d}")
        if caught == 0:
            break

    # =======================================================================
    # C - three operating points
    # =======================================================================
    say("")
    say("=" * 78)
    say("C. THREE OPERATING POINTS")
    say("")
    best_bal, best_con = None, None
    for run, caught, sig, clost in curve:
        n_not = len(lab) - n_x
        fp = sig + clost
        bal = 0.5 * (caught / n_x + (n_not - fp) / n_not)
        if best_bal is None or bal > best_bal[0]:
            best_bal = (bal, run, caught, sig, clost)
        if sig <= MAX_SIGNAL_DESTROYED:
            # best_con[2] is `caught`. Comparing against best_con[1] here would
            # compare a count against a threshold and silently pick wrong.
            if best_con is None or caught > best_con[2]:
                best_con = (bal, run, caught, sig, clost)
    say(f"  1. balanced accuracy   run >= {best_bal[1]}   "
        f"X caught {best_bal[2]}/{n_x}   signal lost {best_bal[3]}   "
        f"bal.acc {best_bal[0]:.0%}")
    if best_con:
        say(f"  2. signal-protected    run >= {best_con[1]}   "
            f"X caught {best_con[2]}/{n_x}   signal lost {best_con[3]}   "
            f"bal.acc {best_con[0]:.0%}")
    strict = [c for c in curve if c[2] == 0]
    if strict:
        b = max(strict, key=lambda c: c[1])
        say(f"  3. zero signal loss    run >= {b[0]}   "
            f"X caught {b[1]}/{n_x}   signal lost 0")
    else:
        say("  3. zero signal loss    unreachable at any span cut")
    say("")
    say("  Point 2 is the declared criterion: at most "
        f"{MAX_SIGNAL_DESTROYED} hand-labelled L/I destroyed.")
    if best_con is None:
        say("  UNREACHABLE - no span cut satisfies it. The echo rule needs a")
        say("  different feature, not a different threshold. Stopping here")
        say("  rather than picking the least bad option silently.")
        chosen = dict(STEP12_CUTS)
        chosen_order = STEP12_ORDER
    else:
        chosen = dict(STEP12_CUTS, r_cut=best_con[1])
        chosen_order = STEP12_ORDER
        say(f"  Adopting question_overlap >= 0.75 AND max_copied_run >= "
            f"{best_con[1]}.")
        say(f"  Residual contamination: {n_x - best_con[2]} of {n_x} hand-labelled")
        say("  X survive the filter. That is a REPORTED RATE, not a hidden loss.")

    # rule order, re-checked under the new cut
    say("")
    MAP = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
           "echo": "X", "degenerate": "X", "untestable": None}
    for order in ("A", "B"):
        ok = sum(1 for r in lab if MAP.get(classify(r, chosen, order)) == r["hand"])
        say(f"  order {order} reproduces {ok}/{len(lab)} = {ok/len(lab):.0%}")
    chosen_order = max("AB", key=lambda o: sum(
        1 for r in lab if MAP.get(classify(r, chosen, o)) == r["hand"]))
    say(f"  Using order {chosen_order}.")

    say("")
    say("              hand L   hand I   hand C   hand X")
    for code in ("locked_in", "interleaving", "inconsistent", "echo",
                 "degenerate", "untestable"):
        cnts = [sum(1 for r in lab if classify(r, chosen, chosen_order) == code
                    and r["hand"] == h) for h in ("L", "I", "C", "X")]
        if sum(cnts):
            say(f"  {code:<13}" + "".join(f"{c:>9}" for c in cnts))

    # =======================================================================
    # D - reclassify all 11,120 from stored features
    # =======================================================================
    say("")
    say("=" * 78)
    say("D. FULL SCALE, RECLASSIFIED")
    say("")
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
    say(f"  {len(full):,} rows, {len(wrong):,} wrong")

    # D0 - the round-trip checksum
    repro = Counter(classify(r, STEP12_CUTS, STEP12_ORDER) for r in wrong)
    match = all(repro[k] == v for k, v in STEP12_EXPECTED.items())
    say("")
    say("  Round-trip check: re-deriving Step 12's split from the stored")
    say("  features must reproduce Step 12's report exactly.")
    for k, v in STEP12_EXPECTED.items():
        flag = "ok" if repro[k] == v else f"MISMATCH (expected {v})"
        say(f"    {k:<14} {repro[k]:6,}  {flag}")
    if not match:
        say("")
        say("  STOPPED - the CSV does not reproduce Step 12. Something was lost")
        say("  writing or reading it, so nothing below can be trusted.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  All match - the stored features are lossless.")

    for r in wrong:
        r["mode"] = classify(r, chosen, chosen_order)
    say("")
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
    say("  change against Step 12:")
    old_tot = Counter(r["old_mode"] for r in wrong)
    for m in ("locked_in", "interleaving", "inconsistent", "echo"):
        d = tot[m] - old_tot[m]
        say(f"    {m:<14} {old_tot[m]:6,} -> {tot[m]:6,}   {d:+,}")

    # =======================================================================
    # E - interleaving supply
    # =======================================================================
    say("")
    say("=" * 78)
    say("E. SUPPLY AGAINST THE 150-PER-CELL TARGET")
    say("")
    say("  dataset         mode              have   target   shortfall")
    say("  --------------  -------------  -------  -------  ----------")
    shortfalls = []
    for ds in DATASETS:
        c = Counter(r["mode"] for r in by.get(ds, []))
        for m in ("locked_in", "interleaving", "inconsistent"):
            gap = max(0, TARGET_PER_CELL - c[m])
            if gap:
                shortfalls.append((ds, m, c[m], gap))
            say(f"  {ds:<14}  {m:<13}  {c[m]:7,}  {TARGET_PER_CELL:7,}  "
                f"{('-' if not gap else f'{gap:,} short'):>10}")
    say("")
    if shortfalls:
        say("  Short cells exist. DO NOT size another generation run on this.")
        say("  The judge moves the wrong-set in both directions, and every count")
        say("  above is computed on a string-matched wrong-set. Sizing on")
        say("  uncalibrated numbers is the mistake that produced the 32-step")
        say("  decoder config and the 10 s/question cost estimate.")
        say("")
        say("  Re-read this table after Step 13. If a cell is still short then,")
        say("  it is a real shortfall and generation resumes - which costs one")
        say("  command, because step11 skips what exists.")
    else:
        say("  Every cell meets the target.")

    # ---- CSV --------------------------------------------------------------
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
    say(f"  rule   : overlap >= {chosen['e_cut']:.2f} AND run >= {chosen['r_cut']}"
        f", order {chosen_order}")
    say(f"  CSV    : {OUT_CSV}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
