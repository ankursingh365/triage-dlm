#!/usr/bin/env python3
"""
Step 19a - which measure fails to transfer between the two arms?
================================================================

    modal run modal_app.py::run_cpu --script step19a_measure_transfer.py

CPU only. Seconds, free. Reads three CSVs and no trajectories.

THE PROBLEM
===========
Step 17d applied the LLaDA-fitted rules to Dream and the split did not survive:

    mode            LLaDA    Dream     diff
    locked_in       18.3%    71.0%   +52.7
    inconsistent    49.1%    20.3%   -28.8
    echo            22.2%     2.2%   -20.0

71% locked-in is not a finding about Dream. Something in the measurement is
responding to the sampler rather than to the model, and **which** measure is
doing it decides what happens next:

    if `stable_top3` drifted   the schedule broke it - see below - and the fix
                               is a schedule-invariant measure, not new cuts
    if `cands_top3` drifted    the token distributions genuinely differ and the
                               cuts need refitting on Dream hand labels
    if `q_overlap` drifted     Dream simply echoes less; nothing is broken
    if none drifted            the cuts are fine and the classifier's ORDER or
                               some interaction is at fault

THE HYPOTHESIS, STATED BEFORE THE NUMBERS
=========================================
`stable_frac` is `held / len(history)`, where the denominator counts the
**rounds** before a position was revealed. Step 13c measured Dream's schedule
as adaptive: 0-5 positions per round, with **37% of rounds revealing nothing at
all**. A round that reveals nothing is very unlikely to change the argmax at a
position that is already settled, so it adds 1 to the numerator and 1 to the
denominator and pushes the ratio towards 1.

LLaDA reveals exactly one position per round, every round, so it has no such
rounds and the measure was calibrated in a regime that does not exist for
Dream.

If that is right, `stable_top3` shifts sharply upward for Dream while
`cands_top3` - a count of DISTINCT values, which does not care how many rounds
passed - barely moves. Part B tests exactly that, and Part C asks what a
cands-only rule would give, since `claude/failure-mode-calibration.md` already
records `cands_top3` as the better discriminator (Cohen's d 1.65 vs 1.47).

This script does not change any rule. It identifies the culprit so the fix is
chosen with evidence.
"""

import csv
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

LLADA_CSVS = [config.TAB_DIR / "step12c_final_modes.csv",
              config.TAB_DIR / "step12b_csqa_modes.csv"]
DREAM_CSV = config.TAB_DIR / "step17d_dream_modes.csv"
REPORT_PATH = config.OUT_DIR / "step19a_measure_transfer.txt"

CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6)
MEASURES = ("stable_top3", "cands_top3", "q_overlap", "copy_run", "n_content")
MODES = ("locked_in", "interleaving", "inconsistent", "echo", "degenerate",
         "untestable")

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def fnum(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load(paths) -> list:
    out = []
    for p in ([paths] if isinstance(paths, Path) else paths):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out.append(dict(
                    dataset=r.get("dataset", ""),
                    correct=str(r.get("correct", "")).lower() == "true",
                    mode=r.get("mode", ""),
                    truncated=str(r.get("truncated", "")).lower() == "true",
                    gold_ever=str(r.get("gold_ever", "")).lower() == "true",
                    gold_testable=str(r.get("gold_testable", "")).lower() == "true",
                    **{m: fnum(r.get(m)) for m in MEASURES}))
    return out


def q(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 19a: which measure fails to transfer?")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    llada, dream = load(LLADA_CSVS), load(DREAM_CSV)
    if not llada or not dream:
        say("\nNeed step12b/12c and step17d CSVs. Run those first.")
        sys.exit(1)
    lw = [r for r in llada if not r["correct"]]
    dw = [r for r in dream if not r["correct"]]
    say(f"  LLaDA {len(llada):,} rows, {len(lw):,} wrong")
    say(f"  Dream {len(dream):,} rows, {len(dw):,} wrong")
    say("")
    say("  Comparison is on WRONG answers only, because that is the set the")
    say("  failure-mode rules are applied to.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("A. DISTRIBUTION OF EVERY MEASURE, BOTH ARMS")
    say("")
    say("  measure         arm      p10     p25   median     p75     p90    mean")
    say("  -------------  ------  ------  ------  ------  ------  ------  ------")
    shifts = {}
    for m in MEASURES:
        a = np.array([r[m] for r in lw], float)
        b = np.array([r[m] for r in dw], float)
        for tag, v in (("LLaDA", a), ("Dream", b)):
            say(f"  {m if tag=='LLaDA' else '':<13}  {tag:<6}  "
                f"{q(v,10):6.2f}  {q(v,25):6.2f}  {q(v,50):6.2f}  "
                f"{q(v,75):6.2f}  {q(v,90):6.2f}  {v.mean():6.2f}")
        # standardised shift: median difference in LLaDA standard deviations
        sd = a.std() or 1.0
        shifts[m] = (q(b, 50) - q(a, 50)) / sd
        say("")

    say("  median shift, in LLaDA standard deviations")
    say("  ----------------------------------------")
    for m in sorted(shifts, key=lambda k: -abs(shifts[k])):
        bar = "#" * min(40, int(abs(shifts[m]) * 10))
        say(f"    {m:<13} {shifts[m]:+6.2f}  {bar}")
    worst = max(shifts, key=lambda k: abs(shifts[k]))
    say("")
    say(f"  Largest drift: {worst} ({shifts[worst]:+.2f} sd)")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. WHICH SIDE OF THE CUT EACH ARM FALLS ON")
    say("")
    say("  The rule fires when stable_top3 >= 0.26 AND cands_top3 <= 3.4.")
    say("  A measure that moved the classification must have moved the share")
    say("  of rows on the firing side of its own cut.")
    say("")
    say("  condition                      LLaDA     Dream    diff")
    say("  ---------------------------  -------  --------  ------")
    conds = [("stable_top3 >= 0.26", lambda r: r["stable_top3"] >= CUTS["s_cut"]),
             ("cands_top3  <= 3.4", lambda r: r["cands_top3"] <= CUTS["c_cut"]),
             ("BOTH (the locked_in rule)",
              lambda r: r["stable_top3"] >= CUTS["s_cut"]
              and r["cands_top3"] <= CUTS["c_cut"]),
             ("q_overlap   >= 0.75", lambda r: r["q_overlap"] >= CUTS["e_cut"]),
             ("copy_run    >= 6", lambda r: r["copy_run"] >= CUTS["r_cut"])]
    for label, fn in conds:
        a = sum(1 for r in lw if fn(r)) / max(1, len(lw))
        b = sum(1 for r in dw if fn(r)) / max(1, len(dw))
        say(f"  {label:<27}  {a:6.1%}  {b:8.1%}  {b-a:+6.1%}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. VERDICT, AND WHAT IT IMPLIES")
    say("")
    s_gap = (sum(1 for r in dw if r["stable_top3"] >= CUTS["s_cut"]) / max(1, len(dw))
             - sum(1 for r in lw if r["stable_top3"] >= CUTS["s_cut"]) / max(1, len(lw)))
    c_gap = (sum(1 for r in dw if r["cands_top3"] <= CUTS["c_cut"]) / max(1, len(dw))
             - sum(1 for r in lw if r["cands_top3"] <= CUTS["c_cut"]) / max(1, len(lw)))
    say(f"  stable_top3 firing share moved {s_gap:+.1%}")
    say(f"  cands_top3  firing share moved {c_gap:+.1%}")
    say("")
    s_big, c_big = abs(s_gap) > 0.20, abs(c_gap) > 0.20
    if s_big and c_big:
        # Not an either/or. Forcing a single culprit here would hide half the
        # problem and make a measure swap look like a complete fix.
        say("  BOTH MOVED. stable_top3 moved further "
            f"({s_gap:+.1%} vs {c_gap:+.1%}), so the schedule is the larger")
        say("  effect and a schedule-invariant measure is still the right")
        say("  first move - but cands_top3 shifting by more than 20 points")
        say("  means Dream's token distributions genuinely differ too.")
        say("")
        say("  So a measure swap alone will NOT close the gap. Expect to need")
        say("  both: a schedule-invariant measure AND a refit on Dream hand")
        say("  labels. Part D shows how much of the gap the swap recovers.")
        verdict = "both"
    elif s_big:
        say("  stable_top3 IS THE CULPRIT, and the hypothesis holds: it is not")
        say("  schedule-invariant. Its denominator counts rounds, and 37% of")
        say("  Dream's rounds reveal nothing, so a settled position accrues")
        say("  'stability' for free.")
        say("")
        say("  This is NOT fixed by refitting the cut on Dream. A cut refitted")
        say("  on a measure that responds to the sampler would make the two")
        say("  arms incomparable in a way no reader could check - the same")
        say("  number would mean different things in each column.")
        say("")
        say("  The fix is a measure whose denominator does not count rounds.")
        say("  Part D prices the obvious candidate.")
        verdict = "stable"
    elif c_big:
        say("  cands_top3 moved but stable_top3 did not, so this is not a")
        say("  schedule effect -")
        say("  Dream's token distributions differ. Refitting on Dream hand")
        say("  labels is then the honest route, not a measure change.")
        verdict = "cands"
    else:
        say("  Neither cut's firing share moved much, so the explosion comes")
        say("  from the rule ORDER or an interaction, not from a single")
        say("  measure. Investigate classify() before changing anything.")
        verdict = "order"

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. WHAT WOULD A CANDS-ONLY RULE GIVE?")
    say("")
    say("  claude/failure-mode-calibration.md already records cands_top3 as")
    say("  the better discriminator - Cohen's d 1.65 against stable_top3's")
    say("  1.47 - and it counts DISTINCT values, so it does not care how many")
    say("  rounds passed. If dropping stable_top3 brings the two arms into")
    say("  line, that is evidence the schedule was the whole problem.")
    say("")
    say("  locked_in share of wrong answers, under three rules")
    say("")
    say("  rule                                  LLaDA     Dream    diff")
    say("  ----------------------------------  -------  --------  ------")

    def split(rows, rule):
        n = 0
        for r in rows:
            if r["truncated"]:
                continue
            if r["q_overlap"] >= CUTS["e_cut"] and r["copy_run"] >= CUTS["r_cut"]:
                continue
            if not r["gold_testable"]:
                continue
            if r["gold_ever"]:
                continue
            if rule(r):
                n += 1
        return n / max(1, len(rows))

    rules = [
        ("current: stable>=0.26 AND cands<=3.4",
         lambda r: r["stable_top3"] >= CUTS["s_cut"]
         and r["cands_top3"] <= CUTS["c_cut"]),
        ("cands only: cands<=3.4", lambda r: r["cands_top3"] <= CUTS["c_cut"]),
        ("cands only, tighter: cands<=2.5",
         lambda r: r["cands_top3"] <= 2.5),
    ]
    best = None
    for label, rule in rules:
        a, b = split(lw, rule), split(dw, rule)
        say(f"  {label:<34}  {a:6.1%}  {b:8.1%}  {b-a:+6.1%}")
        if best is None or abs(b - a) < abs(best[1]):
            best = (label, b - a)
    say("")
    say(f"  Closest agreement: {best[0]}  ({best[1]:+.1%})")
    say("")
    say("  Closing the gap is necessary, not sufficient. A rule that agrees")
    say("  across arms can still be wrong about both. Whichever rule is")
    say("  adopted must be re-validated against hand labels before the")
    say("  per-mode table is built - LLaDA's existing 59 labels are a free")
    say("  test for any candidate, and Dream needs its own blind round.")

    say("")
    say("=" * 78)
    say("  NOTHING WAS CHANGED. This script only names the culprit.")
    if verdict in ("stable", "both"):
        say("  Next: a schedule-invariant replacement for stable_frac,")
        say("  re-validated on LLaDA's 59 hand labels, then a Dream round.")
        if verdict == "both":
            say("  The Dream round is required, not optional - cands moved too.")
    elif verdict == "cands":
        say("  Next: a blind two-annotator round on Dream, then refit.")
    else:
        say("  Next: read classify() and the rule order before touching cuts.")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
