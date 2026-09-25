#!/usr/bin/env python3
"""
Step 19a - one labelling rule, two arms: the table Phase E is built on
======================================================================

    modal run modal_app.py::run_cpu --script step19a_final_modes.py

CPU only, seconds, free. FIRST STEP OF PHASE E.

WHY THIS COMES BEFORE src/features.py
=====================================
The execution plan records packaging debt with a condition attached: it "must be
repaid before step 19 writes src/features.py against them". The debt that is now
actively dangerous is the labelling rule. `classify()` currently lives in four
scripts - 17c, 17d, 18d and 18e - each restating it so it could be LOCKed
against the others. The LOCKs passed, 16,781 rows with zero mismatches, so the
copies agree today. But every Phase E and F script needs mode labels, and a
fifth and sixth copy would be the moment one of them drifts - silently, because
a relabelled row does not crash anything.

So the rule moves into `src/label_failure_mode.py`, once, and this script:

  1. proves the module reproduces every label already published, on both arms;
  2. applies the repaired rule - gate 1 fixed, per-arm cuts - to both arms;
  3. writes ONE table, `step19a_final_modes.csv`, carrying every generation of
     both arms, correct answers included, with the measures alongside;
  4. re-asks the power question on the final counts, because the counts just
     changed and the power decision says any future power question is a
     question about interleaving.

From here on, Phase E and F read that one table and import that one module.

WHAT CHANGES, AND FOR WHOM
==========================
    Dream   s_cut 0.26 -> 0.63, c_cut 3.4 -> 3.0 (step 18e), gate 1 repaired.
            Identical to step18e_dream_modes.csv - LOCK 1 proves it.
    LLaDA   cuts UNCHANGED at 0.26 / 3.4. Gate 1 repaired, because gate 1 is
            shared: its 220 untestable rows now reach the cuts. LOCK 3 proves
            that exactly those rows move and nothing else does.

The LLaDA change deserves a plain statement. Step 18e's Part F checked the
repair against LLaDA's 59 hand labels and found agreement unchanged at 28/37 -
but step 18d had already shown that none of those 59 is untestable, so the
check had nothing to act on. The repair is unvalidated by hand on LLaDA; it is
adopted because its logic does not depend on the arm and it was validated by
hand on Dream. That is recorded here rather than implied by a passing check.

THE LOCKS - nothing is written unless all five pass
===================================================
    LOCK 1  module, dream     == step18e_dream_modes.csv         every wrong row
    LOCK 2  module, legacy    == step12c / 12b stored modes      every LLaDA row
    LOCK 3  repaired - legacy  changes ONLY the untestable rows, and each lands
                               in locked_in or inconsistent
    LOCK 4  CUTS["dream"]     == step18e_dream_cuts.json
    LOCK 5  correct == True   <=>  mode == "correct", in every source CSV
"""

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
    from src.label_failure_mode import (CUTS, LEGACY_MODES, MODES, classify,
                                        classify_legacy, int_untested)
except ImportError as exc:                                  # pragma: no cover
    print(f"Could not import from src/ ({exc}).")
    print("Is src/label_failure_mode.py saved next to src/config.py?")
    sys.exit(1)

DREAM_SRC = config.TAB_DIR / "step17d_dream_modes.csv"
DREAM_18E = config.TAB_DIR / "step18e_dream_modes.csv"
DREAM_JSON = config.TAB_DIR / "step18e_dream_cuts.json"
LLADA_SRC = [config.TAB_DIR / "step12c_final_modes.csv",
             config.TAB_DIR / "step12b_csqa_modes.csv"]
OUT_CSV = config.TAB_DIR / "step19a_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step19a_final_modes.txt"

NUMERIC = ("stable_top3", "cands_top3", "q_overlap", "copy_run")
BOOLEAN = ("correct", "truncated", "gold_ever", "gold_testable")
NEED = {"qid", "dataset", "mode", *NUMERIC, *BOOLEAN}

# Power, exactly as claude/sample-size-power-decision.md computed it.
AUC_ASSUMED = 0.65
Z_SUM = 1.959964 + 0.841621            # alpha 0.05 two-sided, 80% power
CLAIMED_EFFECT = 0.10
PRIMARY_DATASETS = ("triviaqa", "hotpotqa")   # CSQA stays out - see the doc

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def rule(char: str = "=") -> None:
    say(char * 78)


def finish(code: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(code)


def load(paths, label: str) -> list:
    """Every row of the source CSVs. Header validated; nothing defaulted."""
    out = []
    for p in ([paths] if isinstance(paths, Path) else paths):
        if not p.exists():
            say(f"  FATAL - {label}: {p.name} not found.")
            finish(1)
        with open(p, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            missing = sorted(NEED - set(rd.fieldnames or ()))
            if missing:
                say(f"  FATAL - {p.name} lacks: {', '.join(missing)}")
                say("  No default is substituted for a missing column.")
                finish(1)
            for r in rd:
                row = dict(qid=r["qid"], dataset=r["dataset"],
                           stored=(r["mode"] or "").strip(), source=p.name)
                for c in NUMERIC:
                    try:
                        row[c] = float(r[c])
                    except (TypeError, ValueError):
                        row[c] = None
                for c in BOOLEAN:
                    row[c] = str(r[c]).strip().lower() == "true"
                out.append(row)
    return out


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) standard error of an AUROC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var)


def mde(n_a: int, n_b: int, n_neg: int, auc: float = AUC_ASSUMED) -> float:
    """Minimum detectable AUROC difference between two failure modes that share
    the same negatives. Independence is assumed, which is conservative: shared
    negatives correlate the two estimates and shrink the SE of their
    difference. Identical to step 14's calculation."""
    if min(n_a, n_b, n_neg) < 2:
        return float("nan")
    return Z_SUM * math.sqrt(hanley_mcneil_se(auc, n_a, n_neg) ** 2
                             + hanley_mcneil_se(auc, n_b, n_neg) ** 2)


# ---------------------------------------------------------------------------

def main() -> None:
    rule()
    say("  TRIAGE - Step 19a: one labelling rule, two arms")
    say(f"  {datetime.now():%Y-%m-%d %H:%M}")
    rule()
    say("")
    say("  PHASE E BEGINS HERE.")
    say("")
    say("  Rule source: src/label_failure_mode.py")
    for arm, c in CUTS.items():
        say(f"    {arm:<6} s_cut={c['s_cut']:.2f}  c_cut={c['c_cut']:.1f}  "
            f"e_cut={c['e_cut']:.2f}  r_cut={c['r_cut']}")
    say("")

    dream = load(DREAM_SRC, "Dream")
    llada = load(LLADA_SRC, "LLaDA")
    arms = {"dream": dream, "llada": llada}
    for arm, rows in arms.items():
        w = sum(1 for r in rows if not r["correct"])
        say(f"  {arm:<6} {len(rows):6d} rows   {w:6d} wrong   "
            f"{len(rows)-w:6d} correct")
    say("")

    ok_all = True

    # --- LOCK 5 first: it is the cheapest and it guards every other one ----
    rule("-")
    say("  LOCK 5 - correct == True  <=>  mode == 'correct'")
    rule("-")
    for arm, rows in arms.items():
        bad = [r for r in rows if r["correct"] != (r["stored"] == "correct")]
        unknown = Counter(r["stored"] for r in rows
                          if r["stored"] not in LEGACY_MODES + ("correct",))
        say(f"    {arm:<6} {len(rows)-len(bad):6d} / {len(rows):<6d} consistent"
            f"{'' if not bad else f'   {len(bad)} CONTRADICT'}")
        if unknown:
            say(f"    {arm:<6} unrecognised stored modes: {dict(unknown)}")
            ok_all = False
        ok_all = ok_all and not bad
    say("")

    # --- LOCK 1: the module reproduces step 18e on Dream --------------------
    rule("-")
    say("  LOCK 1 - module, arm='dream'  ==  step18e_dream_modes.csv")
    rule("-")
    if not DREAM_18E.exists():
        say(f"    {DREAM_18E.name} not found. Run step 18e first.")
        finish(1)
    with open(DREAM_18E, encoding="utf-8") as fh:
        ref = {r["qid"]: r["mode"] for r in csv.DictReader(fh)}
    wrong_d = [r for r in dream if not r["correct"]]
    mism = [(r["qid"], ref.get(r["qid"]), classify(r, "dream"))
            for r in wrong_d if ref.get(r["qid"]) != classify(r, "dream")]
    say(f"    {len(wrong_d)-len(mism):6d} / {len(wrong_d):<6d} reproduced"
        f"{'' if not mism else f'   {len(mism)} MISMATCH'}")
    for qid, want, got in mism[:5]:
        say(f"      {qid}: 18e says {want}, module says {got}")
    ok_all = ok_all and not mism
    say("")

    # --- LOCK 2: legacy rule reproduces LLaDA's stored modes ---------------
    rule("-")
    say("  LOCK 2 - module, legacy rule  ==  LLaDA's stored modes")
    rule("-")
    wrong_l = [r for r in llada if not r["correct"]]
    mism2 = [r for r in wrong_l
             if classify_legacy(r, CUTS["llada"]) != r["stored"]]
    say(f"    {len(wrong_l)-len(mism2):6d} / {len(wrong_l):<6d} reproduced"
        f"{'' if not mism2 else f'   {len(mism2)} MISMATCH'}")
    for r in mism2[:5]:
        say(f"      {r['qid']}: stored {r['stored']}, legacy rule says "
            f"{classify_legacy(r, CUTS['llada'])}")
    ok_all = ok_all and not mism2
    say("")

    # --- LOCK 3: the repair moves exactly the untestable rows --------------
    rule("-")
    say("  LOCK 3 - on LLaDA the repair moves ONLY the untestable rows")
    rule("-")
    moved = [r for r in wrong_l
             if classify(r, "llada") != classify_legacy(r, CUTS["llada"])]
    moved_untestable = [r for r in moved if r["stored"] == "untestable"]
    stray = [r for r in moved if r["stored"] != "untestable"]
    n_untestable = sum(1 for r in wrong_l if r["stored"] == "untestable")
    landing = Counter(classify(r, "llada") for r in moved)
    bad_landing = {m: c for m, c in landing.items()
                   if m not in ("locked_in", "inconsistent")}
    say(f"    untestable rows            {n_untestable}")
    say(f"    rows the repair moved      {len(moved)}")
    say(f"    moved and were untestable  {len(moved_untestable)}")
    say(f"    moved but were NOT         {len(stray)}")
    say("    landed in: " + ", ".join(f"{m} {c}"
                                     for m, c in landing.most_common()))
    lock3 = (len(moved_untestable) == n_untestable == len(moved)
             and not stray and not bad_landing)
    say(f"    {'PASS' if lock3 else 'FAIL'}")
    ok_all = ok_all and lock3
    say("")

    # --- LOCK 4: module cuts == the JSON 18e wrote -------------------------
    rule("-")
    say("  LOCK 4 - CUTS['dream'] in the module == step18e_dream_cuts.json")
    rule("-")
    if not DREAM_JSON.exists():
        say(f"    {DREAM_JSON.name} not found.")
        ok_all = False
    else:
        js = json.loads(DREAM_JSON.read_text(encoding="utf-8"))["cuts"]
        same = all(abs(float(js[k]) - float(CUTS["dream"][k])) < 1e-12
                   for k in ("s_cut", "c_cut", "e_cut", "r_cut"))
        say(f"    json   {js}")
        say(f"    module {CUTS['dream']}")
        say(f"    {'PASS' if same else 'FAIL - the module and the fit disagree'}")
        ok_all = ok_all and same
    say("")

    if not ok_all:
        rule()
        say("  A LOCK FAILED. Nothing is written.")
        say("  The module does not reproduce what was published, so any table")
        say("  built on it would disagree with every number already reported.")
        rule()
        finish(1)

    # --- write the one table -----------------------------------------------
    rule("-")
    say("  ALL FIVE LOCKS PASSED - writing the table")
    rule("-")
    final = []
    for arm, rows in arms.items():
        for r in rows:
            if r["correct"]:
                mode, unt = "correct", ""
            else:
                mode, unt = classify(r, arm), int_untested(r)
            final.append(dict(arm=arm, dataset=r["dataset"], qid=r["qid"],
                              correct=r["correct"], mode=mode,
                              int_untested=unt,
                              **{c: r[c] for c in NUMERIC},
                              truncated=r["truncated"],
                              gold_ever=r["gold_ever"],
                              gold_testable=r["gold_testable"]))
    unknown = sum(1 for r in final if r["mode"] == "unknown")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(final[0]))
        wr.writeheader()
        wr.writerows(final)
    say("")
    say(f"    {OUT_CSV}")
    say(f"    {len(final)} rows. Key is (arm, qid): both arms share questions.")
    say(f"    unmeasurable rows labelled 'unknown': {unknown}")
    say("")

    # --- Part B: the dataset ------------------------------------------------
    rule("-")
    say("  THE DATASET - share of WRONG answers, per arm and dataset")
    rule("-")
    say("")
    datasets = sorted({r["dataset"] for r in final})
    for arm in arms:
        say(f"  {arm.upper()}")
        say("    dataset          wrong  " + "".join(f"{m:>13}" for m in MODES))
        for ds in datasets + ["ALL"]:
            sub = [r for r in final if r["arm"] == arm and r["mode"] != "correct"
                   and (ds == "ALL" or r["dataset"] == ds)]
            if not sub:
                continue
            c = Counter(r["mode"] for r in sub)
            say(f"    {ds:<15} {len(sub):6d}  " + "".join(
                f"{c[m]/len(sub):13.1%}" for m in MODES))
        say("")
    say("  CommonsenseQA is in the table and out of every primary analysis:")
    say("  its answers average ~4.6 tokens, which the trajectory measures were")
    say("  never calibrated on. See claude/sample-size-power-decision.md.")
    say("")

    # --- Part C: is the final dataset still powered? -----------------------
    rule("-")
    say("  POWER - the headline contrast on the FINAL counts")
    rule("-")
    say("")
    say(f"  locked_in vs interleaving, pooled over "
        f"{' + '.join(PRIMARY_DATASETS)}, negatives = correct answers,")
    say(f"  AUROC assumed {AUC_ASSUMED}, 80% power, alpha 0.05 - the same")
    say("  assumptions as the step 14 decision, so the numbers compare.")
    say("")
    say("    arm     subset              locked_in  interleaving  correct"
        "     MDE")
    worst = 0.0
    for arm in arms:
        for tag, keep in (("all rows", lambda r: True),
                          ("int_untested dropped",
                           lambda r: r["int_untested"] is not True)):
            prim = [r for r in final if r["arm"] == arm
                    and r["dataset"] in PRIMARY_DATASETS and keep(r)]
            n_l = sum(1 for r in prim if r["mode"] == "locked_in")
            n_i = sum(1 for r in prim if r["mode"] == "interleaving")
            n_c = sum(1 for r in prim if r["mode"] == "correct")
            m = mde(n_l, n_i, n_c)
            worst = max(worst, m if m == m else 0.0)
            say(f"    {arm:<7} {tag:<20} {n_l:9d} {n_i:13d} {n_c:8d}   "
                f"{m:.3f}")
    say("")
    say(f"  Claimed effect: >= {CLAIMED_EFFECT:.2f}.  Worst MDE above: "
        f"{worst:.3f}.")
    if worst < CLAIMED_EFFECT:
        say("  Every arm and subset can detect the claimed effect. The dataset")
        say("  is complete for the primary contrast; no more generation.")
    else:
        say("  At least one arm or subset CANNOT detect the claimed effect.")
        say("  Its per-mode result is a consistency check, not a finding.")
    say("")
    say("  Interleaving is the binding cell everywhere. Its count here is")
    say("  still set by gate 2's whole-string matcher, whose recall on Dream")
    say("  step 18d put near 40%: repairing gate 2 (step 18f) should RAISE")
    say("  these counts and LOWER the MDE. The table above is the")
    say("  pessimistic case.")
    say("")

    rule()
    say("  STEP 19a COMPLETE.")
    say("")
    say("  One rule (src/label_failure_mode.py), one table")
    say("  (step19a_final_modes.csv). Every Phase E and F script reads these")
    say("  and nothing else.")
    say("")
    say("  Next: step 19b - src/features.py.")
    rule()
    finish(0)


# ---------------------------------------------------------------------------
#  Regression tests:  python step19a_final_modes.py --test
# ---------------------------------------------------------------------------

def _test() -> None:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {got!r} (want {want!r})")

    # The power calculation must reproduce the published decision exactly:
    # LLaDA triviaqa + hotpotqa, 264 interleaving, 1,406 locked_in, 2,898
    # correct -> pooled MDE 0.059.
    check("MDE reproduces step 14's pooled 0.059",
          round(mde(1406, 264, 2898), 3), 0.059)

    def row(**kw):
        r = dict(stable_top3=0.7, cands_top3=2.0, q_overlap=0.1, copy_run=1,
                 truncated=False, gold_ever=False, gold_testable=True)
        r.update(kw)
        return r

    check("dream cuts are exact decimals",
          (CUTS["dream"]["s_cut"], CUTS["dream"]["c_cut"]), (0.63, 3.0))
    check("llada cuts unchanged",
          (CUTS["llada"]["s_cut"], CUTS["llada"]["c_cut"]), (0.26, 3.4))
    check("the repaired rule never emits 'untestable'",
          classify(row(gold_testable=False), "llada"), "locked_in")
    check("the legacy rule still does, for the LOCK",
          classify_legacy(row(gold_testable=False), CUTS["llada"]),
          "untestable")
    check("an untested row never becomes interleaving",
          classify(row(gold_testable=False, gold_ever=True), "dream"),
          "locked_in")
    check("the same row splits by arm at s=0.50",
          (classify(row(stable_top3=0.50), "llada"),
           classify(row(stable_top3=0.50), "dream")),
          ("locked_in", "inconsistent"))
    check("c_cut boundary is inclusive on Dream",
          classify(row(cands_top3=3.0), "dream"), "locked_in")
    check("just past it is not",
          classify(row(cands_top3=3.1), "dream"), "inconsistent")
    check("truncated beats everything",
          classify(row(truncated=True, gold_ever=True), "dream"), "degenerate")
    check("NaN is unknown, never a number",
          classify(row(stable_top3=float("nan")), "dream"), "unknown")
    try:
        classify(row(), "gpt")
        check("an unknown arm raises", "no error", "ValueError")
    except ValueError:
        check("an unknown arm raises", "ValueError", "ValueError")
    try:
        r = row()
        del r["copy_run"]
        classify(r, "dream")
        check("a missing field raises", "no error", "KeyError")
    except KeyError:
        check("a missing field raises", "KeyError", "KeyError")

    print("\n  " + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    main()
