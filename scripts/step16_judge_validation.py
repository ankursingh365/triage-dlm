#!/usr/bin/env python3
"""
Step 13b - validate the Qwen3 judge against the human consensus
================================================================

    modal run modal_app.py::run_cpu --script step13b_judge_validation.py

CPU only, seconds, free. Reads one CSV.

THE PASS CRITERION, DECLARED BEFORE THE NUMBERS ARE SEEN
========================================================
This script was written before `step13a_judge_verdicts.csv` was opened by
either annotator. The criterion below is therefore pre-registered rather than
chosen to fit a result:

    PASS   accuracy against the human consensus >= 90%
           AND at least 10 points better than string matching on the same 100

    GATE   80-89%. The judge is used only where it is confident; the rest are
           either held out or sent to a human. Part E sizes that.

    FAIL   below 80%. The judge does not replace string matching, and the
           project needs a different approach to correctness.

Accuracy is meaningful here rather than a majority-class artefact: the
consensus splits 52 correct / 48 wrong.

**Beating string matching is not optional.** A judge that merely matches the
incumbent has bought nothing for its cost and complexity. Part C runs the
incumbent through the identical test so the comparison is like for like.

THE ANNOTATION, REPORTED HONESTLY
=================================
Two annotators, 100 items, blind to the judge's verdicts and to each other's
labels. Raw agreement 100/100, Cohen's kappa 1.000.

That number is real but it must be reported with what produced it, or it
overstates the independence:

    - Both annotators worked from the SAME written codebook: judge the fact
      the question asks for; a wrong detail that was not asked does not make
      the answer wrong. A shared codebook is correct practice - it is what
      makes labels comparable - but it means the two passes are not
      independent draws.
    - One worked example (#1) was resolved in discussion BEFORE labelling, so
      that item is not independent evidence at all.
    - Four further items (#30, #39, #46, #56) are the same pattern as #1 and
      follow mechanically from the codebook.
    - One (#43, corrupted output text) was covered by a stated rule.
    - Three genuinely ambiguous items were NOT covered by any rule and were
      decided independently: #47 (gold "English", answer "British"), #69 (gold
      "Vatican City", answer "Rome, Italy"), #98 (gold "Andaman Islands",
      answer "North Sentinel Island"). Both annotators reached y, n, n.

    - The LLM annotator also wrote the judge's prompt. Its labels alone are
      therefore not independent evidence about the judge.

The defensible claim is: **the correctness task is well defined enough that two
annotators working from one codebook did not diverge on any of 100 items.** It
is not: "two independent raters agree perfectly."

THE SAMPLE IS STRATIFIED, SO RAW ACCURACY IS NOT POPULATION ACCURACY
====================================================================
The 100 were drawn 50 from string-correct and 50 from string-wrong. The
population is 2,898 and 8,222. String-correct is over-sampled roughly 2x, so
reading the raw accuracy as the population accuracy would weight the easy
stratum twice as heavily as it deserves. Part F reweights by stratum size, and
uses the same weights to estimate the TRUE error rate of Phase C - the number
every table in the paper currently reports as 74% on string matching alone.
"""

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

VERDICT_CSV = config.TAB_DIR / "step13a_judge_verdicts.csv"
REPORT_PATH = config.OUT_DIR / "step13b_validation_report.txt"
OUT_CSV = config.TAB_DIR / "step13b_validation_detail.csv"

# Population stratum sizes, from step12c Part D.
N_POP_CORRECT, N_POP_WRONG = 2898, 8222
N_POP = N_POP_CORRECT + N_POP_WRONG

PASS_ACC, GATE_ACC, MIN_MARGIN = 0.90, 0.80, 0.10

# ---------------------------------------------------------------------------
# The two label sets. Kept separately rather than merged, so the agreement
# calculation is reproducible from this file alone.
# ---------------------------------------------------------------------------
ANKUR = """y n y n n n n y n n n y y n y y y n y y n y y n n n y n y y
n n y y n y y y y n n y y n n y y y n n y n n y n y y y n y
y y y n n n n n n n y n y n y y n n n y n y n n y y y y n n
y y n y y y n n y y""".split()

CLAUDE = """y n y n n n n y n n n y y n y y y n y y n y y n n n y n y y
n n y y n y y y y n n y y n n y y y n n y n n y n y y y n y
y y y n n n n n n n y n y n y y n n n y n y n n y y y y n n
y y n y y y n n y y""".split()

assert len(ANKUR) == 100 and len(CLAUDE) == 100

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval. Correct at k == n, where the normal
    approximation collapses to a zero-width interval and lies."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a: list, b: list) -> float:
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    pa, pb = a.count("y") / len(a), b.count("y") / len(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def rates(pred: list, truth: list, pos: str = "y") -> dict:
    tp = sum(1 for p, t in zip(pred, truth) if p == pos and t == pos)
    fp = sum(1 for p, t in zip(pred, truth) if p == pos and t != pos)
    fn = sum(1 for p, t in zip(pred, truth) if p != pos and t == pos)
    tn = sum(1 for p, t in zip(pred, truth) if p != pos and t != pos)
    n = len(pred)
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=(tp + tn) / n,
                recall=rec, spec=spec, bal=0.5 * (rec + spec),
                prec=tp / (tp + fp) if tp + fp else float("nan"))


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 13b: judge validation against human consensus")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not VERDICT_CSV.exists():
        say(f"\nMissing {VERDICT_CSV}. Run step13a first.")
        sys.exit(1)

    rows = []
    with open(VERDICT_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    rows.sort(key=lambda r: int(r["n"]))
    if len(rows) != 100:
        say(f"\nExpected 100 verdict rows, found {len(rows)}. Stopping.")
        sys.exit(1)

    judge = ["y" if r["judge_verdict"] == "correct" else "n" for r in rows]
    strm = ["y" if r["string_correct"].lower() == "true" else "n" for r in rows]
    conf = [float(r["judge_p_correct"]) for r in rows]
    mass = [float(r.get("yesno_mass", 1.0)) for r in rows]

    # =======================================================================
    say("")
    say("=" * 78)
    say("A. ANNOTATOR AGREEMENT")
    say("")
    agree = sum(a == c for a, c in zip(ANKUR, CLAUDE))
    lo, hi = wilson(agree, 100)
    say(f"  raw agreement   {agree}/100    95% CI [{lo:.3f}, {hi:.3f}]")
    say(f"  Cohen's kappa   {kappa(ANKUR, CLAUDE):.3f}")
    say(f"  annotator 1     y {ANKUR.count('y')}  n {ANKUR.count('n')}  "
        f"? {ANKUR.count('?')}")
    say(f"  annotator 2     y {CLAUDE.count('y')}  n {CLAUDE.count('n')}  "
        f"? {CLAUDE.count('?')}")
    diffs = [i for i, (a, c) in enumerate(zip(ANKUR, CLAUDE), 1) if a != c]
    say(f"  disagreements   {diffs if diffs else 'none'}")
    say("")
    say("  Read the docstring before quoting this. Both annotators used one")
    say("  written codebook, one item was resolved in discussion beforehand,")
    say("  and the LLM annotator wrote the judge's prompt. The claim this")
    say("  supports is that the TASK is well defined - not that the raters")
    say("  were independent.")

    # consensus. With no disagreements this is either set; the branch stays
    # so a future round with disagreements does not need a code change.
    consensus, unresolved = [], []
    for i, (a, c) in enumerate(zip(ANKUR, CLAUDE), 1):
        if a == c:
            consensus.append(a)
        else:
            consensus.append(None)
            unresolved.append(i)
    keep = [i for i, v in enumerate(consensus) if v is not None]
    if unresolved:
        say(f"  {len(unresolved)} items need adjudication and are excluded: "
            f"{unresolved}")
    C = [consensus[i] for i in keep]
    J = [judge[i] for i in keep]
    S = [strm[i] for i in keep]
    P = [conf[i] for i in keep]
    STRAT = [strm[i] for i in keep]          # which stratum each item came from

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. JUDGE vs CONSENSUS")
    say("")
    j = rates(J, C)
    say("                  consensus y   consensus n")
    say(f"  judge y         {j['tp']:11d}   {j['fp']:11d}")
    say(f"  judge n         {j['fn']:11d}   {j['tn']:11d}")
    jlo, jhi = wilson(j["tp"] + j["tn"], len(C))
    say("")
    say(f"  accuracy          {j['acc']:.0%}   95% CI [{jlo:.2f}, {jhi:.2f}]")
    say(f"  balanced accuracy {j['bal']:.0%}")
    say(f"  recall (correct)  {j['recall']:.0%}      "
        f"precision {j['prec']:.0%}")
    say(f"  specificity       {j['spec']:.0%}")
    say(f"  Cohen's kappa     {kappa(J, C):.3f}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. STRING MATCHING vs CONSENSUS - the incumbent, same test")
    say("")
    s = rates(S, C)
    say("                  consensus y   consensus n")
    say(f"  string y        {s['tp']:11d}   {s['fp']:11d}")
    say(f"  string n        {s['fn']:11d}   {s['tn']:11d}")
    say("")
    say(f"  accuracy          {s['acc']:.0%}")
    say(f"  balanced accuracy {s['bal']:.0%}")
    say(f"  Cohen's kappa     {kappa(S, C):.3f}")
    say("")
    margin = j["acc"] - s["acc"]
    say(f"  JUDGE MARGIN OVER STRING MATCHING: {margin:+.0%}")
    say("")
    say("  Remember this sample is stratified 50/50, not population-weighted.")
    say("  Part F reweights both.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. EVERY ITEM THE JUDGE GOT WRONG")
    say("")
    errs = [(int(rows[i]['n']), rows[i], judge[i], consensus[i], conf[i])
            for i in keep if judge[i] != consensus[i]]
    if not errs:
        say("  None.")
    else:
        say("    #   consensus  judge    P(correct)  stratum        mode")
        say("   ---  ---------  -------  ----------  -------------  ----------")
        for n, r, jv, cv, p in errs:
            strat = "string-correct" if r["string_correct"].lower() == "true" \
                else "string-wrong"
            say(f"   {n:3d}  {cv:<9}  {jv:<7}  {p:10.3f}  {strat:<13}  "
                f"{r['mode']}")
        say("")
        near = sum(1 for *_x, p in errs if 0.2 <= p <= 0.8)
        say(f"  {near} of {len(errs)} errors sit in the uncertain band "
            f"(0.2 - 0.8).")
        say("  Errors made confidently are the dangerous kind: a confidence")
        say("  gate cannot catch them, and they enter the dataset silently.")

    say("")
    low_mass = [int(rows[i]["n"]) for i in keep if mass[i] < 0.5]
    say(f"  items where yes/no held under half the distribution: "
        f"{low_mass if low_mass else 'none'}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("E. WOULD A CONFIDENCE GATE HELP?")
    say("")
    say("   band kept        n    accuracy   coverage")
    say("   ---------------  ---  ---------  --------")
    for band in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        idx = [i for i, p in enumerate(P) if p >= band or p <= 1 - band]
        if not idx:
            continue
        acc = sum(1 for i in idx if J[i] == C[i]) / len(idx)
        say(f"   |p-0.5| >= {band - 0.5:.2f}    {len(idx):3d}   {acc:8.1%}   "
            f"{len(idx)/len(C):7.0%}")
    say("")
    say("  A gate only earns its complexity if accuracy climbs materially as")
    say("  coverage falls. If the top row is already at the bottom row's")
    say("  accuracy, the judge's confidence carries no information and the")
    say("  gate is theatre.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("F. REWEIGHTED TO THE POPULATION")
    say("")
    wA = N_POP_CORRECT / N_POP
    wB = N_POP_WRONG / N_POP
    say(f"  stratum weights: string-correct {wA:.3f} ({N_POP_CORRECT:,}), "
        f"string-wrong {wB:.3f} ({N_POP_WRONG:,})")
    say("")

    def by_stratum(pred):
        out = {}
        for tag, want in (("string-correct", "y"), ("string-wrong", "n")):
            idx = [i for i, sv in enumerate(STRAT) if sv == want]
            hit = sum(1 for i in idx if pred[i] == C[i])
            out[tag] = (hit, len(idx), hit / len(idx) if idx else float("nan"))
        return out

    js, ss = by_stratum(J), by_stratum(S)
    say("  accuracy within each stratum")
    say("                    judge          string")
    for tag in ("string-correct", "string-wrong"):
        say(f"    {tag:<14}  {js[tag][0]:2d}/{js[tag][1]:<2d} {js[tag][2]:5.0%}"
            f"    {ss[tag][0]:2d}/{ss[tag][1]:<2d} {ss[tag][2]:5.0%}")
    jpop = wA * js["string-correct"][2] + wB * js["string-wrong"][2]
    spop = wA * ss["string-correct"][2] + wB * ss["string-wrong"][2]
    say("")
    say(f"  population-weighted accuracy   judge {jpop:.1%}   "
        f"string {spop:.1%}   margin {jpop - spop:+.1%}")

    # true error rate of Phase C
    trueA = sum(1 for i, sv in enumerate(STRAT) if sv == "y" and C[i] == "y")
    nA = sum(1 for sv in STRAT if sv == "y")
    trueB = sum(1 for i, sv in enumerate(STRAT) if sv == "n" and C[i] == "y")
    nB = sum(1 for sv in STRAT if sv == "n")
    pA, pB = trueA / nA, trueB / nB
    pop_correct = wA * pA + wB * pB
    say("")
    say("  TRUE correctness rate of Phase C, estimated from the consensus")
    say(f"    of the string-CORRECT stratum, actually correct: "
        f"{trueA}/{nA} = {pA:.0%}")
    say(f"    of the string-WRONG   stratum, actually correct: "
        f"{trueB}/{nB} = {pB:.0%}")
    say("")
    say(f"    population correct  {pop_correct:.1%}    "
        f"population error  {1 - pop_correct:.1%}")
    say(f"    vs string matching  {N_POP_CORRECT/N_POP:.1%} correct, "
        f"{N_POP_WRONG/N_POP:.1%} error")
    say("")
    est_wrong = (1 - pop_correct) * N_POP
    say(f"    estimated true wrong answers: {est_wrong:,.0f} "
        f"(string matching says {N_POP_WRONG:,})")
    say("")
    say("  Bootstrap 95% CI on the population error rate:")
    rng = np.random.default_rng(config.SEED)
    idxA = [i for i, sv in enumerate(STRAT) if sv == "y"]
    idxB = [i for i, sv in enumerate(STRAT) if sv == "n"]
    boots = []
    for _ in range(10000):
        a = rng.choice(idxA, size=len(idxA), replace=True)
        b = rng.choice(idxB, size=len(idxB), replace=True)
        qa = np.mean([C[i] == "y" for i in a])
        qb = np.mean([C[i] == "y" for i in b])
        boots.append(1 - (wA * qa + wB * qb))
    blo, bhi = np.percentile(boots, [2.5, 97.5])
    say(f"    [{blo:.1%}, {bhi:.1%}]   (n=50 per stratum, so it is wide)")

    # =======================================================================
    say("")
    say("=" * 78)
    say("G. VERDICT against the pre-registered criterion")
    say("")
    say(f"  required : accuracy >= {PASS_ACC:.0%} AND margin over string "
        f"matching >= {MIN_MARGIN:+.0%}")
    say(f"  measured : accuracy {j['acc']:.0%}   margin {margin:+.0%}")
    say("")
    if j["acc"] >= PASS_ACC and margin >= MIN_MARGIN:
        say("  PASS - run the judge on all 11,120.")
        say("")
        say("      modal run modal_app.py::run --script step13c_judge_full.py")
        say("")
        say(f"  The {len(errs)} errors above are the residual error of the final")
        say("  dataset. They are reported in the paper's data section as a rate,")
        say("  not assumed away.")
    elif j["acc"] >= GATE_ACC and margin >= MIN_MARGIN:
        say("  GATE - accuracy is between the two thresholds. Use the judge")
        say("  only inside the confident band from Part E, and decide what")
        say("  happens to the rest before running on 11,120.")
    else:
        say("  FAIL - the judge does not clear the bar that was set before")
        say("  the numbers were seen. Do NOT run it on 11,120 and do NOT move")
        say("  the threshold. The options are a better prompt, a bigger judge,")
        say("  or few-shot examples - each re-validated on a FRESH sample,")
        say("  because this 100 has now been used to select and cannot")
        say("  honestly be used to evaluate again.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "dataset", "qid", "mode", "string_correct",
                    "ankur", "claude", "consensus", "judge",
                    "judge_p_correct", "judge_right"])
        for i in range(len(rows)):
            c = consensus[i]
            w.writerow([rows[i]["n"], rows[i]["dataset"], rows[i]["qid"],
                        rows[i]["mode"], rows[i]["string_correct"],
                        ANKUR[i], CLAUDE[i], c or "", judge[i],
                        f"{conf[i]:.4f}",
                        "" if c is None else str(judge[i] == c)])
    say("")
    say("=" * 78)
    say(f"  detail CSV : {OUT_CSV}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
