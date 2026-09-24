#!/usr/bin/env python3
"""
Step 14 - how much data does the key experiment actually need?
===============================================================

    modal run modal_app.py::run_cpu --script step14_power_and_supply.py

CPU only, under a minute, free. Reads one CSV.

WHY THIS EXISTS
===============
Steps 10d, 12, 12b and 12c all measured supply against **150 per mode per
dataset** and all reported a shortfall in interleaving. That 150 was chosen
early, written into `TARGET_PER_CELL`, and never justified. It is a round
number, and four scripts have now reported against it as though it meant
something.

The quantity that matters is not a count. It is **the smallest difference in
detector AUROC between two failure modes that this data can detect**, because
that contrast IS the paper's result: hesitation-based detectors should be near
chance on locked-in errors and better on inconsistent ones. A sample size is
adequate when it can resolve the effect being claimed, and no earlier.

So this script replaces the target with a power calculation, and either
authorises more generation with a number attached or closes the question.

THE METHOD
==========
Hanley & McNeil (1982) give the standard error of an AUROC in closed form:

    Q1 = A/(2-A)        Q2 = 2A^2/(1+A)
    SE = sqrt( [ A(1-A) + (n_pos-1)(Q1 - A^2) + (n_neg-1)(Q2 - A^2) ]
               / (n_pos * n_neg) )

Part B cross-checks it against a bootstrap simulation at the real counts, so
the closed form is verified on this data rather than trusted from the paper.

For the DIFFERENCE between two modes, the two AUROCs share their negatives -
the same correct answers - which correlates them and SHRINKS the standard error
of their difference. Treating them as independent is therefore conservative,
and that is what is done here: the true MDE is a little smaller than reported.

    MDE = (z(1-a/2) + z(power)) * sqrt(SE1^2 + SE2^2)
        = (1.96 + 0.8416) * sqrt(SE1^2 + SE2^2)     at 80% power, a = 0.05

WHAT IT CANNOT TELL YOU
=======================
It assumes a true AUROC near 0.65, the level TraceDet reports on LLaDA. If a
detector lands near 0.50 on locked-in - which is what the thesis predicts - its
SE is slightly larger. Part C reports the MDE at several assumed AUROCs so the
sensitivity is visible rather than hidden in one number.

Planning figures are not results. The final per-mode table must carry bootstrap
CIs computed on the real scores.
"""

import csv
import math
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

MODES_CSV = config.TAB_DIR / "step12c_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step14_power_report.txt"

DATASETS = ("triviaqa", "hotpotqa")
MODES = ("locked_in", "interleaving", "inconsistent")

OLD_TARGET = 150            # the round number being retired
PLANNING_AUROC = 0.65       # TraceDet on LLaDA
POWER_Z = 1.96 + 0.8416     # 80% power, alpha 0.05 two-sided
SEC_PER_Q, USD_PER_HOUR = 3.09, 2.10

# The effect the thesis predicts. Declared here rather than compared against
# after the fact: if hesitation-based detectors are near chance on locked-in
# and useful on inconsistent, the gap is at least this wide.
CLAIMED_EFFECT = 0.10

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on erf. No scipy dependency."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def hm_se(A: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil standard error of an AUROC."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1 = A / (2 - A)
    q2 = 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n_pos - 1) * (q1 - A * A)
                      + (n_neg - 1) * (q2 - A * A)) / (n_pos * n_neg))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC. Ties get average ranks via argsort of argsort."""
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 14: power, and whether to generate more")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not MODES_CSV.exists():
        say(f"\nMissing {MODES_CSV}. Run step12c first.")
        sys.exit(1)

    counts, n_correct = defaultdict(Counter), Counter()
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ds = r["dataset"]
            if str(r["correct"]).lower() == "true":
                n_correct[ds] += 1
            else:
                counts[ds][r["mode"]] += 1

    say("")
    say("=" * 78)
    say("A. WHAT IS ACTUALLY ON DISK")
    say("")
    say("  Negatives for every AUROC are that dataset's CORRECT answers.")
    say("")
    say("  dataset      mode             n_pos   n_neg   vs old target")
    say("  -----------  -------------  -------  ------  --------------")
    for ds in DATASETS:
        for m in MODES:
            gap = OLD_TARGET - counts[ds][m]
            note = "-" if gap <= 0 else f"{gap} short"
            say(f"  {ds:<11}  {m:<13}  {counts[ds][m]:7,}  "
                f"{n_correct[ds]:6,}  {note:>14}")
    pooled = {m: sum(counts[ds][m] for ds in DATASETS) for m in MODES}
    pooled_neg = sum(n_correct[ds] for ds in DATASETS)
    say(f"  {'POOLED':<11}  {'':13}  {'':7}  {pooled_neg:6,}")
    for m in MODES:
        say(f"  {'':<11}  {m:<13}  {pooled[m]:7,}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. IS THE CLOSED FORM RIGHT? bootstrap cross-check at the real counts")
    say("")
    say("  Simulating scores whose true AUROC is "
        f"{PLANNING_AUROC}, then bootstrapping.")
    say("")
    say("  dataset      mode            analytic   simulated   diff")
    say("  -----------  -------------  ---------  ----------  ------")
    rng = np.random.default_rng(config.SEED)
    d = norm_ppf(PLANNING_AUROC) * math.sqrt(2)
    fatal, conservative = [], []
    for ds in DATASETS:
        for m in MODES:
            npos, nneg = counts[ds][m], n_correct[ds]
            if npos < 2 or nneg < 2:
                continue
            ana = 1.96 * hm_se(PLANNING_AUROC, npos, nneg)
            widths = []
            for _ in range(12):
                pos = rng.normal(d, 1, npos)
                neg = rng.normal(0, 1, nneg)
                bs = [auroc(rng.choice(pos, npos, True), rng.choice(neg, nneg, True))
                      for _ in range(150)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                widths.append((hi - lo) / 2)
            sim = float(np.mean(widths))
            gap = ana - sim
            say(f"  {ds:<11}  {m:<13}  {ana:9.3f}  {sim:10.3f}  {gap:+6.3f}")
            # The DIRECTION decides, not the size. The closed form is asymptotic
            # and loses accuracy at small n_pos - but there it over-states the
            # SE, which over-states the MDE, which asks for MORE data than
            # needed. That is the safe direction and is allowed to pass with a
            # note. Under-stating the SE is never safe: it would claim the data
            # resolves an effect it cannot.
            if gap < -0.02:
                fatal.append((ds, m, npos, gap))
            elif gap > 0.02:
                conservative.append((ds, m, npos, gap))
    say("")
    if fatal:
        say("  STOPPED - the closed form UNDER-states the standard error here:")
        for ds, m, npos, gap in fatal:
            say(f"    {ds} {m} (n_pos={npos}): analytic is {gap:+.3f} below simulated")
        say("  That would claim more resolving power than the data has, so")
        say("  Part C cannot be trusted. Fix before continuing.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    if conservative:
        say("  The closed form OVER-states the SE in these cells, which are too")
        say("  small for the asymptotics:")
        for ds, m, npos, gap in conservative:
            say(f"    {ds} {m} (n_pos={npos}): analytic is {gap:+.3f} above simulated")
        say("  Tolerated: it errs towards demanding more data, never less.")
    else:
        say("  The closed form holds on this data, in every cell.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. MINIMUM DETECTABLE DIFFERENCE, locked_in vs interleaving")
    say("")
    say("  80% power, alpha 0.05 two-sided. Conservative: the two AUROCs share")
    say("  their negatives, which shrinks the SE of the difference.")
    say("")
    say("  assumed AUROC   triviaqa   hotpotqa    POOLED")
    say("  -------------  ---------  ---------  --------")
    pooled_mde = None
    for A in (0.55, 0.60, 0.65, 0.70):
        row = []
        for ds in DATASETS:
            se = math.sqrt(hm_se(A, counts[ds]["locked_in"], n_correct[ds]) ** 2
                           + hm_se(A, counts[ds]["interleaving"], n_correct[ds]) ** 2)
            row.append(POWER_Z * se)
        se = math.sqrt(hm_se(A, pooled["locked_in"], pooled_neg) ** 2
                       + hm_se(A, pooled["interleaving"], pooled_neg) ** 2)
        p = POWER_Z * se
        if abs(A - PLANNING_AUROC) < 1e-9:
            pooled_mde = p
        mark = "  <- planning" if abs(A - PLANNING_AUROC) < 1e-9 else ""
        say(f"       {A:.2f}        {row[0]:9.3f}  {row[1]:9.3f}  {p:8.3f}{mark}")

    say("")
    say(f"  The effect the thesis predicts: at least {CLAIMED_EFFECT:.2f} AUROC")
    say("  points, because a locked-in error contains no hesitation for a")
    say("  hesitation-based detector to read.")
    say("")
    if pooled_mde <= CLAIMED_EFFECT:
        say(f"  POOLED MDE {pooled_mde:.3f} < claimed effect {CLAIMED_EFFECT:.2f}.")
        say("  The data already resolves the effect. Generation is NOT the")
        say("  bottleneck.")
    else:
        say(f"  POOLED MDE {pooled_mde:.3f} > claimed effect {CLAIMED_EFFECT:.2f}.")
        say("  More data IS needed. See Part D for how much.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. WHAT WOULD MORE GENERATION BUY?")
    say("")
    say("  Extra questions yield each mode at its observed rate. AUROC precision")
    say("  improves with sqrt(n), and interleaving is "
        f"{pooled['interleaving']/sum(pooled.values()):.1%} of usable wrong")
    say("  answers, so each extra point of precision costs about four times the")
    say("  last one.")
    say("")
    n_now = sum(sum(counts[ds].values()) for ds in DATASETS) + pooled_neg
    say("     extra q   interleaving n     MDE     cost")
    say("   ---------  --------------  ------  -------")
    for extra in (0, 2000, 6000, 12000, 24000):
        f = 1 + extra / n_now
        nI = round(pooled["interleaving"] * f)
        nL = round(pooled["locked_in"] * f)
        nN = round(pooled_neg * f)
        se = math.sqrt(hm_se(PLANNING_AUROC, nL, nN) ** 2
                       + hm_se(PLANNING_AUROC, nI, nN) ** 2)
        cost = extra * SEC_PER_Q / 3600 * USD_PER_HOUR
        say(f"   {extra:9,}  {nI:14,}  {POWER_Z*se:6.3f}  ${cost:6.2f}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("E. DECISION")
    say("")
    per_ds = [POWER_Z * math.sqrt(
        hm_se(PLANNING_AUROC, counts[ds]["locked_in"], n_correct[ds]) ** 2
        + hm_se(PLANNING_AUROC, counts[ds]["interleaving"], n_correct[ds]) ** 2)
        for ds in DATASETS]
    if pooled_mde <= CLAIMED_EFFECT:
        say("  STOP GENERATING.")
        say("")
        say(f"  The pooled contrast resolves differences of {pooled_mde:.3f} AUROC")
        say(f"  points; the claimed effect is {CLAIMED_EFFECT:.2f}. The 150-per-cell")
        say("  target is retired - it was a round number, and the shortfalls it")
        say("  reported were against nothing.")
        say("")
        say(f"  Per-dataset MDE is {per_ds[0]:.3f} and {per_ds[1]:.3f}, so a single")
        say("  dataset can only confirm a large effect. **Report the POOLED")
        say("  contrast as primary and per-dataset as a consistency check** - and")
        say("  fix that now, before seeing which one looks better.")
        say("")
        say("  Next: src/features.py (plan step 19), then the Ave Entropy anchor")
        say("  at 62-65 AUROC (step 20), which is the gate that makes every")
        say("  number after it defensible.")
    else:
        need = None
        for extra in range(0, 60001, 1000):
            f = 1 + extra / n_now
            se = math.sqrt(
                hm_se(PLANNING_AUROC, round(pooled["locked_in"] * f),
                      round(pooled_neg * f)) ** 2
                + hm_se(PLANNING_AUROC, round(pooled["interleaving"] * f),
                        round(pooled_neg * f)) ** 2)
            if POWER_Z * se <= CLAIMED_EFFECT:
                need = extra
                break
        say(f"  GENERATE MORE: about {need:,} additional questions "
            f"(${need*SEC_PER_Q/3600*USD_PER_HOUR:.2f}).")
        say("  Raise the targets in step11_phasec_generate.py and re-run it; it")
        say("  skips everything already on disk.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    say("")
    say("=" * 78)
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
