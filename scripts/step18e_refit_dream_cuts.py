#!/usr/bin/env python3
"""
Step 18e - fix gate 1, then refit Dream's locked_in cuts on the hand labels
==========================================================================

    modal run modal_app.py::run_cpu --script step18e_refit_dream_cuts.py

CPU only, seconds, free.

WHERE THIS COMES FROM
=====================
Step 18c: two blind annotators, kappa 0.797, 60/60 settled. Population-weighted,
Dream's locked_in share is 43.2% [27.4, 58.7] by hand against the classifier's
71.0% - a 27.8-point gap on the class the paper is about.

Step 18d decomposed that gap and found the two upstream gates net out to about
zero:

    0  as shipped                                 71.0%
    1  untestable falls through to L/C            73.8%   +2.8
    2  gold match repaired at recall 39%          70.6%   -3.2
       hand labels                                43.2%
       still unexplained after both gates        +27.5

So the cuts carry essentially the whole gap. This script closes it.

WHAT MOVES AND WHAT DOES NOT - declared before any number is computed
====================================================================
**Gate 1 is repaired.** An item whose gold cannot be tested for interleaving
skips the test it genuinely cannot run and continues to the locked_in /
inconsistent decision, carrying `int_untested = True` so the row still says so.
280 Dream rows and 220 LLaDA rows were being deleted from the analysis; three of
the four sampled ones are locked-in by hand. This must happen before the fit,
because it changes which rows the cuts decide.

**Only `s_cut` and `c_cut` move.** `e_cut`, `r_cut` and the rule ORDER stay at
LLaDA's values for Dream too. The echo stratum agreed 6/8 with the hand labels
and echo is 2.2% of Dream's wrong answers: there is no evidence to move it, and
every extra free parameter costs generalisation on sixty items.

**Gate 2 is NOT widened here.** Step 18d measured the gold matcher behaving very
differently on the two arms - raw precision 11/12 on Dream against 5/9 on LLaDA -
and two of LLaDA's four false fires sit on hand label X, which is a
correctness-label artefact rather than a matcher error, since `classify()` has no
correctness test at all. Widening it responsibly needs a per-arm reweighted
precision analysis, and LLaDA's round was sampled by `stable_frac` deciles with
the rare modes forced in, so its numbers need a different reweighting from
Dream's. That is step 18f.

Deferring it is defensible only because its size is known: gate 2 moves locked_in
by about 3.2 points, well inside the hand-label interval's +/-15. Part E
quantifies it anyway - the fit is repeated with the three known-missed items
relabelled I, and if the chosen cuts move, the deferral is withdrawn.

THE OBJECTIVE, DECLARED BEFORE THE FIT
======================================
Step 10d fitted these cuts by maximising balanced accuracy. Step 12b found that
objective to be wrong for this problem: balanced accuracy weights a missed X the
same as a destroyed locked-in case, and the paper's claim is not symmetric.

The claim is that trajectory detectors sit at chance ON LOCKED-IN ERRORS. That
claim needs the locked_in class to be PURE - a contaminated class would put the
at-chance result on a mixed bag and the finding would mean nothing - and it needs
the class large enough to have power. Precision first, then size. So:

    maximise   population-weighted RECALL of locked_in
    subject to (a) population-weighted PRECISION of locked_in >= 0.85
               (b) the resulting ARM SHARE of locked_in lands inside step 18c's
                   bootstrap interval for the hand labels, [27.4%, 58.7%]
    ties       -> the larger locked_in class (more power)
    ties       -> the more conservative pair (higher s_cut, then lower c_cut)

If no grid point satisfies both, nothing is adopted and the script says so. A
criterion that is quietly relaxed when it fails is not a criterion.

Constraint (b) was ADDED on 2026-09-25, after a dry run on synthetic data and
BEFORE any real fit. Precision alone let the criterion buy purity by collapsing
the class: the dry run chose s=0.87, c=3.9 at precision 0.873 and recall 0.216.
A rule that calls a seventh of the wrong answers locked_in when the hand labels
say two fifths is not more accurate than one that over-calls - it is the same
size of error pointing the other way, and it would quietly drain the power the
per-mode table needs. The interval is not invented here; it is the one step 18c
already published, so this is a pre-registered bound being enforced rather than
a new parameter being chosen.

Both rates are population-weighted, because the worksheet was drawn with
per-mode quotas and is not a random sample of the arm. Three numbers in step 18
were wrong for exactly that reason before they were caught.

Grids are the exact decimals of `claude/numeric-thresholds-rule.md`:
`np.arange(0.40, 1.01, 0.01)` produced an element printing as 0.75 that was
really 0.7500000000000003, which moved 4% of the dataset across a threshold.

IN-SAMPLE AGREEMENT IS NOT REPORTED AS A RESULT
===============================================
Two free parameters fitted on sixty items will agree with those sixty items
better than they will agree with anything else. The headline agreement figure is
leave-one-out cross-validated: for each item, refit on the other 59 and score
that item with cuts it did not see. In-sample is printed beside it so the gap -
the size of the overfitting - is visible rather than hidden.

LLADA IS NOT REFITTED
=====================
LLaDA's cuts were fitted and validated against LLaDA's own 59 hand labels and a
Dream round says nothing about them. But gate 1 is SHARED, so Part F applies the
gate-1 repair to LLaDA with its cuts unchanged and re-checks it against those 59
labels. A shared fix that helps one arm and hurts the other is not adopted.
"""

import csv
import json
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:                                  # pragma: no cover
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

CUTS_LLADA = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6)

# Exact decimals. NOT np.arange - see claude/numeric-thresholds-rule.md.
STABLE_GRID = [i / 100 for i in range(5, 90)]        # 0.05 .. 0.89
CANDS_GRID = [i / 10 for i in range(15, 55)]         # 1.5 .. 5.4

MIN_PRECISION = 0.85                                 # declared above
# step 18c's stratified bootstrap interval for the hand-label locked_in share.
# Copied, not recomputed, so the constraint cannot drift with a rerun.
SHARE_LO, SHARE_HI = 0.274, 0.587

DREAM_CSV = config.TAB_DIR / "step17d_dream_modes.csv"
LLADA_CSVS = [config.TAB_DIR / "step12c_final_modes.csv",
              config.TAB_DIR / "step12b_csqa_modes.csv"]
LLADA_CAL_CSV = config.TAB_DIR / "step10d_final_modes.csv"
LLADA_KEY_CSV = config.TAB_DIR / "step10c_answer_key.csv"
CONSENSUS_CSV = config.TAB_DIR / "step18c_consensus.csv"

OUT_CSV = config.TAB_DIR / "step18e_dream_modes.csv"
CUTS_JSON = config.TAB_DIR / "step18e_dream_cuts.json"
REPORT_PATH = config.OUT_DIR / "step18e_refit_dream_cuts.txt"

# step 18d, Part B: hand-labelled interleaving the gold matcher missed.
GATE2_MISSED = {11, 33, 48}

LLADA_HAND = ("L C C X C X C I X C . X X X C C I L C C I L L I L C C C C C "
              "L C L X X X X X X C C C I X X L C C C L L L X I X X X X L C").split()

MODES = ("locked_in", "interleaving", "inconsistent", "echo", "untestable",
         "degenerate")
NUMERIC = ("stable_top3", "cands_top3", "q_overlap", "copy_run")
BOOLEAN = ("correct", "truncated", "gold_ever", "gold_testable")
FULL_NEED = {"qid", "mode", *NUMERIC, *BOOLEAN}

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


# ---------------------------------------------------------------------------
#  Loading. Never substitutes a default for a column that does not exist.
# ---------------------------------------------------------------------------

def load(paths, label: str, need=None, fatal: bool = True) -> list:
    need = FULL_NEED if need is None else need
    out = []
    for p in ([paths] if isinstance(paths, Path) else paths):
        if not p.exists():
            say(f"  {label}: {p.name} not found - skipped.")
            continue
        with open(p, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            missing = sorted(need - set(rd.fieldnames or ()))
            if missing:
                say(f"  {'FATAL' if fatal else 'SKIPPED'} - {p.name} lacks: "
                    + ", ".join(missing))
                if fatal:
                    say("  No default is substituted for a missing column.")
                    finish(1)
                continue
            for r in rd:
                row = dict(qid=r["qid"], mode=(r["mode"] or "").strip(),
                           dataset=r.get("dataset", ""))
                for c in NUMERIC:
                    try:
                        row[c] = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        row[c] = None
                for c in BOOLEAN:
                    row[c] = str(r.get(c, "")).strip().lower() == "true"
                row["bad"] = any(row[c] is None for c in NUMERIC)
                out.append(row)
    return out


def wrong(rows: list) -> list:
    return [r for r in rows if r["mode"] in MODES]


# ---------------------------------------------------------------------------
#  The rule. Gate 1 repaired; everything else exactly as shipped.
# ---------------------------------------------------------------------------

def upstream(r, cuts) -> str:
    """The verdict BEFORE the cuts are consulted, or "" if the cuts decide.

    Separating this from the cut test is what lets the fit loop over 3,400
    candidate pairs without re-deciding the upstream classes 3,400 times, and
    makes it impossible for a cut to change an upstream verdict by accident.
    """
    if r["bad"]:
        return "unknown"
    if r["truncated"]:
        return "degenerate"
    if r["q_overlap"] >= cuts["e_cut"] and r["copy_run"] >= cuts["r_cut"]:
        return "echo"
    if r["gold_testable"] and r["gold_ever"]:
        return "interleaving"
    # gate 1 repaired: an untestable row no longer returns here. It reaches the
    # cuts, and `int_untested` records that interleaving could not be tested.
    return ""


def cut_verdict(r, cuts) -> str:
    return ("locked_in"
            if r["stable_top3"] >= cuts["s_cut"] and r["cands_top3"] <= cuts["c_cut"]
            else "inconsistent")


def classify(r, cuts, gate1_fixed: bool = True) -> str:
    if not gate1_fixed:
        if r["bad"]:
            return "unknown"
        if r["truncated"]:
            return "degenerate"
        if r["q_overlap"] >= cuts["e_cut"] and r["copy_run"] >= cuts["r_cut"]:
            return "echo"
        if not r["gold_testable"]:
            return "untestable"
        if r["gold_ever"]:
            return "interleaving"
        return cut_verdict(r, cuts)
    up = upstream(r, cuts)
    return up or cut_verdict(r, cuts)


# ---------------------------------------------------------------------------
#  Population-weighted precision and recall of locked_in
# ---------------------------------------------------------------------------

def make_weights(pool: list, arm_rows: list) -> dict:
    """qid -> weight, from the stratum the item was SAMPLED from.

    The worksheet's quota was applied to step 17d's stored `mode`, so that
    column - not any recomputed verdict - is the sampling stratum. Weight is
    (stratum's share of the arm) / (items from that stratum on the sheet), so
    the weighted counts below are estimates of arm counts.
    """
    n_arm = len(arm_rows)
    pop = Counter(r["mode"] for r in arm_rows)
    on_sheet = Counter(r["mode"] for _n, _lab, r in pool)
    return {r["qid"]: (pop[r["mode"]] / n_arm) / on_sheet[r["mode"]]
            for _n, _lab, r in pool if on_sheet[r["mode"]]}


def arm_share_table(rows: list, base: dict) -> list:
    """share[a][b] = the EXACT fraction of `rows` that (STABLE_GRID[a],
    CANDS_GRID[b]) calls locked_in.

    The prevalence constraint is a statement about the whole arm, and the whole
    arm is on disk - there is no reason to estimate it from sixty items. An
    earlier version did exactly that, using the sheet weights, and the estimate
    sat 8.5 points from the truth on a dry run. With n=20 in the largest
    stratum the standard error on a within-stratum rate is about 0.11, which at
    a stratum weight of 0.71 is +/-8 points on the share: the weighted estimate
    is simply too noisy to constrain anything.

    Computing it exactly for all 3,400 pairs costs one pass, not 3,400. A row
    with stable v and cands u is called locked_in by every pair with
    s_cut <= v and c_cut >= u, which is a RECTANGLE of the grid, so each row is
    a rectangle increment on a 2D difference array and one prefix sum turns the
    whole table out at the end. O(n + |grid|) instead of O(n * |grid|).

    `upstream` does not depend on s_cut or c_cut - e_cut and r_cut do not move -
    so the eligible set is fixed and computed once.
    """
    ns, nc = len(STABLE_GRID), len(CANDS_GRID)
    d = [[0] * (nc + 1) for _ in range(ns + 1)]
    for r in rows:
        if upstream(r, base):
            continue                      # decided before the cuts
        ia = bisect_right(STABLE_GRID, r["stable_top3"])   # flagged for a < ia
        ib = bisect_left(CANDS_GRID, r["cands_top3"])      # flagged for b >= ib
        if ia == 0 or ib >= nc:
            continue                      # no grid pair flags this row
        d[0][ib] += 1                     # rectangle [0,ia) x [ib,nc)
        d[ia][ib] -= 1
        d[0][nc] -= 1
        d[ia][nc] += 1
    for a in range(1, ns + 1):            # prefix down the rows
        for b in range(nc + 1):
            d[a][b] += d[a - 1][b]
    for a in range(ns + 1):               # prefix across the columns
        for b in range(1, nc + 1):
            d[a][b] += d[a][b - 1]
    n = len(rows) or 1
    return [[d[a][b] / n for b in range(nc)] for a in range(ns)]


def score(pool: list, w: dict, cuts: dict):
    """(precision, recall, weighted class size) of locked_in on the sheet.

    Only items whose consensus is L or C are scored: those are the only two
    verdicts the cuts can produce, so an item the hand labels call I or X is an
    error no (s_cut, c_cut) can repair. Part A counts those separately as the
    ceiling on any refit. The arm share is NOT computed here - see
    arm_share_table.
    """
    tp = fp = fn = 0.0
    for _n, lab, r in pool:
        if lab not in ("L", "C"):
            continue
        wt = w.get(r["qid"], 0.0)
        said_l = cut_verdict(r, cuts) == "locked_in"
        if said_l and lab == "L":
            tp += wt
        elif said_l:
            fp += wt
        elif lab == "L":
            fn += wt
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, tp + fp


def fit(pool: list, w: dict, base: dict, shares: list):
    """The declared criterion, over the exact decimal grids.

    Returns the pick and the feasible set, so the surface can be described
    rather than just its argmax reported. `shares` is arm_share_table's output:
    the prevalence constraint is exact, the precision constraint is not.
    """
    feasible = []
    for a, s in enumerate(STABLE_GRID):
        for b, c in enumerate(CANDS_GRID):
            share = shares[a][b]
            if not SHARE_LO <= share <= SHARE_HI:
                continue                  # cheapest test first
            prec, rec, size = score(pool, w, dict(base, s_cut=s, c_cut=c))
            if prec >= MIN_PRECISION:
                feasible.append((rec, size, s, -c, prec, share))
    # max recall; then larger class; then higher s_cut; then lower c_cut
    return (max(feasible) if feasible else None), feasible


# ---------------------------------------------------------------------------

def main() -> None:
    rule()
    say("  TRIAGE - Step 18e: gate 1 repair + refit Dream's locked_in cuts")
    say(f"  {datetime.now():%Y-%m-%d %H:%M}")
    rule()
    say("")
    say(f"  Declared criterion: maximise population-weighted RECALL of")
    say(f"  locked_in subject to population-weighted PRECISION >= "
        f"{MIN_PRECISION:.2f}.")
    say(f"  Grids: stable {STABLE_GRID[0]:.2f}..{STABLE_GRID[-1]:.2f} "
        f"({len(STABLE_GRID)} points), cands {CANDS_GRID[0]:.1f}.."
        f"{CANDS_GRID[-1]:.1f} ({len(CANDS_GRID)} points) = "
        f"{len(STABLE_GRID)*len(CANDS_GRID)} pairs.")
    say("  Only s_cut and c_cut move. LLaDA is not refitted.")
    say("")

    dream = wrong(load(DREAM_CSV, "Dream"))
    if not dream:
        say("  Dream's mode CSV is required. Run step17d first.")
        finish(1)
    llada = wrong(load(LLADA_CSVS, "LLaDA"))
    say(f"  Dream {len(dream):6d} wrong answers")
    say(f"  LLaDA {len(llada):6d} wrong answers")
    say("")

    # --- LOCK ------------------------------------------------------------
    rule("-")
    say("  LOCK - reproduce the stored modes with gate 1 DISABLED")
    rule("-")
    say("")
    say("  Cheap insurance: if this file's rule does not reproduce step 17d's")
    say("  output when the repair is switched off, the repair's measured")
    say("  effect is not the repair's effect.")
    say("")
    for name, rows in (("Dream", dream), ("LLaDA", llada)):
        if not rows:
            continue
        bad = sum(1 for r in rows
                  if classify(r, CUTS_LLADA, gate1_fixed=False) != r["mode"])
        say(f"    {name:<6} {len(rows)-bad:6d} / {len(rows):<6d} reproduced"
            f"{'' if not bad else f'   {bad} MISMATCH'}")
        if bad:
            say("")
            say("  LOCK FAILED. Stopping. Nothing is fitted or written.")
            finish(1)
    say("    LOCK PASSED.")
    say("")

    # --- the fit pool ----------------------------------------------------
    rule("-")
    say("  PART A - which sheet items the cuts actually decide")
    rule("-")
    if not CONSENSUS_CSV.exists():
        say(f"\n  {CONSENSUS_CSV.name} not found. Run step18c first.")
        finish(1)
    cons = {}
    with open(CONSENSUS_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("consensus") and r.get("qid"):
                cons[r["qid"]] = (int(r["n"]), r["consensus"])
    by_qid = {r["qid"]: r for r in dream}
    joined = [(n, lab, by_qid[q]) for q, (n, lab) in cons.items() if q in by_qid]
    say("")
    say(f"  {len(joined)}/{len(cons)} consensus items joined to a Dream row.")
    if len(joined) < len(cons):
        say("  A partial join means the consensus CSV and step17d came from")
        say("  different runs. Fix that before fitting anything.")
        finish(1)

    pool = [(n, lab, r) for n, lab, r in joined
            if not upstream(r, CUTS_LLADA)]
    excluded = [(n, lab, r) for n, lab, r in joined if upstream(r, CUTS_LLADA)]
    say("")
    say(f"  {len(pool)} items reach the cuts; {len(excluded)} are decided "
        "upstream.")
    say("")
    say("    decided upstream by   n   consensus says")
    up_by = defaultdict(list)
    for _n, lab, r in excluded:
        up_by[upstream(r, CUTS_LLADA)].append(lab)
    for k, labs in sorted(up_by.items()):
        c = Counter(labs)
        say(f"    {k:<20} {len(labs):3d}   "
            + " ".join(f"{l}{c[l]}" for l in "LICX" if c[l]))
    say("")
    pc = Counter(lab for _n, lab, _r in pool)
    say("    reaching the cuts:    " + " ".join(f"{l}{pc[l]}" for l in "LICX"
                                                if pc[l]))
    fixable = pc["L"] + pc["C"]
    say(f"    of those, {fixable} have consensus L or C - the only two verdicts")
    say(f"    the cuts can produce. The other {len(pool)-fixable} are errors no")
    say("    cut pair can repair; they are the ceiling on this refit.")
    say("")
    n_gate1 = sum(1 for _n, _lab, r in pool if not r["gold_testable"])
    say(f"    {n_gate1} of them reach the cuts only because gate 1 was repaired.")
    say("")
    if fixable < 20:
        say("  Fewer than twenty fittable items. Two free parameters on this")
        say("  many points will not generalise. Nothing is fitted.")
        finish(1)

    w = make_weights(joined, dream)

    # --- the fit ---------------------------------------------------------
    rule("-")
    say("  PART B - the fit")
    rule("-")
    say("")
    shares = arm_share_table(dream, CUTS_LLADA)
    ia0 = STABLE_GRID.index(CUTS_LLADA["s_cut"])
    ib0 = CANDS_GRID.index(CUTS_LLADA["c_cut"])
    prec0, rec0, size0 = score(pool, w, CUTS_LLADA)
    say(f"    LLaDA's cuts on Dream  s={CUTS_LLADA['s_cut']:.2f} "
        f"c={CUTS_LLADA['c_cut']:.1f}   precision {prec0:.3f}  recall "
        f"{rec0:.3f}  share {shares[ia0][ib0]:.1%}")
    say("")
    say("    The share is EXACT - computed over all "
        f"{len(dream)} wrong answers for every")
    say("    grid pair in one pass, not estimated from the sixty. Precision and")
    say("    recall can only come from the labels, so those carry the sampling")
    say("    noise and the share does not.")
    say(f"    constraints            precision >= {MIN_PRECISION:.2f}  AND  "
        f"share in [{SHARE_LO:.1%}, {SHARE_HI:.1%}]")
    say("")
    best, feasible = fit(pool, w, CUTS_LLADA, shares)
    if not best:
        say(f"    NO grid point satisfies both constraints.")
        say("    The declared criterion fails. Nothing is adopted and nothing")
        say("    is written. Relaxing a criterion after it fails is not a")
        say("    criterion.")
        say("")
        # Size the shortfall on each constraint separately, so the next move is
        # informed rather than guessed.
        all_pts = [score(pool, w, dict(CUTS_LLADA, s_cut=s, c_cut=c))
                   + (shares[a][b], s, c)
                   for a, s in enumerate(STABLE_GRID)
                   for b, c in enumerate(CANDS_GRID)]
        by_prec = max(all_pts)
        in_share = [p for p in all_pts if SHARE_LO <= p[3] <= SHARE_HI]
        say(f"    best precision anywhere:            {by_prec[0]:.3f} "
            f"(share {by_prec[3]:.1%})")
        if in_share:
            b = max(in_share)
            say(f"    best precision inside the share band: {b[0]:.3f} at "
                f"s={b[4]:.2f} c={b[5]:.1f}")
            say(f"    -> the share band is reachable; PRECISION is what fails.")
            say("")
            say("    The frontier - best precision reachable at each share:")
            say("")
            say("      share band     best precision   at")
            lo = SHARE_LO
            while lo < SHARE_HI - 1e-9:
                hi = min(lo + 0.05, SHARE_HI)
                bucket = [p for p in all_pts if lo <= p[3] < hi]
                if bucket:
                    t = max(bucket)
                    say(f"      {lo:5.1%}-{hi:5.1%}      {t[0]:12.3f}   "
                        f"s={t[4]:.2f} c={t[5]:.1f}  recall {t[1]:.3f}")
                lo = hi
            say("")
            say("    Read the frontier before deciding anything. If the best")
            say("    precision anywhere in the band is close to the bar, the")
            say("    problem is sixty labels' worth of noise and a larger")
            say("    labelled sample settles it. If it is far below, the two")
            say("    measures do not separate locked-in from inconsistent on an")
            say("    adaptive schedule, and the answer is a THIRD measure -")
            say("    see claude/stable-top3-measure-decision.md - not a new")
            say("    threshold on these two.")
            say("")
            say("    Either way, do NOT lower MIN_PRECISION to make this pass.")
            say("    A locked_in class that is one third inconsistent cases")
            say("    would put the paper's at-chance result on a mixed bag.")
        else:
            say("    NO point lands inside the share band at all.")
            say("    -> the two measures cannot produce the hand labels'")
            say("       prevalence at any threshold. A new measure is needed,")
            say("       not a new threshold.")
        finish(0)

    rec, size, s_pick, neg_c, prec, share = best
    c_pick = -neg_c
    cuts_new = dict(CUTS_LLADA, s_cut=s_pick, c_cut=c_pick)
    say(f"    feasible grid points   {len(feasible)} of "
        f"{len(STABLE_GRID)*len(CANDS_GRID)}")
    say(f"    chosen                 s={s_pick:.2f}  c={c_pick:.1f}"
        f"   precision {prec:.3f}  recall {rec:.3f}  share {share:.1%}")
    say("")
    say(f"    movement from LLaDA:   s_cut {CUTS_LLADA['s_cut']:.2f} -> "
        f"{s_pick:.2f}   c_cut {CUTS_LLADA['c_cut']:.1f} -> {c_pick:.1f}")
    say("")
    # Is the pick on a cliff or a plateau? A cut sitting on a cliff is an
    # overfit waiting to happen, and the only way to know is to look.
    neigh = []
    for ds in (-0.02, -0.01, 0.0, 0.01, 0.02):
        for dc in (-0.2, -0.1, 0.0, 0.1, 0.2):
            s2 = round(s_pick + ds, 2)
            c2 = round(c_pick + dc, 1)
            if s2 in STABLE_GRID and c2 in CANDS_GRID:
                p2, r2, _sz = score(pool, w,
                                    dict(CUTS_LLADA, s_cut=s2, c_cut=c2))
                neigh.append((p2, r2))
    if neigh:
        ps = [p for p, _ in neigh]
        rs = [r for _, r in neigh]
        say(f"    within +/-0.02 s and +/-0.2 c ({len(neigh)} points):")
        say(f"      precision {min(ps):.3f} .. {max(ps):.3f}   "
            f"recall {min(rs):.3f} .. {max(rs):.3f}")
        say("      A wide spread here means the pick sits on a cliff and the")
        say("      third decimal is doing work it cannot support.")
    say("")

    # --- Part C: honest agreement, leave-one-out -------------------------
    rule("-")
    say("  PART C - agreement: in-sample against leave-one-out")
    rule("-")
    say("")
    in_hits = sum(1 for _n, lab, r in pool
                  if lab in ("L", "C")
                  and (cut_verdict(r, cuts_new) == "locked_in") == (lab == "L"))
    say(f"    in-sample            {in_hits}/{fixable} = "
        f"{in_hits/fixable:.1%}")
    loo_hits, loo_picks = 0, Counter()
    for i, (n, lab, r) in enumerate(pool):
        if lab not in ("L", "C"):
            continue
        rest = pool[:i] + pool[i + 1:]
        b, _f = fit(rest, w, CUTS_LLADA, shares)
        if not b:
            continue
        cuts_i = dict(CUTS_LLADA, s_cut=b[2], c_cut=-b[3])
        loo_picks[(b[2], -b[3])] += 1
        if (cut_verdict(r, cuts_i) == "locked_in") == (lab == "L"):
            loo_hits += 1
    say(f"    leave-one-out        {loo_hits}/{fixable} = "
        f"{loo_hits/fixable:.1%}   <- the honest number")
    say(f"    overfitting gap      {(in_hits-loo_hits)/fixable:+.1%}")
    say("")
    say(f"    the LOO refits chose {len(loo_picks)} distinct cut pair(s):")
    for (s2, c2), k in loo_picks.most_common(5):
        say(f"      s={s2:.2f} c={c2:.1f}   {k:3d} folds"
            + ("   <- the full-sample pick"
               if (s2, c2) == (s_pick, c_pick) else ""))
    if len(loo_picks) > 4:
        say("      Many different pairs across folds means the surface is flat")
        say("      and the exact pick is not identified by sixty items. Report")
        say("      the pair, but do not read meaning into its precise value.")
    say("")

    # --- Part D: the whole arm, against the hand-label estimate ----------
    rule("-")
    say("  PART D - the refitted rule on all of Dream")
    rule("-")
    say("")
    n_d = len(dream)
    before = Counter(r["mode"] for r in dream)
    after = Counter(classify(r, cuts_new) for r in dream)
    say("    mode            step 17d   refitted      diff")
    for m in MODES:
        say(f"    {m:<15} {before[m]/n_d:8.1%} {after[m]/n_d:10.1%}"
            f"  {(after[m]-before[m])/n_d:+9.1%}")
    say("")
    # the hand-label target, recomputed by the same stratified estimator
    sheet_by_mode = defaultdict(list)
    for _n, lab, r in joined:
        sheet_by_mode[r["mode"]].append(lab)
    num = den = 0.0
    for m, cnt in Counter(r["mode"] for r in dream).items():
        obs = sheet_by_mode.get(m)
        if not obs:
            continue
        wt = cnt / n_d
        den += wt
        num += wt * (obs.count("L") / len(obs))
    hand = num / den if den else None
    if hand is not None:
        got = after["locked_in"] / n_d
        say(f"    hand labels, stratified estimator   {hand:7.1%}")
        say(f"    refitted rule                       {got:7.1%}")
        say(f"    remaining gap                      {got-hand:+8.1%}")
        say("")
        say("    Step 18c's bootstrap interval for the hand estimate was")
        say("    [27.4%, 58.7%]. A refitted share inside that interval is")
        say("    consistent with the labels; it is not a match, and sixty")
        say("    items cannot narrow it further.")
    say("")

    # --- Part E: is deferring gate 2 safe? -------------------------------
    rule("-")
    say("  PART E - sensitivity to gate 2, which was NOT repaired here")
    rule("-")
    say("")
    say("  Step 18d found three sheet items the gold matcher missed: "
        + ", ".join("#" + str(n) for n in sorted(GATE2_MISSED)) + ".")
    say("  If gate 2 were repaired they would leave the cuts' pool for")
    say("  interleaving. Refitting without them tests whether the deferral")
    say("  changed the answer.")
    say("")
    pool2 = [(n, lab, r) for n, lab, r in pool if n not in GATE2_MISSED]
    dropped = len(pool) - len(pool2)
    b2, _f2 = fit(pool2, w, CUTS_LLADA, shares)
    say(f"    {dropped} of the three were in the cuts' pool.")
    if b2:
        s2, c2 = b2[2], -b2[3]
        say(f"    refit without them:  s={s2:.2f}  c={c2:.1f}"
            f"   precision {b2[4]:.3f}  recall {b2[0]:.3f}")
        if (s2, c2) == (s_pick, c_pick):
            say("    IDENTICAL to the full-pool pick. Deferring gate 2 does not")
            say("    change the cuts, so the deferral stands and step 18f can")
            say("    take the matcher on its own terms.")
        else:
            say("    DIFFERENT from the full-pool pick. The deferral is")
            say("    withdrawn: gate 2 must be repaired before these cuts are")
            say("    adopted, because the fit is sensitive to it.")
    else:
        say("    No feasible point without them - report, adopt nothing.")
    say("")

    # --- Part F: the shared gate 1 repair must not hurt LLaDA ------------
    rule("-")
    say("  PART F - gate 1 is SHARED: does the repair hurt LLaDA?")
    rule("-")
    say("")
    say("  LLaDA's cuts are NOT refitted. Only the gate 1 repair is applied,")
    say("  and only to check that a shared fix does not break the arm it was")
    say("  not diagnosed on.")
    say("")
    if not llada:
        say("  LLaDA's mode CSVs are absent - this check is OWED.")
    elif not LLADA_KEY_CSV.exists():
        say(f"  {LLADA_KEY_CSV.name} absent, so LLaDA's 59 hand labels cannot")
        say("  be attached. The share table below stands; the agreement check")
        say("  is OWED.")
        n_l = len(llada)
        b4 = Counter(r["mode"] for r in llada)
        af = Counter(classify(r, CUTS_LLADA) for r in llada)
        for m in MODES:
            say(f"    {m:<15} {b4[m]/n_l:8.1%} -> {af[m]/n_l:7.1%}")
    else:
        pos2qid = {}
        with open(LLADA_KEY_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    pos2qid[int(r["n"])] = r.get("qid", "")
                except (KeyError, TypeError, ValueError):
                    continue
        poolL = {r["qid"]: r for r in llada}
        for r in wrong(load(LLADA_CAL_CSV, "LLaDA-cal", fatal=False)):
            poolL.setdefault(r["qid"], r)
        pairs = [(i, LLADA_HAND[i - 1], poolL[pos2qid[i]])
                 for i in range(1, 61)
                 if LLADA_HAND[i - 1] in ("L", "I", "C", "X")
                 and pos2qid.get(i) in poolL]
        say(f"  {len(pairs)}/59 LLaDA hand labels attached.")
        say("")
        for tag, fixed in (("as shipped", False), ("gate 1 repaired", True)):
            hit = tot = 0
            for _i, lab, r in pairs:
                v = classify(r, CUTS_LLADA, gate1_fixed=fixed)
                if v in ("locked_in", "inconsistent"):
                    tot += 1
                    if (v == "locked_in") == (lab == "L"):
                        hit += 1
            say(f"    {tag:<18} L-vs-C agreement {hit}/{tot}"
                + (f" = {hit/tot:.1%}" if tot else ""))
        say("")
        say("    If the repair lowers this, it is not adopted for either arm")
        say("    and step 18f revisits what `untestable` should mean.")
    say("")

    # --- write ------------------------------------------------------------
    rule("-")
    say("  WRITING")
    rule("-")
    say("")
    say(f"  step17d_dream_modes.csv is NOT modified. The refitted labels go to")
    say(f"  a new file, so every number already reported stays reproducible.")
    say("")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["qid", "dataset", "old_mode", "mode", "int_untested",
                     "stable_top3", "cands_top3", "q_overlap", "copy_run",
                     "truncated", "gold_ever", "gold_testable"])
        for r in dream:
            wr.writerow([r["qid"], r["dataset"], r["mode"],
                         classify(r, cuts_new), not r["gold_testable"],
                         r["stable_top3"], r["cands_top3"], r["q_overlap"],
                         r["copy_run"], r["truncated"], r["gold_ever"],
                         r["gold_testable"]])
    CUTS_JSON.write_text(json.dumps(dict(
        arm="dream", fitted=datetime.now().isoformat(timespec="seconds"),
        cuts=cuts_new, llada_cuts=CUTS_LLADA, gate1_fixed=True,
        gate2_fixed=False, criterion=f"max recall s.t. precision >= "
                                     f"{MIN_PRECISION}",
        n_fit=fixable, loo_agreement=round(loo_hits / fixable, 4),
        in_sample_agreement=round(in_hits / fixable, 4)), indent=2),
        encoding="utf-8")
    say(f"    {OUT_CSV}")
    say(f"    {CUTS_JSON}")
    say("")
    say("  `int_untested` marks the rows gate 1 used to delete. Any per-mode")
    say("  table in step 23 must be reported with and without them.")
    say("")

    rule()
    say(f"  DREAM CUTS: s_cut={s_pick:.2f}  c_cut={c_pick:.1f}  "
        f"e_cut={CUTS_LLADA['e_cut']:.2f}  r_cut={CUTS_LLADA['r_cut']}")
    say(f"  LLADA CUTS: unchanged at s_cut={CUTS_LLADA['s_cut']:.2f}  "
        f"c_cut={CUTS_LLADA['c_cut']:.1f}")
    say("")
    say("  Two arms, two operating points, each validated against its own hand")
    say("  labels. That is reported as a finding about the two samplers, not")
    say("  hidden as a fix.")
    rule()
    finish(0)


# ---------------------------------------------------------------------------
#  Regression tests:  python step18e_refit_dream_cuts.py --test
# ---------------------------------------------------------------------------

def _test() -> None:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {got!r} (want {want!r})")

    def row(**kw):
        r = dict(qid="q", dataset="d", mode="locked_in", stable_top3=0.5,
                 cands_top3=2.0, q_overlap=0.1, copy_run=1, correct=False,
                 truncated=False, gold_ever=False, gold_testable=True,
                 bad=False)
        r.update(kw)
        return r

    C = CUTS_LLADA
    check("grids are exact decimals",
          (0.75 in STABLE_GRID, 3.4 in CANDS_GRID), (True, True))
    check("no float noise in the stable grid",
          any(abs(x - 0.75) > 1e-12 and repr(x).startswith("0.75")
              for x in STABLE_GRID), False)
    check("gate 1 off: untestable returns early",
          classify(row(gold_testable=False), C, gate1_fixed=False),
          "untestable")
    check("gate 1 on: untestable reaches the cuts",
          classify(row(gold_testable=False), C), "locked_in")
    check("gate 1 on: untestable never becomes interleaving",
          classify(row(gold_testable=False, gold_ever=True), C), "locked_in")
    check("truncated still wins",
          classify(row(truncated=True, gold_ever=True), C), "degenerate")
    check("echo still needs both terms",
          classify(row(q_overlap=0.9, copy_run=2), C), "locked_in")

    # score(): a pool where the perfect cut exists, and precision/recall are
    # hand-computable. Equal weights so the arithmetic is checkable by eye.
    pool = [(1, "L", row(qid="a", stable_top3=0.9, cands_top3=1.5)),
            (2, "L", row(qid="b", stable_top3=0.9, cands_top3=1.5)),
            (3, "C", row(qid="c", stable_top3=0.1, cands_top3=9.0)),
            (4, "C", row(qid="d", stable_top3=0.9, cands_top3=1.5))]
    w = {k: 1.0 for k in "abcd"}
    prec, rec, size = score(pool, w, dict(C, s_cut=0.26, c_cut=3.4))
    check("precision = 2 of 3 flagged", round(prec, 4), round(2 / 3, 4))
    check("recall = 2 of 2 real L", rec, 1.0)
    check("weighted class size", size, 3.0)

    # Items whose consensus is I or X must be invisible to precision, recall
    # and class size - no cut pair can fix them - but they MUST count towards
    # the arm share, because the share is what the rule does to the arm and
    # does not care what the labels say. Two rows that the cuts flag therefore
    # leave the first three numbers alone and add 2.0 to the share.
    pool2 = pool + [(5, "I", row(qid="e", stable_top3=0.9, cands_top3=1.5)),
                    (6, "X", row(qid="f", stable_top3=0.9, cands_top3=1.5))]
    w2 = dict(w, e=1.0, f=1.0)
    a = score(pool, w, dict(C, s_cut=0.26, c_cut=3.4))
    b = score(pool2, w2, dict(C, s_cut=0.26, c_cut=3.4))
    check("I and X rows do not enter precision, recall or class size", b, a)

    # arm_share_table against brute force. Every grid pair, both ways, on rows
    # with values that land on and between grid points, including exact ties -
    # the float bug of step 12b lived exactly there.
    rng = __import__("random").Random(7)
    arm = []
    for i in range(400):
        arm.append(row(qid=f"x{i}",
                       stable_top3=rng.choice(STABLE_GRID + [0.0, 1.0,
                                              round(rng.uniform(0, 1), 3)]),
                       cands_top3=rng.choice(CANDS_GRID + [0.5, 9.9,
                                             round(rng.uniform(1, 6), 2)]),
                       truncated=rng.random() < 0.05,
                       gold_ever=rng.random() < 0.1,
                       gold_testable=rng.random() > 0.05,
                       q_overlap=round(rng.uniform(0, 1), 2),
                       copy_run=rng.randint(0, 9)))
    tab = arm_share_table(arm, C)
    worst, n_arm = 0.0, len(arm)
    for ai, s2 in enumerate(STABLE_GRID):
        for bi, c2 in enumerate(CANDS_GRID):
            cuts2 = dict(C, s_cut=s2, c_cut=c2)
            brute = sum(1 for r in arm
                        if classify(r, cuts2) == "locked_in") / n_arm
            worst = max(worst, abs(brute - tab[ai][bi]))
    check("share table matches brute force on all 3,400 pairs",
          round(worst, 12), 0.0)

    # The criterion must refuse rather than settle for low precision.
    # Precision cannot be met: ten C rows and one L row all above the cuts,
    # so anything the rule flags is at best 1/11 precise. Weights are scaled so
    # the share constraint is satisfiable and only precision can bite.
    allC = [(i, "C", row(qid=f"z{i}", stable_top3=0.9, cands_top3=1.5))
            for i in range(10)]
    allC.append((99, "L", row(qid="zl", stable_top3=0.9, cands_top3=1.5)))
    wC = {f"z{i}": 0.04 for i in range(10)}
    wC["zl"] = 0.04
    flat = [[0.4] * len(CANDS_GRID) for _ in STABLE_GRID]   # share always OK
    best, feas = fit(allC, wC, C, flat)
    check("precision constraint refuses rather than settles",
          (best is None, len(feas)), (True, 0))

    # The share constraint must bite on its own: one pure L row whose weight
    # puts the arm share far below SHARE_LO. Precision is a perfect 1.0.
    # One L row the cuts flag, plus enough rows they never flag that the arm
    # share is 0.10 - below SHARE_LO. Precision is a perfect 1.0, and it is
    # still rejected, because purity bought by shrinking the class is exactly
    # what constraint (b) exists to refuse.
    tiny = [(1, "L", row(qid="t1", stable_top3=0.95, cands_top3=1.5))]
    tiny += [(i, "C", row(qid=f"t{i}", stable_top3=0.01, cands_top3=9.9))
             for i in range(2, 11)]
    tiny_rows = [r for _n, _l, r in tiny]
    tiny_shares = arm_share_table(tiny_rows, C)
    wt_t = {r["qid"]: 0.1 for r in tiny_rows}
    bt, ft = fit(tiny, wt_t, C, tiny_shares)
    pt, rt, _s = score(tiny, wt_t, dict(C, s_cut=0.26, c_cut=3.4))
    ia, ib = STABLE_GRID.index(0.26), CANDS_GRID.index(3.4)
    check("purity bought by shrinking the class is rejected on prevalence",
          (round(pt, 3), round(rt, 3), round(tiny_shares[ia][ib], 3),
           bt is None, len(ft)),
          (1.0, 1.0, 0.1, True, 0))

    print("\n  " + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    main()
