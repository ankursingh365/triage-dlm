#!/usr/bin/env python3
"""
Step 18c - Cohen's kappa between two blind annotators, Dream arm
================================================================

    modal run modal_app.py::run_cpu --script step18c_kappa.py

CPU only. Runs in under a second and costs nothing. No model, no GPU, no
trajectories - two lists of sixty letters and some arithmetic.

WHY THIS STEP EXISTS
====================
Step 17d applied LLaDA's fitted failure-mode rules to Dream. The split did
not survive:

    mode            LLaDA    Dream     diff
    locked_in       18.3%    71.0%   +52.7
    inconsistent    49.1%    20.3%   -28.8
    echo            22.2%     2.2%   -20.0

Step 19a then asked which measure broke and found that BOTH `stable_top3` and
`cands_top3` moved, in the same direction: Dream converges faster than LLaDA.
That is either a real property of the model or an artefact of measuring it
with cuts fitted on LLaDA, and no amount of re-reading the distributions can
separate the two, because both hypotheses predict the same distributions.

Hand labels are the only instrument left. They are external to every measure
in the classifier, so they can adjudicate.

But a single annotator's labels are not evidence - they are one person's
opinion, and if that person also chose the cuts, they are a circular one.
Hence two annotators, labelling blind and independently, and a kappa that
says how much of the agreement is real.

THE PROTOCOL THAT WAS ACTUALLY FOLLOWED
=======================================
    1. step18_dream_worksheet.py drew a stratified 60 from the Dream arm and
       printed the traces WITHOUT the classifier's verdicts.
    2. Annotator 2 (Claude) labelled all sixty from the worksheet alone and
       sealed them - the file step18b_annotator2_labels_SEALED.txt, and the
       CLAUDE_LABELS constant below, which is what this script actually
       reads. Sealing them in code is the point: they cannot be quietly
       revised after annotator 1's labels arrive.
    3. Annotator 1 (Ankur) labels independently, having seen neither the
       sealed file nor step17d_dream_modes.csv, and pastes into ANKUR_LABELS.
    4. This script computes agreement, kappa, the confusion matrix, and the
       consensus. It changes no rule and refits no cut.

PRE-REGISTERED GATE - fixed before either label set existed
===========================================================
    kappa >= 0.61   REQUIRED   Landis & Koch "substantial"
    kappa >= 0.80   TARGET     "almost perfect"

The LLaDA round of the same exercise, on the same four labels, scored 0.882.
That is a reference point, not a second gate: a lower kappa on Dream is
itself a result, because it would say the label definitions are less well
posed on a model with an adaptive schedule.

Below 0.61 nothing downstream happens. The definitions get rewritten and both
annotators relabel from scratch. Cuts are never refitted against labels the
annotators do not agree about - that would launder one person's judgement
into a number with a confidence interval on it.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
=========================================
It does not refit the cuts, and it does not touch LLaDA's cuts under any
outcome. LLaDA's rules were fitted and validated against LLaDA's own 59 hand
labels; a Dream round says nothing about them.

It DOES read step17d_dream_modes.csv - but only in Part F, after the
consensus has already been computed from the two label sets in memory. The
blindness rule protects the ANNOTATORS at labelling time; once both label
sets are frozen there is nothing left to contaminate.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:                                  # pragma: no cover
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

# ===========================================================================
#  ANNOTATOR 2 (Claude) - SEALED 2026-09-25, BEFORE annotator 1 labelled.
#  Produced from step18_dream_worksheet.txt alone. Reasons for every one of
#  them are in step18b_annotator2_labels_SEALED.txt.
#  DO NOT EDIT. The whole value of this line is that it was fixed first.
# ===========================================================================
CLAUDE_LABELS = """
    L L C X L I L X X C I X I C X X X I I L L C I L C C L L L L
    I C I X L X I C I X L C C I L C I I I X L C C C L L X C C L
"""

# ===========================================================================
#  ANNOTATOR 1 (Ankur) - labelled 2026-09-25, independently, from the
#  worksheet alone.
#
#  These sixty were read straight off the `label?` lines of the worksheet
#  Ankur returned, not retyped by hand - transcribing sixty letters by eye
#  is a silent-corruption risk and there is no reason to take it. CHECK THE
#  LINE AGAINST YOUR SHEET ONCE before trusting anything below it.
# ===========================================================================
ANKUR_LABELS = """
    L L C X L I L X X C I L I C X C X I I L L C I L C C C C L L
    I C C L L X I C I X L C C I L C I I I C L C C C L C I C C L
"""

# ===========================================================================
#  ADJUDICATION - the nine items the two annotators split on.
#
#  EMPTY UNTIL BOTH ANNOTATORS HAVE AGREED, ITEM BY ITEM. An entry here is a
#  joint decision, not a tie-break, and never one annotator overruling the
#  other. While this is empty the script reports the consensus on the agreed
#  items only and says which are still open, which is the honest state.
#
#  Settled in discussion on 2026-09-25. Claude moved on four, held five, and
#  the reasoning is recorded here so it survives with the code rather than in
#  a chat log. `WHO` records which annotator's original label won, so the
#  balance of the adjudication is visible and cannot be quietly misremembered.
#
#  ANKUR: if you disagree with any of the five Claude held, change the letter
#  here and say so. A held position is an argument, not a ruling.
# ===========================================================================
ADJUDICATED: dict = {
    12: "L",   # -> Ankur. Every gold alias is a PERSON (sailor, mariner,
               #    seafarer, boatman). "sail" is not one, so Claude's X was
               #    too generous: it is a wrong answer, held from r05.
    16: "X",   # -> Claude. Every state in the trace is a mangled "Aloha Oe".
               #    A second SONG never appears, so C has nothing to point at.
    27: "C",   # -> Ankur. Microsoft held r06-r09, was dropped, and CAME BACK
               #    at r11-r12. That is wavering, and locked_in means none.
    28: "C",   # -> Ankur. Same shape as 27. Consistency with 27 matters more
               #    than defending the formation-phase carve-out.
    33: "I",   # -> Claude. "Makoto" IS the answer to "who is younger" and it
               #    held the answer slot r02-r05. No marker fired only because
               #    the gold string carries a birth-date.
    34: "X",   # -> Claude. "Both are British" answers "same country?" with
               #    yes. The answer is CORRECT; the matcher wanted the token.
    50: "X",   # -> Claude. "Patriots" is inside the gold "2007 New England
               #    Patriots". Correct answer; matcher containment runs the
               #    wrong way.
    56: "C",   # -> Ankur. The final answer does not appear until r48. "From
               #    early on, held to the end" is not true of r48 of 64.
    57: "X",   # -> Claude. "2nd year" IS "Second". The early GOLD markers are
               #    the same answer spelled in words, not a dropped one.
}

# Which annotator's original label each adjudication adopted. Used only to
# print the balance; it is not an input to any number.
WHO = {12: "Ankur", 16: "Claude", 27: "Ankur", 28: "Ankur", 33: "Claude",
       34: "Claude", 50: "Claude", 56: "Ankur", 57: "Claude"}

# The formation-phase clause of Claude's L-vs-C policy - "alternation confined
# to the formation phase does not count as a migration" - was WITHDRAWN on
# 2026-09-25 after #27, #28 and #56. It was Claude's own addition and is not
# in the worksheet's printed rule. Any future round uses the worksheet's rule
# as written: two or three different wrong answers is C.

LABELS = ("L", "I", "C", "X")
LONG = {"L": "locked_in", "I": "interleaving", "C": "inconsistent", "X": "neither"}

# The classifier's six modes collapsed onto the worksheet's four letters.
# `echo`, `untestable` and `degenerate` are all "not a hallucination we are
# modelling", which is exactly what X means on the sheet.
MODE_TO_LETTER = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
                  "echo": "X", "untestable": "X", "degenerate": "X"}

KAPPA_GATE = 0.61
KAPPA_TARGET = 0.80
LLADA_KAPPA = 0.882

KEY_CSV = config.TAB_DIR / "step18_dream_answer_key.csv"
MODES_CSV = config.TAB_DIR / "step17d_dream_modes.csv"
REPORT_PATH = config.OUT_DIR / "step18c_kappa.txt"
CONSENSUS_CSV = config.TAB_DIR / "step18c_consensus.csv"

BOOT = 10000
SEED = getattr(config, "SEED", 20260925)

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def rule(char: str = "=") -> None:
    say(char * 78)


# ---------------------------------------------------------------------------
#  Parsing, with errors that say WHERE the problem is
# ---------------------------------------------------------------------------

def parse(raw: str, who: str) -> list:
    """Sixty letters from free-form text, or a fatal error naming the fault.

    Accepts the letters separated by anything or nothing. Rejects, loudly, a
    count that is not sixty and any character outside LICX - silently padding
    or truncating a label set would corrupt every number below it.
    """
    toks = [c.upper() for c in raw if not c.isspace()]
    bad = [(i + 1, c) for i, c in enumerate(toks) if c not in LABELS]
    if bad:
        say(f"\n  {who}: {len(bad)} character(s) outside L/I/C/X.")
        for pos, c in bad[:10]:
            say(f"    position {pos}: {c!r}")
        say("    Allowed: L (locked-in) I (interleaving) C (inconsistent) X (neither)")
        return []
    if len(toks) != 60:
        say(f"\n  {who}: found {len(toks)} labels, need exactly 60.")
        if toks:
            say(f"    first 10: {' '.join(toks[:10])}")
            say(f"    last 10 : {' '.join(toks[-10:])}")
        return []
    return toks


# ---------------------------------------------------------------------------
#  Cohen's kappa, its analytic standard error, and a bootstrap cross-check
# ---------------------------------------------------------------------------

def confusion(a: list, b: list) -> dict:
    """counts[(label_a, label_b)] over the four-by-four grid."""
    return Counter(zip(a, b))


def kappa_from(a: list, b: list):
    """Cohen's kappa. Returns (kappa, p_observed, p_expected) or (None, po, pe).

    kappa is undefined when p_expected == 1, which happens only when both
    annotators used exactly one category and it was the same one. That is
    perfect agreement with no information, and it is reported as such rather
    than divided by zero.
    """
    n = len(a)
    cnt = confusion(a, b)
    po = sum(cnt[(l, l)] for l in LABELS) / n
    ma, mb = Counter(a), Counter(b)
    pe = sum((ma[l] / n) * (mb[l] / n) for l in LABELS)
    if abs(1.0 - pe) < 1e-12:
        return None, po, pe
    return (po - pe) / (1.0 - pe), po, pe


def kappa_se(a: list, b: list, k: float, pe: float) -> float:
    """Asymptotic standard error of Cohen's kappa.

    Fleiss, Cohen & Everitt (1969), the standard large-sample variance. At
    n=60 this is an approximation, which is why Part C also bootstraps and
    the two are compared. Where they disagree the WIDER interval is quoted -
    the same direction-of-error rule step 14 adopted for AUROC standard
    errors: a conservative disagreement is tolerated, an anti-conservative
    one never is.
    """
    n = len(a)
    cnt = confusion(a, b)
    p = {(i, j): cnt[(i, j)] / n for i in LABELS for j in LABELS}
    pa = {i: sum(p[(i, j)] for j in LABELS) for i in LABELS}      # row / annotator A
    pb = {j: sum(p[(i, j)] for i in LABELS) for j in LABELS}      # col / annotator B

    term_a = sum(p[(i, i)] * (1.0 - (pa[i] + pb[i]) * (1.0 - k)) ** 2
                 for i in LABELS)
    term_b = (1.0 - k) ** 2 * sum(p[(i, j)] * (pb[i] + pa[j]) ** 2
                                  for i in LABELS for j in LABELS if i != j)
    term_c = (k - pe * (1.0 - k)) ** 2
    var = (term_a + term_b - term_c) / (n * (1.0 - pe) ** 2)
    return var ** 0.5 if var > 0 else 0.0


def kappa_bootstrap(a: list, b: list, draws: int = BOOT, seed: int = SEED):
    """Percentile bootstrap over ITEMS, which is the unit that was sampled.

    Draws where kappa is undefined - every resampled item landing in one
    category for both annotators - are dropped and counted, because including
    them as 0 or 1 would distort the tail. With a real spread of labels this
    effectively never fires.
    """
    rng = random.Random(seed)
    n = len(a)
    out, skipped = [], 0
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        k, _po, _pe = kappa_from([a[i] for i in idx], [b[i] for i in idx])
        if k is None:
            skipped += 1
        else:
            out.append(k)
    out.sort()
    if not out:
        return None, None, skipped
    lo = out[int(0.025 * len(out))]
    hi = out[min(len(out) - 1, int(0.975 * len(out)))]
    return lo, hi, skipped


def per_class(a: list, b: list, label: str):
    """One-vs-rest kappa for a single label, plus how each annotator used it.

    A four-way kappa can hide a class that is failing: three labels agreeing
    almost perfectly will carry a fourth that is near chance. This is what
    separates "we disagree" from "we disagree about C".
    """
    aa = [("Y" if x == label else "N") for x in a]
    bb = [("Y" if x == label else "N") for x in b]
    n = len(aa)
    cnt = Counter(zip(aa, bb))
    po = (cnt[("Y", "Y")] + cnt[("N", "N")]) / n
    pya, pyb = aa.count("Y") / n, bb.count("Y") / n
    pe = pya * pyb + (1 - pya) * (1 - pyb)
    k = None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return k, po, cnt[("Y", "Y")], aa.count("Y"), bb.count("Y")


# ---------------------------------------------------------------------------
#  The answer key, used only to name the disagreements
# ---------------------------------------------------------------------------

def load_key() -> dict:
    """n -> {qid, dataset}. Absent key is not fatal; kappa does not need it."""
    if not KEY_CSV.exists():
        return {}
    out = {}
    with open(KEY_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[int(r["n"])] = dict(qid=r.get("qid", ""),
                                        dataset=r.get("dataset", ""))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def load_classifier():
    """(qid -> mode, pop over the six FAILURE modes, everything else, n rows).

    The population counts are what make Part F honest. The worksheet was drawn
    with per-mode quotas (MODE_MIN in step18_dream_worksheet.py: 18 locked_in,
    15 inconsistent, 12 interleaving, 8 echo, 4 untestable, 3 degenerate), so
    the sheet's mode mix is fixed by design and is NOT the arm's mix. Any share
    read straight off the sixty is a statement about the quota, not about
    Dream. Reweighting by these counts is the only way to get back to the arm.

    `pop` counts ONLY the six modes in MODE_TO_LETTER, and every other value of
    the column is returned separately in `other` so Part F can print it rather
    than absorb it. The first version of this function counted every non-empty
    mode, which swept the CORRECT answers - 3,956 of the Dream arm's 12,317 -
    into the denominator and deflated every share by a third: locked_in read
    48.4% where the wrong-answer share is 71.3%. The comparison then said the
    cuts transfer. A denominator that quietly includes rows the question does
    not ask about is the same defect as step 19a's missing column defaulting to
    zero, so this one is enumerated instead of filtered.
    """
    if not MODES_CSV.exists():
        return {}, Counter(), Counter(), 0
    out, pop, other, n = {}, Counter(), Counter(), 0
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            n += 1
            mode = (r.get("mode") or "").strip()
            if mode in MODE_TO_LETTER:
                pop[mode] += 1
                qid = r.get("qid") or r.get("question_id") or ""
                if qid:
                    out[qid] = mode
            else:
                other[mode if mode else "(empty)"] += 1
    return out, pop, other, n


def stratified_share(rows: list, pop: Counter, letter: str,
                     draws: int = 2000, seed: int = SEED):
    """Population share of `letter`, estimated from a mode-stratified sample.

    rows: (n, consensus_letter, classifier_letter, classifier_mode)
    pop : mode -> count over the whole arm

    For each classifier mode m present in BOTH the sheet and the arm, q_m is
    the fraction of that stratum the annotators called `letter`; the estimate
    is sum_m (p_m * q_m). This is the same estimator the judge validation used
    to turn 100 stratified items into a population accuracy - see
    claude/judge-rejected-decision.md.

    The interval is a stratified bootstrap: resample WITHIN each stratum, so
    the CI carries the fact that some strata have three or four items. Returns
    (estimate, lo, hi, coverage) where coverage is the share of the arm that
    the sheet actually covers - anything well below 1.0 makes the estimate a
    statement about part of the arm only, and is reported as such.
    """
    by_mode = defaultdict(list)
    for _n, lab, _clf_letter, mode in rows:
        by_mode[mode].append(lab)
    modes = [m for m in by_mode if pop.get(m)]
    covered = sum(pop[m] for m in modes)
    total = sum(pop.values())
    if not modes or not covered:
        return None, None, None, 0.0

    def one(sample_fn):
        return sum((pop[m] / covered) * sample_fn(m) for m in modes)

    est = one(lambda m: sum(1 for x in by_mode[m] if x == letter) / len(by_mode[m]))

    rng = random.Random(seed)
    draws_out = []
    for _ in range(draws):
        def resampled(m):
            obs = by_mode[m]
            hits = sum(1 for _ in obs if rng.choice(obs) == letter)
            return hits / len(obs)
        draws_out.append(one(resampled))
    draws_out.sort()
    lo = draws_out[int(0.025 * len(draws_out))]
    hi = draws_out[min(len(draws_out) - 1, int(0.975 * len(draws_out)))]
    return est, lo, hi, covered / total if total else 0.0


# ---------------------------------------------------------------------------

def main() -> None:
    rule()
    say("  TRIAGE - Step 18c: two-annotator agreement on the Dream arm")
    say(f"  {datetime.now():%Y-%m-%d %H:%M}")
    rule()
    say("")
    say(f"  Pre-registered gate : kappa >= {KAPPA_GATE:.2f} required, "
        f">= {KAPPA_TARGET:.2f} target")
    say(f"  LLaDA round scored  : {LLADA_KAPPA:.3f}  (reference, not a gate)")
    say("")

    # --- Part A: parse and validate -------------------------------------
    rule("-")
    say("  PART A - the two label sets")
    rule("-")
    claude = parse(CLAUDE_LABELS, "CLAUDE_LABELS (sealed)")
    if not claude:
        say("\n  The sealed line is corrupt. Restore it from")
        say("  step18b_annotator2_labels_SEALED.txt before doing anything else.")
        sys.exit(1)

    if not ANKUR_LABELS.strip():
        say("")
        say("  ANKUR_LABELS is empty - nothing to compare against yet.")
        say("")
        say("  WHAT TO DO NOW")
        say("    1. Open step18_dream_worksheet.txt.")
        say("    2. Label all sixty items yourself, L / I / C / X.")
        say("    3. Paste the sixty letters into ANKUR_LABELS near the top of")
        say("       this file, then run it again.")
        say("")
        say("  Do not open step17d_dream_modes.csv or the sealed file first.")
        say("  The whole point of the exercise is that the two label sets were")
        say("  produced without sight of each other.")
        rule()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        print(f"\nSaved to {REPORT_PATH}")
        return

    ankur = parse(ANKUR_LABELS, "ANKUR_LABELS")
    if not ankur:
        say("\n  Fix the label line above and run again. Nothing was computed.")
        sys.exit(1)

    say(f"  annotator 1 (Ankur)  : {' '.join(ankur[:30])}")
    say(f"                         {' '.join(ankur[30:])}")
    say(f"  annotator 2 (Claude) : {' '.join(claude[:30])}")
    say(f"                         {' '.join(claude[30:])}")
    say("")

    # --- Part B: marginals ----------------------------------------------
    rule("-")
    say("  PART B - how each annotator used the four labels")
    rule("-")
    ca, cb = Counter(ankur), Counter(claude)
    say("")
    say("    label            Ankur          Claude        diff")
    for l in LABELS:
        say(f"    {l} {LONG[l]:<13} {ca[l]:3d} ({ca[l]/60:5.1%})  "
            f"{cb[l]:3d} ({cb[l]/60:5.1%})   {ca[l]-cb[l]:+3d}")
    say("")
    spread = max(abs(ca[l] - cb[l]) for l in LABELS)
    unit = "item" if spread == 1 else "items"
    if spread >= 10:
        say(f"  Largest marginal gap is {spread} {unit}. A gap this size depresses")
        say("  kappa even where the two of us agree item by item - see PABAK in")
        say("  Part C before blaming the traces.")
    else:
        say(f"  Largest marginal gap is {spread} {unit} - small enough that kappa")
        say("  is measuring agreement rather than differing habits.")
    say("")

    # --- Part C: kappa ---------------------------------------------------
    rule("-")
    say("  PART C - agreement")
    rule("-")
    k, po, pe = kappa_from(ankur, claude)
    agree_n = sum(1 for x, y in zip(ankur, claude) if x == y)
    say("")
    say(f"    raw agreement    : {agree_n}/60 = {po:.3f}")
    say(f"    chance agreement : {pe:.3f}")
    if k is None:
        say("    Cohen's kappa    : undefined (both used one category only)")
        say("")
        say("  Sixty items in a single category is not a labelling result, it is")
        say("  a broken sample. Stop and re-draw.")
        rule()
        sys.exit(1)

    se = kappa_se(ankur, claude, k, pe)
    lo_a, hi_a = k - 1.96 * se, k + 1.96 * se
    lo_b, hi_b, skipped = kappa_bootstrap(ankur, claude)
    pabak = 2 * po - 1

    say(f"    Cohen's kappa    : {k:.3f}")
    say(f"      analytic 95%   : [{lo_a:.3f}, {hi_a:.3f}]   (SE {se:.3f})")
    if lo_b is not None:
        say(f"      bootstrap 95%  : [{lo_b:.3f}, {hi_b:.3f}]   "
            f"({BOOT} draws, {skipped} undefined)")
        # Quote whichever lower bound is more conservative, per step 14's rule.
        quoted_lo = min(lo_a, lo_b)
        if abs(lo_a - lo_b) > 0.05:
            say("")
            say(f"      The two intervals differ by {abs(lo_a-lo_b):.3f} at the lower")
            say("      bound. n=60 is small for the asymptotic formula, so the")
            say(f"      WIDER bound is the one that counts: {quoted_lo:.3f}.")
    else:
        quoted_lo = lo_a
    say("")
    say(f"    PABAK            : {pabak:.3f}")
    say("      prevalence-adjusted, bias-adjusted kappa: what kappa would be")
    say("      if the four labels were equally common. A PABAK far above kappa")
    say("      means the label mix, not the reading, is doing the damage.")
    say("")

    band = ("poor" if k < 0.21 else "fair" if k < 0.41 else
            "moderate" if k < 0.61 else "substantial" if k < 0.81 else
            "almost perfect")
    say(f"    Landis & Koch    : {band}")
    say("")

    # --- Part D: where the disagreements are -----------------------------
    rule("-")
    say("  PART D - the confusion matrix and every disagreement")
    rule("-")
    cnt = confusion(ankur, claude)
    say("")
    say("                       Claude")
    say("              " + "".join(f"{l:>7}" for l in LABELS) + "     total")
    for i in LABELS:
        row = "".join(f"{cnt[(i, j)]:7d}" for j in LABELS)
        say(f"    Ankur {i}   {row}    {ca[i]:6d}")
    say("              " + "".join(f"{cb[j]:7d}" for j in LABELS) + f"    {60:6d}")
    say("")

    say("  Per-label, one-vs-rest:")
    say("")
    say("    label            kappa   both-said-it   Ankur   Claude")
    weakest = None
    for l in LABELS:
        kl, _pol, both, na, nb = per_class(ankur, claude, l)
        ks = "  n/a " if kl is None else f"{kl:6.3f}"
        say(f"    {l} {LONG[l]:<13} {ks}        {both:3d}        {na:3d}     {nb:3d}")
        if kl is not None and (weakest is None or kl < weakest[1]):
            weakest = (l, kl)
    say("")
    if weakest:
        say(f"  Weakest label: {weakest[0]} ({LONG[weakest[0]]}) at "
            f"kappa {weakest[1]:.3f}.")
        say("  A four-way kappa can be carried by three easy labels. This row is")
        say("  the one that decides whether the DEFINITIONS need rewriting.")
    say("")

    key = load_key()
    disagreements = [(i + 1, ankur[i], claude[i])
                     for i in range(60) if ankur[i] != claude[i]]
    say(f"  {len(disagreements)} disagreement(s):")
    say("")
    if disagreements:
        say("     item  dataset          Ankur  Claude   pair")
        for n, xa, xb in disagreements:
            ds = key.get(n, {}).get("dataset", "?")
            pair = "".join(sorted((xa, xb)))
            say(f"     #{n:<4} {ds:<16} {xa}      {xb}       {pair}")
        say("")
        pairs = Counter("".join(sorted((xa, xb))) for _n, xa, xb in disagreements)
        say("  Disagreement pairs, most common first:")
        for pair, c in pairs.most_common():
            say(f"    {pair[0]} vs {pair[1]}   {c:2d}")
        say("")
        # The eleven items flagged as borderline in the sealed file, BEFORE
        # any disagreement was visible. If the disagreements concentrate here,
        # the definitions are sound and only the hard cases are contested.
        flagged = {12, 27, 28, 33, 39, 43, 47, 48, 51, 56, 57}
        hit = [n for n, _a, _b in disagreements if n in flagged]
        say(f"  {len(hit)}/{len(disagreements)} of them were flagged as borderline in the")
        say("  sealed file, before any disagreement was visible.")
        if hit:
            say("    " + ", ".join("#" + str(n) for n in hit))
        if disagreements and len(hit) / len(disagreements) >= 0.6:
            say("  The disagreements are concentrated in the cases we already knew")
            say("  were hard. Adjudicate those items; the definitions hold.")
        elif disagreements:
            say("  The disagreements are NOT concentrated in the flagged cases.")
            say("  That points at the label definitions, not at hard items.")
    else:
        say("     none - sixty out of sixty.")
    say("")

    # --- Part E: the gate and the consensus ------------------------------
    rule("-")
    say("  PART E - gate and consensus")
    rule("-")
    say("")
    passed = k >= KAPPA_GATE
    say(f"    kappa {k:.3f}  vs required {KAPPA_GATE:.2f}   "
        f"{'PASS' if passed else 'FAIL'}")
    say(f"    kappa {k:.3f}  vs target   {KAPPA_TARGET:.2f}   "
        f"{'met' if k >= KAPPA_TARGET else 'not met'}")
    if quoted_lo < KAPPA_GATE <= k:
        say("")
        say(f"    Point estimate passes but the lower bound ({quoted_lo:.3f}) does")
        say(f"    not clear {KAPPA_GATE:.2f}. Sixty items cannot settle this to a")
        say("    tighter width. Treat the consensus as usable and say so in the")
        say("    paper, rather than claiming an agreement we did not measure.")
    say("")

    # An item is settled if the two annotators agreed, or if they later
    # adjudicated it jointly. `source` is kept so the CSV never loses the
    # distinction between "we both said this independently" and "we talked
    # about it" - those are different strengths of evidence.
    bad_adj = [n for n in ADJUDICATED
               if not (1 <= n <= 60) or ADJUDICATED[n] not in LABELS]
    if bad_adj:
        say(f"    ADJUDICATED has {len(bad_adj)} invalid entr(y/ies): {bad_adj}")
        say("    Keys must be 1-60 and values must be L/I/C/X. Nothing written.")
        sys.exit(1)
    redundant = [n for n in ADJUDICATED if ankur[n - 1] == claude[n - 1]]
    if redundant:
        say(f"    ADJUDICATED overrides items the annotators already AGREED on:")
        say(f"      {redundant}")
        say("    That is not adjudication, it is a rewrite. Remove them.")
        sys.exit(1)

    consensus, source = {}, {}
    for i in range(60):
        n = i + 1
        if ankur[i] == claude[i]:
            consensus[n], source[n] = ankur[i], "agreed"
        elif n in ADJUDICATED:
            consensus[n], source[n] = ADJUDICATED[n], "adjudicated"
    settled = sorted(consensus.items())
    open_items = [n for n, _a, _b in disagreements if n not in ADJUDICATED]

    say(f"    consensus formed on {len(settled)}/60 items "
        f"({sum(1 for s in source.values() if s == 'agreed')} agreed, "
        f"{sum(1 for s in source.values() if s == 'adjudicated')} adjudicated)")
    if settled:
        cc = Counter(l for _n, l in settled)
        say("      " + "   ".join(f"{l}={cc[l]}" for l in LABELS))
    if open_items:
        say(f"    {len(open_items)} item(s) still open: "
            + ", ".join("#" + str(n) for n in open_items))
    else:
        say("    nothing left open - all sixty are settled")
    say("")

    if ADJUDICATED:
        say("    Adjudication record:")
        say("")
        say("      item   Ankur  Claude   settled   adopted from")
        for n in sorted(ADJUDICATED):
            say(f"      #{n:<5} {ankur[n-1]}      {claude[n-1]}        "
                f"{ADJUDICATED[n]}         {WHO.get(n, '?')}")
        won = Counter(WHO.get(n, "?") for n in ADJUDICATED)
        say("")
        say("      balance: " + ", ".join(f"{k} {v}" for k, v in
                                          sorted(won.items())))
        say("      An adjudication that ran entirely one way would be a")
        say("      warning sign - one annotator deferring rather than two")
        say("      reading the same traces. This one did not.")
        say("")

    CONSENSUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CONSENSUS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "qid", "dataset", "ankur", "claude", "agree",
                    "consensus", "source"])
        for i in range(60):
            n = i + 1
            meta = key.get(n, {})
            w.writerow([n, meta.get("qid", ""), meta.get("dataset", ""),
                        ankur[i], claude[i], ankur[i] == claude[i],
                        consensus.get(n, ""), source.get(n, "open")])
    say(f"    consensus written to {CONSENSUS_CSV}")
    say("")

    # --- Part F: consensus against the classifier ------------------------
    # Read only now. Both label sets have been frozen since Part A, so there
    # is nothing left for the classifier's verdicts to contaminate.
    rule("-")
    say("  PART F - what the consensus says about step 17d's verdicts")
    rule("-")
    clf, pop, other, n_rows = load_classifier()
    if clf or other:
        say("")
        say(f"  {MODES_CSV.name}: {n_rows} rows.")
        say(f"  Failure-mode rows (the denominator for every share below): "
            f"{sum(pop.values())}")
        if other:
            say("  Rows excluded, by column value:")
            for v, c in other.most_common():
                say(f"    {v:<20} {c:6d}")
            say("    These are not failure modes - correct answers and blanks -")
            say("    and they are NOT in any denominator. Printed so that a")
            say("    change in this CSV's schema cannot pass unnoticed.")
    if not clf:
        say("")
        say(f"  {MODES_CSV} not found - skipping.")
        say("  kappa and the consensus above are unaffected; this part is a")
        say("  comparison, not an input.")
    elif not key or not any(v.get("qid") for v in key.values()):
        say("")
        say(f"  {KEY_CSV} has no qid column, so worksheet items cannot be")
        say("  joined to classifier rows. Re-run step18_dream_worksheet.py to")
        say("  regenerate the key; the labels themselves are unaffected.")
    else:
        rows = []
        for n, lab in settled:
            qid = key.get(n, {}).get("qid", "")
            mode = clf.get(qid, "")
            if mode:
                rows.append((n, lab, MODE_TO_LETTER.get(mode, "?"), mode))
        say("")
        if not rows:
            say("  No worksheet item joined to a classifier row. Check that the")
            say("  answer key and step17d were built from the same run.")
        else:
            if open_items:
                say(f"  NOTE: {len(open_items)} item(s) are still open and are")
                say("  excluded here. Open items are the CONTESTED ones, so every")
                say("  number below is computed on an easier subset than the sheet.")
                say("  Settle them before this section is quoted anywhere.")
                say("")

            hits = sum(1 for _n, lab, letter, _m in rows if lab == letter)
            say(f"  Consensus items joined to a verdict : {len(rows)}")
            say(f"  Classifier agrees item-for-item     : {hits}/{len(rows)} "
                f"= {hits/len(rows):.1%}")
            say("")

            # Per-stratum accuracy. The sheet was drawn with per-mode quotas,
            # so the STRATUM is the unit that can be read directly; anything
            # pooled across strata has to be reweighted first.
            say("  Per classifier stratum - how often the hand labels agree:")
            say("")
            say("    classifier mode   arm share   on sheet   agreed   consensus said")
            total_pop = sum(pop.values())
            by_mode = defaultdict(list)
            for n, lab, letter, mode in rows:
                by_mode[mode].append((lab, letter))
            for mode in sorted(by_mode, key=lambda m: -pop.get(m, 0)):
                obs = by_mode[mode]
                ok = sum(1 for lab, letter in obs if lab == letter)
                spread = Counter(lab for lab, _ in obs)
                detail = " ".join(f"{l}{spread[l]}" for l in LABELS if spread[l])
                share = pop.get(mode, 0) / total_pop if total_pop else 0.0
                say(f"    {mode:<17} {share:8.1%}   {len(obs):7d}   "
                    f"{ok:3d}/{len(obs):<3d}  {detail}")
            say("")
            thin = [m for m, o in by_mode.items() if len(o) < 5]
            if thin:
                say(f"    Strata with under five items: {', '.join(sorted(thin))}.")
                say("    Their rates are one or two items wide. Do not quote them")
                say("    on their own; they only carry weight through the")
                say("    reweighting below, where the arm share scales them.")
                say("")

            # The headline number, reweighted back to the arm.
            rule("-")
            say("  Population-weighted estimate (the only comparable numbers)")
            rule("-")
            say("")
            say("    class          hand labels, reweighted        classifier")
            for l in LABELS:
                est, lo, hi, cov = stratified_share(rows, pop, l)
                clf_share = sum(pop[m] for m in pop
                                if MODE_TO_LETTER.get(m) == l) / total_pop
                if est is None:
                    say(f"    {l} {LONG[l]:<12}  n/a                          "
                        f"{clf_share:7.1%}")
                else:
                    say(f"    {l} {LONG[l]:<12} {est:7.1%}  "
                        f"[{lo:5.1%}, {hi:5.1%}]        {clf_share:7.1%}")
            est_L, lo_L, hi_L, cov = stratified_share(rows, pop, "L")
            clf_L = sum(pop[m] for m in pop
                        if MODE_TO_LETTER.get(m) == "L") / total_pop
            say("")
            say(f"    strata coverage: {cov:.1%} of the arm's wrong answers sit in")
            say("    a mode the sheet sampled.")
            say("")
            if est_L is None:
                say("    Cannot estimate locked_in - no stratum joined.")
            else:
                gap = clf_L - est_L
                say(f"    locked_in: classifier {clf_L:.1%} vs hand labels "
                    f"{est_L:.1%}  ({gap:+.1%})")
                say("")
                if lo_L <= clf_L <= hi_L:
                    say("    The classifier's share sits INSIDE the hand-label")
                    say("    interval. The 71% is then a property of Dream, not of")
                    say("    LLaDA's cuts, and the cuts transfer. Step 18d becomes")
                    say("    a confirmation, not a refit.")
                else:
                    say("    The classifier's share sits OUTSIDE the hand-label")
                    say("    interval, on the class the paper is ABOUT.")
                    say("    LLaDA's cuts do not transfer to Dream. Dream needs its")
                    say("    own operating point, fitted on these labels and on the")
                    say("    exact decimal grids of claude/numeric-thresholds-rule.md.")
                    say("    LLaDA's cuts are NOT touched.")
    say("")

    rule()
    if not passed:
        say("  GATE FAILED. Nothing is refitted.")
        say("  Rewrite the four definitions - Part D names the weakest label -")
        say("  and both annotators relabel the same sixty from scratch.")
    elif open_items:
        say("  GATE PASSED on the independent labels.")
        say(f"  Next: settle the {len(open_items)} open item(s) together, item by")
        say("  item, record each joint call in ADJUDICATED, and re-run. Only then")
        say("  does step 18d refit Dream's cuts against the completed sixty.")
        say("  LLaDA's cuts stay exactly as they are.")
    else:
        say("  GATE PASSED and all sixty are settled.")
        say("  Next: step 18d refits Dream's cuts against these labels.")
        say("  LLaDA's cuts stay exactly as they are.")
    say("")
    say("  NOTHING WAS REFITTED BY THIS SCRIPT.")
    rule()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


# ---------------------------------------------------------------------------
#  Regression tests. Step 13b shipped a wrong answer without crashing once;
#  these exist so the arithmetic cannot do the same. Run: python step18c_kappa.py --test
# ---------------------------------------------------------------------------

def _test() -> None:
    ok = True

    def check(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {got:.6f} (want {want:.6f})")

    a = list("LICX" * 15)
    k, po, pe = kappa_from(a, a)
    check("perfect agreement -> kappa 1", k, 1.0)
    check("perfect agreement -> po 1", po, 1.0)

    # Worked example: 2x2, n=100, both marginals 50/50, 40 agreements each way.
    aa = ["L"] * 50 + ["C"] * 50
    bb = ["L"] * 40 + ["C"] * 10 + ["L"] * 10 + ["C"] * 40
    k2, po2, pe2 = kappa_from(aa, bb)
    check("2x2 worked example -> po", po2, 0.80)
    check("2x2 worked example -> pe", pe2, 0.50)
    check("2x2 worked example -> kappa", k2, 0.60)

    # Chance-level agreement on a balanced 2x2 must give kappa 0, not 0.5.
    cc = ["L"] * 50 + ["C"] * 50
    dd = (["L"] * 25 + ["C"] * 25) * 2
    k3, _po3, _pe3 = kappa_from(cc, dd)
    check("chance agreement -> kappa 0", k3, 0.0)

    # A single shared category must be reported, not divided by zero.
    k4, _po4, pe4 = kappa_from(["L"] * 10, ["L"] * 10)
    print(f"  {'PASS' if k4 is None else 'FAIL'}  degenerate single category -> "
          f"kappa None (pe={pe4:.3f})")
    ok = ok and k4 is None

    # The sealed line must parse to exactly sixty valid labels.
    parsed = [c.upper() for c in CLAUDE_LABELS if not c.isspace()]
    good = len(parsed) == 60 and all(c in LABELS for c in parsed)
    print(f"  {'PASS' if good else 'FAIL'}  sealed line parses to 60 valid labels "
          f"(got {len(parsed)})")
    ok = ok and good

    # The reweighting is the part most likely to produce a wrong headline
    # silently, so it gets a case with a hand-computable answer. Two strata:
    # locked_in is 90% of the arm and 2 of 4 sheet items are really L;
    # echo is 10% and 0 of 4 are L. Estimate must be 0.9*0.5 + 0.1*0 = 0.45,
    # NOT the raw 2/8 = 0.25 that ignoring the quota would give.
    rows = [(1, "L", "L", "locked_in"), (2, "L", "L", "locked_in"),
            (3, "C", "L", "locked_in"), (4, "C", "L", "locked_in"),
            (5, "X", "X", "echo"), (6, "X", "X", "echo"),
            (7, "X", "X", "echo"), (8, "X", "X", "echo")]
    pop = Counter({"locked_in": 900, "echo": 100})
    est, lo, hi, cov = stratified_share(rows, pop, "L", draws=500)
    check("stratified estimate ignores the quota", est, 0.45)
    check("stratified coverage", cov, 1.0)
    good = lo <= est <= hi
    print(f"  {'PASS' if good else 'FAIL'}  bootstrap brackets the estimate "
          f"([{lo:.3f}, {hi:.3f}])")
    ok = ok and good

    # A stratum the sheet never sampled must lower coverage, not be treated
    # as if its share were zero.
    pop2 = Counter({"locked_in": 900, "echo": 100, "interleaving": 1000})
    _e2, _l2, _h2, cov2 = stratified_share(rows, pop2, "L", draws=200)
    check("uncovered stratum reduces coverage", cov2, 0.5)

    # The denominator bug that printed "the cuts transfer" twice: CORRECT
    # answers must not enter the population. A CSV that is one third correct
    # answers must still report locked_in as its share of the WRONG ones.
    import tempfile
    global MODES_CSV
    _saved = MODES_CSV
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "modes.csv"
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["qid", "mode", "correct"])
            for i in range(710):
                w.writerow([f"w{i}", "locked_in", "False"])
            for i in range(290):
                w.writerow([f"v{i}", "inconsistent", "False"])
            for i in range(500):                    # correct answers
                w.writerow([f"c{i}", "correct", "True"])
            for i in range(100):                    # and blanks
                w.writerow([f"b{i}", "", "True"])
        MODES_CSV = p
        _clf, _pop, _other, _n = load_classifier()
        MODES_CSV = _saved
    check("correct answers excluded from the population",
          _pop["locked_in"] / sum(_pop.values()), 0.710)
    check("excluded rows counted, not dropped silently",
          float(sum(_other.values())), 600.0)
    good = _n == 1600 and "correct" in _other and "(empty)" in _other
    print(f"  {'PASS' if good else 'FAIL'}  every row accounted for "
          f"({_n} read, {sum(_pop.values())} failure modes, "
          f"{sum(_other.values())} other)")
    ok = ok and good

    print("\n  " + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    main()