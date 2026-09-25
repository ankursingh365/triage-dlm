"""
Failure-mode labelling - the single copy of the rule.
=====================================================

Every wrong answer in TRIAGE is assigned a failure mode by this module and by
nothing else.

    from src.label_failure_mode import classify, int_untested
    mode = classify(row, arm="dream")

Why one copy
------------
Until step 19a the rule lived in four scripts - 17c, 17d, 18d and 18e - each
restating `classify()` so it could be LOCKed against the others. That was worth
doing once: 16,781 rows reproduced with zero mismatches proves the four agree.
After that, four copies are four chances to drift apart, and a drift would not
crash anything - it would silently relabel rows. Every script from Phase E on
imports this module instead of restating the rule.

The rule
--------
Applied in this order; the first test that fires decides.

    1. truncated                               -> degenerate
    2. question echo: overlap >= e_cut AND
                      copied run >= r_cut      -> echo
    3. gold testable AND gold ever in trace    -> interleaving
    4. stable_top3 >= s_cut AND
       cands_top3  <= c_cut                    -> locked_in
    5. otherwise                               -> inconsistent

Gate 1, repaired at step 18e
----------------------------
Step 3 used to be preceded by "gold not testable -> untestable", which RETURNED:
an answer whose gold could not be tested for interleaving was removed from the
analysis as though it were not a hallucination. Those are different claims -
"we cannot test whether the right answer appeared" is not "this answer is not
wrong" - and the Dream hand labels settled it: three of the four sampled
untestable answers are locked-in. 500 rows across the two arms were being
deleted.

Such a row now skips only the test it genuinely cannot run and continues to
steps 4-5. `int_untested(row)` records that interleaving was not testable on it,
so every per-mode table can be reported with and without those rows.
`classify()` can no longer return "untestable"; the name survives only in
`LEGACY_MODES` so older CSVs can still be read.

Gate 2, NOT yet repaired
------------------------
`gold_ever` is a whole-string match: a gold of `Makoto, born July 15, 1980`
never matches a trace holding `Makoto Kinaka`, and a gold shorter than four
characters is never testable at all. Its recall on Dream is about 40%. Step 18e
showed repairing it does not move the fitted cuts; it does change the size of
the interleaving class, so it must be repaired before step 23 and is tracked as
step 18f.

Two operating points
--------------------
The cuts differ by arm, and that is a finding about the two samplers rather than
a patch. Dream's schedule is adaptive - zero to five positions revealed per
round, 37% of rounds revealing nothing - and step 17e measured both of Dream's
measures shifted towards "converged": stable_top3 up, cands_top3 down from 4.00
to 2.50. Under LLaDA's cuts that shift swept 71% of Dream's wrong answers into
locked_in against 43% by hand. Each arm's cuts were fitted and validated against
that arm's own blind, two-annotator hand labels.

Nothing in this module substitutes a default for a missing or unparseable value.
A row that cannot be measured is "unknown", and `validate_row` refuses a row
that lacks a field outright.
"""

from typing import Mapping

# ---------------------------------------------------------------------------
#  Cuts. Exact decimals - never np.arange; see claude/numeric-thresholds-rule.md.
# ---------------------------------------------------------------------------
CUTS = {
    # Step 17c. Fitted on LLaDA's 60-item round (kappa 0.882) with a declared
    # asymmetric criterion; echo operating point fixed on exact grids.
    "llada": dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6),
    # Step 18e. Refitted on Dream's 60-item round (kappa 0.797): maximise
    # population-weighted recall s.t. precision >= 0.85 and the arm share
    # inside the hand labels' bootstrap interval. Chosen in 30 of 33 LOO folds;
    # LOO agreement 81.8%; arm share 39.6% against 43.2% by hand. Only s_cut
    # and c_cut were free - e_cut and r_cut are LLaDA's.
    "dream": dict(s_cut=0.63, c_cut=3.0, e_cut=0.75, r_cut=6),
}
ARMS = tuple(CUTS)

# What this module can emit.
MODES = ("locked_in", "interleaving", "inconsistent", "echo", "degenerate")
# What older CSVs may contain. Read-only.
LEGACY_MODES = MODES + ("untestable",)

# The worksheet's four letters. X is "not a hallucination we model".
MODE_TO_LETTER = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
                  "echo": "X", "degenerate": "X", "untestable": "X"}

NUMERIC = ("stable_top3", "cands_top3", "q_overlap", "copy_run")
BOOLEAN = ("truncated", "gold_ever", "gold_testable")
REQUIRED = NUMERIC + BOOLEAN


def validate_row(row: Mapping) -> None:
    """Raise if a field the rule reads is absent. Never fills one in.

    step 17e's loader defaulted a missing column to 0.0; `n_content` did not
    exist for LLaDA, read 0.00 everywhere, and was then reported as the largest
    drift between the arms. An absent field is a schema error, and it is
    reported as one.
    """
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise KeyError(f"row lacks {missing}; refusing to substitute defaults")


def _measurable(row: Mapping) -> bool:
    return all(isinstance(row[k], (int, float)) and row[k] == row[k]
               for k in NUMERIC)                       # rejects None and NaN


def upstream(row: Mapping, cuts: Mapping) -> str:
    """The verdict reached BEFORE the cuts are consulted, or "" if they decide.

    Kept separate so a calibration script can sweep s_cut and c_cut without
    re-deciding the upstream classes, and so no cut can change an upstream
    verdict by accident.
    """
    validate_row(row)
    if row["truncated"]:
        return "degenerate"
    if not _measurable(row):
        return "unknown"
    if row["q_overlap"] >= cuts["e_cut"] and row["copy_run"] >= cuts["r_cut"]:
        return "echo"
    if row["gold_testable"] and row["gold_ever"]:
        return "interleaving"
    return ""


def cut_verdict(row: Mapping, cuts: Mapping) -> str:
    """Steps 4-5 alone. Both comparisons inclusive, on exact decimals."""
    if (row["stable_top3"] >= cuts["s_cut"]
            and row["cands_top3"] <= cuts["c_cut"]):
        return "locked_in"
    return "inconsistent"


def classify_with(row: Mapping, cuts: Mapping) -> str:
    """The rule under explicit cuts. For LOCKs and calibration only - analysis
    code should call `classify(row, arm)` so it cannot pass the wrong cuts."""
    return upstream(row, cuts) or cut_verdict(row, cuts)


def classify(row: Mapping, arm: str) -> str:
    """The failure mode of one WRONG answer under its arm's operating point.

    Returns one of MODES, or "unknown" for a row whose measures could not be
    computed. Never "untestable" - see the module docstring.
    """
    if arm not in CUTS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return classify_with(row, CUTS[arm])


def int_untested(row: Mapping) -> bool:
    """True when interleaving could not be tested on this row (gate 1)."""
    validate_row(row)
    return not row["gold_testable"]


def classify_legacy(row: Mapping, cuts: Mapping) -> str:
    """The rule AS SHIPPED before step 18e, with the untestable early return.

    Exists only so step 19a can LOCK this module against the CSVs that were
    written before the repair. Do not use it for analysis.
    """
    validate_row(row)
    if row["truncated"]:
        return "degenerate"
    if not _measurable(row):
        return "unknown"
    if row["q_overlap"] >= cuts["e_cut"] and row["copy_run"] >= cuts["r_cut"]:
        return "echo"
    if not row["gold_testable"]:
        return "untestable"
    if row["gold_ever"]:
        return "interleaving"
    return cut_verdict(row, cuts)
