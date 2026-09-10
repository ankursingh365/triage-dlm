#!/usr/bin/env python3
"""
Step 7b - measure and filter dataset defects
============================================

    python scripts/step7b_quality.py

No GPU, no tokenizer, no model. Loads a larger sample from each dataset, applies
`src/data_quality`, and reports what was flagged and why.

Why this step exists
--------------------
Twenty sampled records in Step 7 surfaced four defects, three of which corrupt
the positive class - the hallucinations this project exists to detect:

  * CommonsenseQA: `['A. clumsy', 'B. ineffectual', 'C. dull', 'D. clumsy',
    'E. stupid']`. A and D identical. This is the NEGATIVE CONTROL; an
    unanswerable question there breaks the control before it is run.

  * TriviaQA: 23 aliases per question, including `Šachmatai`, `Sjakk` and
    `Ajedrez` for an answer of "Chess Records", plus raw
    `Rudolph (disambiguation)` stubs. Matching against these marks wrong answers
    correct, shrinking the positive class - the most damaging direction of error
    available to a hallucination-detection project.

  * HotpotQA: `"who is the younger brother of The episode guest stars of The
    Hard Easy"`. Ungrammatical. The model fails it, Stage 3 labels it incorrect,
    and a hallucination label lands on a question that was never answerable.

Sample size
-----------
300 per dataset rather than 20. Defect rates in the low single digits cannot be
estimated from 20 records, and the rate is a number the paper has to report.

What to do with the output
--------------------------
The flag counts belong in the paper's data section. "We excluded N% of HotpotQA
for malformed questions" is checkable; "we used HotpotQA" is not.

The excluded examples are printed so the heuristics can be judged by eye. If they
are discarding good questions, loosen `EXCLUDE_BY_DEFAULT` in
`src/data_quality.py` - under-flagging is the safer error, because the manual
spot check at Step 10 catches what the heuristics miss.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step7b_quality.py")
    sys.exit(1)

N_SAMPLE = 300
N_SHOW = 4

REPORT_PATH = config.OUT_DIR / "step7b_quality_report.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 7b: dataset quality")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"sampling {N_SAMPLE} per dataset, seed {config.SEED}")

    totals = {}

    for name in ("triviaqa", "hotpotqa", "commonsenseqa"):
        say("")
        say("=" * 78)
        say(name.upper())
        say("-" * 78)
        try:
            recs = data.load_records(name, n=N_SAMPLE, seed=config.SEED)
        except Exception as exc:
            say(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        kept, excluded, rep = data_quality.clean_records(recs)
        totals[name] = rep

        say(f"  loaded            : {rep['n_in']}")
        say(f"  kept              : {rep['n_kept']}  "
            f"({rep['n_kept'] / max(rep['n_in'], 1):.1%})")
        say(f"  excluded          : {rep['n_excluded']}  "
            f"({rep['n_excluded'] / max(rep['n_in'], 1):.1%})")

        if rep["flag_counts"]:
            say("")
            say("  flags raised:")
            for flag, n in sorted(rep["flag_counts"].items(),
                                  key=lambda kv: -kv[1]):
                marker = ("EXCLUDES" if flag in data_quality.EXCLUDE_BY_DEFAULT
                          else "info")
                say(f"    {flag:<24} {n:4d}  ({n/max(rep['n_in'],1):5.1%})  {marker}")
        else:
            say("  no flags raised")

        # ---- TriviaQA alias pollution, quantified -------------------------
        if name == "triviaqa" and rep["aliases_before"]:
            before = rep["aliases_before"] / max(rep["n_in"], 1)
            after = rep["aliases_after"] / max(rep["n_in"], 1)
            say("")
            say("  alias cleaning:")
            say(f"    mean aliases before : {before:.1f}")
            say(f"    mean aliases after  : {after:.1f}")
            say(f"    removed             : {before - after:.1f} per question")
            if rep["alias_examples"]:
                say(f"    examples dropped    : {rep['alias_examples'][:8]}")

        # ---- what got excluded, so the heuristics can be judged ----------
        if excluded:
            say("")
            say(f"  excluded examples (showing {min(N_SHOW, len(excluded))}):")
            for rec, flags in excluded[:N_SHOW]:
                say(f"    [{','.join(flags)}]")
                say(f"      Q: {rec.question[:88]}")
                if rec.choices:
                    say(f"      choices: {rec.choices}")

        # ---- a couple of survivors, for contrast --------------------------
        if kept:
            say("")
            say(f"  kept examples (showing 2):")
            for rec in kept[:2]:
                say(f"      Q: {rec.question[:88]}")
                say(f"      gold: {rec.gold_answers[:6]}")

    # =====================================================================
    say("")
    say("=" * 78)
    say("SUMMARY - these numbers go in the paper's data section")
    say("")
    say("  dataset         loaded   kept   excluded   rate")
    say("  --------------  ------  -----   --------   -----")
    for name, rep in totals.items():
        say(f"  {name:<14}  {rep['n_in']:6d}  {rep['n_kept']:5d}   "
            f"{rep['n_excluded']:8d}   {rep['n_excluded']/max(rep['n_in'],1):5.1%}")

    say("")
    say("STILL REQUIRES HUMAN JUDGEMENT")
    say(f"  {data_quality.MANUAL_CHECK_NOTE}")
    say("")
    say("  Heuristics catch obvious breakage. A question that is grammatical")
    say("  but unanswerable will pass every automatic check and still put a")
    say("  hallucination label on something that was never answerable.")

    say("")
    say("SCOPE DECISION TO CONFIRM")
    say("  TriviaQA rc.nocontext strips entity_pages, so there is no gold")
    say("  evidence for it. Step 26's upper bound therefore runs on HotpotQA")
    say("  only, where supporting_facts is real gold evidence and where")
    say("  retrieval was already expected to be the weak link. State this in")
    say("  the paper rather than leaving it implicit.")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
