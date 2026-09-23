#!/usr/bin/env python3
"""
Step 9b - re-score the cached trajectories, correctly
=====================================================

    modal run modal_app.py::run_cpu --script step9b_rescore.py

CPU only. No GPU, no model, no generation - it reads the 300 trajectories Step 9
already wrote to the Volume. Costs fractions of a cent and takes seconds.

Why
---
Step 9's rough labels were wrong in two ways.

**1. Substring matching on single-letter gold answers.** CommonsenseQA's gold
list includes the option letter, so gold "D" normalised to "d" - and "d" is a
substring of "agreement", "and", "would", and almost every English sentence.
This was scored CORRECT:

    Q: When drinking booze what can you do to stay busy?
    A: 'The best option is A. reach tentative agreement.'
    gold: examine thing | D                                  -> [OK] (wrong!)

The model answered A, the gold is D. CommonsenseQA's reported 97% accuracy is
therefore meaningless, and the negative control depends on that number.

The same bug hit TriviaQA more quietly: gold "Dolores HAZE" against answer
"Dolores Humbert" scored correct on a partial alias match. Lolita's surname is
Haze; Humbert is the narrator.

**Fix: word-boundary matching.** `\\bd\\b` matches "is d" but not "agreement",
and "dolores haze" no longer matches "dolores humbert".

**2. Question echo, counted as hallucination.** HotpotQA answers like

    Q: Roger O. Egeberg was Assistant Secretary for Health and Scientific...
    A: 'Roger O. Egeberg was Assistant Secretary for Health and Scientif'

are the model restating the question and running out of the 64-token budget.
That is a truncation artefact, not a hallucination. If it is common, part of
HotpotQA's 82% error rate measures `gen_length` rather than the model's
knowledge - and those cases would pollute the failure-mode taxonomy in Step 17,
because an echo has no meaningful trajectory dynamics to classify.

This script measures both and reports the corrected base rates side by side with
the old ones, so the size of each correction is visible rather than silently
absorbed.

Still rough
-----------
Word boundaries fix a specific bug; they do not turn string matching into a
judge. "the Nile" still fails gold "Nile River". Stage 3 uses Qwen3-8B and
Step 16 verifies 100 labels by hand. These numbers size a sample; they do not
enter the paper.
"""

import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

TRAJ_SUBDIR = config.TRAJ_DIR / "step9" / config.RUN_TAG
REPORT_PATH = config.OUT_DIR / "step9b_rescore_report.txt"
CSV_PATH = config.TAB_DIR / "step9b_rescored.csv"

DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
N_PER_DATASET = 100
ECHO_WORDS = 6          # consecutive leading words shared with the question
N_SHOW = 8

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Scoring
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Cut at the first control token. Everything after is block padding."""
    first = SPECIAL_RE.search(text)
    if first:
        text = text[:first.start()]
    return text.strip()


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def match_substring(answer: str, golds: list[str]) -> bool:
    """Step 9's original rule. Kept only to quantify how wrong it was."""
    a = normalise(answer)
    if not a:
        return False
    return any(normalise(g) and normalise(g) in a for g in golds)


def match_wordbound(answer: str, golds: list[str]) -> bool:
    """Corrected rule: the gold must appear as whole words.

    `\\bd\\b` matches "is d" but not "agreement". This is what makes a
    single-letter multiple-choice label usable as a gold answer at all.
    """
    a = normalise(answer)
    if not a:
        return False
    for g in golds:
        gn = normalise(g)
        if not gn:
            continue
        if re.search(r"\b" + re.escape(gn) + r"\b", a):
            return True
    return False


def is_question_echo(answer: str, question: str, n: int = ECHO_WORDS) -> bool:
    """True if the answer opens by restating the question.

    Compares the first `n` words of each after normalisation. A model that
    begins "The battle in which Giuseppe Arimondi lost his life secured..." is
    repeating its prompt, and with a 64-token budget it will be cut off before
    reaching an actual answer.

    Deliberately requires the echo to be at the START. An answer that happens to
    reuse the question's vocabulary later on is a normal full-sentence reply.
    """
    a = normalise(answer).split()
    q = normalise(question).split()
    if len(a) < n or len(q) < n:
        return False
    return a[:n] == q[:n]


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 9b: re-score cached trajectories")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"reading {TRAJ_SUBDIR}")

    if not TRAJ_SUBDIR.exists():
        say("")
        say("No trajectories found. Run step9_base_rate.py first.")
        sys.exit(1)

    # Rebuild the same record set Step 9 used, for gold answers and questions.
    records = {}
    for dataset in DATASETS:
        recs = data.load_records(dataset, n=N_PER_DATASET * 2, seed=config.SEED)
        kept, _exc, _rep = data_quality.clean_records(recs)
        for rec in kept[:N_PER_DATASET]:
            records[(dataset, rec.qid)] = rec

    rows = []
    for path in sorted(TRAJ_SUBDIR.glob("*.npz")):
        stem = path.stem
        dataset = stem.split("_", 1)[0]
        qid = stem.split("_", 1)[1] if "_" in stem else stem
        rec = records.get((dataset, qid))
        if rec is None:
            continue

        traj = logging_patch.Trajectory.load(path)

        # Decode without a tokenizer: the trajectory stores token ids, and we
        # only need the answer TEXT. Rebuilding a tokenizer here would mean
        # downloading one, so instead Step 9's CSV is joined if present.
        rows.append(dict(traj=traj, rec=rec, dataset=dataset, qid=qid))

    # Answer text comes from Step 9's CSV - already decoded, already cleaned.
    step9_csv = config.TAB_DIR / "step9_per_question.csv"
    answers = {}
    if step9_csv.exists():
        with open(step9_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                answers[(r["dataset"], r["qid"])] = r.get("answer", "")

    if not answers:
        say("")
        say(f"Could not read {step9_csv}. It holds the decoded answers.")
        say("Re-run step9_base_rate.py, then this script.")
        sys.exit(1)

    say(f"{len(rows)} trajectories, {len(answers)} decoded answers")

    # ---- score -----------------------------------------------------------
    out = []
    for row in rows:
        rec, traj = row["rec"], row["traj"]
        answer = clean_answer(answers.get((row["dataset"], row["qid"]), ""))

        old = match_substring(answer, rec.gold_answers)
        new = match_wordbound(answer, rec.gold_answers)
        echo = is_question_echo(answer, rec.question)

        masked = traj.mask_state == 1
        content = traj.content_mask() & masked

        out.append(dict(
            dataset=row["dataset"], qid=row["qid"],
            question=rec.question[:100], answer=answer[:100],
            gold=" | ".join(rec.gold_answers[:3]),
            old_correct=old, new_correct=new, changed=(old != new),
            question_echo=echo,
            ans_tokens=sum(1 for t in traj.final_ids.tolist()
                           if t not in (traj.eos_id, traj.mask_id)),
            ent_content=round(float(traj.entropy[content].mean()), 4)
                        if content.any() else float("nan"),
            flips=int(traj.flips_per_round().sum()),
            regret=int(traj.regret_per_round().sum()),
        ))

    # ---- corrected base rates --------------------------------------------
    say("")
    say("=" * 78)
    say("CORRECTED BASE RATES")
    say("")
    say("  dataset          n   OLD err   NEW err   changed   echo")
    say("  --------------  ---  -------   -------   -------   ----")
    by_ds = defaultdict(list)
    for r in out:
        by_ds[r["dataset"]].append(r)

    for dataset in DATASETS:
        rs = by_ds.get(dataset, [])
        if not rs:
            continue
        old_err = 1 - sum(r["old_correct"] for r in rs) / len(rs)
        new_err = 1 - sum(r["new_correct"] for r in rs) / len(rs)
        changed = sum(r["changed"] for r in rs)
        echo = sum(r["question_echo"] for r in rs)
        say(f"  {dataset:<14}  {len(rs):3d}  {old_err:6.1%}   {new_err:6.1%}   "
            f"{changed:7d}   {echo:4d}")

    # ---- what the letter bug did -----------------------------------------
    say("")
    say("LABELS THAT FLIPPED (the single-letter substring bug)")
    flipped = [r for r in out if r["changed"]]
    say(f"  {len(flipped)} of {len(out)} labels changed")
    for r in flipped[:N_SHOW]:
        direction = ("OK -> WRONG" if r["old_correct"] else "WRONG -> OK")
        say("")
        say(f"    [{direction}]  {r['dataset']}")
        say(f"      Q: {r['question'][:70]}")
        say(f"      A: {r['answer'][:70]!r}")
        say(f"      gold: {r['gold'][:70]}")

    # ---- echo analysis ----------------------------------------------------
    say("")
    say("=" * 78)
    say("QUESTION ECHO (truncation artefact, not hallucination)")
    echoes = [r for r in out if r["question_echo"]]
    say(f"  {len(echoes)} of {len(out)} answers open by restating the question")
    for dataset in DATASETS:
        rs = by_ds.get(dataset, [])
        if not rs:
            continue
        e = [r for r in rs if r["question_echo"]]
        if not e:
            continue
        wrong_e = sum(1 for r in e if not r["new_correct"])
        say(f"    {dataset:<14} {len(e):3d} echoes, {wrong_e} of them scored wrong")
    for r in echoes[:4]:
        say("")
        say(f"    {r['dataset']}")
        say(f"      Q: {r['question'][:70]}")
        say(f"      A: {r['answer'][:70]!r}")

    say("")
    say("  These need a decision before Step 17. An echoed, truncated answer")
    say("  has no meaningful trajectory dynamics to classify, so it fits none")
    say("  of the three failure modes and would be noise in the taxonomy.")
    say("  Options: exclude them, or add a fourth 'degenerate' category and")
    say("  report its size. Excluding is cleaner; reporting the rate is honest.")

    # ---- negative control health -----------------------------------------
    say("")
    say("=" * 78)
    say("NEGATIVE CONTROL HEALTH (CommonsenseQA)")
    csqa = by_ds.get("commonsenseqa", [])
    if csqa:
        err = 1 - sum(r["new_correct"] for r in csqa) / len(csqa)
        say(f"  corrected error rate : {err:.1%}")
        say(f"  wrong answers per 100: {int(round(err * 100))}")
        say("")
        say("  CommonsenseQA validation has 1,221 examples; train has 9,741.")
        avail = int(round(err * (1221 + 9741)))
        say(f"  At this rate the ENTIRE dataset yields ~{avail} wrong answers.")
        say("")
        if err < 0.10:
            say("  PROBLEM. The negative control needs enough wrong answers to")
            say("  show that evidence gives ZERO gain. Too few and the result is")
            say("  'no significant difference', which is not the same claim and")
            say("  is much weaker.")
            say("")
            say("  Options, best first:")
            say("   1. Drop the multiple-choice options - ask CommonsenseQA")
            say("      open-ended. The answers still are not in Wikipedia, which")
            say("      is what makes it a control, but the task stops being")
            say("      5-way guessing. Costs one 100-question test to measure.")
            say("   2. Use train + validation and subsample the correct answers")
            say("      to balance. Works, but 3% positives is severe imbalance.")
            say("   3. Replace the control with a harder commonsense set.")
            say("      Deviates from TraceDet's setup.")
            say("")
            say("  Check what TraceDet did before choosing - if they presented")
            say("  options they had the same problem, and how they handled it")
            say("  matters for comparability.")
        else:
            say("  Workable. Enough wrong answers for the control to mean")
            say("  something.")

    # ---- output -----------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    say("")
    say(f"  CSV: {CSV_PATH}")
    say("=" * 78)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
