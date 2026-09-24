#!/usr/bin/env python3
"""
Step 12 - fix the echo rule, then classify all 11,120 Phase C trajectories
===========================================================================

    modal run modal_app.py::run_cpu --script step12_fullscale_modes.py

CPU only. Roughly 15-25 minutes, free. No GPU, no model - only the tokenizer,
which is already on the cache Volume.

WHY THIS STEP EXISTS
====================
Everything the project currently believes about failure modes rests on 150
questions, and one of the three rules used to classify them is known to be
broken. Both facts are fixed here, in that order, because fixing the rule after
classifying 11,120 questions would mean classifying them twice.

**The broken rule.** Step 10d's echo detector is a single number:

    echo/copy  :  question_overlap >= 0.75

It fires on any answer whose content words all occur in the question. That is
correct for the two failures it was built for:

    echo   "Roger O. Egeberg was Assistant Secretary for Health and Scien..."
           40 tokens, never reaches the years the question asked for

    copy   "Billie Jean King's maiden name was Billie Jean King."
           the answer slot is filled with a word lifted from the question

and wrong for a third thing that looks identical to it:

    selection  "The Rescuers was released earlier."
               9 tokens, and a completely correct answer

A comparison question - "which of A or B was released earlier" - offers the
answer inside the question. Naming A is not reuse of the input, it is the task.
On the 59 hand-labelled trajectories the overlap rule ate 3 locked-in and 1
interleaving case this way. The bias runs one direction: it UNDER-counts the
modes the paper is about.

WHAT SEPARATES THEM
===================
Not length. Length is a symptom, and thresholding on it would be a hack that no
reviewer should accept. The mechanism is what differs:

    an echo or a copy reproduces a SPAN of the question
    a selection reproduces a TOKEN the question offered as an alternative

So the added measurement is the longest run of consecutive answer words that
appears consecutively in the question - `max_copied_run` below.

    "roger o egeberg was assistant secretary for health and"     run 9
    "billie jean king s maiden name was"                         run 7
    "rescuers was released earlier"                              run 2

The cut is FITTED to the hand labels alongside the other two, not asserted. The
old rule is the special case `max_copied_run >= 1`, so the fit is a nested-model
comparison and can only tie or improve - and the report prints both, together
with the names of the cases that change, so the improvement is visible in the
output rather than claimed in a comment.

WHAT THIS DOES NOT FIX
======================
Right/wrong here is still decided by whole-word string matching. That rule has
a known failure in the other direction: the gold string genuinely appears in an
answer that is not the answer, so some wrong answers are scored correct. Step 9b
measured this and could not repair it - it needs a model that reads meaning.

That is the Qwen3-8B judge, and it is the next step. This script deliberately
runs BEFORE it, because the judge's scope and cost depend on numbers only this
script can produce: how many answers there actually are to judge. Sizing the
expensive GPU step is the job of the free CPU step.

Every number in Part C is therefore **provisional on the judge** and labelled so.

OUTPUT
======
    outputs/step12_fullscale_report.txt
    outputs/tables/step12_phasec_modes.csv    one row per Phase C question
"""

import csv
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

# --- the calibration set: Step 9's 150, which carry the hand labels ---------
CAL_TRAJ_DIR = config.TRAJ_DIR / "step9" / config.RUN_TAG
CAL_MODES_CSV = config.TAB_DIR / "step10_modes.csv"
CAL_KEY_CSV = config.TAB_DIR / "step10c_answer_key.csv"
CAL_DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
CAL_N_PER_DATASET = 100

# --- the full set: Phase C -------------------------------------------------
PHASEC_ROOT = config.TRAJ_DIR / "phasec" / config.RUN_TAG
PHASEC_ANSWERS_CSV = config.TAB_DIR / "step11_phasec_answers.csv"
PHASEC_TARGETS = (("triviaqa", 3750), ("hotpotqa", 7500))

REPORT_PATH = config.OUT_DIR / "step12_fullscale_report.txt"
CSV_PATH = config.TAB_DIR / "step12_phasec_modes.csv"

PROGRESS_EVERY = 1000

# Optional smoke run: `--limit 300` classifies only the first 300 Phase C
# trajectories per dataset. Part A is unaffected - it always uses all 150
# calibration questions, because a threshold fitted on a subset is not a
# threshold.
LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Text handling. Identical rules to Step 9b / 10 / 10d, restated so this
# script is standalone and so a future edit to one cannot silently desync
# "gold seen at round r" from "correct at the end".
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

STOP = {"is", "was", "are", "were", "be", "been", "of", "in", "on", "at", "to",
        "for", "and", "or", "but", "by", "with", "from", "that", "which",
        "who", "what", "when", "where", "how", "it", "its", "this", "these",
        "as", "has", "have", "had", "do", "does", "did", "not", "no", "yes",
        "s", "t"}

MIN_GOLD_CHARS = 4
STOP_GOLDS = {"yes", "no", "true", "false", "none", "both", "all"}


def clean_answer(text: str) -> str:
    """Cut at the first control token. Everything after it is block padding."""
    first = SPECIAL_RE.search(str(text))
    if first:
        text = str(text)[: first.start()]
    return str(text).strip()


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def match_wordbound(text: str, golds: list) -> bool:
    """Gold must appear as whole words. Step 9b's corrected rule."""
    a = normalise(text)
    if not a:
        return False
    for g in golds:
        gn = normalise(g)
        if not gn:
            continue
        if re.search(r"\b" + re.escape(gn) + r"\b", a):
            return True
    return False


def testable_golds(golds: list) -> list:
    """Golds usable for the INTERLEAVING test. Not for scoring.

    A gold short or common enough to appear by accident - a bare option letter,
    or `no` - makes the shadow prediction match at round 0 and reports
    interleaving that never happened. A question with no usable alias is
    UNTESTABLE for interleaving, which is not the same as negative, and the
    report counts those separately so the blind spot stays visible.
    """
    return [g for g in golds
            if len(normalise(g)) >= MIN_GOLD_CHARS
            and normalise(g) not in STOP_GOLDS]


# ===========================================================================
# The two reuse measurements
# ===========================================================================

def question_overlap(answer: str, question: str) -> float:
    """Share of the answer's content words that already occur in the question.

    High for echo, for copy, AND for a correct selection from a comparison
    question. On its own it cannot tell them apart, which is the defect this
    step repairs. Returns 1.0 for an empty answer: nothing was contributed.
    """
    a = [w for w in normalise(answer).split() if w not in STOP]
    q = set(normalise(question).split())
    if not a:
        return 1.0
    return sum(1 for w in a if w in q) / len(a)


def max_copied_run(answer: str, question: str) -> int:
    """Longest run of consecutive answer words that runs consecutively in the
    question. Words in, stopwords kept.

    THE ECHO / SELECTION SEPARATOR.

    Stopwords are deliberately KEPT here, unlike in `question_overlap`. A span
    copied out of the question carries its function words with it - "was
    assistant secretary for health and" - and stripping them would destroy the
    very contiguity being measured.

        echo       "roger o egeberg was assistant secretary for health and"  9
        copy       "billie jean king s maiden name was"                      7
        selection  "rescuers was released earlier"                           2

    Classic longest-common-substring over word lists. The arrays are at most
    ~64 x ~64, so the quadratic table costs nothing.
    """
    a = normalise(answer).split()
    q = normalise(question).split()
    if not a or not q:
        return 0
    prev = [0] * (len(q) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(q) + 1)
        ai = a[i - 1]
        for j in range(1, len(q) + 1):
            if ai == q[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ===========================================================================
# Trajectory measurement
# ===========================================================================

def top_by_entropy(ents: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-entropy positions, selected BY RANK.

    A value threshold (`ents >= np.quantile(ents, .75)`) selects far more than
    a quarter whenever the high-entropy positions are a small minority, which
    is the normal case.
    """
    k = max(1, min(int(k), len(ents)))
    return np.argsort(-ents, kind="stable")[:k]


def check_config(traj, path) -> None:
    """Refuse a trajectory generated under different decoder settings.

    `RUN_TAG` in the cache path is meant to make this impossible, but a path is
    a convention and a check is a check. This has already gone wrong twice.
    """
    want = (config.GEN_LENGTH, config.DENOISING_STEPS, config.BLOCK_LENGTH)
    got = (int(traj.gen_length), int(traj.steps), int(traj.block_length))
    if got != want:
        raise SystemExit(
            f"\nSTOPPED - {path}\n"
            f"  config.py : gen {want[0]} steps {want[1]} block {want[2]}\n"
            f"  this file : gen {got[0]} steps {got[1]} block {got[2]}\n")


def shadow_guesses(traj, tokenizer) -> list:
    """The model's full current guess, decoded, at every round.

    `pred_ids[r]` is the raw argmax at EVERY position, recorded BEFORE the
    reveal overwrite - what the model would emit if it committed everything
    now. Interleaving is a question about this, not about committed tokens: a
    revealed position is frozen and cannot change, so "had it and lost it" is
    only meaningful at the shadow level.
    """
    rows = [[int(t) for t in traj.pred_ids[r].tolist()
             if t not in (traj.eos_id, traj.mask_id)]
            for r in range(traj.pred_ids.shape[0])]
    return tokenizer.batch_decode(rows, skip_special_tokens=True) if rows else []


def measure(traj) -> dict:
    """stable_top3 and cands_top3 over the three highest-entropy positions.

    Why the top three rather than every content position:
    `claude/stable-top3-measure-decision.md`. Averaging over all of them
    correlates +0.44 with answer length, because template and echoed tokens are
    low-entropy and trivially stable and outvote the one position that carries
    the claim. Restricting to the high-entropy positions drops that to -0.07.
    """
    final = traj.final_ids.tolist()
    pos = [i for i, t in enumerate(final) if t not in (traj.eos_id, traj.mask_id)]
    if not pos:
        return {}
    rev = traj.revealed_at()
    fr, ent, cn = [], [], []
    for i in pos:
        r = int(rev[i])
        hist = traj.pred_ids[: r + 1, i].tolist()
        committed = hist[-1]
        held = 0
        for v in reversed(hist):
            if v == committed:
                held += 1
            else:
                break
        fr.append(held / len(hist))
        ent.append(float(np.mean(traj.entropy[: r + 1, i])))
        cn.append(len(set(hist)))
    fr, ent, cn = np.array(fr), np.array(ent), np.array(cn)
    sel = top_by_entropy(ent, 3)
    return {
        "n_content": len(pos),
        "stable_top3": float(np.mean(fr[sel])),
        "cands_top3": float(np.mean(cn[sel])),
        "truncated": len(pos) >= traj.gen_length - 2,
    }


# ===========================================================================
# Fitting
# ===========================================================================

def fit_locked(rows) -> tuple:
    """Grid-search the (stable_top3, cands_top3) cut on hand-labelled L vs C."""
    LC = [r for r in rows if r.get("hand") in ("L", "C")]
    best = None
    for sc in np.arange(0.05, 0.90, 0.01):
        for cn in np.arange(1.5, 5.5, 0.1):
            tp = sum(1 for r in LC if r["hand"] == "L"
                     and r["stable_top3"] >= sc and r["cands_top3"] <= cn)
            fn = sum(1 for r in LC if r["hand"] == "L") - tp
            fp = sum(1 for r in LC if r["hand"] == "C"
                     and r["stable_top3"] >= sc and r["cands_top3"] <= cn)
            tn = sum(1 for r in LC if r["hand"] == "C") - fp
            if tp + fn == 0 or fp + tn == 0:
                continue
            bal = 0.5 * (tp / (tp + fn) + tn / (fp + tn))
            if best is None or bal > best[0]:
                best = (bal, float(sc), float(cn), tp, fn, fp, tn)
    return best


def fit_echo(rows, run_grid) -> tuple:
    """Grid-search (question_overlap, max_copied_run) on X vs everything else.

    `run_grid = [1]` reproduces Step 10d's one-dimensional rule exactly, since
    every non-empty answer has a copied run of at least 1 whenever its content
    words are all in the question. That makes the two-dimensional fit a proper
    nested comparison: same data, same objective, one added degree of freedom.

    X also contains "the model was actually right and the matcher disagreed",
    which no string rule can catch, so perfect separation is neither expected
    nor the target. What is being fitted is the split between mechanical reuse
    and genuine selection.
    """
    lab = [r for r in rows if r.get("hand") in ("L", "I", "C", "X")]
    best = None
    for cut in np.arange(0.40, 1.01, 0.01):
        for run in run_grid:
            tp = sum(1 for r in lab if r["hand"] == "X"
                     and r["q_overlap"] >= cut and r["copy_run"] >= run)
            fn = sum(1 for r in lab if r["hand"] == "X") - tp
            fp = sum(1 for r in lab if r["hand"] != "X"
                     and r["q_overlap"] >= cut and r["copy_run"] >= run)
            tn = sum(1 for r in lab if r["hand"] != "X") - fp
            if tp + fn == 0 or fp + tn == 0:
                continue
            bal = 0.5 * (tp / (tp + fn) + tn / (fp + tn))
            if best is None or bal > best[0]:
                best = (bal, float(cut), int(run), tp, fn, fp, tn)
    return best


CUTS = {}          # filled by Part A, read by classify()


def classify(r, order="B") -> str:
    """Apply the fitted rules. `order` selects which of the two orderings runs.

    Order changes the answer, so it is fitted in Part B rather than assumed.
    A protects against a gold alias that appears inside an echo; B protects a
    genuine interleaving case whose answer happens to restate the question.
    """
    if r.get("truncated"):
        return "degenerate"
    echo = (r["q_overlap"] >= CUTS["e_cut"]
            and r["copy_run"] >= CUTS["r_cut"])
    gold = r.get("gold_ever", False)
    testable = r.get("gold_testable", True)

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

    if r["stable_top3"] >= CUTS["s_cut"] and r["cands_top3"] <= CUTS["c_cut"]:
        return "locked_in"
    return "inconsistent"


# ---------------------------------------------------------------------------
# The hand labels, consensus of two independent blind passes, Cohen's kappa
# 0.882. Provenance and the two adjudications are recorded in
# `claude/failure-mode-calibration.md`. `.` is the one worksheet-defect drop.
# ---------------------------------------------------------------------------
HAND = ("L C C X C X C I X C . X X X C C I L C C I L L I L C C C C C "
        "L C L X X X X X X C C C I X X L C C C L L L X I X X X X L C").split()
assert len(HAND) == 60, len(HAND)
DROPPED = {i for i, h in enumerate(HAND, 1) if h == "."}

MAP = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
       "echo": "X", "degenerate": "X", "untestable": None}


# ===========================================================================

def load_tokenizer():
    """Tokenizer only - no model, no GPU.

    `src.config` sets HF_HOME before transformers is imported, so this resolves
    against the model already on the cache Volume and downloads nothing.
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.MODEL_LLADA, trust_remote_code=True)


def build_rows(traj_dir, records, tokenizer, wrong_only_prev=None, limit=None):
    """Measure every trajectory in `traj_dir` and score it against its golds.

    `wrong_only_prev` is Step 10's CSV for the calibration set, where
    correctness is already recorded. For Phase C it is None and correctness is
    computed here from the decoded answer.
    """
    out, n_seen, n_skipped, n_empty = [], 0, 0, 0
    for path in sorted(traj_dir.glob("*.npz")):
        if limit is not None and n_seen >= limit:
            break
        stem = path.stem
        ds = traj_dir.name if traj_dir.name in dict(PHASEC_TARGETS) else stem.split("_", 1)[0]
        qid = stem if traj_dir.name in dict(PHASEC_TARGETS) else (
            stem.split("_", 1)[1] if "_" in stem else stem)
        rec = records.get((ds, qid))
        if rec is None:
            n_skipped += 1
            continue
        n_seen += 1

        traj = logging_patch.Trajectory.load(path)
        check_config(traj, path)

        if wrong_only_prev is not None:
            p = wrong_only_prev.get((ds, qid))
            if p is None or p["correct"].lower() == "true":
                continue
            answer = p.get("answer", "")
            correct = False
        else:
            answer = clean_answer(tokenizer.decode(traj.final_ids.tolist()))
            correct = match_wordbound(answer, rec.gold_answers)

        m = measure(traj)
        if not m:
            # Every generated position is EOS or still masked: the model output
            # nothing at all. Counted rather than silently dropped - it is a
            # decoder failure and belongs in the report.
            n_empty += 1
            continue

        row = dict(
            dataset=ds, qid=qid, question=rec.question, answer=answer,
            gold=" | ".join(rec.gold_answers[:5]), correct=correct,
            q_overlap=question_overlap(answer, rec.question),
            copy_run=max_copied_run(answer, rec.question),
            **m)

        # The interleaving test costs 64 decodes, so it only runs where it can
        # matter - on answers that are wrong. A correct answer has nothing to
        # have held and lost.
        if not correct:
            usable = testable_golds(rec.gold_answers)
            gold_round = -1
            if usable:
                for r_i, text in enumerate(shadow_guesses(traj, tokenizer)):
                    if match_wordbound(text, usable):
                        gold_round = r_i
                        break
            row["gold_round"] = gold_round
            row["gold_ever"] = gold_round >= 0
            row["gold_testable"] = bool(usable)
        out.append(row)
    return out, n_skipped, n_empty


def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 12: fixed echo rule, applied to all of Phase C")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  config   : gen {config.GEN_LENGTH}  steps {config.DENOISING_STEPS}"
        f"  block {config.BLOCK_LENGTH}   (tag {config.RUN_TAG})")
    if LIMIT:
        say(f"  SMOKE RUN: Phase C limited to {LIMIT} per dataset")

    for p in (CAL_TRAJ_DIR, CAL_MODES_CSV, CAL_KEY_CSV, PHASEC_ROOT):
        if not p.exists():
            say(f"\nMissing {p}. Run the earlier steps first.")
            sys.exit(1)

    say("")
    say("  Loading tokenizer (no model, no GPU)...")
    tokenizer = load_tokenizer()
    say(f"  vocab {len(tokenizer):,}")

    # =======================================================================
    # PART A - refit the three rules on the 150 calibration questions
    # =======================================================================
    say("")
    say("=" * 78)
    say("A. REFIT ON THE HAND LABELS")
    say("")

    prev = {}
    with open(CAL_MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            prev[(r["dataset"], r["qid"])] = r
    key = {}
    with open(CAL_KEY_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key[int(r["n"])] = (r["dataset"], r["qid"])
    hand_by_qid = {key[n]: HAND[n - 1] for n in key
                   if n not in DROPPED and HAND[n - 1] != "."}

    cal_records = {}
    for ds in CAL_DATASETS:
        recs = data.load_records(ds, n=CAL_N_PER_DATASET * 2, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        for rec in kept[:CAL_N_PER_DATASET]:
            cal_records[(ds, rec.qid)] = rec

    cal, _sk, _emp = build_rows(CAL_TRAJ_DIR, cal_records, tokenizer,
                                wrong_only_prev=prev)
    for r in cal:
        r["hand"] = hand_by_qid.get((r["dataset"], r["qid"]))
        # Step 10's CSV carries gold_ever for this set; recomputing it would
        # decode 150 x 64 rows to reproduce a number already measured.
        p = prev[(r["dataset"], r["qid"])]
        r["gold_ever"] = str(p.get("gold_ever", "")).lower() == "true"
        r["gold_testable"] = str(p.get("gold_testable", "True")).lower() == "true"
    n_hand = sum(1 for r in cal if r["hand"])
    say(f"  {len(cal)} wrong answers, {n_hand} hand-labelled")

    fitted = fit_locked(cal)
    if fitted is None:
        say("\nSTOPPED - no usable L/C hand labels reached the fitter. The join")
        say("between the worksheet key and the trajectories is broken.")
        sys.exit(1)
    bal, s_cut, c_cut, tp, fn, fp, tn = fitted
    say("")
    say(f"  locked_in : stable_top3 >= {s_cut:.2f}  AND  cands_top3 <= {c_cut:.1f}")
    say(f"              bal.acc {bal:.0%}  recall {tp/(tp+fn):.0%}  "
        f"specificity {tn/(fp+tn):.0%}  (n_L={tp+fn}, n_C={fp+tn})")

    old = fit_echo(cal, run_grid=[1])
    new = fit_echo(cal, run_grid=list(range(1, 13)))
    if old is None or new is None:
        say("\nSTOPPED - no usable X hand labels reached the echo fitter.")
        sys.exit(1)
    say("")
    say("  echo/copy, one-dimensional (Step 10d's rule):")
    say(f"              question_overlap >= {old[1]:.2f}")
    say(f"              bal.acc {old[0]:.0%}  recall {old[3]/(old[3]+old[4]):.0%}  "
        f"specificity {old[6]/(old[5]+old[6]):.0%}  (n_X={old[3]+old[4]})")
    say("")
    say("  echo/copy, with the span term added:")
    say(f"              question_overlap >= {new[1]:.2f}  AND  "
        f"max_copied_run >= {new[2]}")
    say(f"              bal.acc {new[0]:.0%}  recall {new[3]/(new[3]+new[4]):.0%}  "
        f"specificity {new[6]/(new[5]+new[6]):.0%}  (n_X={new[3]+new[4]})")
    say("")
    say(f"  balanced accuracy {old[0]:.0%} -> {new[0]:.0%}   "
        f"false positives {old[5]} -> {new[5]}")

    # Name the cases that change. This is the claim that the fix works, and it
    # belongs in the output rather than in a comment.
    say("")
    say("  Hand-labelled cases the old rule called echo but a human did not:")
    changed = 0
    for r in cal:
        if not r["hand"] or r["hand"] == "X":
            continue
        was = r["q_overlap"] >= old[1]
        now = r["q_overlap"] >= new[1] and r["copy_run"] >= new[2]
        if was:
            changed += 1
            verdict = "STILL EATEN" if now else "recovered"
            say(f"    [{r['hand']}] run={r['copy_run']:2d} "
                f"n_content={r['n_content']:2d}  {verdict}")
            say(f"          {r['answer'][:66]!r}")
    if not changed:
        say("    (none)")

    CUTS.update(s_cut=s_cut, c_cut=c_cut, e_cut=new[1], r_cut=new[2])

    # =======================================================================
    # PART B - rule order, and the confusion matrix against the hand labels
    # =======================================================================
    say("")
    say("=" * 78)
    say("B. RULE ORDER AND AGREEMENT WITH THE HAND LABELS")
    say("")
    lab = [r for r in cal if r["hand"]]
    for order, desc in (("A", "echo before interleaving"),
                        ("B", "interleaving before echo")):
        ok = sum(1 for r in lab if MAP.get(classify(r, order)) == r["hand"])
        say(f"  order {order} ({desc:<26}) reproduces {ok}/{len(lab)} = {ok/len(lab):.0%}")
    best_order = max("AB", key=lambda o: sum(
        1 for r in lab if MAP.get(classify(r, o)) == r["hand"]))
    say("")
    say(f"  Using order {best_order}.")
    for r in cal:
        r["mode"] = classify(r, best_order)

    say("")
    say("              hand L   hand I   hand C   hand X")
    for code in ("locked_in", "interleaving", "inconsistent", "echo",
                 "degenerate", "untestable"):
        cnts = [sum(1 for r in lab if r["mode"] == code and r["hand"] == h)
                for h in ("L", "I", "C", "X")]
        if sum(cnts):
            say(f"  {code:<13}" + "".join(f"{c:>9}" for c in cnts))
    ok = sum(1 for r in lab if MAP.get(r["mode"]) == r["hand"])
    say("")
    say(f"  overall {ok}/{len(lab)} = {ok/len(lab):.0%}")
    say("")
    say("  The residual in `inconsistent` that humans called X is the part no")
    say("  string rule can reach: answers that were CORRECT and the matcher")
    say("  disagreed. That is the judge's job, not a threshold's.")

    # =======================================================================
    # PART C - apply to all of Phase C
    # =======================================================================
    say("")
    say("=" * 78)
    say("C. FULL SCALE - all Phase C trajectories")
    say("")

    pc_rows, t_pc = [], time.time()
    for ds, want in PHASEC_TARGETS:
        recs = data.load_records(ds, n=want * 2, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        records = {(ds, rec.qid): rec for rec in kept[:want]}
        sub = PHASEC_ROOT / ds
        if not sub.exists():
            say(f"  WARNING {ds}: {sub} does not exist, skipping")
            continue
        n_files = len(list(sub.glob("*.npz")))
        say(f"  {ds:<14} {n_files:,} trajectories on disk, "
            f"{len(records):,} questions selected")
        got, skipped, empty = build_rows(sub, records, tokenizer, limit=LIMIT)
        if skipped:
            say(f"  {'':<14} {skipped:,} files had no matching record "
                f"(dataset selection drifted - investigate before trusting)")
        if empty:
            say(f"  {'':<14} {empty:,} produced no output at all "
                f"(every position EOS) - counted, not analysed")
        pc_rows.extend(got)
        say(f"  {'':<14} measured {len(got):,}   "
            f"[{timedelta(seconds=int(time.time() - t_pc))}]")

    if not pc_rows:
        say("\nNo Phase C rows measured. Stopping.")
        sys.exit(1)

    # Integrity check on the answers CSV. Phase C wrote it only at the end of
    # each invocation, and the first invocation's container was killed, so the
    # CSV is expected to be short. Nothing depends on it - answers here are
    # decoded from the trajectories - but a silent mismatch is worth naming.
    n_csv = 0
    if PHASEC_ANSWERS_CSV.exists():
        with open(PHASEC_ANSWERS_CSV, encoding="utf-8") as fh:
            n_csv = sum(1 for _ in csv.DictReader(fh))
    say("")
    say(f"  answers CSV holds {n_csv:,} rows against {len(pc_rows):,} "
        f"trajectories measured.")
    if n_csv < len(pc_rows):
        say("  The CSV is short because it is written once per invocation and one")
        say("  invocation was killed. This script does not read it - every answer")
        say("  above was decoded from the .npz - so nothing is lost.")

    wrong = [r for r in pc_rows if not r["correct"]]
    for r in wrong:
        r["mode"] = classify(r, best_order)
    for r in pc_rows:
        r.setdefault("mode", "correct")

    say("")
    say("  PROVISIONAL - right/wrong below is whole-word string matching, which")
    say("  scores some wrong answers correct. The judge revises these numbers.")
    say("")
    say("  dataset          total    wrong   err%  locked-in  interleav  "
        "inconsist  echo  degen  untest")
    say("  --------------  -------  -------  ----  ---------  ---------  "
        "---------  ----  -----  ------")
    by = defaultdict(list)
    for r in pc_rows:
        by[r["dataset"]].append(r)
    for ds, _w in PHASEC_TARGETS:
        rs = by.get(ds, [])
        if not rs:
            continue
        w = [r for r in rs if not r["correct"]]
        c = Counter(r["mode"] for r in w)
        say(f"  {ds:<14}  {len(rs):7,}  {len(w):7,}  {len(w)/len(rs):4.0%}  "
            f"{c['locked_in']:9,}  {c['interleaving']:9,}  "
            f"{c['inconsistent']:9,}  {c['echo']:4,}  {c['degenerate']:5,}  "
            f"{c['untestable']:6,}")
    tot = Counter(r["mode"] for r in wrong)
    say(f"  {'ALL':<14}  {len(pc_rows):7,}  {len(wrong):7,}  "
        f"{len(wrong)/len(pc_rows):4.0%}  "
        f"{tot['locked_in']:9,}  {tot['interleaving']:9,}  "
        f"{tot['inconsistent']:9,}  {tot['echo']:4,}  {tot['degenerate']:5,}  "
        f"{tot['untestable']:6,}")

    contaminated = tot["echo"] + tot["degenerate"]
    say("")
    say(f"  echo + degenerate = {contaminated:,} of {len(wrong):,} "
        f"({contaminated/max(1,len(wrong)):.0%}) are NOT hallucinations.")
    say("  They are the model reusing its input or running out of budget, and")
    say("  they are reported as a rate in the paper's data section rather than")
    say("  counted as errors of belief.")

    # =======================================================================
    # PART D - did the 150-question pilot hold?
    # =======================================================================
    say("")
    say("=" * 78)
    say("D. PILOT vs FULL SCALE")
    say("")
    say("  Share of wrong answers in each mode. The pilot drove the Phase C")
    say("  sample size; if the shares moved, the sizing assumption moved with")
    say("  them. Both columns use the SAME refitted rules, so any difference is")
    say("  sample size, not a rule change.")
    say("")
    say("  dataset         mode            pilot n   pilot%   full n   full%")
    say("  --------------  -------------  --------  -------  -------  ------")
    cal_by = defaultdict(list)
    for r in cal:
        cal_by[r["dataset"]].append(r)
    for ds, _w in PHASEC_TARGETS:
        pilot = Counter(r["mode"] for r in cal_by.get(ds, []))
        full = Counter(r["mode"] for r in by.get(ds, []) if not r["correct"])
        n_p, n_f = sum(pilot.values()), sum(full.values())
        if not n_p or not n_f:
            continue
        for m in ("locked_in", "interleaving", "inconsistent", "echo",
                  "degenerate", "untestable"):
            say(f"  {ds:<14}  {m:<13}  {pilot[m]:8,}  {pilot[m]/n_p:6.1%}  "
                f"{full[m]:7,}  {full[m]/n_f:5.1%}")
        say("")

    # =======================================================================
    # PART E - the judge work-order
    # =======================================================================
    say("=" * 78)
    say("E. WORK-ORDER FOR THE QWEN3-8B JUDGE")
    say("")
    say("  String matching fails in BOTH directions, so the judge's scope is not")
    say("  'the wrong answers':")
    say("")
    say("    false wrong   the model was right and the matcher disagreed")
    say("                  'is considered a giant dog breed' vs 'is a giant dog")
    say("                  breed'. These inflate every error rate above.")
    say("")
    say("    false right   the gold string occurs in an answer that is not the")
    say("                  answer. Step 9b measured this and could not repair it.")
    say("                  These are hallucinations currently counted as correct.")
    say("")
    say("  Only the first set is reachable by re-reading the wrong answers. The")
    say("  second needs the correct ones read too, so the honest scope is all of")
    say("  them, and the two options are priced separately:")
    say("")
    say("    scope                       questions")
    say(f"    wrong answers only          {len(wrong):9,}")
    say(f"    every answer                {len(pc_rows):9,}")
    say("")
    say("  Cost depends on judge throughput, which has NOT been measured for")
    say("  this model on this hardware. Guessing it is how the Phase C estimate")
    say("  went wrong by 5x, so it is left as a variable here:")
    say("")
    say("    s/question   wrong only          everything")
    say("    ----------   ----------------    ----------------")
    for sec in (0.25, 0.5, 1.0, 2.0):
        hw = len(wrong) * sec / 3600
        he = len(pc_rows) * sec / 3600
        say(f"    {sec:>5.2f}        {hw:5.2f} h  ${hw*2.10:6.2f}    "
            f"{he:5.2f} h  ${he*2.10:6.2f}")
    say("")
    say("  Step 13 therefore measures the rate on 100 questions before")
    say("  committing to a full run, and validates the judge against the 59")
    say("  hand labels - an unvalidated judge is not publishable.")

    # ---- CSV --------------------------------------------------------------
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["dataset", "qid", "correct", "mode", "stable_top3", "cands_top3",
            "q_overlap", "copy_run", "n_content", "truncated", "gold_round",
            "gold_ever", "gold_testable", "question", "answer", "gold"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in pc_rows:
            r["question"] = str(r["question"])[:200]
            r["answer"] = str(r["answer"])[:300]
            r["gold"] = str(r["gold"])[:200]
        w.writerows(pc_rows)

    say("")
    say("=" * 78)
    say(f"  rows written : {len(pc_rows):,}")
    say(f"  CSV          : {CSV_PATH}")
    say(f"  wall clock   : {timedelta(seconds=int(time.time() - t0))}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
