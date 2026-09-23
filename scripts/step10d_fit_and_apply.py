#!/usr/bin/env python3
"""
Step 10d - fit the failure-mode rules to hand labels, apply to all 150
=======================================================================

    modal run modal_app.py::run_cpu --script step10d_fit_and_apply.py

CPU only. Minutes, free.

WHY
===
Step 10's thresholds were chosen by reading definitions, three times, and were
wrong three times. Step 10c produced a blind labelling worksheet; 60 wrong
answers were then labelled independently by two annotators.

    raw agreement   55/60 = 92%
    Cohen's kappa   0.882          ("almost perfect", >0.80)

Both label sets independently fit the SAME candidate-count cut (<= 3.3), which
is the strongest evidence so far that the taxonomy is a property of the data
rather than of whoever drew the line.

This script stops asserting thresholds. It FITS them to the 60 hand labels,
reports how well the fitted rules reproduce those labels, and only then applies
them to all 150 wrong answers to get the Phase C sample size.

THE TWO THINGS BEING FITTED
===========================
**1. locked_in.** From the hand labels, maximising balanced accuracy on L vs C:

    stable_top3 >= 0.25  AND  cands_top3 <= 3.3     87% bal.acc, 92% recall

Compare the unfitted guess it replaces - `stable_all >= 0.70 AND cands <= 2.0`
caught 5 of 12 true locked-in cases, 42% recall. That gap is the whole reason
locked-in kept coming out too small.

**2. The X category, which the code currently cannot see at all.**

Hand labels put 32% of wrong answers in X - not hallucinations, but:

    echo    the model restates the question and never fills the answer slot
            "Roger O. Egeberg was Assistant Secretary for Health and Scient..."
            (64 rounds, never gives the years)

    copy    the answer slot is filled with a word lifted from the question
            "Billie Jean King's maiden name was Billie Jean King."
            "Shari Lewis' SASSY sock puppet was named Sassy."

Step 10 only detects truncation, which is 11%. So roughly a fifth of "wrong
answers" are currently mislabelled as hallucinations. For a hallucination
dataset that is a serious contamination, and it inflates every base rate.

Both echo and copy have the same signature: **the answer's content words
already occur in the question.** That is one number, `question_overlap`, and
its threshold is fitted here the same way.

A third kind of X cannot be detected from strings at all: the model was simply
RIGHT and the matcher disagreed.

    model 'The Leonberger is considered a giant dog breed.'
    gold  'The Leonberger is a giant dog breed.'

    model 'Whole Foods'   gold 'hydrogenated'   <- HotpotQA gold is broken

Those need the Qwen3 judge (Stage 3). This script reports the residual so its
size is known rather than assumed.

RULE ORDER
==========
Order changes the answer, so it is fitted rather than assumed: the script tries
both plausible orderings and reports which reproduces the hand labels better.

    A:  degenerate -> echo -> untestable -> interleaving -> locked_in -> inconsistent
    B:  degenerate -> interleaving -> echo -> untestable -> locked_in -> inconsistent

B protects genuine interleaving cases whose answers happen to restate the
question (#43, the AVN Awards one, overlaps the question at ~0.75 and is a real
interleaving). A protects against spurious gold matches inside echoes (#6,
where a gold alias appears in the question itself).

WHAT COMES OUT
==============
The Phase C sample size, computed from a calibrated split instead of an
assumed one - which is the last thing standing between this project and
generation.
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

TRAJ_SUBDIR = config.TRAJ_DIR / "step9" / config.RUN_TAG
MODES_CSV = config.TAB_DIR / "step10_modes.csv"
KEY_CSV = config.TAB_DIR / "step10c_answer_key.csv"
REPORT_PATH = config.OUT_DIR / "step10d_fitted_report.txt"
CSV_PATH = config.TAB_DIR / "step10d_final_modes.csv"

DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
N_PER_DATASET = 100
TARGET_PER_CELL = 150

# ---------------------------------------------------------------------------
# The hand labels. Worksheet position 1-60; KEY_CSV maps position -> qid.
#
# Consensus of two independent blind passes (kappa 0.882). Where the two
# annotators disagreed the human annotator's call is kept, with two exceptions
# recorded here so the provenance is auditable:
#
#   #36  human said L, consensus X - the model's answer is CORRECT ('is
#        considered a giant dog breed' vs gold 'is a giant dog breed') and
#        labelling it locked-in would count a correct answer as a
#        hallucination. Fitting with L instead gives an IDENTICAL threshold,
#        so this affects data quality, not calibration.
#   #11  dropped - the worksheet truncates at ~62 characters and this answer's
#        content lies beyond that, so neither annotator could read it. A
#        worksheet defect, not an unlabellable trajectory.
# ---------------------------------------------------------------------------
HAND = ("L C C X C X C I X C . X X X C C I L C C I L L I L C C C C C "
        "L C L X X X X X X C C C I X X L C C C L L L X I X X X X L C").split()
assert len(HAND) == 60, len(HAND)
DROPPED = {i for i, h in enumerate(HAND, 1) if h == "."}

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Measurement (identical to step10c, restated so this script is standalone)
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

# Words too common to count as evidence that the answer was lifted from the
# question. Deliberately small: the aim is to catch content reuse, and an
# aggressive stoplist would hide exactly the reuse being measured.
STOP = {"is", "was", "are", "were", "be", "been", "of", "in", "on", "at", "to",
        "for", "and", "or", "but", "by", "with", "from", "that", "which",
        "who", "what", "when", "where", "how", "it", "its", "this", "these",
        "as", "has", "have", "had", "do", "does", "did", "not", "no", "yes",
        "s", "t"}


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def question_overlap(answer: str, question: str) -> float:
    """Share of the answer's content words that already occur in the question.

    THE ECHO AND COPY DETECTOR, and the one measurement Step 10 was missing.

    An echo restates the question and never fills the answer slot, so nearly
    every content word is borrowed and this approaches 1.0. A copy fills the
    slot with a word lifted from the question - "Billie Jean King's maiden name
    was Billie Jean King" - and scores almost as high. A real answer introduces
    at least one word the question did not contain, and scores lower.

    Both failures are the model reusing its input rather than retrieving a
    belief, so one number covers them, and its cut is fitted to the hand labels
    rather than guessed.

    Returns 1.0 for an empty answer: nothing was contributed.
    """
    a = [w for w in normalise(answer).split() if w not in STOP]
    q = set(normalise(question).split())
    if not a:
        return 1.0
    return sum(1 for w in a if w in q) / len(a)


def top_by_entropy(ents: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-entropy positions, selected BY RANK.

    A value threshold (`ents >= np.quantile(ents, .75)`) selects far more than
    a quarter when the high-entropy positions are a small minority, which is
    the normal case - `The best answer is A. bookstore.` is eight positions of
    which one carries the claim.
    """
    k = max(1, min(int(k), len(ents)))
    return np.argsort(-ents, kind="stable")[:k]


def measure(traj) -> dict:
    """stable_top3 and cands_top3 over the three highest-entropy positions.

    Why the top three and not all of them: `claude/stable-top3-measure-
    decision.md`. Averaging over every content position correlates +0.44 with
    answer length; restricting to the high-entropy positions drops that to
    -0.07, because template and echoed tokens are low-entropy and trivially
    stable, and averaging lets them outvote the one position carrying the claim.
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
    n_content = len(pos)
    return {
        "n_content": n_content,
        "stable_top3": float(np.mean(fr[sel])),
        "cands_top3": float(np.mean(cn[sel])),
        "truncated": n_content >= traj.gen_length - 2,
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


def fit_echo(rows) -> tuple:
    """Grid-search the question_overlap cut on hand-labelled X vs everything else.

    X also contains 'the model was actually right', which no overlap threshold
    can catch, so perfect separation is not expected and not the target. What
    matters is whether echo and copy - the two mechanical kinds - separate.
    """
    lab = [r for r in rows if r.get("hand") in ("L", "I", "C", "X")]
    best = None
    for cut in np.arange(0.40, 1.01, 0.01):
        tp = sum(1 for r in lab if r["hand"] == "X" and r["q_overlap"] >= cut)
        fn = sum(1 for r in lab if r["hand"] == "X") - tp
        fp = sum(1 for r in lab if r["hand"] != "X" and r["q_overlap"] >= cut)
        tn = sum(1 for r in lab if r["hand"] != "X") - fp
        if tp + fn == 0 or fp + tn == 0:
            continue
        bal = 0.5 * (tp / (tp + fn) + tn / (fp + tn))
        if best is None or bal > best[0]:
            best = (bal, float(cut), tp, fn, fp, tn)
    return best


def classify(r, s_cut, c_cut, e_cut, order="B") -> str:
    """Apply the fitted rules. `order` selects which of the two orderings runs."""
    if r.get("truncated"):
        return "degenerate"
    echo = r["q_overlap"] >= e_cut
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

    if r["stable_top3"] >= s_cut and r["cands_top3"] <= c_cut:
        return "locked_in"
    return "inconsistent"


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 10d: fit rules to hand labels, apply to all 150")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    for p in (MODES_CSV, KEY_CSV):
        if not p.exists():
            say(f"Need {p}. Run step10 and step10c first.")
            sys.exit(1)

    prev = {}
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            prev[(r["dataset"], r["qid"])] = r

    # worksheet position -> qid, so the hand labels attach to the right rows
    key = {}
    with open(KEY_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key[int(r["n"])] = (r["dataset"], r["qid"])
    hand_by_qid = {key[n]: HAND[n - 1] for n in key
                   if n in key and n not in DROPPED and HAND[n - 1] != "."}
    say(f"{len(prev)} rows from Step 10, {len(hand_by_qid)} hand labels attached")

    records = {}
    for ds in DATASETS:
        recs = data.load_records(ds, n=N_PER_DATASET * 2, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        for rec in kept[:N_PER_DATASET]:
            records[(ds, rec.qid)] = rec

    say("Measuring...")
    rows = []
    for path in sorted(TRAJ_SUBDIR.glob("*.npz")):
        stem = path.stem
        ds = stem.split("_", 1)[0]
        qid = stem.split("_", 1)[1] if "_" in stem else stem
        rec, p = records.get((ds, qid)), prev.get((ds, qid))
        if rec is None or p is None or p["correct"].lower() == "true":
            continue
        traj = logging_patch.Trajectory.load(path)
        want = (config.GEN_LENGTH, config.DENOISING_STEPS, config.BLOCK_LENGTH)
        got = (int(traj.gen_length), int(traj.steps), int(traj.block_length))
        if got != want:
            raise SystemExit(f"\nSTOPPED - {path} is gen/steps/block {got}, "
                             f"config.py says {want}. Re-run step9.\n")
        m = measure(traj)
        if not m:
            continue
        rows.append(dict(
            dataset=ds, qid=qid, question=rec.question,
            answer=p.get("answer", ""), gold=p.get("gold", ""),
            gold_ever=str(p.get("gold_ever", "")).lower() == "true",
            gold_testable=str(p.get("gold_testable", "True")).lower() == "true",
            q_overlap=question_overlap(p.get("answer", ""), rec.question),
            hand=hand_by_qid.get((ds, qid)), **m))
    say(f"{len(rows)} wrong answers measured, "
        f"{sum(1 for r in rows if r['hand'])} of them hand-labelled")

    # ---- fit --------------------------------------------------------------
    say("")
    say("=" * 78)
    say("A. FITTED THRESHOLDS")
    say("")
    bal, s_cut, c_cut, tp, fn, fp, tn = fit_locked(rows)
    say(f"  locked_in : stable_top3 >= {s_cut:.2f}  AND  cands_top3 <= {c_cut:.1f}")
    say(f"              balanced acc {bal:.0%}   recall {tp/(tp+fn):.0%}   "
        f"specificity {tn/(fp+tn):.0%}   (n_L={tp+fn}, n_C={fp+tn})")
    say("")
    ebal, e_cut, etp, efn, efp, etn = fit_echo(rows)
    say(f"  echo/copy : question_overlap >= {e_cut:.2f}")
    say(f"              balanced acc {ebal:.0%}   recall {etp/(etp+efn):.0%}   "
        f"specificity {etn/(efp+etn):.0%}   (n_X={etp+efn})")
    say("")
    say("  The unfitted guess these replace: stable_all >= 0.70 AND cands <= 2.0,")
    say("  which caught 5 of 12 true locked-in cases - 42% recall. Everything")
    say("  Step 10 reported about locked-in was measured through that gap.")

    # ---- which rule order reproduces the hand labels better ----------------
    say("")
    say("=" * 78)
    say("B. RULE ORDER")
    say("")
    MAP = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
           "echo": "X", "degenerate": "X", "untestable": None}
    for order, desc in (("A", "echo before interleaving"),
                        ("B", "interleaving before echo")):
        lab = [r for r in rows if r["hand"]]
        ok = sum(1 for r in lab
                 if MAP.get(classify(r, s_cut, c_cut, e_cut, order)) == r["hand"])
        say(f"  order {order} ({desc:<28}) reproduces {ok}/{len(lab)} = {ok/len(lab):.0%}")
    lab = [r for r in rows if r["hand"]]
    best_order = max("AB", key=lambda o: sum(
        1 for r in lab if MAP.get(classify(r, s_cut, c_cut, e_cut, o)) == r["hand"]))
    say(f"")
    say(f"  Using order {best_order}.")

    for r in rows:
        r["mode"] = classify(r, s_cut, c_cut, e_cut, best_order)

    # ---- confusion against the hand labels --------------------------------
    say("")
    say("=" * 78)
    say("C. FITTED RULES vs HAND LABELS")
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
    say("  Read the off-diagonal. Cases landing in `inconsistent` that the")
    say("  annotators called X are the residual the string rules cannot reach:")
    say("  answers that were actually CORRECT and the matcher disagreed. Those")
    say("  need the Qwen3 judge (Stage 3), not a better threshold.")

    # ---- the split --------------------------------------------------------
    say("")
    say("=" * 78)
    say("D. FINAL SPLIT - all wrong answers, fitted rules")
    say("")
    say("  dataset         wrong  locked-in  interleav  inconsist  echo  degen  untest")
    say("  --------------  -----  ---------  ---------  ---------  ----  -----  ------")
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    split = {}
    for ds in DATASETS:
        c = Counter(r["mode"] for r in by[ds])
        split[ds] = (len(by[ds]), c)
        say(f"  {ds:<14}  {len(by[ds]):5d}  {c['locked_in']:9d}  "
            f"{c['interleaving']:9d}  {c['inconsistent']:9d}  {c['echo']:4d}  "
            f"{c['degenerate']:5d}  {c['untestable']:6d}")
    tot = Counter()
    for _n, c in split.values():
        tot.update(c)
    n_wrong = sum(v[0] for v in split.values())
    say(f"  {'ALL':<14}  {n_wrong:5d}  {tot['locked_in']:9d}  "
        f"{tot['interleaving']:9d}  {tot['inconsistent']:9d}  {tot['echo']:4d}  "
        f"{tot['degenerate']:5d}  {tot['untestable']:6d}")
    say("")
    contaminated = tot["echo"] + tot["degenerate"]
    say(f"  echo + degenerate = {contaminated} of {n_wrong} "
        f"({contaminated/n_wrong:.0%}) are NOT hallucinations.")
    say("  They are the model reusing its input or running out of budget. They")
    say("  must be excluded from Phase C and reported as a rate in the paper's")
    say("  data section - a hallucination benchmark containing them measures")
    say("  the decoder, not the model's beliefs.")

    # ---- sample size ------------------------------------------------------
    say("")
    say("=" * 78)
    say("E. PHASE C SAMPLE SIZE")
    say("")
    say(f"  Target {TARGET_PER_CELL} per mode per dataset, driven by the rarest.")
    say("  Usable = locked_in + interleaving + inconsistent (echo, degenerate")
    say("  and untestable are excluded, so the error rate below is the USABLE")
    say("  rate, not the raw one.")
    say("")
    say("  dataset         usable%  rarest mode     share   questions   GPU-hr    cost")
    say("  --------------  -------  --------------  ------  ---------  -------  ------")
    SEC = 2.01
    total_q = 0
    for ds in DATASETS:
        n_all, c = split[ds]
        usable = {m: c[m] for m in ("locked_in", "interleaving", "inconsistent")}
        n_use = sum(usable.values())
        n_total = len([1 for k in prev if k[0] == ds])
        if not n_total or not n_use:
            say(f"  {ds:<14}  (no usable wrong answers)")
            continue
        rate = n_use / n_total
        rarest = min(usable, key=lambda m: usable[m])
        share = usable[rarest] / n_use
        if share == 0:
            say(f"  {ds:<14}  {rate:6.0%}   {rarest:<14}   0.0%   unbounded")
            continue
        need = int(TARGET_PER_CELL / (rate * share))
        hrs = need * SEC / 3600
        total_q += need
        say(f"  {ds:<14}  {rate:6.0%}   {rarest:<14}  {share:5.1%}  {need:9d}  "
            f"{hrs:7.1f}  ${hrs*2.10:5.2f}")
    if total_q:
        hrs = total_q * SEC / 3600
        say(f"  {'TOTAL':<14}  {'':6}   {'':14}  {'':6}  {total_q:9d}  "
            f"{hrs:7.1f}  ${hrs*2.10:5.2f}")
    say("")
    say(f"  At the measured {SEC:.2f} s/question on A100 at $2.10/hr.")
    say("  A mode that never appears is 'unbounded' - it is not absent, it is")
    say("  rarer than 1-in-(wrong answers seen), and sizing from zero")
    say("  observations is extrapolation rather than estimation.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["dataset", "qid", "mode", "hand", "stable_top3", "cands_top3",
            "q_overlap", "n_content", "truncated", "gold_ever", "gold_testable",
            "question", "answer", "gold"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    say("")
    say(f"  CSV: {CSV_PATH}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
