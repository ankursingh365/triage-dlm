#!/usr/bin/env python3
"""
Step 10c - is `stable_frac` a real measurement, and what are the 58
unclassified questions actually doing?
===================================================================

    modal run modal_app.py::run_cpu --script step10c_confound_check.py

CPU only. Reads the 300 cached trajectories and Step 10's CSV. Minutes.

Why this exists
---------------
Step 10, run on trajectories from the corrected decoder (`g64s64b32`), gives a
working split:

    locked_in     16 of 150 (10.7%)
    interleaving   9
    inconsistent  44
    unclassified  60  (40%)  <- the LARGEST bucket
    degenerate    17
    untestable     4

Sample size from those shares: 3,000 TriviaQA + 7,500 HotpotQA, about $18.
Affordable. But the numbers rest on thresholds that were chosen by reading
definitions rather than fitted to anything, and the unclassified bucket says
those thresholds are in the wrong place:

    all wrong answers   median stable 0.51   candidates 3.0
    the 60 unclassified median stable 0.61   candidates 2.5

Those are the same distribution. The unclassified are not a fourth kind of
failure - they are the CENTRE of the data, and the two rules carve off the
tails and leave the bulk unlabelled. A threshold has now been guessed three
times and been wrong three times. This script stops guessing.


PART 1 - THE CONFOUND SCREEN
============================
Before fitting anything, the measure itself has to be shown to be sound. Two
warning signs that `mean_stable_frac` may be tracking answer length rather
than model behaviour:

**CommonsenseQA contributes 2 locked-in from 11 wrong answers (18%) against
TriviaQA's 5 from 61 (8.2%).** CSQA answers are four tokens (`'C. envelope'`);
HotpotQA answers run past twenty. Averaging stability over 4 positions and
over 20 are not the same measurement.

**One locked-in example is not a failure at all.**

    Q: Which dog is considered a giant dog breed, the Leonberger or the Ba...
    A: 'The Leonberger is considered a giant dog breed.'
    gold: The Leonberger is a giant dog breed.

Same answer, one extra word, scored wrong by the string matcher and then
classified locked-in. And this one is a genuine near-miss:

    'The 22nd AVN Awards took place in Las Vegas, Nevada.'  gold: Clark County
        -> Las Vegas IS IN Clark County.

High stability in long answers often comes from TEMPLATE or ECHO positions,
which are trivially stable. `The best answer is A.` is stable because it is
boilerplate, not because the model is confident about the answer. Averaging
over the whole sentence lets the boilerplate outvote the one position that
carries the claim.

So this script applies the Step 6b screens to four candidate measures:

    stable_all    mean over every content position         (what Step 10 used)
    stable_hi     top QUARTILE by entropy                  (constant fraction)
    stable_top3   top THREE by entropy                     (constant count)
    stable_top1   the single highest-entropy position

**Entropy is the right selector.** Template tokens and echoed question words
are low-entropy - the model finds them easy, which is exactly why they are
stable. The positions where the model was genuinely deciding something are the
high-entropy ones, and a failure mode is a statement about those positions and
no others.

Two ways of saying "the high-entropy positions" are tested because they fail
differently. A fixed COUNT has the better prior: gold answers are a few tokens
wide whatever surrounds them - `Clark County` is two tokens whether the model
says `Clark County` or `The 22nd AVN Awards took place in Clark County,
Nevada.` A fixed FRACTION still grows with the sentence, which is the confound
being tested for. The screen decides between them rather than this docstring.

Screens, both inherited from Step 6b:

    LENGTH   |r| with content-token count.  >0.80 fails, 0.50-0.80 cautions.
    VARIANCE coefficient of variation.      <0.02 means schedule-determined,
                                            not a measurement.


PART 2 - THE BLIND LABELLING WORKSHEET
======================================
Thresholds invented by reading definitions have now been wrong three times:
`frac_early >= 0.5` was mathematically unreachable under the old reveal
schedule and returned locked_in = 0; the 0.70 / 2.0 pair sits in the tails of
a distribution whose median is 0.51 / 3.0; and the gold test fired on bare
option letters and on the word "no". The way out is not a fourth guess. It is
to fit them.

Step 18 was always going to hand-verify 100 trajectories with a second
annotator. **That work should happen now, before Phase C, not after.** It costs
no compute, it converts every threshold from an assertion into a fitted
parameter, and it is a paper requirement regardless: a Q1 reviewer asking "how
did you define locked-in" is much better answered by "calibrated against 100
hand-labelled trajectories" than by "we chose 0.70".

The worksheet is **blind** - it shows the trace and the gold answer, never the
label this code assigned. Seeing the machine's guess first would anchor the
annotator and destroy the agreement statistic the paper needs. The key is
written to a separate file and compared afterwards.

Traces are **compressed to change-points**. In the locked-in example from Step
10, rounds 5 through 31 are identical; printing all 32 makes 60 questions
unreadable. Showing only the rounds where the guess changed turns a 32-line
trace into 3-8 lines and makes the whole worksheet about a 30-minute job.
"""

import csv
import random
import re
import sys
from collections import defaultdict
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
REPORT_PATH = config.OUT_DIR / "step10c_confound_report.txt"
WORKSHEET_PATH = config.OUT_DIR / "step10c_labelling_worksheet.txt"
KEY_PATH = config.TAB_DIR / "step10c_answer_key.csv"

DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
N_PER_DATASET = 100

N_WORKSHEET = 60          # questions to hand-label
HI_ENT_QUANTILE = 0.75    # top quartile by entropy
LABEL_SEED = 20260909     # same seed discipline as the rest of the project

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


SPECIAL_RE = re.compile(r"<\|[^|]*\|>")


# ===========================================================================
# The three candidate measures
# ===========================================================================

def assert_config_matches(traj, path) -> None:
    """Refuse to analyse a trajectory generated under different settings.

    `RUN_TAG` in the cache path is meant to make this impossible, but a path is
    only as good as every script that builds one, and that has now failed twice
    - once from stale untagged files, once from a script being replaced by a
    copy that predated the tagging fix. A path is a convention; this is a check.
    The trajectory records the settings it was made with, so the comparison is
    free and turns a silent wrong answer into a stopped run.
    """
    want = (config.GEN_LENGTH, config.DENOISING_STEPS, config.BLOCK_LENGTH)
    got = (int(traj.gen_length), int(traj.steps), int(traj.block_length))
    if got != want:
        raise SystemExit(
            "\n".join([
                "",
                "=" * 78,
                "  STOPPED - trajectory does not match the current configuration",
                "=" * 78,
                f"  file      : {path}",
                f"  config.py : gen {want[0]}  steps {want[1]}  block {want[2]}",
                f"  this file : gen {got[0]}  steps {got[1]}  block {got[2]}",
                "",
                "  Re-run step9_base_rate.py under the current settings, then",
                "  step10_failure_modes.py, then this script.",
                "=" * 78,
            ])
        )


def position_stats(traj, positions: list) -> tuple:
    """Per-position stability and mean pre-reveal entropy.

    `stable_frac` is unchanged from Step 10: walking back from a position's own
    reveal round, the share of its pre-reveal life spent holding the value it
    finally committed. Schedule-free by construction.

    `mean_entropy` is averaged over the same pre-reveal window - how uncertain
    the model was about this position while it was still deciding it. That is
    what separates a template token from a claim.
    """
    rev = traj.revealed_at()
    fracs, ents, cands = [], [], []

    for i in positions:
        r = int(rev[i])
        history = traj.pred_ids[: r + 1, i].tolist()
        committed = history[-1]

        held = 0
        for v in reversed(history):
            if v == committed:
                held += 1
            else:
                break

        fracs.append(held / len(history))
        ents.append(float(np.mean(traj.entropy[: r + 1, i])))
        cands.append(len(set(history)))

    return np.array(fracs), np.array(ents), np.array(cands)


def top_by_entropy(ents: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` highest-entropy positions. Selection is by RANK.

    A value threshold does not work here, and getting this wrong is easy. The
    obvious implementation - `ents >= np.quantile(ents, 0.75)` - returns a
    VALUE, and `>=` on that value selects far more than a quarter of the
    positions whenever the high-entropy ones are a small minority. On a
    12-position answer with 10 low-entropy template tokens and 2 real ones, the
    75th percentile still falls inside the template block, so the comparison
    selects all 12 and the filter does nothing at all.

    That minority case is not a corner case, it is the normal one: `The best
    answer is A. bookstore.` is eight positions of which one carries the claim.
    Ranking with argsort selects exactly `k` and is immune to the distribution's
    shape.
    """
    k = max(1, min(int(k), len(ents)))
    return np.argsort(-ents, kind="stable")[:k]


def three_measures(traj) -> dict:
    """Stability under four position selections, plus candidate counts.

    If they all agree, `stable_all` was fine and Step 10's CSQA imbalance came
    from somewhere else. If they disagree, the high-entropy variants are the
    ones to trust: they are the only ones measuring the position that carries
    the answer rather than the sentence wrapped around it.

    Two ways of saying "the high-entropy positions", because they fail
    differently and only the screen can say which is right here:

        stable_hi    top QUARTILE - a constant fraction of the answer
        stable_top3  top THREE    - a constant count

    A fixed count has the better prior. Gold answers are a few tokens wide
    whatever surrounds them: `Clark County` is two tokens whether the model
    says `Clark County` or `The 22nd AVN Awards took place in Clark County,
    Nevada.` A fixed fraction still grows with the sentence, which is the
    confound being tested for. Section A reports both and the screen decides.
    """
    final = traj.final_ids.tolist()
    positions = [i for i, t in enumerate(final)
                 if t not in (traj.eos_id, traj.mask_id)]
    if not positions:
        return {}

    fracs, ents, cands = position_stats(traj, positions)

    hi = top_by_entropy(ents, int(np.ceil(len(ents) * (1.0 - HI_ENT_QUANTILE))))
    top3 = top_by_entropy(ents, 3)
    top1 = top_by_entropy(ents, 1)

    return {
        "n_content": len(positions),
        "stable_all": float(np.mean(fracs)),
        "stable_hi": float(np.mean(fracs[hi])),
        "stable_top3": float(np.mean(fracs[top3])),
        "stable_top1": float(np.mean(fracs[top1])),
        "cands_all": float(np.mean(cands)),
        "cands_hi": float(np.mean(cands[hi])),
        "cands_top3": float(np.mean(cands[top3])),
        "cands_top1": float(np.mean(cands[top1])),
        "n_hi": int(len(hi)),
    }


# ===========================================================================
# Compressed traces for the worksheet
# ===========================================================================

def compressed_trace(traj, tokenizer, width: int = 62) -> list:
    """The shadow prediction, but only at rounds where it CHANGED.

    A locked-in trajectory holds the same guess for 25+ consecutive rounds.
    Printing every round makes 60 questions unreadable; printing change-points
    makes the shape of the failure visible at a glance and is what makes a
    30-minute hand-labelling session possible at all.

    Returns lines like "r00-r04 | The best option is A." - the round range over
    which that guess was held, then the guess.
    """
    rows = []
    for r in range(traj.pred_ids.shape[0]):
        ids = [int(t) for t in traj.pred_ids[r].tolist()
               if t not in (traj.eos_id, traj.mask_id)]
        rows.append(ids)
    if not rows:
        return []

    texts = tokenizer.batch_decode(rows, skip_special_tokens=True)
    texts = [" ".join(str(t).split())[:width] for t in texts]

    out, start, prev = [], 0, texts[0]
    for r in range(1, len(texts)):
        if texts[r] != prev:
            span = f"r{start:02d}" if start == r - 1 else f"r{start:02d}-r{r-1:02d}"
            out.append(f"{span:<9} | {prev}")
            start, prev = r, texts[r]
    span = f"r{start:02d}" if start == len(texts) - 1 else f"r{start:02d}-r{len(texts)-1:02d}"
    out.append(f"{span:<9} | {prev}")
    return out


def screen(name: str, values: list, lengths: list) -> str:
    """Step 6b's two screens, applied to one candidate measure."""
    v, L = np.array(values), np.array(lengths)
    if len(v) < 3 or np.std(v) == 0:
        return f"  {name:<14}  (degenerate, no variance)"
    r = float(np.corrcoef(v, L)[0, 1])
    cv = float(np.std(v) / np.mean(v)) if np.mean(v) else 0.0

    if abs(r) > 0.80:
        verdict = "FAIL  length-driven"
    elif abs(r) > 0.50:
        verdict = "CAUTION"
    elif cv < 0.02:
        verdict = "FAIL  schedule-determined"
    else:
        verdict = "PASS"
    return (f"  {name:<14}  r_len {r:+.2f}   CV {cv:.3f}   "
            f"median {np.median(v):.2f}   {verdict}")


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 10c: confound screen + blind labelling worksheet")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not MODES_CSV.exists():
        say(f"Need {MODES_CSV}. Run step10_failure_modes.py first.")
        sys.exit(1)

    modes = {}
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            modes[(r["dataset"], r["qid"])] = r
    say(f"{len(modes)} rows from Step 10")

    say("Loading tokenizer (no model, no GPU)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_LLADA,
                                              trust_remote_code=True)

    records = {}
    for dataset in DATASETS:
        recs = data.load_records(dataset, n=N_PER_DATASET * 2, seed=config.SEED)
        kept, _exc, _rep = data_quality.clean_records(recs)
        for rec in kept[:N_PER_DATASET]:
            records[(dataset, rec.qid)] = rec

    say("Recomputing stability under three position selections...")
    rows = []
    for path in sorted(TRAJ_SUBDIR.glob("*.npz")):
        stem = path.stem
        dataset = stem.split("_", 1)[0]
        qid = stem.split("_", 1)[1] if "_" in stem else stem
        rec = records.get((dataset, qid))
        prev = modes.get((dataset, qid))
        if rec is None or prev is None:
            continue

        traj = logging_patch.Trajectory.load(path)
        assert_config_matches(traj, path)
        m = three_measures(traj)
        if not m:
            continue

        rows.append(dict(dataset=dataset, qid=qid,
                         correct=prev["correct"].lower() == "true",
                         mode=prev["mode"],
                         question=rec.question, answer=prev.get("answer", ""),
                         gold=prev.get("gold", ""),
                         gold_round=int(prev.get("gold_round", -1) or -1),
                         _traj=traj, **m))

    wrong = [r for r in rows if not r["correct"] and r["mode"] != "degenerate"]
    say(f"{len(rows)} measured, {len(wrong)} wrong and non-degenerate")

    # =====================================================================
    say("")
    say("=" * 78)
    say("A. CONFOUND SCREEN - is stable_frac measuring length?")
    say("")
    say("  Step 6b's two screens. |r| with content-token count above 0.80 means")
    say("  the measure is answer length wearing a different name; CV below 0.02")
    say("  means it is fixed by the reveal schedule and not a measurement.")
    say("")
    lengths = [r["n_content"] for r in wrong]
    for name in ("stable_all", "stable_hi", "stable_top3", "stable_top1",
                 "cands_all", "cands_hi", "cands_top3", "cands_top1"):
        say(screen(name, [r[name] for r in wrong], lengths))

    say("")
    say("  stable_all   = every content position        (what Step 10 used)")
    say("  stable_hi    = top QUARTILE by entropy       (constant fraction)")
    say("  stable_top3  = top THREE by entropy          (constant count)")
    say("  stable_top1  = the single highest-entropy position")
    say("")
    say("  Template and echoed tokens are LOW entropy - easy, therefore stable.")
    say("  The high-entropy variants measure the positions where the model was")
    say("  actually deciding, which is the only place a failure mode lives.")
    say("")
    say("  Pick the measure that PASSES the length screen and still separates")
    say("  the modes in section C. If stable_top3 passes and stable_all fails,")
    say("  Step 10's locked-in count was partly a sentence-length artefact and")
    say("  every threshold in it has to be refitted to the new measure.")

    # ---- the CSQA imbalance, directly -------------------------------------
    say("")
    say("=" * 78)
    say("B. THE CommonsenseQA IMBALANCE")
    say("")
    say("  Step 10 gave CSQA 3 of 7 locked-in from 11 wrong answers (27%) and")
    say("  TriviaQA 1 of 66 (1.5%). If that gap is answer length rather than")
    say("  model behaviour, it will shrink under the high-entropy measures.")
    say("")
    say("  dataset         n   ans_len   stable_all   stable_hi   stable_top3")
    say("  --------------  --  -------   ----------   ---------   -----------")
    by_ds = defaultdict(list)
    for r in wrong:
        by_ds[r["dataset"]].append(r)
    for dataset in DATASETS:
        rs = by_ds.get(dataset, [])
        if not rs:
            continue
        say(f"  {dataset:<14} {len(rs):3d}   {np.mean([r['n_content'] for r in rs]):6.1f}   "
            f"{np.mean([r['stable_all'] for r in rs]):10.2f}   "
            f"{np.mean([r['stable_hi'] for r in rs]):9.2f}   "
            f"{np.mean([r['stable_top3'] for r in rs]):11.2f}")
    say("")
    say("  If stable_all tracks ans_len across the three rows and stable_hi does")
    say("  not, the confound is confirmed and stable_hi replaces it everywhere -")
    say("  in Step 10's classifier, in Step 17's, and in features.py.")

    # ---- what happens to the unclassified bucket --------------------------
    say("")
    say("=" * 78)
    say("C. WHERE THE 58 UNCLASSIFIED SIT")
    say("")
    say("  mode            n   stable_all   stable_top3   cands_top3")
    say("  -------------  --   ----------   -----------   ----------")
    for mode in ("locked_in", "interleaving", "inconsistent", "unclassified"):
        rs = [r for r in wrong if r["mode"] == mode]
        if not rs:
            continue
        say(f"  {mode:<13} {len(rs):3d}   "
            f"{np.mean([r['stable_all'] for r in rs]):10.2f}   "
            f"{np.mean([r['stable_top3'] for r in rs]):11.2f}   "
            f"{np.mean([r['cands_top3'] for r in rs]):10.1f}")
    say("")
    say("  If unclassified sits between locked_in and inconsistent on stable_top3,")
    say("  it is not a fourth thing - it is the two real modes with the cut in")
    say("  the wrong place, and the hand labels will say where the cut goes.")

    # =====================================================================
    # The worksheet
    # =====================================================================
    rng = random.Random(LABEL_SEED)
    rare = [r for r in wrong if r["mode"] in ("locked_in", "interleaving")]
    rest = [r for r in wrong if r not in rare]
    rng.shuffle(rest)
    sample = rare + rest[: max(0, N_WORKSHEET - len(rare))]
    rng.shuffle(sample)                      # so modes are not grouped

    wlines = []

    def w(text: str = "") -> None:
        wlines.append(text)

    w("=" * 78)
    w("  TRIAGE - Step 10c: BLIND failure-mode labelling worksheet")
    w("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    w("=" * 78)
    w("")
    w("HOW TO USE THIS")
    w("---------------")
    w(f"{len(sample)} wrong answers, in random order. For each one, read the")
    w("trace and write ONE letter in the blank. About 30 minutes.")
    w("")
    w("The machine's own label is deliberately NOT shown. If you see it first")
    w("you will agree with it, and the agreement number this produces is the")
    w("evidence the paper needs that the taxonomy is real rather than an")
    w("artefact of where a threshold was put.")
    w("")
    w("THE FOUR LABELS")
    w("---------------")
    w("  L = LOCKED-IN     One wrong answer, fixed early, never seriously")
    w("                    challenged. The right answer never appears. Reads")
    w("                    as confident and wrong.")
    w("")
    w("  I = INTERLEAVING  The RIGHT answer appears at some round, then the")
    w("                    model moves away from it and commits to a wrong one.")
    w("                    Marked <-- GOLD where it was spotted, but check by")
    w("                    eye: a common word like 'yes' may match by accident.")
    w("")
    w("  C = INCONSISTENT  Several unrelated guesses, none of them held. Often")
    w("                    decays into word salad near the end.")
    w("")
    w("  X = NEITHER       Truncated, an echo of the question, or a near-miss")
    w("                    that is arguably correct (e.g. 'Las Vegas' scored")
    w("                    against gold 'Clark County'). Say which in the note.")
    w("")
    w("Trust your eye over the numbers. If a trace looks locked-in, write L,")
    w("whatever the statistics next to it say. Disagreement between your labels")
    w("and the code is the POINT of this exercise - it is what moves the")
    w("thresholds.")
    w("")
    w("=" * 78)

    key = []
    for n, r in enumerate(sample, 1):
        w("")
        w(f"--- {n:02d} / {len(sample)} " + "-" * 56)
        w(f"  dataset : {r['dataset']}")
        w(f"  Q       : {r['question'][:70]}")
        w(f"  MODEL   : {str(r['answer'])[:70]!r}")
        w(f"  GOLD    : {str(r['gold'])[:70]}")
        w("")
        w(f"  stable_all {r['stable_all']:.2f}   stable_top3 {r['stable_top3']:.2f}   "
          f"cands_top3 {r['cands_top3']:.1f}   ans_len {r['n_content']}")
        w("")
        w("  shadow prediction, change-points only:")
        for line in compressed_trace(r["_traj"], tokenizer):
            gold_r = r["gold_round"]
            mark = ""
            if gold_r >= 0:
                lo = int(line.split("|")[0].strip().lstrip("r").split("-")[0])
                hi_s = line.split("|")[0].strip().split("-")
                hi_r = int(hi_s[-1].lstrip("r")) if len(hi_s) > 1 else lo
                if lo <= gold_r <= hi_r:
                    mark = "   <-- GOLD"
            w(f"      {line}{mark}")
        w("")
        w("  YOUR LABEL  [ ]      (L / I / C / X)")
        w("  note: ______________________________________________")

        key.append(dict(n=n, dataset=r["dataset"], qid=r["qid"],
                        code_label=r["mode"],
                        stable_all=round(r["stable_all"], 3),
                        stable_hi=round(r["stable_hi"], 3),
                        stable_top3=round(r["stable_top3"], 3),
                        cands_top3=round(r["cands_top3"], 1),
                        n_content=r["n_content"],
                        question=r["question"][:80]))

    w("")
    w("=" * 78)
    w("WHEN YOU ARE DONE")
    w("-----------------")
    w("Send back just the letters, in order, e.g.:")
    w("")
    w("   1 L   2 C   3 X   4 I   5 L   ...")
    w("")
    w("I will fit the thresholds to them, report agreement with the code's own")
    w("labels, and recompute the Phase C sample size from a calibrated split")
    w("rather than an assumed one.")
    w("=" * 78)

    WORKSHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORKSHEET_PATH.write_text("\n".join(wlines) + "\n", encoding="utf-8")

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KEY_PATH, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(key[0].keys()))
        wr.writeheader()
        wr.writerows(key)

    say("")
    say("=" * 78)
    say("D. WORKSHEET")
    say("")
    say(f"  {len(sample)} questions to hand-label, blind, in random order.")
    say(f"  All {len(rare)} locked-in and interleaving cases are included, so")
    say("  the rare modes are actually represented; the rest are random.")
    say("")
    say(f"  worksheet : {WORKSHEET_PATH}")
    say(f"  key       : {KEY_PATH}   (do not read before labelling)")
    say("")
    say("  Fetch it with:")
    say("      modal volume get triage-outputs step10c_labelling_worksheet.txt .")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()