#!/usr/bin/env python3
"""
Step 10 - the failure-mode split, and the final sample size  [CORRECTED]
========================================================================

    modal run modal_app.py::run_cpu --script step10_failure_modes.py

CPU only. Reads the 300 cached trajectories. Minutes, fractions of a cent.

What this step is for
---------------------
Phase C's size is not set by the error rate. It is set by the RAREST failure
mode, because the paper's central table compares detector AUROC in three
separate columns and a column with 30 items in it cannot support a claim.

Corrected base rates from Step 9b:

    triviaqa       66% wrong
    hotpotqa       83% wrong
    commonsenseqa  11% wrong


WHY THIS FILE WAS REWRITTEN
===========================
The first version returned **locked_in = 0** for both TriviaQA and HotpotQA -
zero instances of the failure mode this project exists to study, out of 149
wrong answers. That was not a finding. Three separate bugs produced it.

**Bug 1 - rule ordering starved locked-in.**
`gold_seen_then_lost` was tested first, and it fired whenever ANY gold token
appeared at ANY position in ANY round. A question could be textbook locked-in
and still be stamped "interleaving" because one subword flickered once. The
result was 58/160 interleaving (36%), which is implausible on its face.

**Bug 2 - the gold token set was far too permissive.**
It unioned the subword ids of up to eight alias variants. For gold "Lincoln
Steffens" that set ends up holding fragments like "Lin", "col", "n", "Ste",
which occur constantly in unrelated words. Almost every trajectory "saw" a gold
token by accident. Step 9b already fixed exactly this class of bug for answer
scoring - word-boundary matching on decoded text - and that fix was never
carried into the mode classifier.

**Bug 3 - the early-settle test was structurally impossible to pass.**
`frac_early >= 0.5` asked whether answer positions settle in the first 30% of
rounds. But `get_num_transfer_tokens` fixes the reveal schedule in advance, and
low-confidence remasking reveals the EASY positions first: EOS, padding,
punctuation. The uncertain positions - which is to say the answer - are
structurally revealed LAST. Most observed examples sat at settle round ~30 of
32 for that reason alone. This is the same trap that produced a constant
median settle round of 15 in Step 6b, and the same correction applies:

    **Measure a position's settling relative to its own reveal round, never
    against the absolute round number.**

A position revealed at round r has r+1 rounds of pre-reveal life during which
the model was already predicting something there. The schedule-free question
is: what fraction of that life did it spend holding the value it finally
committed? That is `stable_frac`, and it is comparable across positions
revealed at round 3 and at round 31.


THE CORRECTED DEFINITIONS
=========================
LOCKED-IN     the wrong answer was fixed almost immediately and never
              seriously challenged: high `stable_frac`, few candidates, and
              the gold answer never appeared at any round.

INTERLEAVING  the model HELD the right answer at some round and moved away
              from it. Operationalised on the model's full shadow prediction -
              decode the argmax at every content position, at every round, and
              word-boundary match the gold. If the gold appears at some round
              and not in the final answer, the model had it and lost it.

INCONSISTENT  cycled through several unrelated candidates and never held any
              of them: high candidate count, low `stable_frac`, gold never
              seen.

DEGENERATE    used the whole 64-token budget without emitting EOS. A decoding
              artefact, not a hallucination. Its own bucket, excluded from the
              three-way table, reported as a rate.


HOW THE THRESHOLDS ARE SET
==========================
They are not asserted. The script prints the observed distribution of
`mean_stable_frac` and `mean_candidates` FIRST, then classifies, then prints a
sensitivity grid showing how the counts move as the cuts move. If the split
only exists at one particular threshold it is an artefact, and the grid makes
that visible instead of hiding it.

Still rough
-----------
Correctness comes from Step 9b's word-boundary matching, not the Qwen3 judge.
Step 17 builds the real classifier; Step 18 verifies 100 by hand with a second
annotator. This step sizes a sample, and a sample size needs to be right to
within a factor of two, not exact.
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
RESCORE_CSV = config.TAB_DIR / "step9b_rescored.csv"
REPORT_PATH = config.OUT_DIR / "step10_failure_modes_report.txt"
CSV_PATH = config.TAB_DIR / "step10_modes.csv"

DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
N_PER_DATASET = 100

# --- thresholds -------------------------------------------------------------
# Starting values, read off the definitions. The sensitivity grid at the end of
# the report shows how much the split depends on them; if it depends a lot,
# these are wrong and Step 17 must derive them from hand-labelled data.
LOCKED_STABLE_FRAC = 0.70     # held its final value for 70%+ of its pre-reveal life
LOCKED_MAX_CANDIDATES = 2.0   # essentially one guess, allowing for one revision
INCONSISTENT_MIN_CANDIDATES = 3.0
INCONSISTENT_MAX_STABLE = 0.50
TRUNCATION_SLACK = 2          # positions left unused before we call it terminated
TARGET_PER_CELL = 150         # wrong answers wanted per mode per dataset
N_EXAMPLES = 5                # examples printed per mode

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Text matching - identical rules to Step 9b, so "gold seen" at round r means
# exactly what "correct" means at the end. Using a different rule here would
# make interleaving incomparable with the correctness label.
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def match_wordbound(text: str, golds: list) -> bool:
    """Gold must appear as whole words. Step 9b's corrected rule.

    `\\bd\\b` matches "is d" but not "agreement"; "dolores haze" no longer
    matches "dolores humbert".
    """
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


MIN_GOLD_CHARS = 4
STOP_GOLDS = {"yes", "no", "true", "false", "none", "both", "all"}


def testable_golds(golds: list) -> list:
    """Golds usable for the INTERLEAVING test. Not for scoring.

    Interleaving asks whether the model ever held the right answer and let it
    go. That is a question about the shadow prediction at some round, and a
    gold string short or common enough to appear by accident answers it wrongly
    every time.

    Two failures observed on real data, both of which put questions in the
    interleaving bucket that do not belong there:

    **Bare option letters.** CommonsenseQA gold lists carry the letter, so
    `moon | E` makes `\\be\\b` a gold match, and the shadow `E. desktop` scores
    as "the model had the answer" at round 1. It did not - it had the letter E
    next to the wrong word. The real interleaving in that trajectory happens at
    round 18 (`D. moon`), and the marker pointed at the wrong round. This is
    the same single-letter bug Step 9b fixed for scoring; it was never carried
    into the mode classifier.

    **yes / no.** HotpotQA comparison questions have gold `no`, two characters,
    and "no" occurs constantly in ordinary English. Two of the first five
    interleaving examples were yes/no questions matching at round 0.

    A question whose every gold alias is filtered out here is **untestable for
    interleaving** - not "not interleaving". The distinction matters: silently
    treating untestable as negative would understate interleaving, and the
    report counts them separately so the size of the blind spot is visible.

    A yes/no question genuinely cannot be tested this way. "Did the model hold
    the right answer" is not answerable by looking for the word "no" in text
    that is mostly prose. Step 17 needs a position-aware test for those.
    """
    out = []
    for g in golds:
        gn = normalise(g)
        if len(gn) < MIN_GOLD_CHARS or gn in STOP_GOLDS:
            continue
        out.append(g)
    return out


# ===========================================================================
# Per-trajectory measurement
# ===========================================================================

def assert_config_matches(traj, path) -> None:
    """Refuse to analyse a trajectory generated under different settings.

    `RUN_TAG` in the cache path is meant to make this impossible, but a path is
    only as good as every script that builds one. This has now gone wrong twice:
    once when a config change was ignored because untagged files already
    existed, and once when a script was replaced by a copy that predated the
    tagging fix and silently reverted it.

    A path is a convention; this is a check. The trajectory itself records the
    settings it was generated with, so comparing them against config.py costs
    nothing and turns a silent wrong answer into a stopped run. Silent is the
    dangerous failure here: the report prints whatever config.py says in its
    banner while analysing something else entirely, and the only visible tell
    is a small "of 32" buried in one line of section A.
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
                "  These trajectories were generated under different decoder",
                "  settings, so every number computed from them would describe",
                "  a configuration this project no longer uses.",
                "",
                "  Fix: re-run step9_base_rate.py to generate trajectories under",
                "  the current settings, then re-run this script. If you meant to",
                "  analyse the older run, set config.py back to those values.",
                "=" * 78,
            ])
        )


def content_positions(traj) -> list:
    """Positions holding real answer content in the final output.

    Charter Warning #2: EOS and padding settle early for reasons unrelated to
    the model's confidence in the answer, so any mode assigned over them
    measures the padding schedule rather than the model.
    """
    final = traj.final_ids.tolist()
    return [i for i, t in enumerate(final)
            if t not in (traj.eos_id, traj.mask_id)]


def is_truncated(traj) -> bool:
    """Generation used the whole budget without ever terminating.

    Step 9b's echo test flagged any answer opening with the question's words,
    which caught well-formed replies too. Truncation is the real signature.
    """
    final = traj.final_ids.tolist()
    n_content = sum(1 for t in final if t not in (traj.eos_id, traj.mask_id))
    return n_content >= traj.gen_length - TRUNCATION_SLACK


def stability(traj, positions: list) -> tuple:
    """Schedule-free settling measure. THE CENTRAL CORRECTION IN THIS FILE.

    For each content position, walk backwards from its own reveal round and
    count how many consecutive rounds it held the value it finally committed.
    Divide by the length of its pre-reveal life.

        stable_frac = 1.0   the position showed its final token from round 0
                            and never wavered  -> locked in
        stable_frac = 0.1   the model changed its mind right up to the moment
                            the schedule forced a commitment -> hesitant

    Because the denominator is the position's OWN reveal round, a position
    revealed at round 3 and one revealed at round 31 are directly comparable.
    The absolute-round version of this test could not distinguish them, and
    answer tokens are revealed late by construction, which is why the previous
    version of this script found no locked-in cases at all.

    Returns (stable_fracs, n_candidates, reveal_rounds), one entry per position.
    """
    rev = traj.revealed_at()
    stable_fracs, n_candidates, reveal_rounds = [], [], []

    for i in positions:
        r = int(rev[i])
        history = traj.pred_ids[: r + 1, i].tolist()   # every guess up to commit
        committed = history[-1]

        # Walk back from the commit while the prediction is unchanged.
        held = 0
        for v in reversed(history):
            if v == committed:
                held += 1
            else:
                break

        stable_fracs.append(held / len(history))
        n_candidates.append(len(set(history)))
        reveal_rounds.append(r)

    return stable_fracs, n_candidates, reveal_rounds


def shadow_guesses(traj, tokenizer) -> list:
    """The model's full current guess, decoded, at every round.

    At round r, `pred_ids[r]` is the raw argmax at EVERY position - what the
    model would output if it committed everything right now. That is the shadow
    prediction, and it is only visible because `logging_patch` records the
    argmax BEFORE the reveal overwrite. Every published trajectory method logs
    after that line and cannot see this.

    This is what makes a faithful interleaving test possible: "did the model
    ever hold the right answer" is a question about the shadow prediction, not
    about committed tokens. Committed tokens cannot change - the confidence
    freeze makes a revealed position unselectable forever - so "had it and lost
    it" is meaningless at the committed level and meaningful here.
    """
    rows = []
    for r in range(traj.pred_ids.shape[0]):
        ids = [int(t) for t in traj.pred_ids[r].tolist()
               if t not in (traj.eos_id, traj.mask_id)]
        rows.append(ids)
    if not rows:
        return []
    return tokenizer.batch_decode(rows, skip_special_tokens=True)


def measure(traj, tokenizer, golds: list) -> dict:
    """All per-question quantities. No thresholds applied here - measurement
    and classification are kept apart so the thresholds can be varied without
    recomputing anything."""
    positions = content_positions(traj)
    if not positions:
        return {"empty": True, "truncated": False}

    stable_fracs, n_candidates, reveal_rounds = stability(traj, positions)

    # Did the right answer ever appear in the shadow prediction? Only aliases
    # long and distinctive enough to mean something are used - see
    # `testable_golds`.
    usable = testable_golds(golds)
    gold_round = -1
    if usable:
        for r, text in enumerate(shadow_guesses(traj, tokenizer)):
            if match_wordbound(text, usable):
                gold_round = r
                break

    # Where EOS positions get revealed, for the schedule diagnostic.
    rev = traj.revealed_at()
    eos_positions = [i for i in range(traj.gen_length) if i not in set(positions)]
    eos_reveal = [int(rev[i]) for i in eos_positions]

    return {
        "empty": False,
        "truncated": is_truncated(traj),
        "n_content": len(positions),
        "mean_stable_frac": float(np.mean(stable_fracs)),
        "min_stable_frac": float(np.min(stable_fracs)),
        "mean_candidates": float(np.mean(n_candidates)),
        "max_candidates": int(np.max(n_candidates)),
        "mean_reveal": float(np.mean(reveal_rounds)),
        "mean_eos_reveal": float(np.mean(eos_reveal)) if eos_reveal else float("nan"),
        "gold_round": gold_round,
        "gold_ever": gold_round >= 0,
        "gold_testable": bool(usable),
    }


def classify(m: dict,
             locked_stable: float = LOCKED_STABLE_FRAC,
             locked_cands: float = LOCKED_MAX_CANDIDATES,
             incons_cands: float = INCONSISTENT_MIN_CANDIDATES,
             incons_stable: float = INCONSISTENT_MAX_STABLE) -> str:
    """Assign one wrong answer to a failure mode.

    Order is deliberate and is the fix for Bug 1.

    Degenerate first: a truncated answer has no dynamics worth classifying, so
    it must leave before anything else is measured on it.

    Locked-in second, and crucially it now requires `not gold_ever`. In the
    previous version interleaving was tested first and swallowed everything;
    the honest statement of the distinction is that a model which never once
    produced the right answer cannot have interleaved, whatever else it did.

    Interleaving third, on the strict whole-word shadow-prediction test rather
    than the old subword union.

    Inconsistent last, as the residual "many guesses, held none" case.
    """
    if m.get("empty"):
        return "degenerate"
    if m["truncated"]:
        return "degenerate"

    # Both locked-in and interleaving are claims about whether the model ever
    # held the right answer. When no gold alias is distinctive enough to test
    # that (yes/no questions, bare option letters - see `testable_golds`),
    # neither claim can be made and the question goes in its own bucket.
    #
    # Forcing these into "not interleaving" would inflate locked-in with every
    # yes/no question the model got wrong, which is the failure mode this
    # project cares most about getting right. An explicit bucket keeps the
    # three-way table honest and makes the size of the blind spot reportable.
    if not m.get("gold_testable", True):
        return "untestable"

    if (not m["gold_ever"]
            and m["mean_stable_frac"] >= locked_stable
            and m["mean_candidates"] <= locked_cands):
        return "locked_in"

    if m["gold_ever"]:
        return "interleaving"

    if (m["mean_candidates"] >= incons_cands
            and m["mean_stable_frac"] <= incons_stable):
        return "inconsistent"

    return "unclassified"


# ===========================================================================
# Reporting helpers
# ===========================================================================

def histogram(values: list, lo: float, hi: float, bins: int = 10,
              width: int = 40, label: str = "") -> None:
    """Print a text histogram. The thresholds in this file are only defensible
    if the underlying distribution is visible next to them."""
    if not values:
        return
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    peak = max(counts) if len(counts) else 1
    say(f"  {label}  (n={len(values)}, median {np.median(values):.2f})")
    for b in range(bins):
        bar = "#" * int(round(width * counts[b] / peak)) if peak else ""
        say(f"    {edges[b]:5.2f}-{edges[b+1]:5.2f} {counts[b]:4d} {bar}")


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 10 (CORRECTED): failure-mode split and sample size")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not RESCORE_CSV.exists():
        say(f"Need {RESCORE_CSV}. Run step9b_rescore.py first.")
        sys.exit(1)

    labels = {}
    with open(RESCORE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            labels[(r["dataset"], r["qid"])] = (
                r["new_correct"].lower() == "true", r.get("answer", ""))
    say(f"{len(labels)} corrected labels from Step 9b")

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

    # ---- measure everything once ------------------------------------------
    say("Measuring trajectories (decoding 32 shadow predictions each)...")
    rows = []
    for path in sorted(TRAJ_SUBDIR.glob("*.npz")):
        stem = path.stem
        dataset = stem.split("_", 1)[0]
        qid = stem.split("_", 1)[1] if "_" in stem else stem
        rec = records.get((dataset, qid))
        label = labels.get((dataset, qid))
        if rec is None or label is None:
            continue
        correct, answer = label

        traj = logging_patch.Trajectory.load(path)
        assert_config_matches(traj, path)
        m = measure(traj, tokenizer, rec.gold_answers)
        mode = "CORRECT" if correct else classify(m)

        rows.append(dict(dataset=dataset, qid=qid, correct=correct, mode=mode,
                         question=rec.question[:90], answer=answer[:90],
                         gold=" | ".join(rec.gold_answers[:3])[:60],
                         _traj=traj, _rec=rec, **m))

    say(f"{len(rows)} trajectories measured")
    wrong_all = [r for r in rows if not r["correct"] and not r.get("empty")]

    # =====================================================================
    # DISTRIBUTIONS FIRST. Thresholds come after, so they can be judged
    # against the data rather than asserted ahead of it.
    # =====================================================================
    say("")
    say("=" * 78)
    say("A. THE REVEAL SCHEDULE - why absolute settle rounds are unusable")
    say("")
    content_rev = [r["mean_reveal"] for r in wrong_all]
    eos_rev = [r["mean_eos_reveal"] for r in wrong_all
               if not np.isnan(r["mean_eos_reveal"])]
    if content_rev:
        say(f"  answer positions  revealed at round  {np.mean(content_rev):5.1f} "
            f"(median {np.median(content_rev):.1f}) of {rows[0]['_traj'].steps}")
    if eos_rev:
        say(f"  EOS/pad positions revealed at round  {np.mean(eos_rev):5.1f} "
            f"(median {np.median(eos_rev):.1f})")
    say("")
    # Which way round the schedule runs is a property of block_length, and it
    # FLIPPED when the decoder was fixed in Step 11. Describing it from the
    # measurement rather than from memory is the point - an earlier version of
    # this text asserted the old ordering and was left contradicting the two
    # numbers printed directly above it.
    if content_rev and eos_rev and np.mean(content_rev) < np.mean(eos_rev):
        say("  The ANSWER is written first and the padding fills in afterwards.")
        say("  That is semi-autoregressive decoding working as intended: with")
        say("  block_length < gen_length the first block is decoded before the")
        say("  second, and the answer lives in the first block.")
        say("")
        say("  Under the old single-block configuration this was the other way")
        say("  round - EOS at round 13.6, the answer at 26.7 of 32 - because")
        say("  low-confidence remasking reveals the easy positions first and")
        say("  padding is trivially easy. That is what left the answer with two")
        say("  rounds of denoising and produced 'Londonasgow'.")
    else:
        say("  EOS and padding are being revealed BEFORE the answer. Under")
        say("  block_length < gen_length that should not happen, and it means")
        say("  the decoder configuration is not what config.py reports. Check")
        say("  BLOCK_LENGTH and the trajectory cache tag before reading on.")
    say("")
    say("  Either way the lesson is unchanged: an absolute round number is a")
    say("  fact about the schedule, not about the model. Everything below uses")
    say("  stability relative to each position's OWN reveal round.")

    say("")
    say("=" * 78)
    say("B. OBSERVED DISTRIBUTIONS (wrong answers only)")
    say("")
    histogram([r["mean_stable_frac"] for r in wrong_all], 0.0, 1.0, 10,
              label="mean_stable_frac - fraction of pre-reveal life held at final value")
    say("")
    cands = [r["mean_candidates"] for r in wrong_all]
    histogram(cands, 1.0, max(6.0, float(np.max(cands)) if cands else 6.0), 10,
              label="mean_candidates - distinct guesses per answer position")
    say("")
    say(f"  Thresholds applied below:")
    say(f"    locked-in    : stable_frac >= {LOCKED_STABLE_FRAC} "
        f"AND candidates <= {LOCKED_MAX_CANDIDATES} AND gold never seen")
    say(f"    interleaving : gold answer appears in the shadow prediction "
        f"at some round")
    say(f"    inconsistent : candidates >= {INCONSISTENT_MIN_CANDIDATES} "
        f"AND stable_frac <= {INCONSISTENT_MAX_STABLE}")
    say("")
    say("  Read these against the histograms. If a threshold sits in the middle")
    say("  of a dense region, small changes will move many questions and the")
    say("  split is fragile - the sensitivity grid in section F tests exactly")
    say("  that.")

    # =====================================================================
    say("")
    say("=" * 78)
    say("C. FAILURE-MODE SPLIT (wrong answers only)")
    say("")
    say("  dataset         wrong  locked-in  interleav  inconsist  unclass  degen  untest")
    say("  --------------  -----  ---------  ---------  ---------  -------  -----  ------")

    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    split = {}
    for dataset in DATASETS:
        wrong = [r for r in by_ds[dataset] if not r["correct"]]
        c = Counter(r["mode"] for r in wrong)
        split[dataset] = (len(wrong), c)
        say(f"  {dataset:<14}  {len(wrong):5d}  {c['locked_in']:9d}  "
            f"{c['interleaving']:9d}  {c['inconsistent']:9d}  "
            f"{c['unclassified']:7d}  {c['degenerate']:5d}  {c['untestable']:6d}")

    total_wrong = sum(v[0] for v in split.values())
    total = Counter()
    for _n, c in split.values():
        total.update(c)
    say(f"  {'ALL':<14}  {total_wrong:5d}  {total['locked_in']:9d}  "
        f"{total['interleaving']:9d}  {total['inconsistent']:9d}  "
        f"{total['unclassified']:7d}  {total['degenerate']:5d}  "
        f"{total['untestable']:6d}")

    say("")
    say("  untest = no gold alias distinctive enough to test whether the model")
    say("           ever held the right answer (yes/no questions, bare option")
    say("           letters). NOT a failure mode - a measurement blind spot.")

    # ---- where the unclassified actually sit ------------------------------
    unc = [r for r in wrong_all if r["mode"] == "unclassified"]
    if unc:
        say("")
        say(f"  The {len(unc)} UNCLASSIFIED sit at stable "
            f"{np.median([r['mean_stable_frac'] for r in unc]):.2f} / candidates "
            f"{np.median([r['mean_candidates'] for r in unc]):.1f} (medians),")
        say(f"  against {np.median([r['mean_stable_frac'] for r in wrong_all]):.2f} / "
            f"{np.median([r['mean_candidates'] for r in wrong_all]):.1f} for all wrong answers.")
        say("")
        say("  If those two lines are close, the unclassified bucket is not a")
        say("  fourth kind of failure - it is the CENTRE of the distribution,")
        say("  and the two rules have been placed so that they carve off the")
        say("  tails and leave the bulk unlabelled. Widening a threshold to")
        say("  absorb them would be guessing a third time. Hand-label first")
        say("  (step10c writes the worksheet) and fit the cuts to the labels.")

    say("")
    if total["locked_in"] == 0:
        say("  LOCKED-IN IS STILL ZERO. Do not proceed. Either the stability")
        say("  measure is wrong, or LLaDA genuinely never commits early to a")
        say("  wrong answer - and the second would mean the paper's premise")
        say("  needs restating before any more compute is spent. Check the")
        say("  stable_frac histogram above: if its mass sits below 0.7, lower")
        say("  the threshold and look at the examples by eye.")
    else:
        pct = total["locked_in"] / total_wrong if total_wrong else 0
        say(f"  Locked-in is {pct:.1%} of wrong answers. This is the number the")
        say("  whole Phase C budget is sized from.")

    # =====================================================================
    say("")
    say("=" * 78)
    say("D. SAMPLE SIZE")
    say("")
    say(f"  Target: {TARGET_PER_CELL} wrong answers per mode per dataset,")
    say("  driven by the RAREST mode - the others get more than they need.")
    say("")
    say("  dataset         err%   rarest mode      share   questions needed")
    say("  --------------  -----  ---------------  ------  ----------------")

    for dataset in DATASETS:
        n_wrong, c = split[dataset]
        n_total = len(by_ds[dataset])
        if n_wrong == 0 or n_total == 0:
            continue
        err = n_wrong / n_total
        usable = {m: c[m] for m in ("locked_in", "interleaving", "inconsistent")}
        if not any(usable.values()):
            say(f"  {dataset:<14}  {err:4.0%}   (no modes classified)")
            continue
        rarest = min(usable, key=lambda m: usable[m])
        share = usable[rarest] / n_wrong
        need = int(TARGET_PER_CELL / (err * share)) if share > 0 else -1
        say(f"  {dataset:<14}  {err:4.0%}   {rarest:<15}  {share:5.1%}  "
            f"{need if need > 0 else 'unbounded':>16}")

    say("")
    say("  'unbounded' means a mode never appeared in 100 questions. It is not")
    say("  absent - it is rarer than 1-in-(wrong answers seen). Raise the pilot")
    say("  for that dataset before committing, or the sample size is an")
    say("  extrapolation from zero observations.")

    # =====================================================================
    say("")
    say("=" * 78)
    say("E. DEGENERATE ANSWERS (truncated, never emitted EOS)")
    say("")
    for dataset in DATASETS:
        wrong = [r for r in by_ds[dataset] if not r["correct"]]
        d = [r for r in wrong if r["mode"] == "degenerate"]
        if wrong:
            say(f"  {dataset:<14} {len(d):3d} of {len(wrong):3d} wrong "
                f"({len(d)/len(wrong):5.1%})")
    say("")
    csqa_wrong = [r for r in by_ds["commonsenseqa"] if not r["correct"]]
    csqa_degen = [r for r in csqa_wrong if r["mode"] == "degenerate"]
    if csqa_wrong and len(csqa_degen) / len(csqa_wrong) > 0.30:
        say("  CommonsenseQA's degenerate rate is high. That is a PROMPT problem,")
        say("  not a model problem: asked a multiple-choice question, LLaDA writes")
        say("  a verbose justification that fills all 64 positions and never")
        say("  reaches an EOS. Combined with Step 9b's finding that CSQA's error")
        say("  rate is only 11%, the negative control is in trouble from two")
        say("  directions at once.")
        say("")
        say("  Decide before Phase C, cheapest first:")
        say("   1. Raise gen_length for CSQA only, to 128. Costs 2x compute on")
        say("      one dataset and changes nothing else.")
        say("   2. Constrain the prompt - 'Answer with the letter only.' Cheap,")
        say("      but it changes the task and TraceDet's comparability with it.")
        say("   3. Ask CSQA open-ended, dropping the options. Step 9b already")
        say("      recommended this for the error-rate problem; it likely fixes")
        say("      the truncation problem at the same time.")
        say("")
        say("  Option 3 addresses both, so test it first: one 100-question run.")

    # =====================================================================
    say("")
    say("=" * 78)
    say("F. THRESHOLD SENSITIVITY - is the split real or an artefact?")
    say("")
    say("  locked-in count as the two locked-in thresholds move:")
    say("")
    say("            candidates<=1.5  <=2.0  <=2.5  <=3.0")
    for sf in (0.50, 0.60, 0.70, 0.80, 0.90):
        cells = []
        for mc in (1.5, 2.0, 2.5, 3.0):
            n = sum(1 for r in wrong_all
                    if classify(r, locked_stable=sf, locked_cands=mc) == "locked_in")
            cells.append(f"{n:6d}")
        say(f"    stable>={sf:.2f}  " + " ".join(cells))
    say("")
    say("  A healthy table changes smoothly. If locked-in jumps from 0 to 80")
    say("  across one cell boundary, the measure is separating noise, not modes.")

    # =====================================================================
    say("")
    say("=" * 78)
    say("G. EXAMPLES PER MODE - CHECK THESE BY EYE")
    say("")
    say("  This is the part that matters. The counts above are only as good as")
    say("  these examples look. Read them before authorising Phase C.")

    for mode in ("locked_in", "interleaving", "inconsistent", "unclassified",
                 "degenerate", "untestable"):
        ex = [r for r in rows if r["mode"] == mode][:N_EXAMPLES]
        if not ex:
            say("")
            say(f"  --- {mode.upper()} --- none")
            continue
        say("")
        say(f"  --- {mode.upper()} ({sum(1 for r in rows if r['mode']==mode)} total) ---")
        for r in ex:
            say("")
            say(f"    {r['dataset']}  stable {r.get('mean_stable_frac', 0):.2f}  "
                f"candidates {r.get('mean_candidates', 0):.1f}  "
                f"gold_first_seen_round {r.get('gold_round', -1)}")
            say(f"      Q: {r['question'][:68]}")
            say(f"      A: {r['answer'][:68]!r}")
            say(f"      gold: {r['gold'][:60]}")

    # ---- one full trace per mode, so the rule can be read against reality --
    say("")
    say("=" * 78)
    say("H. ROUND-BY-ROUND TRACE - one example per mode")
    say("")
    say("  The model's full shadow prediction at each round: the argmax at every")
    say("  position, before the reveal overwrite. No published method can see")
    say("  this. Read down the column and the failure mode should be obvious by")
    say("  eye - if it is not, the rule that produced the label is wrong.")

    for mode in ("locked_in", "interleaving", "inconsistent"):
        ex = [r for r in rows if r["mode"] == mode]
        if not ex:
            continue
        r = ex[0]
        say("")
        say(f"  --- {mode.upper()} ---")
        say(f"    Q: {r['question'][:70]}")
        say(f"    gold: {r['gold'][:60]}")
        say("")
        guesses = shadow_guesses(r["_traj"], tokenizer)
        for rd, text in enumerate(guesses):
            flat = " ".join(str(text).split())[:64]
            mark = " <-- gold" if rd == r.get("gold_round", -1) else ""
            say(f"      r{rd:02d} | {flat}{mark}")

    # =====================================================================
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in clean_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(clean_rows)
    say("")
    say(f"  CSV: {CSV_PATH}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()