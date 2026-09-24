#!/usr/bin/env python3
"""
Step 12b - CommonsenseQA at scale, completing plan step 12
===========================================================

    modal run --detach modal_app.py::run --script step12b_csqa_generate.py

A100. About 65 minutes and $2.30 for the whole validation split.

THE CORRECTED DECISION
======================
Step 12a swept four prompt formats and its own criterion chose `open`. **That
choice was wrong, and the criterion was the reason.**

    format      err%   degen% of wrong   usable-wrong%
    current      12%              72%              3%
    brief        14%               0%             14%
    text_only    12%               0%             12%
    open         85%               2%             82%

`open` wins on usable-wrong by a mile, and it wins for the wrong reason. Its
error rate is 85% where every other format is 12-14%. That is not the model
being worse; it is the question becoming unanswerable once the options are
withheld:

    gold 'bank'               -> 'A revolving door serves as a security
                                  measure at an entrance'
    gold 'listen to each other'-> 'Animals often run away, seek shelter, or
                                  defend themselves.'

Those are not hallucinations. They are reasonable answers to an underspecified
question, marked wrong because an option list said so. CommonsenseQA is
multiple choice precisely because its questions are ambiguous without the
options.

Step 12a's criterion asked for a high yield of wrong answers and never asked
that the wrong answers be genuinely wrong, so it optimised the yield of
prompt artefacts. The missing clause is added here:

    **SAME-TASK GUARD.** A format's error rate must be within 2x of the median
    error rate across formats. A prompt that makes the model six times more
    "wrong" has changed the question, not revealed the model.

Applied below in code, so the rejection and its reason live in the repository
rather than in a conversation.

WHAT 12a DID ESTABLISH
======================
The hypothesis was right. `data.build_prompt_text` gives every dataset except
CommonsenseQA a brevity instruction, because the brevity cue sits in the
no-choices branch. Adding it moves degenerate answers from **72% to 0%**. One
line of prompt was the whole defect.

SIZE
====
`data.load_records` reads the **validation** split; CommonsenseQA's has 1,221
questions. At ~14% usable-wrong that is roughly 170 usable wrong answers, not
the 300 step 12a projected. Part D measures what was actually obtained and
computes what precision it buys plan step 30, rather than asserting it is
enough.

A CAVEAT THAT MUST REACH THE PAPER
==================================
The failure-mode rules were fitted on TriviaQA and HotpotQA answers averaging
15-19 content tokens. CommonsenseQA answers under this format average ~4.6.
`stable_top3` and `cands_top3` are averages over the three highest-entropy
positions, and a four-token answer has barely more than three. The mode split
in Part C is reported for completeness and should NOT be treated as calibrated
for this dataset.
"""

import csv
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

SWEEP_CSV = config.TAB_DIR / "step12a_csqa_sweep.csv"
PHASEC_ROOT = config.TRAJ_DIR / "phasec" / config.RUN_TAG
MODES_CSV = config.TAB_DIR / "step12c_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step12b_csqa_report.txt"
CSV_PATH = config.TAB_DIR / "step12b_csqa_modes.csv"

MAX_DEGENERATE_SHARE = 0.25
SAME_TASK_FACTOR = 2.0            # the clause step 12a was missing

# The rules adopted in step 17c. Restated, not re-fitted.
CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6)
ORDER = "A"

SEC_PER_Q, USD_PER_HOUR = 3.09, 2.10
PROGRESS_EVERY = 100

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Text handling - identical to step 17c
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
    first = SPECIAL_RE.search(str(text))
    return (str(text)[: first.start()] if first else str(text)).strip()


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def match_wordbound(text: str, golds: list) -> bool:
    a = normalise(text)
    if not a:
        return False
    for g in golds:
        gn = normalise(g)
        if gn and re.search(r"\b" + re.escape(gn) + r"\b", a):
            return True
    return False


def text_golds(golds: list) -> list:
    """The option TEXT, never the bare letter - the single-letter trap."""
    return [g for g in golds if len(normalise(g)) > 1]


def testable_golds(golds: list) -> list:
    return [g for g in text_golds(golds)
            if len(normalise(g)) >= MIN_GOLD_CHARS
            and normalise(g) not in STOP_GOLDS]


def question_overlap(answer: str, question: str) -> float:
    a = [w for w in normalise(answer).split() if w not in STOP]
    q = set(normalise(question).split())
    return 1.0 if not a else sum(1 for w in a if w in q) / len(a)


def max_copied_run(answer: str, question: str) -> int:
    a, q = normalise(answer).split(), normalise(question).split()
    if not a or not q:
        return 0
    prev, best = [0] * (len(q) + 1), 0
    for i in range(1, len(a) + 1):
        cur, ai = [0] * (len(q) + 1), a[i - 1]
        for j in range(1, len(q) + 1):
            if ai == q[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def top_by_entropy(ents, k):
    k = max(1, min(int(k), len(ents)))
    return np.argsort(-ents, kind="stable")[:k]


def measure(traj) -> dict:
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
    return {"n_content": len(pos),
            "stable_top3": float(np.mean(fr[sel])),
            "cands_top3": float(np.mean(cn[sel])),
            "truncated": len(pos) >= traj.gen_length - 2}


def shadow_guesses(traj, tokenizer) -> list:
    rows = [[int(t) for t in traj.pred_ids[r].tolist()
             if t not in (traj.eos_id, traj.mask_id)]
            for r in range(traj.pred_ids.shape[0])]
    return tokenizer.batch_decode(rows, skip_special_tokens=True) if rows else []


def classify(r) -> str:
    if r["truncated"]:
        return "degenerate"
    echo = (r["q_overlap"] >= CUTS["e_cut"] and r["copy_run"] >= CUTS["r_cut"])
    if ORDER == "A":
        if echo:
            return "echo"
        if not r["gold_testable"]:
            return "untestable"
        if r["gold_ever"]:
            return "interleaving"
    if r["stable_top3"] >= CUTS["s_cut"] and r["cands_top3"] <= CUTS["c_cut"]:
        return "locked_in"
    return "inconsistent"


def hm_se(A, n_pos, n_neg):
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    return math.sqrt((A * (1 - A) + (n_pos - 1) * (q1 - A * A)
                      + (n_neg - 1) * (q2 - A * A)) / (n_pos * n_neg))


# ===========================================================================

def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 12b: CommonsenseQA at scale (completes plan step 12)")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not SWEEP_CSV.exists():
        say(f"\nMissing {SWEEP_CSV}. Run step12a first.")
        sys.exit(1)

    # =======================================================================
    say("")
    say("=" * 78)
    say("A. THE CORRECTED DECISION")
    say("")
    sweep = defaultdict(list)
    with open(SWEEP_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sweep[r["fmt"]].append(r)

    stats = {}
    for fmt, rs in sweep.items():
        wrong = [r for r in rs if str(r["correct"]).lower() != "true"]
        degen = [r for r in wrong if str(r["degenerate"]).lower() == "true"]
        usable = [r for r in wrong
                  if str(r["degenerate"]).lower() != "true"
                  and str(r["echo"]).lower() != "true"]
        stats[fmt] = dict(n=len(rs), err=len(wrong) / len(rs),
                          d_share=len(degen) / len(wrong) if wrong else 0.0,
                          u_rate=len(usable) / len(rs))

    med_err = float(np.median([s["err"] for s in stats.values()]))
    say(f"  median error rate across formats: {med_err:.0%}")
    say(f"  same-task band: {med_err/SAME_TASK_FACTOR:.0%} to "
        f"{med_err*SAME_TASK_FACTOR:.0%}")
    say("")
    say("  format      err%   degen%   usable%   verdict")
    say("  ---------  -----  -------  --------  ------------------------------")
    eligible = {}
    for fmt in sorted(stats, key=lambda f: -stats[f]["u_rate"]):
        s = stats[fmt]
        if s["err"] > med_err * SAME_TASK_FACTOR:
            v = "REJECT - error rate off-scale"
        elif s["err"] < med_err / SAME_TASK_FACTOR:
            v = "REJECT - error rate collapsed"
        elif s["d_share"] > MAX_DEGENERATE_SHARE:
            v = f"REJECT - degenerate > {MAX_DEGENERATE_SHARE:.0%}"
        else:
            v = "eligible"
            eligible[fmt] = s
        say(f"  {fmt:<9}  {s['err']:5.0%}  {s['d_share']:7.0%}  "
            f"{s['u_rate']:8.0%}  {v}")

    if not eligible:
        say("")
        say("  NO FORMAT is eligible. CommonsenseQA cannot join the dataset and")
        say("  plan step 30 needs a different negative control. Stopping.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    FMT = max(eligible, key=lambda f: eligible[f]["u_rate"])

    # The chosen format must have a template in step12a, or its prompt cannot
    # be rebuilt. Checked HERE, before three minutes of model loading, because
    # a KeyError after the model is resident wastes GPU time to say the same
    # thing.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "sweep", Path(__file__).resolve().parent / "step12a_csqa_prompt_sweep.py")
    sweep_mod = importlib.util.module_from_spec(_spec)
    sys.modules["sweep"] = sweep_mod
    _spec.loader.exec_module(sweep_mod)
    if FMT not in sweep_mod.FORMATS:
        say("")
        say(f"  STOPPED - the sweep CSV names format {FMT!r}, which")
        say("  step12a_csqa_prompt_sweep.py does not define. The CSV and the")
        say("  script have diverged; the prompt cannot be rebuilt.")
        say(f"  defined: {sorted(sweep_mod.FORMATS)}")
        say(f"  in CSV : {sorted(stats)}")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    say("")
    say(f"  ADOPT '{FMT}'   (usable-wrong {eligible[FMT]['u_rate']:.0%}, "
        f"degenerate {eligible[FMT]['d_share']:.0%})")
    # The "before" reference is the worst-degenerate format, found rather than
    # named. Hardcoding 'current' crashes on any sweep that does not contain a
    # format with that exact name.
    worst = max(stats, key=lambda f: stats[f]["d_share"])
    say("")
    say("  What 12a established: the brevity instruction was the whole defect.")
    say(f"  degenerate share {stats[worst]['d_share']:.0%} ({worst}) -> "
        f"{stats[FMT]['d_share']:.0%} ({FMT}), by adding one clause.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. GENERATING")
    say("")
    recs = data.load_records("commonsenseqa", n=None, seed=config.SEED)
    kept, _e, _r = data_quality.clean_records(recs)
    if LIMIT:
        kept = kept[:LIMIT]
    say(f"  {len(recs):,} in the validation split, {len(kept):,} after quality "
        f"filtering")

    out_dir = PHASEC_ROOT / f"commonsenseqa_{FMT}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in kept if not (out_dir / f"{r.qid}.npz").exists()]
    say(f"  {len(kept) - len(todo):,} already on disk, {len(todo):,} to generate")
    est = len(todo) * SEC_PER_Q / 3600
    say(f"  estimate {est:.2f} GPU-hours, ${est * USD_PER_HOUR:.2f}")
    say(f"  writing  {out_dir}")

    say("")
    model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
        config.MODEL_LLADA, log=say)
    mask_id = model_utils.resolve_mask_id(tokenizer, config.MODEL_LLADA)
    say(f"  loaded  : {cfg_label}   mask_id {mask_id}")
    say("")

    # `sweep_mod` was imported in Part A, so the format is byte-identical to
    # the one that was measured. Re-typing the template here would be a second
    # source of truth and the two would drift.
    say(f"  prompt template taken from step12a: {FMT!r}")
    say(f"    {sweep_mod.FORMATS[FMT]!r}")
    say("")

    n_done = 0
    for rec in todo:
        prompt_ids = sweep_mod.build_ids(
            tokenizer, sweep_mod.build_text(FMT, rec))
        _raw, traj = logging_patch.generate_with_logging(
            model, tokenizer, prompt_ids,
            gen_length=config.GEN_LENGTH, steps=config.DENOISING_STEPS,
            block_length=config.BLOCK_LENGTH, temperature=0.0,
            question=rec.question, question_id=rec.qid,
            quant_config=cfg_label, seed=config.SEED, mask_id=mask_id)
        traj.save(out_dir / f"{rec.qid}.npz")
        n_done += 1
        if n_done % PROGRESS_EVERY == 0:
            el = time.time() - t0
            say(f"    {n_done:5,} / {len(todo):,}   "
                f"[{timedelta(seconds=int(el))}, eta "
                f"{timedelta(seconds=int(el/n_done*(len(todo)-n_done)))}]")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. SCORING AND FAILURE MODES")
    say("")
    say("  CAVEAT: the mode rules were fitted on 15-19 token answers.")
    say("  CommonsenseQA answers here are far shorter, and stable_top3 /")
    say("  cands_top3 average over three positions. Treat this split as")
    say("  descriptive, not calibrated for this dataset.")
    say("")
    rows = []
    for rec in kept:
        path = out_dir / f"{rec.qid}.npz"
        if not path.exists():
            continue
        traj = logging_patch.Trajectory.load(path)
        answer = clean_answer(tokenizer.decode(traj.final_ids.tolist()))
        m = measure(traj)
        if not m:
            continue
        tg = text_golds(rec.gold_answers)
        correct = match_wordbound(answer, tg)
        row = dict(dataset="commonsenseqa", qid=rec.qid, fmt=FMT,
                   correct=correct, answer=answer,
                   gold=" | ".join(rec.gold_answers),
                   question=rec.question,
                   q_overlap=question_overlap(answer, rec.question),
                   copy_run=max_copied_run(answer, rec.question), **m)
        if not correct:
            usable = testable_golds(rec.gold_answers)
            gr = -1
            if usable:
                for i, text in enumerate(shadow_guesses(traj, tokenizer)):
                    if match_wordbound(text, usable):
                        gr = i
                        break
            row.update(gold_round=gr, gold_ever=gr >= 0,
                       gold_testable=bool(usable))
            row["mode"] = classify(row)
        else:
            row["mode"] = "correct"
        rows.append(row)

    wrong = [r for r in rows if not r["correct"]]
    c = Counter(r["mode"] for r in wrong)
    say(f"  total {len(rows):,}   wrong {len(wrong):,}  "
        f"({len(wrong)/max(1,len(rows)):.0%})")
    say("")
    say("  mode              n")
    say("  --------------  ----")
    for m in ("locked_in", "interleaving", "inconsistent", "echo",
              "degenerate", "untestable"):
        say(f"  {m:<14}  {c[m]:4d}")
    usable_n = c["locked_in"] + c["interleaving"] + c["inconsistent"]
    n_correct = len(rows) - len(wrong)
    say("")
    say(f"  usable wrong (locked_in + interleaving + inconsistent): {usable_n}")
    say(f"  correct answers, which are the negatives:               {n_correct}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. IS THIS ENOUGH FOR PLAN STEP 30?")
    say("")
    say("  Step 30 is a NEGATIVE control: evidence must give ZERO gain on")
    say("  CommonsenseQA. Claiming 'no effect' needs a tight interval around")
    say("  zero, so what matters is the width, not the point estimate.")
    say("")
    if usable_n and n_correct:
        half = 1.96 * hm_se(0.65, usable_n, n_correct)
        diff = 1.96 * math.sqrt(2) * hm_se(0.65, usable_n, n_correct)
        say(f"    AUROC 95% CI half-width at n_pos={usable_n}, "
            f"n_neg={n_correct}:  +/-{half:.3f}")
        say(f"    CI half-width on a DIFFERENCE of two AUROCs:      "
            f"+/-{diff:.3f}")
        say("")
        if diff <= 0.08:
            say(f"  Enough. The control can state that evidence changes AUROC")
            say(f"  by less than {diff:.2f} on CommonsenseQA.")
        else:
            say(f"  Thin. The control could only rule out gains larger than")
            say(f"  {diff:.2f}. Topping up from the TRAIN split is the cheap fix")
            say("  (9,741 more questions available) and should be decided before")
            say("  step 30, not during it.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("E. PHASE C DATASET, ALL THREE DATASETS")
    say("")
    totals = {}
    if MODES_CSV.exists():
        with open(MODES_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                d = totals.setdefault(r["dataset"], Counter())
                d["n"] += 1
                d["correct" if str(r["correct"]).lower() == "true"
                  else r["mode"]] += 1
    totals["commonsenseqa"] = Counter({"n": len(rows), "correct": n_correct})
    totals["commonsenseqa"].update(c)

    say("  dataset          total   wrong  locked-in  interleav  inconsist")
    say("  --------------  ------  ------  ---------  ---------  ---------")
    for ds in ("triviaqa", "hotpotqa", "commonsenseqa"):
        d = totals.get(ds)
        if not d:
            continue
        w = d["n"] - d["correct"]
        say(f"  {ds:<14}  {d['n']:6,}  {w:6,}  {d['locked_in']:9,}  "
            f"{d['interleaving']:9,}  {d['inconsistent']:9,}")
    gt = Counter()
    for d in totals.values():
        gt.update(d)
    say(f"  {'ALL':<14}  {gt['n']:6,}  {gt['n']-gt['correct']:6,}  "
        f"{gt['locked_in']:9,}  {gt['interleaving']:9,}  "
        f"{gt['inconsistent']:9,}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["dataset", "qid", "fmt", "correct", "mode", "stable_top3",
            "cands_top3", "q_overlap", "copy_run", "n_content", "truncated",
            "gold_round", "gold_ever", "gold_testable", "question", "answer",
            "gold"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["question"] = str(r["question"])[:200]
            r["answer"] = str(r["answer"])[:300]
            r["gold"] = str(r["gold"])[:200]
        w.writerows(rows)

    say("")
    say("=" * 78)
    say(f"  format     : {FMT}")
    say(f"  trajectories: {out_dir}")
    say(f"  CSV        : {CSV_PATH}")
    say(f"  wall clock : {timedelta(seconds=int(time.time() - t0))}")
    say("")
    say("  PLAN STEP 12 IS NOW COMPLETE - LLaDA-8B on all three datasets.")
    say("  Phase C remaining: step 13 (Dream-7B).")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
