#!/usr/bin/env python3
"""
Step 17d - labels for the Dream arm (plan steps 15 and 17), and an
exhaustive reconstruction gate
===================================================================

    modal run modal_app.py::run_cpu --script step17d_dream_labels.py

CPU only. About 30-45 minutes, free. No model, no GPU - only the tokenizer.

WHY THIS EXISTS BEFORE STEP 19
==============================
Phase C finished with 12,317 Dream trajectories on disk. They are unlabelled:
not scored correct/wrong, not classified into failure modes. Plan steps 15 and
17 did that for LLaDA and nothing has done it for Dream, so Phase D is
incomplete again for the second arm. `src/features.py` cannot use data that has
no labels, and racing past a missing step is exactly how two Phase C steps got
skipped the first time.

THREE THINGS, IN ORDER
======================
**1. The gate, exhaustively.** Step 13c passed the reconstruction LOCK on 50
Dream trajectories. LLaDA's gate was run on all 11,120. This pass reads every
Dream trajectory anyway, so the same check runs on all 12,317 for free, and
Dream's gate stops being a sample.

**2. Correctness.** Whole-word matching against the reference answers - the
scorer validated in plan step 16 at 96% against a two-annotator consensus
(`claude/judge-rejected-decision.md`). CommonsenseQA is scored against the
option TEXT only, never the bare letter, because `\\be\\b` matches by accident.

**3. Failure modes**, using the rules adopted in step 17c:

    degenerate    every generated position filled
    echo/copy     question_overlap >= 0.75 AND max_copied_run >= 6
    untestable    no gold alias long or distinctive enough to test
    interleaving  gold appeared in the shadow prediction, then was dropped
    locked_in     stable_top3 >= 0.26 AND cands_top3 <= 3.4
    inconsistent  everything else                      (order A)

THE TRANSFER CAVEAT, STATED BEFORE THE NUMBERS
==============================================
**Those cuts were fitted on LLaDA and have not been refitted for Dream.** Two
reasons to expect strain, and one to expect robustness:

  - Dream's schedule is ADAPTIVE (step 13c: 0-5 positions per round, 37% of
    rounds reveal nothing). A position's pre-reveal history is therefore a
    different length than in LLaDA, and `stable_frac` is a ratio over that
    history.
  - Dream's answers may differ in length, and step 12b showed short answers
    push `stable_top3` up and `cands_top3` down, which manufactures
    `locked_in`. CommonsenseQA's LLaDA split hit exactly that.
  - Against those: `stable_top3` is measured relative to each position's OWN
    reveal round, which is the binding rule in
    `claude/schedule-relative-measurement.md` and the reason it was chosen. A
    schedule-relative measure should survive a schedule change better than an
    absolute one.

Part D therefore puts Dream and LLaDA side by side and asks whether the split
is plausible rather than asserting it is calibrated. A wildly different split
means the cuts need refitting on Dream hand labels - a real cost, and a
decision to take with the numbers in hand.
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

HERE = Path(__file__).resolve().parent
DREAM_TAG = f"dream_g{config.GEN_LENGTH}s{config.DENOISING_STEPS}_origin"
TRAJ_ROOT = config.TRAJ_DIR / "phasec_dream" / DREAM_TAG
LLADA_CSV = config.TAB_DIR / "step12c_final_modes.csv"
CSQA_CSV = config.TAB_DIR / "step12b_csqa_modes.csv"
REPORT_PATH = config.OUT_DIR / "step17d_dream_labels.txt"
CSV_PATH = config.TAB_DIR / "step17d_dream_modes.csv"

TARGETS = (("triviaqa", 3750), ("hotpotqa", 7500), ("commonsenseqa", None))
CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6)
ORDER = "A"
MODES = ("locked_in", "interleaving", "inconsistent", "echo", "degenerate",
         "untestable")
PROGRESS_EVERY = 1000

LIMIT = None
if "--limit" in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index("--limit") + 1])

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Text handling - identical to step 17c, restated so this file is standalone
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


def score_golds(golds: list, dataset: str) -> list:
    """Golds usable for SCORING. CommonsenseQA's list is
    `[option_text, "E"]`, and the bare letter makes `\\be\\b` match by
    accident - the single-letter trap Step 9b fixed for LLaDA."""
    if dataset == "commonsenseqa":
        return [g for g in golds if len(normalise(g)) > 1]
    return list(golds)


def testable_golds(golds: list, dataset: str) -> list:
    return [g for g in score_golds(golds, dataset)
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
    echo = r["q_overlap"] >= CUTS["e_cut"] and r["copy_run"] >= CUTS["r_cut"]
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


# ===========================================================================

def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 17d: Dream labels (plan steps 15 and 17, Dream arm)")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  data : {TRAJ_ROOT}")
    say(f"  cuts : stable>={CUTS['s_cut']} cands<={CUTS['c_cut']} "
        f"overlap>={CUTS['e_cut']} run>={CUTS['r_cut']}, order {ORDER}")
    say("  NOTE : those cuts were fitted on LLaDA. Part D asks whether they")
    say("         transfer; it does not assume they do.")
    if not TRAJ_ROOT.exists():
        say(f"\nMissing {TRAJ_ROOT}. Run step13d first.")
        sys.exit(1)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "csqa_sweep", HERE / "step12a_csqa_prompt_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    sys.modules["csqa_sweep"] = sweep
    spec.loader.exec_module(sweep)

    say("")
    say("  Loading tokenizer (no model, no GPU)...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)
    say(f"  vocab {len(tok):,}")

    records = {}
    for ds, want in TARGETS:
        recs = data.load_records(ds, n=None if want is None else want * 2,
                                 seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        chosen = kept if want is None else kept[:want]
        for rec in chosen:
            records[(ds, rec.qid)] = rec
    say(f"  {len(records):,} records rebuilt for golds and questions")

    # =======================================================================
    say("")
    say("=" * 78)
    say("A. THE RECONSTRUCTION GATE, ON ALL OF DREAM")
    say("")
    say("  Step 13c passed this on 50. LLaDA's ran on all 11,120. These files")
    say("  are being read anyway, so Dream's gate stops being a sample.")
    say("")
    rows, failures, n = [], [], 0
    for ds, _w in TARGETS:
        sub = TRAJ_ROOT / ds
        if not sub.exists():
            say(f"  WARNING {ds}: {sub} missing")
            continue
        paths = sorted(sub.glob("*.npz"))
        if LIMIT:
            paths = paths[:LIMIT]
        say(f"  {ds:<14} {len(paths):,} trajectories")
        for path in paths:
            rec = records.get((ds, path.stem))
            if rec is None:
                continue
            traj = logging_patch.Trajectory.load(path)
            want = (config.GEN_LENGTH, config.DENOISING_STEPS)
            got = (int(traj.gen_length), int(traj.steps))
            if got != want:
                raise SystemExit(f"\nSTOPPED - {path} is gen/steps {got}, "
                                 f"config says {want}\n")
            rebuilt = traj.reconstruct_final()
            ids_ok = bool(np.array_equal(rebuilt, traj.final_ids))
            text_ok = (tok.decode(rebuilt.tolist())
                       == tok.decode(traj.final_ids.tolist()))
            if not (ids_ok and text_ok):
                failures.append((ds, path.stem, ids_ok, text_ok))

            answer = clean_answer(tok.decode(traj.final_ids.tolist()))
            m = measure(traj)
            if not m:
                continue
            sg = score_golds(rec.gold_answers, ds)
            correct = match_wordbound(answer, sg)
            row = dict(dataset=ds, qid=path.stem, correct=correct,
                       answer=answer, gold=" | ".join(rec.gold_answers[:5]),
                       question=rec.question,
                       q_overlap=question_overlap(answer, rec.question),
                       copy_run=max_copied_run(answer, rec.question), **m)
            if not correct:
                usable = testable_golds(rec.gold_answers, ds)
                gr = -1
                if usable:
                    for i, txt in enumerate(shadow_guesses(traj, tok)):
                        if match_wordbound(txt, usable):
                            gr = i
                            break
                row.update(gold_round=gr, gold_ever=gr >= 0,
                           gold_testable=bool(usable))
                row["mode"] = classify(row)
            else:
                row["mode"] = "correct"
            rows.append(row)
            n += 1
            if n % PROGRESS_EVERY == 0:
                el = time.time() - t0
                say(f"    {n:6,} done   {len(failures)} gate failures   "
                    f"[{timedelta(seconds=int(el))}]")

    say("")
    say(f"  checked      {len(rows):,}")
    say(f"  failures     {len(failures):,}")
    if failures:
        say("")
        for ds, qid, i_ok, t_ok in failures[:10]:
            say(f"    {ds}/{qid}  ids_ok={i_ok} text_ok={t_ok}")
        say("")
        say("  GATE FAILED. STOP. Dream's trajectories do not describe what")
        say("  Dream did, and no label computed from them means anything.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  GATE PASSED - every Dream trajectory rebuilds exactly.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. CORRECTNESS")
    say("")
    say("  Whole-word matching, the scorer validated at 96% in plan step 16.")
    say("  PROVISIONAL in the same way LLaDA's is - see")
    say("  claude/judge-rejected-decision.md for its measured error rates.")
    say("")
    say("  dataset          total    wrong   err%")
    say("  --------------  ------  -------  -----")
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    for ds, _w in TARGETS:
        rs = by.get(ds, [])
        if not rs:
            continue
        w = sum(1 for r in rs if not r["correct"])
        say(f"  {ds:<14}  {len(rs):6,}  {w:7,}  {w/len(rs):5.0%}")
    nw = sum(1 for r in rows if not r["correct"])
    say(f"  {'ALL':<14}  {len(rows):6,}  {nw:7,}  {nw/max(1,len(rows)):5.0%}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. FAILURE MODES")
    say("")
    say("  dataset          wrong  locked-in  interleav  inconsist   echo"
        "  degen  untest")
    say("  --------------  ------  ---------  ---------  ---------  -----"
        "  -----  ------")
    for ds, _w in TARGETS:
        wr = [r for r in by.get(ds, []) if not r["correct"]]
        if not wr:
            continue
        c = Counter(r["mode"] for r in wr)
        say(f"  {ds:<14}  {len(wr):6,}  {c['locked_in']:9,}  "
            f"{c['interleaving']:9,}  {c['inconsistent']:9,}  {c['echo']:5,}  "
            f"{c['degenerate']:5,}  {c['untestable']:6,}")
    wrong = [r for r in rows if not r["correct"]]
    tot = Counter(r["mode"] for r in wrong)
    say(f"  {'ALL':<14}  {len(wrong):6,}  {tot['locked_in']:9,}  "
        f"{tot['interleaving']:9,}  {tot['inconsistent']:9,}  {tot['echo']:5,}  "
        f"{tot['degenerate']:5,}  {tot['untestable']:6,}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. DO THE LLaDA-FITTED CUTS TRANSFER?")
    say("")
    llada = Counter()
    n_llada = 0
    for p in (LLADA_CSV, CSQA_CSV):
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                n_llada += 1
                if str(r["correct"]).lower() != "true":
                    llada[r["mode"]] += 1
    if not n_llada:
        say("  LLaDA's mode CSVs are absent; cannot compare.")
    else:
        n_lw = sum(llada.values())
        say("  Share of WRONG answers in each mode.")
        say("")
        say("  mode              LLaDA     Dream    diff")
        say("  --------------  -------  --------  ------")
        big = []
        for m in MODES:
            a = llada[m] / max(1, n_lw)
            b = tot[m] / max(1, len(wrong))
            say(f"  {m:<14}  {a:6.1%}  {b:8.1%}  {b-a:+6.1%}")
            if abs(b - a) > 0.20:
                big.append((m, a, b))
        say("")
        say(f"  LLaDA wrong answers {n_lw:,} of {n_llada:,}"
            f"   Dream {len(wrong):,} of {len(rows):,}")
        say("")
        if not big:
            say("  No mode shifts by more than 20 points. The cuts transfer")
            say("  well enough to proceed, and the paper reports that they")
            say("  were fitted on LLaDA and applied unchanged to Dream.")
        else:
            say("  These modes shift by more than 20 points:")
            for m, a, b in big:
                say(f"    {m}: {a:.0%} -> {b:.0%}")
            say("")
            say("  A shift that large is not a finding about Dream, it is a")
            say("  warning that LLaDA-fitted cuts may not describe Dream's")
            say("  trajectories. Before using Dream's per-mode split in the")
            say("  paper, refit on Dream hand labels - the same blind")
            say("  two-annotator round that produced kappa 0.882 for LLaDA.")
            say("  Dream's OVERALL numbers are unaffected either way.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["dataset", "qid", "correct", "mode", "stable_top3", "cands_top3",
            "q_overlap", "copy_run", "n_content", "truncated", "gold_round",
            "gold_ever", "gold_testable", "question", "answer", "gold"]
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
    say(f"  rows       : {len(rows):,}")
    say(f"  CSV        : {CSV_PATH}")
    say(f"  wall clock : {timedelta(seconds=int(time.time()-t0))}")
    say("")
    say("  Phase D is now complete for BOTH arms. Next: step 19,")
    say("  src/features.py.")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
