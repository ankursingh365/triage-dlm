#!/usr/bin/env python3
"""
Step 18 (Dream arm) - blind failure-mode labelling worksheet
=============================================================

    modal run modal_app.py::run_cpu --script step18_dream_worksheet.py

CPU only. Two or three minutes, free. Tokenizer only, no model, no GPU.

WHY
===
Plan step 18 is "review + second annotator, Cohen's kappa". It was done for
LLaDA: 60 wrong answers, two blind annotators, **kappa 0.882**, recorded in
`claude/failure-mode-calibration.md`. It has **not** been done for Dream.

Step 17d's closing line said "Phase D is now complete for BOTH arms". That line
was wrong and I wrote it. Dream has LABELS; it does not have VALIDATED labels.
Applying LLaDA-fitted rules to Dream produces labels the same way a broken
thermometer produces temperatures.

And step 19a closed the cheap escape routes:

    stable_top3   LLaDA median 0.21 -> Dream 0.58   firing 41.9% -> 88.8%
    cands_top3    LLaDA median 4.00 -> Dream 2.50   firing 37.0% -> 83.6%

My hypothesis was that Dream's zero-reveal rounds inflate `stable_frac`. If
that were the whole story, more rounds before reveal would also mean MORE
distinct candidates - and `cands_top3` went DOWN, not up. Both measures instead
say the same thing: **Dream converges faster than LLaDA.** That may be a real
property of the model or an artefact of measuring it with LLaDA's cuts, and no
distribution can tell the two apart. Part D of 19a also showed a cands-only
rule does not close the gap (+49% and +39%), so no measure swap rescues this.

Hand labels are the only instrument left.

WHAT IS AND IS NOT BLIND
========================
The worksheet carries the question, the reference answers, the model's answer,
and the **trajectory trace** - because L and I cannot be told apart from the
final answer alone. It does NOT carry the classifier's verdict.

The sample is stratified using the current (suspect) classification, purely so
the rare modes appear at all. That biases which items you see; it cannot bias
what you call them, because the labels are not shown. The same was true of the
LLaDA round.

THE FOUR LABELS
===============
    L  locked-in     one wrong answer from early on, held to the end.
                     The trace barely changes.
    I  interleaving  the RIGHT answer appears in the trace, then is dropped.
                     Look for the <-- GOLD marker.
    C  inconsistent  the trace churns between several different answers and
                     settles on a wrong one.
    X  neither       not a hallucination at all: the model restates the
                     question, copies a word from it, runs out of budget, or
                     is actually CORRECT and the matcher disagreed.
"""

import csv
import random
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

DREAM_TAG = f"dream_g{config.GEN_LENGTH}s{config.DENOISING_STEPS}_origin"
TRAJ_ROOT = config.TRAJ_DIR / "phasec_dream" / DREAM_TAG
MODES_CSV = config.TAB_DIR / "step17d_dream_modes.csv"
SHEET_PATH = config.OUT_DIR / "step18_dream_worksheet.txt"
KEY_CSV = config.TAB_DIR / "step18_dream_answer_key.csv"
REPORT_PATH = config.OUT_DIR / "step18_dream_worksheet_report.txt"

N_ITEMS = 60
# Minimums so the rare modes appear. Stratification only - the labels are not
# shown, so this cannot steer what you call each item.
MODE_MIN = {"locked_in": 18, "inconsistent": 15, "interleaving": 12,
            "echo": 8, "untestable": 4, "degenerate": 3}
WRAP = 92
MAX_TRACE_LINES = 12

MIN_GOLD_CHARS = 4
STOP_GOLDS = {"yes", "no", "true", "false", "none", "both", "all"}
SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


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


def testable_golds(golds: list, dataset: str) -> list:
    gs = [g for g in golds if len(normalise(g)) > 1] \
        if dataset == "commonsenseqa" else list(golds)
    return [g for g in gs if len(normalise(g)) >= MIN_GOLD_CHARS
            and normalise(g) not in STOP_GOLDS]


def compressed_trace(traj, tok) -> list:
    """The shadow prediction per round, with unchanged rounds collapsed.

    `pred_ids[r]` is the raw argmax at every position, recorded BEFORE the
    reveal overwrite, so this is what the model would say if it committed
    everything at round r. Collapsing runs is what makes 64 rounds readable:
    a locked-in case becomes one or two lines, an interleaving case shows the
    right answer appearing and then leaving.
    """
    rows = [[int(t) for t in traj.pred_ids[r].tolist()
             if t not in (traj.eos_id, traj.mask_id)]
            for r in range(traj.pred_ids.shape[0])]
    texts = tok.batch_decode(rows, skip_special_tokens=True)
    spans, start = [], 0
    for r in range(1, len(texts) + 1):
        if r == len(texts) or texts[r] != texts[start]:
            label = (f"r{start:02d}" if r - 1 == start
                     else f"r{start:02d}-r{r-1:02d}")
            spans.append((label, texts[start]))
            start = r
    return spans


def trim(spans: list, keep_idx: set) -> list:
    """Keep the first few, the last few, and any span the caller marked."""
    if len(spans) <= MAX_TRACE_LINES:
        return [(a, b, False) for a, b in spans]
    head, tail = 4, 4
    keep = set(range(head)) | set(range(len(spans) - tail, len(spans))) | keep_idx
    out, gap = [], False
    for i, (a, b) in enumerate(spans):
        if i in keep:
            out.append((a, b, False))
            gap = False
        elif not gap:
            out.append(("...", f"({len(spans) - len(keep)} unchanged spans)", True))
            gap = True
    return out


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 18 (Dream arm): blind labelling worksheet")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not MODES_CSV.exists():
        say(f"\nMissing {MODES_CSV}. Run step17d first.")
        sys.exit(1)

    rows = []
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r["correct"]).lower() != "true":
                rows.append(r)
    say(f"  {len(rows):,} Dream wrong answers available")

    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    for v in by_mode.values():
        v.sort(key=lambda r: (r["dataset"], r["qid"]))
    say("  current (suspect) classification, used only for stratification:")
    for m in sorted(by_mode, key=lambda m: -len(by_mode[m])):
        say(f"    {m:<14} {len(by_mode[m]):6,}")

    rng = random.Random(config.SEED)
    picked, remaining = [], N_ITEMS
    for mode, want in MODE_MIN.items():
        pool = by_mode.get(mode, [])
        take = min(want, len(pool), remaining)
        if take:
            picked += rng.sample(pool, take)
            remaining -= take
    if remaining > 0:
        rest = [r for r in rows if r not in picked]
        picked += rng.sample(rest, min(remaining, len(rest)))
    rng.shuffle(picked)
    say("")
    say(f"  {len(picked)} sampled; worksheet order shuffled so it leaks nothing")

    records = {}
    for ds, want in (("triviaqa", 3750), ("hotpotqa", 7500),
                     ("commonsenseqa", None)):
        recs = data.load_records(ds, n=None if want is None else want * 2,
                                 seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        for rec in (kept if want is None else kept[:want]):
            records[(ds, rec.qid)] = rec

    say("  Loading tokenizer (no model, no GPU)...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)

    # ---- build the worksheet ---------------------------------------------
    sheet = []
    sheet.append("=" * WRAP)
    sheet.append("  TRIAGE - Step 18: Dream failure-mode labelling worksheet")
    sheet.append("=" * WRAP)
    sheet.append("")
    sheet.append("  THE FOUR LABELS")
    sheet.append("")
    sheet.append("   L  locked-in     ONE wrong answer from early on, held to the end.")
    sheet.append("                    The trace barely changes.")
    sheet.append("")
    sheet.append("   I  interleaving  the RIGHT answer appears in the trace, then is")
    sheet.append("                    dropped. Look for the  <-- GOLD  marker.")
    sheet.append("")
    sheet.append("   C  inconsistent  the trace CHURNS between several different answers")
    sheet.append("                    and settles on a wrong one.")
    sheet.append("")
    sheet.append("   X  neither       not a hallucination at all: the model restates the")
    sheet.append("                    question, copies a word from it, runs out of")
    sheet.append("                    budget, or is actually CORRECT and the matcher")
    sheet.append("                    disagreed.")
    sheet.append("")
    sheet.append("  HOW TO READ A TRACE")
    sheet.append("")
    sheet.append("  Each line is what the model WOULD have said at those rounds, before")
    sheet.append("  it committed anything. Rounds where nothing changed are collapsed,")
    sheet.append("  so  r13-r40  means the same guess held for 28 rounds.")
    sheet.append("")
    sheet.append("  L vs C is the usual hard call: L is one answer held, C is several")
    sheet.append("  answers tried. If the trace shows two or three DIFFERENT wrong")
    sheet.append("  answers, that is C, not L.")
    sheet.append("")
    sheet.append("  Send back one line of 60 labels, in order, separated by spaces:")
    sheet.append("      L C C X I C L ...")
    sheet.append("")
    sheet.append("  The classifier's verdicts are NOT in this file, deliberately.")
    sheet.append("  Do not open step17d_dream_modes.csv before labelling.")
    sheet.append("")

    key_rows = []
    for n, r in enumerate(picked, 1):
        ds, qid = r["dataset"], r["qid"]
        rec = records.get((ds, qid))
        path = TRAJ_ROOT / ds / f"{qid}.npz"
        if rec is None or not path.exists():
            continue
        traj = logging_patch.Trajectory.load(path)
        spans = compressed_trace(traj, tok)
        tg = testable_golds(rec.gold_answers, ds)
        gold_idx = {i for i, (_a, t) in enumerate(spans)
                    if tg and match_wordbound(t, tg)}

        sheet.append("=" * WRAP)
        sheet.append(f"  #{n:<4} [{ds}]")
        sheet.append("")
        for tag, body in (("Q   ", rec.question),
                          ("GOLD", " | ".join(rec.gold_answers[:5])),
                          ("ANS ", r.get("answer", "") or "(empty)")):
            wrapped = textwrap.wrap(str(body).strip(), WRAP - 10) or ["(empty)"]
            sheet.append(f"  {tag}  {wrapped[0]}")
            for cont in wrapped[1:]:
                sheet.append(" " * 8 + cont)
        sheet.append("")
        sheet.append("  TRACE")
        for i, (label, text, is_gap) in enumerate(trim(spans, gold_idx)):
            mark = ""
            if not is_gap:
                orig = [j for j, (a, _t) in enumerate(spans) if a == label]
                if orig and orig[0] in gold_idx:
                    mark = "   <-- GOLD"
            body = (text or "(nothing yet)")[: WRAP - 26]
            sheet.append(f"    {label:<10} | {body}{mark}")
        sheet.append("")
        sheet.append("  label?  ____")
        sheet.append("")
        key_rows.append(dict(n=n, dataset=ds, qid=qid))

    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHEET_PATH.write_text("\n".join(sheet) + "\n", encoding="utf-8")
    KEY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(KEY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "dataset", "qid"])
        w.writeheader()
        w.writerows(key_rows)

    say("")
    say("=" * 78)
    say(f"  items      : {len(key_rows)}")
    say(f"  worksheet  : {SHEET_PATH}")
    say(f"  answer key : {KEY_CSV}")
    say(f"  widest line: {max(len(x) for x in sheet)}  "
        f"(the LLaDA sheet truncated at 62 and cost one unlabellable item)")
    say("")
    say("  Label it, then a second annotator labels it blind, then kappa.")
    say("  Only after that are Dream's cuts refitted - and the LLaDA cuts")
    say("  stay exactly as they are, because they were validated on LLaDA.")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
