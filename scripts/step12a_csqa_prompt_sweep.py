#!/usr/bin/env python3
"""
Step 12a - why CommonsenseQA is unusable, and which prompt format fixes it
===========================================================================

    modal run modal_app.py::run --script step12a_csqa_prompt_sweep.py

A100. About 35 minutes and $1.20 - 4 formats x 150 questions, plus model load.

WHAT IS BROKEN
==============
CommonsenseQA is plan step 12's missing third dataset and plan step 30's
negative control. As currently prompted it is unusable as either:

    error rate                     11%
    of those wrong answers,
    degenerate (filled all 64
    tokens, never emitted EOS)     73%

A dataset where three quarters of the errors are the decoder running out of
budget measures the decoder, not the model's beliefs.

THE SUSPECT
===========
`data.build_prompt_text` has two branches, and only one of them asks for a
short answer:

    with choices     "{q}\\n\\n{options}\\n\\nAnswer with the single best option."
    without choices  "{q}\\n\\nAnswer as briefly as possible."

CommonsenseQA is the only dataset with choices, so it is the only dataset that
never receives the brevity instruction. A model asked to pick "the single best
option" with no length cue explains its reasoning, and 64 tokens is not enough
to finish - which is exactly what "degenerate" measures.

**That is a hypothesis, not a finding.** The obvious fix - dropping the options
- is probably the wrong one: CommonsenseQA questions are deliberately ambiguous
without them ("where would you find a bookstore" has many defensible answers),
which is why the dataset is multiple choice in the first place. So this script
measures four formats instead of asserting one.

THE FOUR FORMATS
================
    current    the incumbent, measured rather than assumed
    brief      incumbent + an explicit brevity instruction
    text_only  incumbent, but asking for the option's TEXT rather than a letter
    open       no options at all, using the other datasets' wording

THE CRITERION, DECLARED BEFORE THE NUMBERS
==========================================
    HARD    degenerate share of wrong answers <= 25%   (it is 73% today)
    THEN    among formats that clear it, maximise the USABLE WRONG rate -
            questions that are wrong, not degenerate and not echo, as a share
            of all questions asked. That is the quantity step 30 needs, because
            a negative control with no usable errors proves nothing.

If no format clears the hard criterion the script says so and stops. It does
not quietly pick the least bad one.

THE CACHE TRAP THIS AVOIDS
==========================
`config.RUN_TAG` encodes the DECODER settings and nothing else. Changing the
prompt does not change the tag, so trajectories generated under one prompt
would be silently reused under another - the same class of bug that once made a
Step 9 re-run finish in 0.7 minutes instead of 10. Each format therefore writes
to its own directory, keyed by format id.
"""

import csv
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

SWEEP_ROOT = config.TRAJ_DIR / "csqa_sweep" / config.RUN_TAG
REPORT_PATH = config.OUT_DIR / "step12a_csqa_sweep.txt"
CSV_PATH = config.TAB_DIR / "step12a_csqa_sweep.csv"

N_PER_FORMAT = 150
SEC_PER_Q, USD_PER_HOUR = 3.09, 2.10

# Declared before any number is seen.
MAX_DEGENERATE_SHARE = 0.25

# The four formats. `None` for the options slot means the options are withheld.
FORMATS = {
    "current": "{q}\n\n{options}\n\nAnswer with the single best option.",
    "brief": ("{q}\n\n{options}\n\nAnswer with the single best option, "
              "in a few words and nothing else."),
    "text_only": ("{q}\n\n{options}\n\nReply with only the text of the best "
                  "option, and nothing else."),
    "open": "{q}\n\nAnswer as briefly as possible.",
}

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Scoring - identical rules to step 9b / 17c so the numbers are comparable
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
STOP = {"is", "was", "are", "were", "be", "been", "of", "in", "on", "at", "to",
        "for", "and", "or", "but", "by", "with", "from", "that", "which",
        "who", "what", "when", "where", "how", "it", "its", "this", "these",
        "as", "has", "have", "had", "do", "does", "did", "not", "no", "yes",
        "s", "t"}

ECHO_OVERLAP, ECHO_RUN = 0.75, 6      # the rule adopted in step 17c


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
    """The option TEXT, not the bare letter.

    CommonsenseQA's gold list is `[option_text, "E"]`. Scoring against the
    letter makes `\\be\\b` a match, so an answer that happens to contain a
    standalone "e" scores correct. The same single-letter bug that Step 9b
    fixed for scoring and Step 10 fixed for interleaving. Letters are scored
    separately below so the size of the difference is visible.
    """
    return [g for g in golds if len(normalise(g)) > 1]


def letter_golds(golds: list) -> list:
    return [g for g in golds if len(normalise(g)) == 1]


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def build_text(fmt: str, rec) -> str:
    options = "\n".join(rec.choices or [])
    return FORMATS[fmt].format(q=rec.question.strip(), options=options)


def build_ids(tokenizer, text: str) -> list:
    """Mirror of `data.build_prompt_ids`, but for a caller-supplied prompt."""
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    except Exception:
        return tokenizer(f"Question: {text}\nAnswer:",
                         add_special_tokens=True)["input_ids"]


# ===========================================================================

def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 12a: CommonsenseQA prompt format sweep")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  config  : gen {config.GEN_LENGTH}  steps {config.DENOISING_STEPS}"
        f"  block {config.BLOCK_LENGTH}   (tag {config.RUN_TAG})")
    say(f"  formats : {', '.join(FORMATS)}")
    say(f"  {N_PER_FORMAT} questions each = {N_PER_FORMAT * len(FORMATS)} "
        f"generations, about "
        f"{N_PER_FORMAT * len(FORMATS) * SEC_PER_Q / 3600:.1f} GPU-hours")
    say("")
    say(f"  DECLARED CRITERION (before any result): degenerate share of wrong")
    say(f"  answers <= {MAX_DEGENERATE_SHARE:.0%}, then maximise usable-wrong rate.")

    recs = data.load_records("commonsenseqa", n=N_PER_FORMAT * 3,
                             seed=config.SEED)
    kept, _e, _r = data_quality.clean_records(recs)
    chosen = kept[:N_PER_FORMAT]
    say("")
    say(f"  {len(chosen)} questions after quality filtering")
    if len(chosen) < N_PER_FORMAT:
        say(f"  WARNING wanted {N_PER_FORMAT}")

    say("")
    model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
        config.MODEL_LLADA, log=say)
    mask_id = model_utils.resolve_mask_id(tokenizer, config.MODEL_LLADA)
    say(f"  loaded  : {cfg_label}   mask_id {mask_id}")

    say("")
    say("=" * 78)
    say("A. THE FOUR PROMPTS, AS THE MODEL RECEIVES THEM")
    say("")
    demo = chosen[0]
    for fmt in FORMATS:
        say(f"  --- {fmt} " + "-" * (66 - len(fmt)))
        for ln in build_text(fmt, demo).splitlines():
            say("  | " + ln)
        say("")

    # =======================================================================
    say("=" * 78)
    say("B. GENERATING")
    say("")
    rows, n_done = [], 0
    total = len(FORMATS) * len(chosen)
    for fmt in FORMATS:
        out_dir = SWEEP_ROOT / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in chosen:
            path = out_dir / f"{rec.qid}.npz"
            if path.exists():
                traj = logging_patch.Trajectory.load(path)
                answer = clean_answer(tokenizer.decode(traj.final_ids.tolist()))
            else:
                prompt_ids = build_ids(tokenizer, build_text(fmt, rec))
                raw, traj = logging_patch.generate_with_logging(
                    model, tokenizer, prompt_ids,
                    gen_length=config.GEN_LENGTH,
                    steps=config.DENOISING_STEPS,
                    block_length=config.BLOCK_LENGTH,
                    temperature=0.0, question=rec.question,
                    question_id=rec.qid, quant_config=cfg_label,
                    seed=config.SEED, mask_id=mask_id)
                traj.save(path)
                answer = clean_answer(raw)

            n_content = int(sum(1 for t in traj.final_ids.tolist()
                                if t not in (traj.eos_id, traj.mask_id)))
            tg, lg = text_golds(rec.gold_answers), letter_golds(rec.gold_answers)
            rows.append(dict(
                fmt=fmt, qid=rec.qid, question=rec.question, answer=answer,
                gold=" | ".join(rec.gold_answers),
                correct=match_wordbound(answer, tg),
                letter_only=(not match_wordbound(answer, tg)
                             and bool(lg) and match_wordbound(answer, lg)),
                degenerate=n_content >= traj.gen_length - 2,
                echo=(question_overlap(answer, rec.question) >= ECHO_OVERLAP
                      and max_copied_run(answer, rec.question) >= ECHO_RUN),
                n_content=n_content))
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                say(f"    {n_done:4d} / {total}   "
                    f"[{timedelta(seconds=int(el))}, "
                    f"eta {timedelta(seconds=int(el / n_done * (total - n_done)))}]")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. RESULTS")
    say("")
    say("  'usable wrong' = wrong AND not degenerate AND not echo.")
    say("  That is the quantity step 30's negative control actually needs.")
    say("")
    say("  format      n   err%   degen% of wrong   echo%   usable-wrong%   toks")
    say("  ---------  ---  -----  ---------------  ------  --------------  -----")
    summary = {}
    for fmt in FORMATS:
        rs = [r for r in rows if r["fmt"] == fmt]
        n = len(rs)
        wrong = [r for r in rs if not r["correct"]]
        degen = [r for r in wrong if r["degenerate"]]
        echo = [r for r in wrong if r["echo"] and not r["degenerate"]]
        usable = [r for r in wrong if not r["degenerate"] and not r["echo"]]
        d_share = len(degen) / len(wrong) if wrong else float("nan")
        u_rate = len(usable) / n if n else 0.0
        summary[fmt] = dict(n=n, n_wrong=len(wrong), d_share=d_share,
                            u_rate=u_rate, n_usable=len(usable))
        say(f"  {fmt:<9}  {n:3d}  {len(wrong)/n:5.0%}  {d_share:15.0%}  "
            f"{len(echo)/max(1,len(wrong)):6.0%}  {u_rate:14.0%}  "
            f"{np.mean([r['n_content'] for r in rs]):5.1f}")

    say("")
    say("  95% CIs on the two numbers the decision uses:")
    say("")
    say("  format     degenerate share of wrong      usable-wrong rate")
    say("  ---------  -------------------------  ----------------------")
    for fmt in FORMATS:
        s = summary[fmt]
        rs = [r for r in rows if r["fmt"] == fmt]
        n_deg = sum(1 for r in rs if not r["correct"] and r["degenerate"])
        dlo, dhi = wilson(n_deg, max(1, s["n_wrong"]))
        ulo, uhi = wilson(s["n_usable"], s["n"])
        say(f"  {fmt:<9}  {s['d_share']:6.0%}  [{dlo:.2f}, {dhi:.2f}]"
            f"        {s['u_rate']:5.0%}  [{ulo:.2f}, {uhi:.2f}]")

    say("")
    say("  Answers scored correct only by the bare option LETTER (the")
    say("  single-letter matching trap), per format:")
    for fmt in FORMATS:
        n_l = sum(1 for r in rows if r["fmt"] == fmt and r["letter_only"])
        say(f"    {fmt:<10} {n_l}")

    # =======================================================================
    say("")
    say("=" * 78)
    say("D. EXAMPLE ANSWERS - read these, the numbers do not show everything")
    say("")
    for fmt in FORMATS:
        say(f"  --- {fmt} " + "-" * (66 - len(fmt)))
        rs = [r for r in rows if r["fmt"] == fmt]
        for r in rs[:4]:
            tag = "OK  " if r["correct"] else ("DEGN" if r["degenerate"]
                                               else "WRNG")
            say(f"   [{tag}] gold {r['gold'][:28]:<28} | {r['answer'][:60]!r}")
        say("")

    # =======================================================================
    say("=" * 78)
    say("E. DECISION")
    say("")
    eligible = {f: s for f, s in summary.items()
                if s["d_share"] == s["d_share"]
                and s["d_share"] <= MAX_DEGENERATE_SHARE}
    if not eligible:
        say(f"  NO FORMAT clears the declared criterion "
            f"(degenerate <= {MAX_DEGENERATE_SHARE:.0%}).")
        say("")
        say("  Best degenerate share achieved: "
            f"{min(s['d_share'] for s in summary.values()):.0%}")
        say("")
        say("  Not picking the least bad one. CommonsenseQA stays out of the")
        say("  dataset and step 30's negative control needs a different")
        say("  design - options are a different generation length, a different")
        say("  dataset for the control, or dropping the control and saying so.")
    else:
        best = max(eligible, key=lambda f: eligible[f]["u_rate"])
        s = summary[best]
        say(f"  ADOPT format '{best}'.")
        say("")
        say(f"    degenerate share  {s['d_share']:.0%}  "
            f"(limit {MAX_DEGENERATE_SHARE:.0%})")
        say(f"    usable-wrong rate {s['u_rate']:.0%}")
        if len(eligible) > 1:
            say(f"    cleared the limit: {', '.join(sorted(eligible))}")
        say("")
        need = 300
        n_q = int(need / s["u_rate"]) if s["u_rate"] > 0 else 0
        hrs = n_q * SEC_PER_Q / 3600
        say(f"  To reach {need} usable wrong answers at this rate: "
            f"{n_q:,} questions,")
        say(f"  {hrs:.1f} GPU-hours, ${hrs * USD_PER_HOUR:.2f}.")
        say("")
        say("  Next: step 12b generates CommonsenseQA at that size, writing to")
        say("  a directory keyed by this format id so nothing is mixed.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = ["fmt", "qid", "correct", "letter_only", "degenerate", "echo",
            "n_content", "question", "answer", "gold"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["question"] = str(r["question"])[:200]
            r["answer"] = str(r["answer"])[:300]
        w.writerows(rows)

    say("")
    say("=" * 78)
    say(f"  CSV        : {CSV_PATH}")
    say(f"  wall clock : {timedelta(seconds=int(time.time() - t0))}")
    say(f"  cost       : ${(time.time()-t0)/3600*USD_PER_HOUR:.2f}")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
