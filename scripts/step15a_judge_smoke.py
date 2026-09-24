#!/usr/bin/env python3
"""
Step 13a - build the Qwen3 judge, measure it, and emit a blind validation set
=============================================================================

    modal run modal_app.py::run --script step13a_judge_smoke.py

A100. Roughly 5 minutes and $0.20 - almost all of it model download and load.
The judging itself is 100 forward passes.

WHY THE JUDGE IS LOAD-BEARING, NOT A REFINEMENT
===============================================
Whole-word string matching decides right from wrong today, and it is wrong in
both directions:

    false wrong   the model answered correctly and the matcher disagreed
                  model 'The Leonberger is considered a giant dog breed.'
                  gold  'The Leonberger is a giant dog breed.'
                  These inflate the 74% error rate and, worse, they are then
                  classified into a failure mode - five of them currently sit
                  inside `locked_in`, the paper's headline category.

    false right   the gold string occurs in an answer that is not the answer.
                  Step 9b measured this and could not repair it. These are
                  hallucinations counted as correct, so they are invisible.

Every count in `claude/failure-mode-calibration.md` is marked provisional on
this step, including the interleaving shortfall that decides whether more
generation is needed.

HOW IT JUDGES - ONE FORWARD PASS, NOT GENERATION
================================================
The question put to the model is binary, so it is asked as a binary question
and answered by reading the logits at the first answer position:

    P(yes) vs P(no)

No sampling, no decoding loop, no parsing of free text. Three consequences,
all of them good for a load-bearing measurement:

    deterministic   no temperature, no seed sensitivity, nothing to re-run
    fast            one forward pass per question rather than dozens
    calibrated      the softmax over the two tokens is a CONFIDENCE, so
                    borderline cases can be separated from decided ones and
                    sent to a human instead of being silently resolved

Qwen3 is a thinking model; `enable_thinking=False` is passed to the chat
template so the assistant turn starts at the verdict rather than inside a
`<think>` block. Part A prints a complete prompt so the format is auditable
rather than described.

THE BATCHING CHECK
==================
Batched inference on a decoder-only model needs LEFT padding. With right
padding, `logits[:, -1]` is the logit after a run of pad tokens rather than
after the prompt, and every verdict is quietly garbage. Part B runs the same
eight questions at batch 1 and batch 8 and refuses to continue unless the
verdicts and probabilities match. This project has shipped silent measurement
bugs before; this one is cheap to rule out.

WHAT THIS DOES NOT DO
=====================
It does not judge the full set, and it does not decide the judge is good. It
produces a **blind worksheet** of the same 100 questions for a human pass. An
LLM judge validated only against the string matcher it was brought in to
replace has been validated against nothing. Step 13b computes agreement
against the human labels; the full run happens only if that agreement holds.

The worksheet is deliberately written WITHOUT the judge's verdicts - the same
discipline as Step 10c. A labeller who can see the machine's answer is not an
independent annotator.

OUTPUTS
=======
    outputs/step13a_judge_report.txt
    outputs/step13a_validation_worksheet.txt   <- label this one
    outputs/tables/step13a_judge_verdicts.csv  <- do NOT open before labelling
"""

import csv
import random
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

MODES_CSV = config.TAB_DIR / "step12c_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step13a_judge_report.txt"
SHEET_PATH = config.OUT_DIR / "step13a_validation_worksheet.txt"
VERDICT_CSV = config.TAB_DIR / "step13a_judge_verdicts.csv"

PHASEC_TARGETS = (("triviaqa", 3750), ("hotpotqa", 7500))

N_CORRECT = 50      # string-matched correct - finds the FALSE RIGHT cases
N_WRONG = 50        # string-matched wrong   - finds the FALSE WRONG cases
# Minimum per mode inside the wrong half, so the rare modes are represented at
# all. Interleaving is 3.2% of wrong answers; proportional sampling would put
# one or two in the worksheet and tell us nothing about them.
MODE_MIN = {"locked_in": 10, "interleaving": 10, "inconsistent": 10,
            "echo": 10, "degenerate": 5, "untestable": 5}

BATCH = 16
WRAP = 92           # worksheet width. The Step 10c sheet truncated at 62 and
                    # cost one unlabellable item in 60 - see the calibration doc.

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# The prompt
# ===========================================================================

SYSTEM = ("You grade answers to factual questions. You are strict about facts "
          "and lenient about wording.")

TEMPLATE = """Question: {question}

Reference answer(s): {golds}

Model answer: {answer}

Does the model answer state the same fact as one of the reference answers?
Ignore differences in wording, word order, extra explanation, capitalisation
and punctuation. Answer no if the model answer is about something else, does
not contain an answer, or only repeats the question.

Answer with one word, yes or no."""


def thinking_suffix(tokenizer) -> str:
    """The text that closes a thinking block, when the template uses one.

    Qwen3's assistant turn opens `<think>` and reasons before answering, so the
    first token after the generation prompt is reasoning, not a verdict - which
    would make the single-forward-pass reading meaningless. `enable_thinking=
    False` handles this, but only on template versions that accept the flag,
    and the pinned transformers version here is not the one Qwen ships against.

    So the close tag is appended directly when the template did not already do
    it. Guarded on the template actually using thinking, because appending
    `<think></think>` to a model that has never seen it would be worse than
    doing nothing.
    """
    tmpl = str(getattr(tokenizer, "chat_template", "") or "")
    return "<think>\n\n</think>\n\n" if "think" in tmpl.lower() else ""


def build_prompt(tokenizer, question: str, golds: str, answer: str) -> str:
    """Render the chat template with thinking mode off, whatever it takes."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(
                question=question.strip(), golds=golds.strip(),
                answer=answer.strip() or "(empty)")}]
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    if "</think>" not in text:
        text += thinking_suffix(tokenizer)
    return text


def verdict_token_ids(tokenizer) -> tuple:
    """Token ids for yes and no as they appear at the start of a reply.

    Checked rather than assumed: a variant that does not encode to a single
    token is dropped, and if either side ends up empty the script stops instead
    of judging against a token that cannot be produced.
    """
    def ids_for(words):
        out = []
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if len(enc) == 1:
                out.append((w, enc[0]))
        return out

    yes = ids_for(["yes", "Yes", "YES", " yes", " Yes"])
    no = ids_for(["no", "No", "NO", " no", " No"])
    return yes, no


# ===========================================================================

def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 13a: Qwen3 judge, smoke test and validation set")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    if not MODES_CSV.exists():
        say(f"\nMissing {MODES_CSV}. Run step12c first.")
        sys.exit(1)

    # ---- the rows ---------------------------------------------------------
    rows = {}
    with open(MODES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["dataset"], r["qid"])] = r
    say(f"  {len(rows):,} classified rows")

    # Questions, golds and answers. The mode CSV truncates the text, so the
    # full versions come from the dataset loader and the Phase C answers CSV.
    records = {}
    for ds, want in PHASEC_TARGETS:
        recs = data.load_records(ds, n=want * 2, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        for rec in kept[:want]:
            records[(ds, rec.qid)] = rec

    # Answers: the mode CSV does not carry them, so read Step 12's CSV, which
    # does. It is truncated at 300 characters, which is far beyond a 64-token
    # generation, so nothing is lost.
    answers = {}
    src_csv = config.TAB_DIR / "step12_phasec_modes.csv"
    with open(src_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            answers[(r["dataset"], r["qid"])] = r.get("answer", "")
    say(f"  {len(records):,} records, {len(answers):,} answers")

    # ---- stratified sample ------------------------------------------------
    rng = random.Random(config.SEED)
    corr = [k for k, v in rows.items() if v["correct"].lower() == "true"]
    wrong_by_mode = defaultdict(list)
    for k, v in rows.items():
        if v["correct"].lower() != "true":
            wrong_by_mode[v["mode"]].append(k)
    for v in wrong_by_mode.values():
        v.sort()
    corr.sort()

    picked = rng.sample(corr, min(N_CORRECT, len(corr)))
    remaining = N_WRONG
    for mode, n in MODE_MIN.items():
        pool = wrong_by_mode.get(mode, [])
        take = min(n, len(pool), remaining)
        if take:
            picked += rng.sample(pool, take)
            remaining -= take
    sample = [k for k in picked if k in records and k in answers]
    rng.shuffle(sample)          # so the worksheet order leaks nothing
    say(f"  {len(sample)} questions sampled "
        f"({sum(1 for k in sample if rows[k]['correct'].lower() == 'true')} "
        f"string-correct, "
        f"{sum(1 for k in sample if rows[k]['correct'].lower() != 'true')} "
        f"string-wrong)")

    # ---- model ------------------------------------------------------------
    say("")
    say(f"  Loading {config.MODEL_JUDGE} ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODEL_JUDGE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # LEFT padding. With right padding the last position is a pad token and
    # every verdict below would be read from the wrong place.
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_JUDGE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    say(f"  loaded in {time.time() - t0:.0f}s   "
        f"dtype {next(model.parameters()).dtype}   vocab {len(tok):,}")

    yes_ids, no_ids = verdict_token_ids(tok)
    say("")
    say("=" * 78)
    say("A. PROMPT AND VERDICT TOKENS")
    say("")
    say(f"  yes tokens : {yes_ids}")
    say(f"  no  tokens : {no_ids}")
    if not yes_ids or not no_ids:
        say("")
        say("  STOPPED - one side has no single-token form, so the logit")
        say("  comparison cannot be made. Switch to short generation instead.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    y_set = [i for _w, i in yes_ids]
    n_set = [i for _w, i in no_ids]

    demo_key = sample[0]
    demo = build_prompt(tok, records[demo_key].question,
                        " | ".join(records[demo_key].gold_answers[:5]),
                        answers[demo_key])
    say("")
    say("  One complete prompt, exactly as the model receives it:")
    say("  " + "-" * 74)
    for ln in demo.splitlines():
        say("  | " + ln)
    say("  " + "-" * 74)
    say("")
    say(f"  thinking block closed in the prompt : {'</think>' in demo}")
    say("  Part B tests this properly, by measuring how much probability mass")
    say("  actually sits on yes/no at the verdict position.")

    # ---- judging ----------------------------------------------------------
    @torch.no_grad()
    def judge(keys, batch_size):
        """Return {key: (p_yes, verdict, mass)}. One forward pass per question.

        `mass` is how much of the whole next-token distribution sits on yes or
        no at all. It is the check that the model is actually about to deliver
        a verdict here rather than open a reasoning block or start a sentence -
        and unlike grepping the prompt for `<think>`, it works for any model.
        """
        out = {}
        for i in range(0, len(keys), batch_size):
            chunk = keys[i: i + batch_size]
            prompts = [build_prompt(tok, records[k].question,
                                    " | ".join(records[k].gold_answers[:5]),
                                    answers[k]) for k in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            y = torch.logsumexp(lp[:, y_set], dim=-1)
            n = torch.logsumexp(lp[:, n_set], dim=-1)
            p_yes = torch.sigmoid(y - n)        # softmax over the two options
            mass = torch.exp(y) + torch.exp(n)  # share of the whole vocabulary
            for k, p, m in zip(chunk, p_yes.tolist(), mass.tolist()):
                out[k] = (p, "correct" if p >= 0.5 else "wrong", m)
        return out

    say("")
    say("=" * 78)
    say("B. TWO CHECKS BEFORE ANY VERDICT IS BELIEVED")
    say("")
    probe = sample[:8]
    b1 = judge(probe, 1)
    b8 = judge(probe, 8)

    # 1. Is the model answering the question it was asked?
    masses = sorted(b1[k][2] for k in probe)
    med = masses[len(masses) // 2]
    say(f"  yes/no share of the next-token distribution")
    say(f"    median {med:.3f}   min {masses[0]:.3f}   max {masses[-1]:.3f}")
    if med < 0.10:
        say("")
        say("  STOPPED - the model is not about to answer yes or no. Almost")
        say("  certainly a reasoning block opened at the verdict position, so")
        say("  the logit comparison is reading noise. Check the printed prompt")
        say("  in Part A for an unclosed <think>.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  The verdict position really is a verdict position.")

    # 2. Does batching change the answer?
    worst = max(abs(b1[k][0] - b8[k][0]) for k in probe)
    same = all(b1[k][1] == b8[k][1] for k in probe)
    say("")
    say("  8 questions at batch 1 and batch 8")
    say(f"    verdicts identical : {same}")
    say(f"    largest P(yes) gap : {worst:.2e}")
    if not same or worst > 1e-2:
        say("")
        say("  STOPPED - batching changes the answer. Left padding is not")
        say("  taking effect, so batched verdicts are read from pad positions.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  Batched inference is safe.")

    say("")
    say("=" * 78)
    say("C. JUDGING")
    say("")
    t_j = time.time()
    res = judge(sample, BATCH)
    elapsed = time.time() - t_j
    per_q = elapsed / max(1, len(sample))
    say(f"  {len(sample)} questions in {elapsed:.1f}s   {per_q*1000:.0f} ms/question")
    say("")
    n_full = 11120
    hrs = n_full * per_q / 3600
    say(f"  Full run: {n_full:,} questions at this rate")
    say(f"    {hrs:.2f} GPU-hours   ${hrs * 2.10:.2f} at $2.10/hr")
    say("    plus about 3 minutes of model load.")
    say("")
    say("  (Measured, not estimated. The Phase C cost was guessed at 10 s/q and")
    say("   came in at 2.01 - guessing is how that happened.)")

    # ---- what it says, against string matching ----------------------------
    say("")
    say("=" * 78)
    say("D. JUDGE vs STRING MATCHING")
    say("")
    tab = Counter()
    for k in sample:
        s_lab = "correct" if rows[k]["correct"].lower() == "true" else "wrong"
        tab[(s_lab, res[k][1])] += 1
    say("                    judge correct   judge wrong")
    for s_lab in ("correct", "wrong"):
        say(f"  string {s_lab:<8}      {tab[(s_lab,'correct')]:8d}      "
            f"{tab[(s_lab,'wrong')]:8d}")
    dis = tab[("correct", "wrong")] + tab[("wrong", "correct")]
    say("")
    say(f"  They disagree on {dis} of {len(sample)}.")
    say("  Disagreement is the POINT, not a problem - it is the quantity the")
    say("  judge was brought in to find. Whether the judge is right about them")
    say("  is what the human pass decides. Nothing here validates anything yet.")

    say("")
    say("  Confidence spread:")
    for lo, hi in ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01)):
        n = sum(1 for k in sample if lo <= res[k][0] < hi)
        bar = "#" * int(40 * n / max(1, len(sample)))
        say(f"    P(correct) {lo:.1f}-{hi:.1f}  {n:3d}  {bar}")
    band = sum(1 for k in sample if 0.3 <= res[k][0] < 0.7)
    say("")
    say(f"  {band} of {len(sample)} land in the undecided band. A judge that is")
    say("  confident everywhere is not necessarily right, but one that is")
    say("  undecided everywhere cannot carry a dataset.")

    # ---- the blind worksheet ---------------------------------------------
    sheet = []
    sheet.append("=" * WRAP)
    sheet.append("  TRIAGE - Step 13a: correctness validation worksheet")
    sheet.append("=" * WRAP)
    sheet.append("")
    sheet.append("  HOW TO LABEL")
    sheet.append("")
    sheet.append("  For each item, decide ONE thing: does the model answer state")
    sheet.append("  the same fact as one of the reference answers?")
    sheet.append("")
    sheet.append("      y   yes, it is correct")
    sheet.append("      n   no, it is wrong")
    sheet.append("      ?   the reference answer itself looks broken, or the")
    sheet.append("          question cannot be answered as asked")
    sheet.append("")
    sheet.append("  Ignore wording, word order, extra explanation, capitals and")
    sheet.append("  punctuation. An answer that only repeats the question, or")
    sheet.append("  never reaches an answer, is n.")
    sheet.append("")
    sheet.append("  Send back one line of 100 labels, in order, separated by")
    sheet.append("  spaces. Example:  y n n y ? n y y n ...")
    sheet.append("")
    sheet.append("  The judge's verdicts are NOT in this file, deliberately.")
    sheet.append("  Do not open step13a_judge_verdicts.csv before labelling -")
    sheet.append("  an annotator who has seen the machine's answer is not an")
    sheet.append("  independent annotator.")
    sheet.append("")
    for n, k in enumerate(sample, 1):
        rec = records[k]
        sheet.append("=" * WRAP)
        sheet.append(f"  #{n:<4} [{k[0]}]")
        sheet.append("")
        for tag, body in (("Q   ", rec.question),
                          ("GOLD", " | ".join(rec.gold_answers[:5])),
                          ("ANS ", answers[k] or "(empty)")):
            wrapped = textwrap.wrap(str(body).strip(), WRAP - 10) or ["(empty)"]
            sheet.append(f"  {tag}  {wrapped[0]}")
            for cont in wrapped[1:]:
                sheet.append(" " * 8 + cont)
        sheet.append("")
        sheet.append("  correct?  ____")
        sheet.append("")
    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHEET_PATH.write_text("\n".join(sheet) + "\n", encoding="utf-8")

    VERDICT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(VERDICT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "dataset", "qid", "string_correct", "mode",
                    "judge_p_correct", "judge_verdict", "yesno_mass"])
        for n, k in enumerate(sample, 1):
            w.writerow([n, k[0], k[1], rows[k]["correct"], rows[k]["mode"],
                        f"{res[k][0]:.4f}", res[k][1], f"{res[k][2]:.4f}"])

    say("")
    say("=" * 78)
    say(f"  worksheet : {SHEET_PATH}")
    say(f"  verdicts  : {VERDICT_CSV}   (do not open before labelling)")
    say(f"  wall clock: {time.time() - t0:.0f}s")
    say("=" * 78)
    say("")
    say("  Next: label the worksheet, then Step 13b computes agreement. The")
    say("  full 11,120-question run happens only if that agreement holds.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
