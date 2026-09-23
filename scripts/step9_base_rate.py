#!/usr/bin/env python3
"""
Step 9 - measure the base rate, and audit the prompt format
===========================================================

    modal run modal_app.py::run --script step9_base_rate.py

100 questions from each of the three datasets on LLaDA-8B in bf16. About 10
minutes and roughly $0.35 on an A100.

Two jobs
--------
**1. The base rate.** How often does the model get it wrong? That number sets
the sample size for Phase C. The original plan guessed 1500 per dataset, which
assumed a roughly 35% error rate; the real figure decides whether that is
generous or nowhere near enough. What matters is not the number of questions but
the number of WRONG answers, split three ways by failure mode.

**2. Audit the prompt.** Step 8's answers were suspicious:

    'Gerald.'      for "What was President Gerald Ford's middle name?"  (Rudolph)
    'Rep Records.' for "On which label did Chuck Berry record?"          (Chess)
    'Calato.<|eot_id|>'

The first grabbed a word out of its own question. The third leaked `<|eot_id|>`,
a LLaMA-3 chat-control token. `src/data.py` has only ever been tested against
LLaDA-MoE's tokenizer, and LLaDA-8B's chat template is different.

If the prompts are malformed then those are not hallucinations, they are
artefacts - and a hallucination-detection dataset built on artefacts is worthless
no matter how careful everything downstream is. So this script prints the exact
templated prompt, token by token, before it runs anything.

About the correctness labels here
---------------------------------
**These are rough, and deliberately so.** Matching an answer against TriviaQA's
alias list is a cheap approximation, not the judge. Stage 3 uses Qwen3-8B, and
Step 16 verifies 100 of its labels by hand, because TraceDet's 90% human
agreement came from an LLM judge and not from string matching.

Alias matching will be wrong in both directions: it marks "Chess Records"
correct when gold is "Chess" (fine), and marks "the Nile" wrong when gold is
"Nile River" (not fine). For a base-rate estimate that is good enough. For a
label that enters the paper it is not. The report says so in its own output so
the distinction survives being read six months from now.

Trajectories are saved, so these 300 runs are reusable - the Step 6b feature
screen should be re-run on them, since its length correlations were measured on
a deliberately length-diverse question set and will look different on real data.
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

MODEL_ID = config.MODEL_LLADA
N_PER_DATASET = 100
DATASETS = ("triviaqa", "hotpotqa", "commonsenseqa")
N_SHOW = 10
CHECKPOINT_EVERY = 25

REPORT_PATH = config.OUT_DIR / "step9_base_rate_report.txt"
CSV_PATH = config.TAB_DIR / "step9_per_question.csv"
TRAJ_SUBDIR = config.TRAJ_DIR / "step9" / config.RUN_TAG

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def flush():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")


# ===========================================================================
# Answer cleaning and rough matching
# ===========================================================================

# Control tokens seen leaking into LLaDA-8B output. `<|eot_id|>` is LLaMA-3's
# end-of-turn marker - LLaDA-8B inherits a LLaMA-family tokenizer.
SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT_RE = re.compile(r"[^\w\s]")


def clean_answer(text: str) -> str:
    """Strip control tokens and everything after the first one.

    A diffusion model fills a fixed 64-position block, so the answer is followed
    by padding. The first control token marks where the answer actually ended.
    """
    first = SPECIAL_RE.search(text)
    if first:
        text = text[:first.start()]
    return text.strip()


def normalise(text: str) -> str:
    """Lower-case, drop punctuation and articles, squash whitespace."""
    text = SPECIAL_RE.sub(" ", text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def rough_match(answer: str, golds: list[str]) -> bool:
    """True if any gold answer appears in the model's answer.

    Substring containment after normalisation. Crude: generous about extra words
    ("Chess Records" matches gold "Chess"), unforgiving about missing ones ("the
    Nile" fails gold "Nile River"). Adequate for a base rate, not for a label.
    """
    a = normalise(answer)
    if not a:
        return False
    return any(normalise(g) and normalise(g) in a for g in golds)


# ===========================================================================

def audit_prompt(tokenizer, rec, dataset: str) -> None:
    """Print the exact prompt the model will receive, token by token.

    The single most useful diagnostic available. A chat template that silently
    does the wrong thing produces fluent, wrong answers that look exactly like
    hallucinations - and there is no downstream check that would catch it.
    """
    say("")
    say(f"  PROMPT AUDIT - {dataset}")
    text = data.build_prompt_text(rec.question, rec.choices)
    say(f"    user text : {text!r}")

    ids = data.build_prompt_ids(tokenizer, rec)
    say(f"    tokens    : {len(ids)}")
    say(f"    decoded   : {tokenizer.decode(ids)!r}")

    # First and last few token ids with their surface forms - this is where a
    # missing or duplicated turn marker shows up.
    head = [(i, repr(tokenizer.decode([i]))) for i in ids[:6]]
    tail = [(i, repr(tokenizer.decode([i]))) for i in ids[-6:]]
    say(f"    first 6   : {head}")
    say(f"    last 6    : {tail}")
    say(f"    template? : {data.has_chat_template(tokenizer)}")


def main() -> None:
    import csv
    import numpy as np

    say("=" * 78)
    say("  TRIAGE - Step 9: base rate and prompt audit")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    for line in config.describe().splitlines():
        say(line)
    say(f"model  : {MODEL_ID}")
    say(f"dtype  : {model_utils.describe_dtype()}")

    if not model_utils.preflight_memory_check(model_size_gb=16.0, log=say):
        flush()
        sys.exit(1)

    model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
        MODEL_ID, log=say)
    mask_id = model_utils.resolve_mask_id(tokenizer, MODEL_ID)
    say(f"  config: {cfg_label}")

    TRAJ_SUBDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()

    for dataset in DATASETS:
        say("")
        say("=" * 78)
        say(dataset.upper())
        say("-" * 78)

        recs = data.load_records(dataset, n=N_PER_DATASET * 2, seed=config.SEED)
        kept, _exc, _rep = data_quality.clean_records(recs)
        recs = kept[:N_PER_DATASET]
        say(f"{len(recs)} questions after quality filtering")

        audit_prompt(tokenizer, recs[0], dataset)

        say("")
        say("  running...")
        n_right = 0

        for idx, rec in enumerate(recs, 1):
            path = TRAJ_SUBDIR / f"{dataset}_{rec.qid}.npz"
            if path.exists():
                try:
                    traj = logging_patch.Trajectory.load(path)
                    answer = tokenizer.decode(traj.final_ids.tolist())
                except Exception:
                    path.unlink(missing_ok=True)
                    traj = None
            else:
                traj = None

            if traj is None:
                prompt_text = data.build_prompt_text(rec.question, rec.choices)
                data.assert_evidence_withheld(prompt_text, rec)
                prompt_ids = data.build_prompt_ids(tokenizer, rec)
                answer, traj = logging_patch.generate_with_logging(
                    model, tokenizer, prompt_ids,
                    gen_length=config.GEN_LENGTH,
                    steps=config.DENOISING_STEPS,
                    block_length=config.BLOCK_LENGTH,
                    temperature=0.0,
                    question=rec.question, question_id=rec.qid,
                    quant_config=cfg_label, seed=config.SEED,
                    mask_id=mask_id,
                )
                traj.save(path)

            clean = clean_answer(answer)
            correct = rough_match(clean, rec.gold_answers)
            n_right += int(correct)

            masked = traj.mask_state == 1
            content = traj.content_mask() & masked
            pad_frac = (1.0 - content.sum() / masked.sum()) if masked.any() else float("nan")
            ans_tokens = sum(1 for t in traj.final_ids.tolist()
                             if t not in (traj.eos_id, traj.mask_id))

            rows.append(dict(
                dataset=dataset, qid=rec.qid, question=rec.question[:120],
                answer=clean[:120], gold=" | ".join(rec.gold_answers[:4]),
                rough_correct=correct, ans_tokens=ans_tokens,
                pad_frac=round(float(pad_frac), 4),
                ent_content=round(float(traj.entropy[content].mean()), 4)
                             if content.any() else float("nan"),
                flips=int(traj.flips_per_round().sum()),
                regret=int(traj.regret_per_round().sum()),
                elapsed_s=round(traj.elapsed_s, 2),
            ))

            if idx % CHECKPOINT_EVERY == 0:
                say(f"    {idx}/{len(recs)}  rough accuracy so far "
                    f"{n_right/idx:.1%}")
                flush()

        acc = n_right / max(len(recs), 1)
        say("")
        say(f"  ROUGH ACCURACY : {acc:.1%}   "
            f"({n_right} right, {len(recs) - n_right} wrong)")

        say("")
        say(f"  sample answers (first {N_SHOW}) - READ THESE:")
        for r in [x for x in rows if x["dataset"] == dataset][:N_SHOW]:
            mark = "OK  " if r["rough_correct"] else "WRONG"
            say(f"    [{mark}] Q: {r['question'][:64]}")
            say(f"             A: {r['answer'][:64]!r}")
            say(f"             gold: {r['gold'][:64]}")

    elapsed = time.time() - t_start

    # =====================================================================
    say("")
    say("=" * 78)
    say("BASE RATE SUMMARY")
    say("")
    say("  dataset         n    right   wrong   error rate")
    say("  --------------  ---  -----   -----   ----------")
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        if not rs:
            continue
        right = sum(1 for r in rs if r["rough_correct"])
        say(f"  {dataset:<14}  {len(rs):3d}  {right:5d}   {len(rs)-right:5d}   "
            f"{1 - right/len(rs):9.1%}")

    say("")
    say("  THESE ARE ROUGH. Alias substring matching, not the Qwen3 judge.")
    say("  It over-credits ('Chess Records' matches gold 'Chess') and")
    say("  under-credits ('the Nile' fails gold 'Nile River'). Good enough to")
    say("  size the sample; NOT a label that may enter the paper. Stage 3 does")
    say("  the real labelling and Step 16 verifies 100 of those by hand.")

    # ---- what the sample size should be ----------------------------------
    say("")
    say("SAMPLE SIZE IMPLICATION")
    say("")
    say("  What Phase C needs is not questions but WRONG answers, split three")
    say("  ways by failure mode, with enough in each cell to compare AUROCs.")
    say("  Aiming for ~150 wrong answers per failure mode per dataset:")
    say("")
    say("  dataset         error rate   questions needed for 450 wrong")
    say("  --------------  ----------   ------------------------------")
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        if not rs:
            continue
        err = 1 - sum(1 for r in rs if r["rough_correct"]) / len(rs)
        need = int(450 / err) if err > 0.01 else 99999
        say(f"  {dataset:<14}  {err:9.1%}   {need:>6d}"
            f"{'   (capped - error rate too low)' if err <= 0.01 else ''}")
    say("")
    say("  The three modes will NOT split evenly. If locked-in errors are only")
    say("  20% of failures, that column needs 5x the questions the arithmetic")
    say("  above suggests. Step 10 measures the split; do not fix the sample")
    say("  size until then.")

    # ---- Warning #2 on real data -----------------------------------------
    say("")
    say("WARNING #2 ON REAL DATA")
    say("")
    say("  dataset         mean answer tokens   mean padding fraction")
    say("  --------------  ------------------   ---------------------")
    for dataset in DATASETS:
        rs = [r for r in rows if r["dataset"] == dataset]
        if not rs:
            continue
        say(f"  {dataset:<14}  {sum(r['ans_tokens'] for r in rs)/len(rs):18.1f}"
            f"   {sum(r['pad_frac'] for r in rs)/len(rs):21.1%}")

    say("")
    say("TIMING AND COST")
    say(f"  {len(rows)} questions in {elapsed/60:.1f} min")
    gen = [r["elapsed_s"] for r in rows]
    say(f"  mean {sum(gen)/len(gen):.2f}s per question")
    say(f"  Phase C at this rate (9,000): "
        f"{sum(gen)/len(gen)*9000/3600:.1f} GPU-hours")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say("")
    say(f"  per-question CSV : {CSV_PATH}")
    say(f"  trajectories     : {TRAJ_SUBDIR}  ({len(rows)} files, reusable)")
    say("=" * 78)
    flush()
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
