#!/usr/bin/env python3
"""
Step 6 - reconstruction across 50 questions  [HARD GATE]
========================================================

    python scripts/step6_reconstruct_50.py

Charter mandatory sanity check: rebuild the final answer from the logged
trajectory and assert it matches the model's actual output character for
character. **If this fails, every downstream number in the project is garbage.**

50 out of 50 must pass. Not 49.

Why a purpose-built question set rather than TriviaQA
-----------------------------------------------------
The dataset loaders arrive in Step 7. More importantly, a sample drawn from one
dataset would not stress the cases most likely to break reconstruction. These 50
are chosen to span them deliberately:

  * 30 short factual questions - one to five token answers, so the remaining
    ~60 positions are EOS padding. This is the case Step 5 never exercised and
    the one that dominates TriviaQA.
  * 8 medium answers - a sentence or two.
  * 6 with numbers, punctuation, or non-ASCII text, where tokenisation and
    decoding are most likely to disagree.
  * 6 obscure or unanswerable, to produce some wrong answers. Those matter for
    the regret baseline described below.

Three things this produces beyond the pass/fail
-----------------------------------------------
**1. Warning #2, quantified at last.** Step 5 measured 1.0% EOS contamination on
a question that filled all 64 positions. With short answers it should be far
higher, and the gap between "mean entropy over all masked positions" and "mean
entropy over content positions only" is the number that justifies the exclusion
rule in `features.py`.

**2. A regret baseline.** Step 4 found regret rising to 44% of revealed positions
on a single question whose answer contained a factual error. That is one data
point and means nothing on its own. Regret across 50 questions gives the
distribution - and if it is uniformly high regardless of correctness, Finding B
is measuring imperfect self-reconstruction rather than anything interesting.
Better to learn that here than after Phase C.

**3. A timing estimate.** Mean seconds per question on this hardware, which
combined with the Step 8 benchmark on LLaDA-8B decides whether the Phase C
budget is 55 GPU-hours or 250.

Resume
------
Trajectories are written one file per question. A rerun skips any question whose
file already exists and verifies. Phase C depends on exactly this behaviour, so
it is exercised here where a failure costs minutes rather than days.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step6_reconstruct_50.py")
    sys.exit(1)


# ===========================================================================
# The question set - 50 items, `(id, category, text)`
# ===========================================================================

QUESTIONS = [
    # --- 30 short factual: 1-5 token answers, heavy EOS padding -----------
    ("s01", "short", "What is the capital of Japan? Answer in one word."),
    ("s02", "short", "Who wrote the play Hamlet? Answer with a name only."),
    ("s03", "short", "What is the chemical symbol for gold? One word."),
    ("s04", "short", "How many continents are there? Answer with a number."),
    ("s05", "short", "What is the largest ocean on Earth? One word."),
    ("s06", "short", "Which planet is known as the Red Planet? One word."),
    ("s07", "short", "Who painted the Mona Lisa? Name only."),
    ("s08", "short", "What is the tallest mountain in the world? Name only."),
    ("s09", "short", "In which year did the Second World War end? Number only."),
    ("s10", "short", "What is the currency of the United Kingdom? One word."),
    ("s11", "short", "Which gas do plants absorb from the air? One word."),
    ("s12", "short", "What is the longest river in Africa? Name only."),
    ("s13", "short", "Who developed the theory of general relativity? Name only."),
    ("s14", "short", "What is the smallest prime number? Number only."),
    ("s15", "short", "Which country gifted the Statue of Liberty to the USA? One word."),
    ("s16", "short", "What is the hardest natural substance? One word."),
    ("s17", "short", "How many strings does a standard violin have? Number only."),
    ("s18", "short", "What is the capital of Australia? One word."),
    ("s19", "short", "Which element has the atomic number 1? One word."),
    ("s20", "short", "Who wrote the novel Things Fall Apart? Name only."),
    ("s21", "short", "What is the boiling point of water in Celsius? Number only."),
    ("s22", "short", "Which sea separates Europe and Africa? Name only."),
    ("s23", "short", "What is the largest mammal on Earth? Name only."),
    ("s24", "short", "In which city is the Colosseum located? One word."),
    ("s25", "short", "What is the square root of 144? Number only."),
    ("s26", "short", "Which vitamin is produced when skin is exposed to sunlight? One word."),
    ("s27", "short", "Who was the first person to walk on the Moon? Name only."),
    ("s28", "short", "What is the national language of Brazil? One word."),
    ("s29", "short", "How many players are on a football team on the field? Number only."),
    ("s30", "short", "What is the freezing point of water in Fahrenheit? Number only."),

    # --- 8 medium: a sentence or two --------------------------------------
    ("m01", "medium", "In two sentences, explain why the sky appears blue."),
    ("m02", "medium", "Briefly describe what photosynthesis does."),
    ("m03", "medium", "In two sentences, explain what causes ocean tides."),
    ("m04", "medium", "Briefly explain the difference between weather and climate."),
    ("m05", "medium", "In two sentences, describe what DNA is."),
    ("m06", "medium", "Briefly explain why ice floats on water."),
    ("m07", "medium", "In two sentences, explain what inflation means in economics."),
    ("m08", "medium", "Briefly describe how a vaccine works."),

    # --- 6 tokenisation stress: numbers, punctuation, non-ASCII ------------
    ("t01", "tricky", "Write the number 1,234,567 in words."),
    ("t02", "tricky", "What is 17 multiplied by 23? Show only the number."),
    ("t03", "tricky", "How do you say 'thank you' in Japanese? Give the Japanese script."),
    ("t04", "tricky", "Write the chemical formula for sulfuric acid."),
    ("t05", "tricky", "What is the value of pi to five decimal places?"),
    ("t06", "tricky", "Spell the word 'accommodate' letter by letter, separated by hyphens."),

    # --- 6 obscure or unanswerable: expected to produce wrong answers ------
    ("o01", "obscure", "What was the exact population of Reykjavik on 3 March 1987?"),
    ("o02", "obscure", "Who won the 1923 Vardar Valley regional chess championship?"),
    ("o03", "obscure", "What is the middle name of the third assistant director of Casablanca?"),
    ("o04", "obscure", "How many bricks were used to build the Great Wall of China?"),
    ("o05", "obscure", "What did Napoleon eat for breakfast on 14 June 1800?"),
    ("o06", "obscure", "Name the 47th moon of Saturn discovered in 2004."),
]

CHECKPOINT_EVERY = 10
REPORT_PATH = config.OUT_DIR / "step6_reconstruct_report.txt"
CSV_PATH = config.TAB_DIR / "step6_per_question.csv"
TRAJ_SUBDIR = config.TRAJ_DIR / "step6"

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def flush_report():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")


def build_prompt_ids(tokenizer, question: str):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def main() -> None:
    import numpy as np

    say("=" * 78)
    say("  TRIAGE - Step 6: reconstruction across 50 questions   [HARD GATE]")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    for line in config.describe().splitlines():
        say(line)
    if not model_utils.preflight_memory_check(model_size_gb=15.0, log=say):
        flush_report()
        sys.exit(1)

    try:
        model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(log=say)
    except RuntimeError as exc:
        say("\nLOAD FAILED")
        say(str(exc))
        flush_report()
        sys.exit(1)

    TRAJ_SUBDIR.mkdir(parents=True, exist_ok=True)
    say("-" * 78)
    say(f"{len(QUESTIONS)} questions, trajectories -> {TRAJ_SUBDIR}")
    say(f"resume: existing files are reused, so a rerun continues where it stopped")
    say("")
    say("  id    cat      recon  ansTok  EOS%   H(all)  H(cont)  flips  regret  s")
    say("  ----  -------  -----  ------  -----  ------  -------  -----  ------  ----")

    rows = []
    t_start = time.time()
    n_generated = 0

    for idx, (qid, cat, text) in enumerate(QUESTIONS, 1):
        path = TRAJ_SUBDIR / f"{qid}.npz"

        # ---- resume: reuse an existing trajectory ------------------------
        if path.exists():
            try:
                traj = logging_patch.Trajectory.load(path)
                reused = True
            except Exception:
                path.unlink(missing_ok=True)
                traj = None
                reused = False
        else:
            traj = None
            reused = False

        if traj is None:
            prompt_ids = build_prompt_ids(tokenizer, text)
            _, traj = logging_patch.generate_with_logging(
                model, tokenizer, prompt_ids,
                gen_length=config.GEN_LENGTH,
                steps=config.DENOISING_STEPS,
                block_length=config.BLOCK_LENGTH,
                temperature=0.0,
                question=text, question_id=qid,
                quant_config=cfg_label, seed=config.SEED,
            )
            traj.save(path)
            n_generated += 1

        # ---- THE GATE: reconstruct from the log alone --------------------
        rebuilt = traj.reconstruct_final()
        ids_ok = bool(np.array_equal(rebuilt, traj.final_ids))
        text_ok = (tokenizer.decode(rebuilt.tolist())
                   == tokenizer.decode(traj.final_ids.tolist()))
        recon_ok = ids_ok and text_ok

        # ---- statistics ---------------------------------------------------
        masked = traj.mask_state == 1
        content = traj.content_mask() & masked
        eos_frac = 1.0 - (content.sum() / masked.sum()) if masked.any() else float("nan")
        h_all = float(traj.entropy[masked].mean()) if masked.any() else float("nan")
        h_cont = float(traj.entropy[content].mean()) if content.any() else float("nan")

        # Answer length in real content tokens, ignoring EOS and role markers.
        final_list = traj.final_ids.tolist()
        ans_tokens = sum(1 for t in final_list
                         if t not in (traj.eos_id, traj.mask_id))

        flips = int(traj.flips_per_round().sum())
        regret = int(traj.regret_per_round().sum())
        settle_med = int(np.median(traj.revealed_at()))

        rows.append(dict(id=qid, category=cat, recon_ok=recon_ok,
                         ans_tokens=ans_tokens, eos_frac=eos_frac,
                         h_all=h_all, h_content=h_cont, flips=flips,
                         regret=regret, settle_median=settle_med,
                         elapsed_s=traj.elapsed_s, reused=reused,
                         answer=tokenizer.decode(final_list)))

        say(f"  {qid:<4}  {cat:<7}  {'OK ' if recon_ok else 'FAIL':<5}  "
            f"{ans_tokens:6d}  {eos_frac:5.1%}  {h_all:6.3f}  {h_cont:7.3f}  "
            f"{flips:5d}  {regret:6d}  {traj.elapsed_s:4.0f}")

        if idx % CHECKPOINT_EVERY == 0:
            flush_report()

    elapsed = time.time() - t_start

    # ===================================================================
    # Aggregate
    # ===================================================================
    n_pass = sum(1 for r in rows if r["recon_ok"])
    say("")
    say("=" * 78)
    say(f"RECONSTRUCTION : {n_pass} / {len(rows)} exact")

    if n_pass < len(rows):
        say("")
        say("  FAILURES:")
        for r in rows:
            if not r["recon_ok"]:
                say(f"    {r['id']}  {r['answer'][:60]!r}")

    def summarise(key, label, fmt="{:.3f}"):
        vals = [r[key] for r in rows if r[key] == r[key]]
        if not vals:
            return
        vals_sorted = sorted(vals)
        say(f"  {label:<28} min {fmt.format(vals_sorted[0])}  "
            f"median {fmt.format(vals_sorted[len(vals)//2])}  "
            f"max {fmt.format(vals_sorted[-1])}")

    say("")
    say("WARNING #2 - EOS/PADDING CONTAMINATION")
    summarise("eos_frac", "padding fraction", "{:.1%}")
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, rs in by_cat.items():
        fr = [r["eos_frac"] for r in rs if r["eos_frac"] == r["eos_frac"]]
        diffs = [r["h_all"] - r["h_content"] for r in rs
                 if r["h_content"] == r["h_content"]]
        if fr:
            say(f"    {cat:<8} n={len(rs):2d}  padding {sum(fr)/len(fr):5.1%}   "
                f"entropy shift {sum(diffs)/len(diffs):+.4f}"
                if diffs else "")

    say("")
    say("REGRET BASELINE (Finding B)")
    summarise("regret", "regret per question", "{:.0f}")
    for cat, rs in by_cat.items():
        vals = [r["regret"] for r in rs]
        say(f"    {cat:<8} n={len(rs):2d}  mean regret {sum(vals)/len(vals):6.1f}")
    say("")
    say("  Read this carefully. If 'obscure' (mostly wrong answers) shows")
    say("  markedly higher regret than 'short' (mostly right), Finding B is")
    say("  tracking correctness and is worth pursuing. If every category looks")
    say("  the same, regret is measuring imperfect self-reconstruction and")
    say("  should be logged but not built upon.")

    say("")
    say("OTHER SIGNALS")
    summarise("flips", "flips per question", "{:.0f}")
    summarise("settle_median", "median settle round", "{:.0f}")
    summarise("ans_tokens", "answer length (tokens)", "{:.0f}")

    say("")
    say("TIMING")
    gen_times = [r["elapsed_s"] for r in rows if not r["reused"]]
    if gen_times:
        mean_s = sum(gen_times) / len(gen_times)
        say(f"  generated {len(gen_times)} questions, mean {mean_s:.1f}s each")
        say(f"  wall clock {elapsed/60:.1f} min")
        say(f"  Phase C at this rate (9,000 questions): "
            f"{mean_s * 9000 / 3600:.0f} GPU-hours")
        say("  NOTE: this is LLaDA-MoE (~1.4B active). LLaDA-8B is dense and")
        say("  will be slower. Step 8 benchmarks it before Phase C commits quota.")
    else:
        say("  all questions reused from disk; rerun after deleting "
            f"{TRAJ_SUBDIR} for fresh timings")

    # ---- CSV for inspection ----------------------------------------------
    import csv
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say("")
    say(f"  per-question CSV: {CSV_PATH}")

    say("")
    say("=" * 78)
    if n_pass == len(rows):
        say("  PASS - all 50 reconstruct exactly. The log is trustworthy.")
        say("         Phase A is complete. Proceed to Step 7 (dataset loaders).")
    else:
        say("  FAIL - reconstruction is not exact. STOP. Every downstream")
        say("         number depends on this. Send the report.")
    say("=" * 78)
    flush_report()
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
