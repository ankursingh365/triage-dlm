#!/usr/bin/env python3
"""
Step 11 - the generation config sweep
=====================================

    modal run modal_app.py::run --script step11_config_sweep.py

A100, roughly 15 minutes, about $0.50.

WHY THIS STEP EXISTS
====================
Reading 60 trajectories by hand in Step 10c turned up something neither the
base rates nor the failure-mode counts could show. A large share of the "wrong
answers" are not wrong answers. They are **broken text**:

    'Londonasgow'         gold: London           ("London" + "Glasgow")
    'House sprow'         gold: Chicken          ("House sparrow", mangled)
    'Marco van Bast.'     gold: Ryan Guno Babel  ("van Basten", truncated)
    'Lgor Chopovsky.'     gold: Tchaikovsky      ("Igor" + "Chopin" + "-ovsky")
    'Bangiff'             gold: LLANBERIS
    'Theland Islands'     gold: Tonga
    'H.'  'The.'  'L.'  'Not one.'  'BBC.'

A hallucination-detection dataset cannot be built on these. They are not the
model believing something false; they are the decoder failing to emit a word.
Roughly 40% of the hand-read sample looks like this. Every base rate, every
failure-mode share and every AUROC computed on top of it inherits the problem.

THE MECHANISM
=============
Step 10 measured it without either of us recognising what it meant:

    answer positions  revealed at round  26.7  of 32
    EOS/pad positions revealed at round  13.6

`get_num_transfer_tokens` splits the 64 masked positions evenly across 32
rounds: **2 positions revealed per round, fixed in advance**. Low-confidence
remasking then reveals the easiest positions first, and EOS padding is trivially
easy.

For a four-token TriviaQA answer that means 60 EOS positions and 4 content
positions. Sixty EOS positions at two per round is **thirty rounds spent
revealing padding**. The answer itself gets the leftovers - one or two rounds,
out of a nominal thirty-two.

That is the whole pathology, and it explains every symptom at once:

* **Garbage early trajectories.** At rounds 0-25 the answer positions are still
  masked and the model is predicting into a sea of masks. `Thewar name of
  thewar of Hwarwar` is not a wrong belief, it is a prediction with no context.

* **Mashed-together words.** Two positions are revealed in the SAME round, each
  chosen independently, neither conditioned on the other. Position 0 picks
  "London", position 1 picks the "asgow" of "Glasgow", and the result is
  `Londonasgow`. With one position per round, position 1 would see "London"
  already committed and pick something consistent.

* **No locked-in errors.** Nothing can lock in early when nothing is decided
  until the last two rounds. Step 10's locked_in = 7 of 160 was measuring the
  absence of a trajectory, not the absence of a failure mode.

* **CommonsenseQA looking healthier.** Its answers are four tokens behind a
  fixed template, and the template positions are decided early.

WHAT THE REFERENCE DOES
=======================
ML-GSAI/LLaDA's `generate.py` calls:

    generate(model, input_ids, attention_mask, steps=128, gen_length=128,
             block_length=32, temperature=0., cfg_scale=0.,
             remasking='low_confidence')

Two things differ from this project's settings:

    parameter      reference          ours          effect
    -----------    ---------------    ----------    -------------------------
    steps          == gen_length      gen/2         2 positions per round,
                                                    revealed independently
    block_length   gen_length / 4     == gen        one block: no semi-
                                                    autoregressive structure

The LLaDA repository's own GUIDELINES.md is explicit about the second one: for
the **Instruct** model, "low-confidence remasking with semi-autoregressive-
padding effectively mitigates the issue of generating an excessively high
proportion of <EOS> tokens." LLaDA-8B-Instruct is SFT-trained with EOS padding
to a fixed length, so single-block low-confidence remasking is exactly the
configuration that floods the output with EOS - which is what the 26.7-vs-13.6
split is showing.

We are running the configuration the guidelines warn against, at half the
denoising steps.

THE DESIGN
==========
A 2x2 over the two deviations, so each can be attributed separately, plus a
shorter-budget variant:

    0  CURRENT     gen 64  steps 32  block 64    1 block, 2 pos/round
    1  STEPS       gen 64  steps 64  block 64    1 block, 1 pos/round
    2  BLOCKS      gen 64  steps 32  block 32    2 blocks, 2 pos/round
    3  BOTH        gen 64  steps 64  block 32    2 blocks, 1 pos/round  <- reference style
    4  SHORT       gen 32  steps 32  block 32    1 block, 1 pos/round

Config 4 tests a separate idea: with a four-token answer, 64 positions means
94% padding. A shorter budget may help even at one position per round, and it
costs half the compute in Phase C - which matters, because config 3 doubles the
step count and therefore doubles the bill.

WHAT IS MEASURED
================
Error rate is the least interesting number here. These four matter more:

    co_reveal      fraction of ADJACENT content positions revealed in the same
                   round. This is the direct cause of `Londonasgow`, and it is
                   mechanical rather than heuristic: under steps == gen_length
                   it is 0 by construction.

    content_reveal mean reveal round of content positions, as a fraction of
                   total rounds. Near 1.0 means the answer is decided at the
                   very end and there is no trajectory to detect anything in.

    eos_share      fraction of final positions holding EOS. The padding tax.

    commit_conf    the model's own probability at the token it committed,
                   averaged over content positions. Broken text is text the
                   model was not confident about.

And eight sample answers per config, printed in full. **The samples decide
this, not the statistics.** If config 3's answers read like English and config
0's read like `Bangiff`, that is the result, whatever the error rates say.

WHAT HAPPENS NEXT
=================
Whichever config wins becomes the project's setting, `src/config.py` is
updated, and Steps 9, 9b, 10 and 10c are re-run on regenerated trajectories.
That is unwelcome but unavoidable: those four steps measured a decoder
artefact. The 300 existing trajectories stay on the Volume for the record and
for a methods-section paragraph about why the configuration was changed.

Re-running is cheap. Step 9 was about $0.35.
"""

import csv
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, model_utils, logging_patch, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

REPORT_PATH = config.OUT_DIR / "step11_config_sweep_report.txt"
CSV_PATH = config.TAB_DIR / "step11_config_sweep.csv"
TRAJ_ROOT = config.TRAJ_DIR / "step11"

# TriviaQA leads because its answers are short, which is where the pathology is
# worst; HotpotQA is included so the long-answer case is not left untested.
SAMPLE = (("triviaqa", 40), ("hotpotqa", 15))
N_SHOW = 8

# (label, gen_length, steps, block_length)
CONFIGS = [
    ("0 CURRENT", 64, 32, 64),
    ("1 STEPS",   64, 64, 64),
    ("2 BLOCKS",  64, 32, 32),
    ("3 BOTH",    64, 64, 32),
    ("4 SHORT",   32, 32, 32),
]

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Scoring - identical rules to Step 9b so the numbers are comparable
# ===========================================================================

SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
PUNCT_RE = re.compile(r"[^\w\s]")
ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Cut at the first control token. Everything after is block padding."""
    first = SPECIAL_RE.search(text)
    if first:
        text = text[:first.start()]
    return text.strip()


def normalise(text: str) -> str:
    text = SPECIAL_RE.sub(" ", str(text)).lower()
    text = PUNCT_RE.sub(" ", text)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def match_wordbound(answer: str, golds: list) -> bool:
    a = normalise(answer)
    if not a:
        return False
    for g in golds:
        gn = normalise(g)
        if gn and re.search(r"\b" + re.escape(gn) + r"\b", a):
            return True
    return False


# ===========================================================================
# The diagnostics that actually decide this
# ===========================================================================

def diagnose(traj) -> dict:
    """Measure the reveal schedule's effect on the answer itself."""
    final = traj.final_ids.tolist()
    content = [i for i, t in enumerate(final)
               if t not in (traj.eos_id, traj.mask_id)]
    steps = traj.steps
    rev = traj.revealed_at()

    if not content:
        return {"n_content": 0, "co_reveal": float("nan"),
                "content_reveal": float("nan"),
                "eos_share": 1.0, "commit_conf": float("nan")}

    # THE KEY MEASURE. Adjacent content positions revealed in the same round
    # were chosen independently, neither conditioned on the other. That is
    # precisely how "London" and "Glasgow" become "Londonasgow". Under
    # steps == gen_length this is 0 by construction, so it is a direct readout
    # of the deviation rather than a heuristic about text quality.
    adjacent = [(content[k], content[k + 1]) for k in range(len(content) - 1)
                if content[k + 1] == content[k] + 1]
    co = (sum(1 for a, b in adjacent if rev[a] == rev[b]) / len(adjacent)
          if adjacent else float("nan"))

    # How late the answer is decided, as a fraction of the run. Near 1.0 means
    # there is no trajectory before the commitment and nothing to detect in.
    content_reveal = float(np.mean([rev[i] for i in content])) / max(steps - 1, 1)

    # The padding tax.
    eos_share = 1.0 - len(content) / traj.gen_length

    # The model's own probability at the token it committed. Broken text is
    # text the model was not confident about, so this separates "confidently
    # wrong" (which the paper wants) from "decoder failure" (which it does not).
    commit_conf = float(np.mean([traj.pred_probs[int(rev[i]), i] for i in content]))

    return {"n_content": len(content), "co_reveal": co,
            "content_reveal": content_reveal, "eos_share": eos_share,
            "commit_conf": commit_conf}


# ===========================================================================

def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 11: generation config sweep")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say("")
    say("  Reference (ML-GSAI/LLaDA generate.py):")
    say("      steps=128  gen_length=128  block_length=32  remasking='low_confidence'")
    say("  This project until now:")
    say(f"      steps={config.DENOISING_STEPS}   gen_length={config.GEN_LENGTH}"
        f"   block_length={config.BLOCK_LENGTH}")
    say("")
    say("  steps = gen_length/2 reveals TWO positions per round, chosen")
    say("  independently. block_length = gen_length removes the semi-")
    say("  autoregressive structure that LLaDA's GUIDELINES.md recommends for")
    say("  the Instruct model specifically, to stop it flooding the output")
    say("  with EOS. Both deviations are tested separately below.")

    # ---- data -------------------------------------------------------------
    records = []
    for dataset, n in SAMPLE:
        recs = data.load_records(dataset, n=n * 3, seed=config.SEED)
        kept, _exc, _rep = data_quality.clean_records(recs)
        for rec in kept[:n]:
            records.append((dataset, rec))
    say("")
    say(f"  {len(records)} questions "
        + ", ".join(f"{d} {n}" for d, n in SAMPLE))

    # ---- model ------------------------------------------------------------
    say("")
    say("  Loading LLaDA-8B...")
    model, tokenizer, cfg_label = model_utils.load_model_and_tokenizer(
        config.MODEL_LLADA, log=say)
    mask_id = model_utils.resolve_mask_id(tokenizer, config.MODEL_LLADA)
    say(f"  loaded: {cfg_label}   mask_id {mask_id}")

    rows = []
    for label, gen_len, steps, block_len in CONFIGS:
        t0 = time.time()
        say("")
        say("=" * 78)
        say(f"  {label}   gen {gen_len}  steps {steps}  block {block_len}"
            f"   ({gen_len // block_len} block(s), "
            f"{gen_len / steps:.2g} pos/round)")
        say("=" * 78)

        out_dir = TRAJ_ROOT / label.split()[1].lower()
        for dataset, rec in records:
            # Same prompt path as Step 9, so the only thing that differs
            # between this sweep and the existing base rates is the decoder
            # configuration under test.
            prompt_ids = data.build_prompt_ids(tokenizer, rec)

            answer, traj = logging_patch.generate_with_logging(
                model, tokenizer, prompt_ids,
                gen_length=gen_len, steps=steps, block_length=block_len,
                temperature=0.0,
                question=rec.question, question_id=rec.qid,
                quant_config=cfg_label, seed=config.SEED, mask_id=mask_id,
            )
            traj.save(out_dir / f"{dataset}_{rec.qid}.npz")

            clean = clean_answer(answer)
            d = diagnose(traj)
            rows.append(dict(cfg=label, gen=gen_len, steps=steps,
                             block=block_len, dataset=dataset, qid=rec.qid,
                             question=rec.question[:90], answer=clean[:90],
                             gold=" | ".join(rec.gold_answers[:3])[:60],
                             correct=match_wordbound(clean, rec.gold_answers),
                             **d))

        elapsed = time.time() - t0
        mine = [r for r in rows if r["cfg"] == label]
        say(f"  {len(mine)} questions in {elapsed/60:.1f} min")

        # ---- samples. THESE DECIDE IT, not the statistics ------------------
        say("")
        say(f"  --- sample answers ({N_SHOW}) ---")
        for r in mine[:N_SHOW]:
            flag = "OK " if r["correct"] else "   "
            say(f"   {flag} {r['answer'][:66]!r}")
            say(f"        gold: {r['gold'][:56]}")

    # =====================================================================
    say("")
    say("=" * 78)
    say("RESULTS")
    say("")
    say("  config      err%   co_reveal  content_reveal  eos_share  commit_conf  ans_len")
    say("  ----------  -----  ---------  --------------  ---------  -----------  -------")

    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["cfg"]].append(r)

    def m(rs, key):
        v = [r[key] for r in rs if not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.mean(v)) if v else float("nan")

    for label, _g, _s, _b in CONFIGS:
        rs = by_cfg[label]
        if not rs:
            continue
        err = 1 - sum(r["correct"] for r in rs) / len(rs)
        say(f"  {label:<10}  {err:4.0%}   {m(rs,'co_reveal'):9.2f}  "
            f"{m(rs,'content_reveal'):14.2f}  {m(rs,'eos_share'):9.2f}  "
            f"{m(rs,'commit_conf'):11.2f}  {m(rs,'n_content'):7.1f}")

    say("")
    say("  co_reveal      adjacent answer positions revealed in the SAME round,")
    say("                 chosen independently of each other. This is the direct")
    say("                 cause of 'Londonasgow'. Should be 0.00 wherever")
    say("                 steps == gen_length. Lower is better.")
    say("  content_reveal mean reveal round of answer positions / total rounds.")
    say("                 Near 1.00 means the answer is decided in the last")
    say("                 rounds and there is no trajectory to detect in.")
    say("                 Lower is better.")
    say("  eos_share      share of positions holding EOS padding. Lower is")
    say("                 better; it is the budget not spent on the answer.")
    say("  commit_conf    the model's probability at the token it committed.")
    say("                 Higher is better - broken text is unconfident text.")

    # ---- the 2x2, read off ------------------------------------------------
    say("")
    say("=" * 78)
    say("ATTRIBUTION - which deviation matters?")
    say("")
    say("            block 64 (1 blk)   block 32 (2 blk)")
    for steps_val, name in ((32, "steps 32"), (64, "steps 64")):
        cells = []
        for block_val in (64, 32):
            rs = [r for r in rows if r["steps"] == steps_val
                  and r["block"] == block_val and r["gen"] == 64]
            cells.append(f"{1 - sum(x['correct'] for x in rs)/len(rs):16.0%}"
                         if rs else f"{'-':>16}")
        say(f"  {name}  " + " ".join(cells))
    say("")
    say("  Read the rows against each other for the effect of `steps`, and the")
    say("  columns for the effect of `block_length`. If one dominates, only")
    say("  that one has to change and Phase C keeps the cheaper setting.")

    # ---- recommendation ---------------------------------------------------
    say("")
    say("=" * 78)
    say("READ THIS BEFORE CHOOSING")
    say("")
    say("  Error rate is the LEAST reliable number in the table. A config that")
    say("  emits fluent wrong answers will often score a HIGHER error rate than")
    say("  one emitting 'Bangiff', because 'Bangiff' occasionally matches")
    say("  nothing and a fluent wrong answer reliably matches nothing either -")
    say("  while a fluent CORRECT answer is the only thing that moves the number")
    say("  down. Judge on the samples and on commit_conf.")
    say("")
    say("  The question to ask of the sample answers is not 'is this right' but")
    say("  'is this a sentence'. 'Egypt' for a question whose answer is Sudan is")
    say("  a usable hallucination. 'Lgor Chopovsky' is not a hallucination at")
    say("  all and must not enter the dataset.")
    say("")
    say("  Cost note: config 3 doubles `steps` and therefore doubles Phase C's")
    say("  bill. If config 2 or 4 gets most of the way there, take it - the")
    say("  savings buy a larger sample, which is the binding constraint.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say("")
    say(f"  CSV         : {CSV_PATH}")
    say(f"  trajectories: {TRAJ_ROOT}  (kept, so the winner needs no re-run)")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
