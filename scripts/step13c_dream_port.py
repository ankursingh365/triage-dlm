#!/usr/bin/env python3
"""
Step 13c - the Dream logging port, and its OWN reconstruction LOCK
===================================================================

    modal run modal_app.py::run --script step13c_dream_port.py

A100. About 5 minutes and $0.20 (50 questions at the measured 1.66 s/q).

WHAT 13b ESTABLISHED
====================
    hooks fire          129 calls: 1 priming (step=None), then per round
                        logits BEFORE tokens - so the logits hook sees the
                        state ENTERING the round, before the reveal
    logits shape        (1, 98, 152064) - full sequence, full vocabulary
    schedule            ADAPTIVE: 0-4 positions per round, not LLaDA's fixed 1
    confidence freeze   YES, 0 of 2,058 committed positions ever changed
    Finding B           3.1% disagreement -> the hook is pre-commit
    speed               1.66 s/question; all four algs within noise

So the port is a wrapper, not a reimplementation. It records the same four
arrays `src/logging_patch.py` records for LLaDA, from the two callbacks.

THE GATE
========
LLaDA's reconstruction gate says nothing about Dream. This script therefore
re-runs it from scratch on Dream's own trajectories:

    rebuilt[i] = pred_ids[ revealed_at()[i], i ]      must equal final_ids

with `revealed_at()` derived from `mask_state`, and `final_ids` taken from the
tensor Dream actually returned. Two different sources, which is what makes it a
test. 50 out of 50 must pass. **No Dream data is generated at scale until this
passes.**

THE SUSPICION THIS ALSO CHECKS
==============================
13b found exactly **64 disagreements across exactly 64 logits calls**. That may
be genuine regret at ~1 position per round, or it may be an artefact - the
model "regretting" whatever it committed on the previous round, every round,
which would be a property of the sampler rather than a signal about the answer.
3.1% cannot be called a Finding-B rate until that is separated, so Part C
measures the distribution of regret per round rather than its mean. A constant
1-per-round is an artefact; a spread is a signal.

WHY `alg='origin'`
==================
All four remasking algorithms ran within noise of each other (1.66-1.72 s), so
the choice is not about speed. `origin` is Dream's own default and therefore
what any reader reproducing the paper will run. The other three change which
positions are revealed, which changes the trajectory being measured, so the
choice belongs in the methods section rather than in a default.

ON `block_length`
=================
Dream has no block structure - LLaDA's semi-autoregressive blocks do not exist
here. The field is set to `gen_length` so the Trajectory dataclass stays a
single shape across models, and it means "one block" for Dream. Anything
reading it must not infer a block schedule.
"""

import sys
import time
from collections import Counter
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
TRAJ_DIR = config.TRAJ_DIR / "dream_lock" / DREAM_TAG
REPORT_PATH = config.OUT_DIR / "step13c_dream_port.txt"

N_LOCK = 50
ALG = "origin"

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# THE PORT
# ===========================================================================

def generate_with_logging_dream(model, tokenizer, prompt_ids, *,
                                gen_length=64, steps=64, alg="origin",
                                temperature=0.0, mask_id=None,
                                question="", question_id="",
                                quant_config="", seed=0):
    """Dream's diffusion_generate, instrumented through its own callbacks.

    Returns `(text, Trajectory)` with the same four arrays `logging_patch`
    records for LLaDA, so every downstream script works unchanged.

    The recording happens in the LOGITS hook, which 13b showed fires before
    the reveal. `x` there is the state entering the round, so `mask_state[k]`
    means "masked entering round k" - the same convention `revealed_at()`
    expects. Recording in the tokens hook instead would capture the state
    AFTER the commit, and the raw argmax at already-committed positions -
    Finding B, the whole reason this logging exists - would be invisible.
    """
    import torch

    plen = prompt_ids.shape[-1]
    MASK = int(mask_id)

    pred_ids = np.zeros((steps, gen_length), np.int32)
    pred_probs = np.zeros((steps, gen_length), np.float32)
    entropy = np.zeros((steps, gen_length), np.float32)
    mask_state = np.ones((steps, gen_length), np.uint8)
    seen = set()
    errors = []

    def logits_hook(step, x, logits):
        # Must return `logits` whatever happens: raising breaks generation and
        # returning None corrupts it. An error here is recorded, never silent.
        try:
            if step is not None and logits is not None:
                k = int(step)
                if 0 <= k < steps:
                    lg = logits[0, plen:plen + gen_length, :].float()
                    probs = torch.softmax(lg, dim=-1)
                    p, am = probs.max(dim=-1)
                    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                    pred_ids[k] = am.cpu().numpy().astype(np.int32)
                    pred_probs[k] = p.cpu().numpy()
                    entropy[k] = ent.cpu().numpy()
                    mask_state[k] = (
                        x[0, plen:plen + gen_length] == MASK
                    ).cpu().numpy().astype(np.uint8)
                    seen.add(k)
        except Exception as exc:
            errors.append(f"logits hook step={step}: {type(exc).__name__}: {exc}")
        return logits

    def tokens_hook(step, x, logits):
        return x

    t0 = time.time()
    with torch.no_grad():
        out = model.diffusion_generate(
            prompt_ids, max_new_tokens=gen_length, steps=steps,
            temperature=temperature, alg=alg,
            return_dict_in_generate=True,
            generation_tokens_hook_func=tokens_hook,
            generation_logits_hook_func=logits_hook)
    elapsed = time.time() - t0

    seq = out.sequences if hasattr(out, "sequences") else out
    final_ids = seq[0][plen:plen + gen_length].cpu().numpy().astype(np.int32)

    traj = logging_patch.Trajectory(
        pred_ids=pred_ids, pred_probs=pred_probs, entropy=entropy,
        mask_state=mask_state, final_ids=final_ids,
        question=question, question_id=question_id,
        prompt_ids=list(prompt_ids[0].cpu().numpy().tolist()),
        model_id=config.MODEL_DREAM, quant_config=quant_config, seed=seed,
        steps=steps, gen_length=gen_length,
        block_length=gen_length,          # Dream has no blocks; see docstring
        temperature=temperature, mask_id=MASK,
        eos_id=int(tokenizer.eos_token_id), vocab_size=len(tokenizer),
        elapsed_s=elapsed)
    text = tokenizer.decode(final_ids.tolist())
    return text, traj, sorted(seen), errors


# ===========================================================================

def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 13c: Dream logging port + its own reconstruction LOCK")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  model : {config.MODEL_DREAM}")
    say(f"  alg   : {ALG}   (Dream's default; all four within noise on speed)")
    say(f"  tag   : {DREAM_TAG}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.MODEL_DREAM, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda").eval()
    MASK = int(tok.mask_token_id)
    say(f"  loaded  mask_id {MASK}  eos_id {tok.eos_token_id}")

    recs = data.load_records("triviaqa", n=N_LOCK * 2, seed=config.SEED)
    kept, _e, _r = data_quality.clean_records(recs)
    chosen = kept[:N_LOCK]
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    say(f"  {len(chosen)} questions for the gate")

    say("")
    say("=" * 78)
    say("A. THE GATE")
    say("")
    rows, failures, all_errors = [], [], []
    for i, rec in enumerate(chosen, 1):
        text = data.build_prompt_text(rec.question, rec.choices)
        try:
            pid = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt")
        except Exception:
            pid = tok(text, return_tensors="pt")["input_ids"]
        pid = pid.to(model.device)

        answer, traj, seen, errs = generate_with_logging_dream(
            model, tok, pid, gen_length=config.GEN_LENGTH,
            steps=config.DENOISING_STEPS, alg=ALG, mask_id=MASK,
            question=rec.question, question_id=rec.qid, seed=config.SEED)
        all_errors.extend(errs)
        traj.save(TRAJ_DIR / f"{rec.qid}.npz")

        rebuilt = traj.reconstruct_final()
        ids_ok = bool(np.array_equal(rebuilt, traj.final_ids))
        text_ok = (tok.decode(rebuilt.tolist())
                   == tok.decode(traj.final_ids.tolist()))
        if not (ids_ok and text_ok):
            failures.append((rec.qid, ids_ok, text_ok, answer[:60]))

        rev = traj.revealed_at()
        n_content = int(sum(1 for t in traj.final_ids.tolist()
                            if t not in (traj.eos_id, traj.mask_id)))
        rows.append(dict(qid=rec.qid, ids_ok=ids_ok, text_ok=text_ok,
                         rounds_seen=len(seen), n_content=n_content,
                         still_masked=int((traj.mask_state[-1] == 1).sum()),
                         regret=traj.regret_per_round(),
                         flips=int(traj.flips_per_round().sum()),
                         elapsed=traj.elapsed_s, answer=answer))
        if i % 10 == 0:
            say(f"    {i:3d}/{len(chosen)}   {len(failures)} failures   "
                f"[{time.time()-t0:.0f}s]")

    say("")
    if all_errors:
        say(f"  ERRORS INSIDE OUR OWN HOOK: {len(all_errors)}")
        for e in all_errors[:5]:
            say(f"    {e}")
        say("  These are our bug, not Dream's. Fix before reading anything")
        say("  below as a fact about the model.")
        say("")

    n_ok = sum(1 for r in rows if r["ids_ok"] and r["text_ok"])
    say(f"  checked      {len(rows)}")
    say(f"  reconstruct  {n_ok}")
    say(f"  failures     {len(failures)}")
    rounds = Counter(r["rounds_seen"] for r in rows)
    say(f"  rounds recorded per question: {dict(rounds)}  "
        f"(expected {config.DENOISING_STEPS})")
    say(f"  positions still masked at the end: "
        f"mean {np.mean([r['still_masked'] for r in rows]):.1f}")
    say("")
    if failures or all_errors:
        if failures:
            say("  FAILURES (first 10):")
            for qid, i_ok, t_ok, ans in failures[:10]:
                say(f"    {qid}  ids_ok={i_ok} text_ok={t_ok}  {ans!r}")
        say("")
        say("=" * 78)
        say("  GATE FAILED. STOP.")
        say("")
        say("  The port does not describe what Dream did. Do NOT generate")
        say("  Dream data at scale. Most likely causes, in order: the logits")
        say("  hook is not pre-reveal after all; the prompt/generation slice")
        say("  boundary is off; or positions left masked at the end are")
        say("  filled by something other than the last recorded argmax.")
        say("=" * 78)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("  GATE PASSED - every trajectory rebuilds exactly, ids and text.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("B. THE ADAPTIVE SCHEDULE, AT 50 QUESTIONS")
    say("")
    per_round = []
    for r in rows:
        traj = logging_patch.Trajectory.load(TRAJ_DIR / f"{r['qid']}.npz")
        ms = traj.mask_state
        per_round.extend(int(((ms[k] == 1) & (ms[k + 1] == 0)).sum())
                         for k in range(ms.shape[0] - 1))
    c = Counter(per_round)
    say("  positions revealed per round, pooled over 50 questions")
    say("    count   rounds")
    for k in sorted(c):
        say(f"    {k:5d}   {c[k]:6,}  {'#' * int(40*c[k]/max(c.values()))}")
    say("")
    say(f"  mean {np.mean(per_round):.2f}   distinct values {sorted(c)}")
    say("")
    if len(c) <= 2:
        say("  Effectively fixed after all - 13b's single question was not")
        say("  representative. The schedule-relative rule may transfer.")
    else:
        say("  ADAPTIVE confirmed at scale. claude/schedule-relative-")
        say("  measurement.md does NOT transfer to Dream: the reveal count")
        say("  varies per round and is a signal, not a constant. Step 19 must")
        say("  treat Dream's settle-round features differently from LLaDA's.")

    # =======================================================================
    say("")
    say("=" * 78)
    say("C. IS THE 3.1% REGRET REAL, OR ONE PER ROUND?")
    say("")
    say("  13b saw exactly 64 disagreements across exactly 64 logits calls.")
    say("  A constant 1-per-round would be a property of the sampler, not a")
    say("  signal about the answer. The distribution settles it.")
    say("")
    allr = np.concatenate([r["regret"] for r in rows])
    rc = Counter(int(v) for v in allr)
    say("  regret per round, pooled")
    say("    value   rounds")
    for k in sorted(rc)[:10]:
        say(f"    {k:5d}   {rc[k]:6,}  {'#' * int(40*rc[k]/max(rc.values()))}")
    say("")
    say(f"  mean {allr.mean():.2f}   max {int(allr.max())}   "
        f"share of rounds with 0 regret {rc.get(0,0)/len(allr):.0%}")
    say("")
    if len(rc) <= 2 and rc.get(1, 0) > 0.9 * len(allr):
        say("  ARTEFACT. Regret is 1 on essentially every round, which is the")
        say("  sampler, not the model. Finding B does NOT transfer to Dream as")
        say("  measured, and step 19 must not use Dream regret as a feature")
        say("  without understanding this first.")
    else:
        say("  REAL. Regret varies across rounds and questions, so it carries")
        say("  information rather than reflecting a fixed sampler behaviour.")
        say("  13b's exact 64/64 was a coincidence of one question.")

    say("")
    say("=" * 78)
    say(f"  s/question : {np.mean([r['elapsed'] for r in rows]):.2f}")
    say(f"  trajectories: {TRAJ_DIR}")
    say(f"  wall clock : {time.time()-t0:.0f}s")
    say("")
    say("  Next: step 13d generates Dream at LLaDA's sizes - 12,317 questions,")
    say("  about 5.7 GPU-hours and $11.89 at the measured rate.")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
