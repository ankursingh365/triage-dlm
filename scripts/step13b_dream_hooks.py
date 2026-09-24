#!/usr/bin/env python3
"""
Step 13b - can Dream's hooks see what LLaDA's logging sees?   [v2]
==================================================================

    modal run modal_app.py::run --script step13b_dream_hooks.py

A100. About 15 minutes and $0.55.

WHAT v1 GOT WRONG, AND WHY IT MATTERED
======================================
v1 reported "the hooks never fired - the sampler must be reimplemented". That
conclusion was false. The hooks fired exactly as designed; v1's own hook body
crashed on the first call:

    generation_utils.py:408   x = generation_tokens_hook_func(None, x, None)
    step13b:148               step=int(step)   ->  TypeError: int() ... NoneType

Dream primes the tokens hook once before the denoising loop with
`(step=None, x, logits=None)`. v1 assumed `step` was always an integer.

The crash is a small bug. What it exposed is not: v1 caught the exception,
left `captured` empty, and then read that empty list as **evidence about
Dream** rather than evidence about itself. A guard that turns my own error into
a confident claim about the model is worse than no guard at all - the script
did not stop, nothing looked wrong, and the answer was simply incorrect.

v2 fixes three things:

  1. `step` and `logits` may be None. Ordering comes from a call counter, not
     from the reported step value.
  2. Each hook body is guarded internally and ALWAYS returns its input. A hook
     that raises breaks generation; a hook that returns None corrupts it.
  3. Hook failures are recorded separately from hook calls. An empty capture is
     only allowed to say something about Dream when **no error occurred**.
     Otherwise the verdict is "my hook crashed", and it says nothing about the
     model.

Also recorded, because the port needs it: the priming call at line 408 carries
the initial all-masked state, which is round 0.

THE QUESTION
============
Step 13a found that Dream's sampler takes two callbacks:

    generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func",
                                             lambda step, x, logits: x)
    generation_logits_hook_func = kwargs.pop("generation_logits_hook_func",
                                             lambda step, x, logits: logits)

If they fire BEFORE the reveal overwrite, the Dream port is a hundred lines.
If they fire after, Dream's own history is all we can get - which is what every
published method already logs, and Finding B is invisible.

This is measured, not read off the source, because the answer has to describe
the version that is actually installed.

THE FOUR THINGS IT MEASURES
===========================
**1. Fixed or adaptive schedule?** LLaDA's `get_num_transfer_tokens` fixes the
reveal count per round in advance, which is why step 14 found the median settle
round identical (31.0) across all seven failure modes, and why
`claude/schedule-relative-measurement.md` is binding. If Dream is adaptive that
rule does NOT transfer, and the reveal count becomes a feature rather than a
constant.

**2. Confidence freeze?** LLaDA makes a revealed position unselectable forever,
so "held it and lost it" is only meaningful in the shadow prediction. If Dream
can revise a committed position, interleaving is visible in the committed
sequence and the step 17 rules need rewriting rather than reuse.

**3. Pre-commit logits?** At a position revealed on an earlier round, does the
hook's argmax still differ from the committed token? That is Finding B.

**4. Steady-state speed, and which `alg`?** 13a's 8.67 s/question was a single
first call including CUDA warm-up. Dream exposes four remasking algorithms and
the choice changes the trajectory, so it belongs in the paper rather than in a
default.

Nothing is written to the dataset. This measures the interface.
"""

import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

REPORT_PATH = config.OUT_DIR / "step13b_dream_hooks.txt"

N_TIMING = 10          # per algorithm, after a warm-up call
ALGS = ("origin", "maskgit_plus", "topk_margin", "entropy")

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def section(title: str):
    say("")
    say("=" * 78)
    say(title)
    say("")


def guarded(label):
    def deco(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception:
                say(f"  !! {label} FAILED - continuing")
                for ln in traceback.format_exc().splitlines()[-14:]:
                    say("     " + ln)
                return None
        return wrapper
    return deco


def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 13b v2: do Dream's hooks expose Finding B?")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.MODEL_DREAM, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cuda").eval()
    MASK = int(tok.mask_token_id)
    EOS = int(tok.eos_token_id)
    say(f"  model loaded   mask_id {MASK}   eos_id {EOS}")

    recs = data.load_records("triviaqa", n=64, seed=config.SEED)
    kept, _e, _r = data_quality.clean_records(recs)

    def prompt_ids(rec):
        text = data.build_prompt_text(rec.question, rec.choices)
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = tok(text, return_tensors="pt")["input_ids"]
        return ids.to(model.device)

    # =======================================================================
    section("A. WHAT THE HOOKS ACTUALLY RECEIVE")

    captured = []          # one entry per hook call, in call order
    hook_errors = []       # exceptions raised INSIDE our own hook bodies

    def _snap(kind, step, x, logits, want_argmax):
        """Record one call. Never raises: a raising hook breaks generation,
        and an empty capture caused by our own bug must not be mistaken for
        a fact about Dream."""
        try:
            e = dict(idx=len(captured), kind=kind, raw_step=step,
                     step_is_none=step is None,
                     x=None if x is None else x.detach().clone().cpu(),
                     logits_shape=None if logits is None
                     else tuple(logits.shape))
            if want_argmax and logits is not None:
                # Only the argmax is kept. The full tensor is
                # seq x 152,064 floats per step; logging_patch keeps no more
                # than this for LLaDA either.
                e["argmax"] = logits.argmax(-1).detach().clone().cpu()
            else:
                e["argmax"] = None
            captured.append(e)
        except Exception:
            hook_errors.append((kind, traceback.format_exc()))

    def tokens_hook(step, x, logits):
        _snap("tokens", step, x, logits, want_argmax=False)
        return x                       # must return x, whatever happened

    def logits_hook(step, x, logits):
        _snap("logits", step, x, logits, want_argmax=True)
        return logits                  # must return logits, whatever happened

    rec0 = kept[0]
    pid = prompt_ids(rec0)
    plen = pid.shape[-1]
    say(f"  question : {rec0.question}")
    say(f"  gold     : {rec0.gold_answers[:3]}")
    say(f"  prompt   : {plen} tokens")
    say("")

    @guarded("hooked generation")
    def run_hooked():
        with torch.no_grad():
            return model.diffusion_generate(
                pid, max_new_tokens=config.GEN_LENGTH,
                steps=config.DENOISING_STEPS, temperature=0.0,
                alg="origin", output_history=True,
                return_dict_in_generate=True,
                generation_tokens_hook_func=tokens_hook,
                generation_logits_hook_func=logits_hook)
    out = run_hooked()

    say(f"  hook calls captured : {len(captured)}")
    say(f"  errors inside our own hook bodies : {len(hook_errors)}")
    say(f"  by kind             : {dict(Counter(c['kind'] for c in captured))}")

    # --- the distinction v1 got wrong -------------------------------------
    if hook_errors:
        say("")
        say("  STOPPED - OUR OWN HOOK RAISED. This says nothing about Dream.")
        for kind, tb in hook_errors[:2]:
            say(f"  in the {kind} hook:")
            for ln in tb.splitlines()[-8:]:
                say("     " + ln)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    if not captured:
        say("")
        say("  The hooks never fired, and no error of ours explains it. They")
        say("  are not the integration point; the sampler must be")
        say("  reimplemented as LLaDA's was, from the source already saved at")
        say("  outputs/step13a_dream_sampler_source.txt.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    n_primed = sum(1 for c in captured if c["step_is_none"])
    say(f"  calls with step=None (priming): {n_primed}")
    say("")
    say("  first five calls, in order:")
    for c in captured[:5]:
        say(f"    #{c['idx']:<3} {c['kind']:<7} step={c['raw_step']!r:<6} "
            f"x={None if c['x'] is None else tuple(c['x'].shape)}  "
            f"logits={c['logits_shape']}")
    if out is not None:
        say("")
        say(f"  output type: {type(out).__name__}")
        hist = getattr(out, "history", None)
        say(f"  output_history present: {hist is not None}"
            + (f"  ({len(hist)} entries)" if hist is not None else ""))
        say("  (Dream's own history records the committed sequence per step -")
        say("   which is what every published method already logs. Whether we")
        say("   can see more than that is part D.)")

    # =======================================================================
    section("B. IS THE REVEAL SCHEDULE FIXED OR ADAPTIVE?")

    @guarded("schedule analysis")
    def schedule():
        seqs = [c["x"][0][plen:].numpy() for c in captured
                if c["kind"] == "tokens" and c["x"] is not None]
        if len(seqs) < 3:
            say(f"  only {len(seqs)} token-hook states - too few to tell")
            return None
        counts = [int(((a == MASK) & (b != MASK)).sum())
                  for a, b in zip(seqs, seqs[1:])]
        say(f"  token-hook states: {len(seqs)}")
        say(f"  revealed per round: {counts[:16]}"
            + (" ..." if len(counts) > 16 else ""))
        uniq = sorted(set(counts))
        say(f"  distinct values   : {uniq}")
        say("")
        if len(uniq) <= 2:
            say("  FIXED, like LLaDA's get_num_transfer_tokens. The rule in")
            say("  claude/schedule-relative-measurement.md transfers, and")
            say("  absolute settle-round features are dead here too.")
        else:
            say("  ADAPTIVE - the count varies by round. That rule does NOT")
            say("  transfer: for Dream the reveal count is itself a signal,")
            say("  and step 19 may use it as a feature rather than excluding")
            say("  it. It has to be re-derived, not inherited.")
        return counts
    schedule()

    # =======================================================================
    section("C. IS THERE A CONFIDENCE FREEZE?")

    @guarded("freeze analysis")
    def freeze():
        seqs = [c["x"][0][plen:].numpy() for c in captured
                if c["kind"] == "tokens" and c["x"] is not None]
        changed = checked = 0
        for a, b in zip(seqs, seqs[1:]):
            live = (a != MASK)
            checked += int(live.sum())
            changed += int(((a != b) & live).sum())
        say(f"  already-revealed positions examined : {checked:,}")
        say(f"  of those, changed on a later round  : {changed:,}")
        say("")
        if checked == 0:
            say("  Nothing was ever revealed between two captured states.")
        elif changed == 0:
            say("  FROZEN. A revealed position never changes, as in LLaDA. So")
            say("  'held the right answer and lost it' can only be asked of")
            say("  the shadow prediction - which is why part D matters.")
        else:
            say("  NOT FROZEN. Dream can revise a committed position. A real")
            say("  difference from LLaDA: interleaving becomes visible in the")
            say("  committed sequence, and the step 17 rules need rewriting")
            say("  for Dream rather than reuse.")
        return changed
    freeze()

    # =======================================================================
    section("D. DOES THE LOGITS HOOK SEE PRE-COMMIT PREDICTIONS?")
    say("  The whole point. At a position revealed on an EARLIER round, does")
    say("  the hook's argmax still differ from the committed token? If yes,")
    say("  the hook fires before the overwrite and Finding B is visible.")
    say("")

    @guarded("finding-B analysis")
    def finding_b():
        lg = [c for c in captured if c["kind"] == "logits"
              and c["argmax"] is not None and c["x"] is not None]
        if not lg:
            say("  No logits-hook call carried both a sequence and logits.")
            say("  Dream may pass logits=None on the calls we saw. Check the")
            say("  'first five calls' listing in part A.")
            return None
        say(f"  usable logits-hook calls: {len(lg)}")
        a0 = lg[0]["argmax"]
        say(f"  argmax shape: {tuple(a0.shape)}   prompt length {plen}")
        disagree = compared = skipped = 0
        for c in lg:
            seq = c["x"][0][plen:].numpy()
            am = c["argmax"]
            am = am[0] if am.ndim > 1 else am
            am = am.numpy()
            if am.shape[0] > seq.shape[0]:
                am = am[-seq.shape[0]:]
            if am.shape[0] != seq.shape[0]:
                skipped += 1
                continue
            live = (seq != MASK)
            compared += int(live.sum())
            disagree += int(((am != seq) & live).sum())
        if skipped:
            say(f"  calls skipped on a shape mismatch: {skipped}")
        say(f"  committed positions compared : {compared:,}")
        say(f"  where the hook's argmax differs : {disagree:,}")
        say("")
        if compared == 0:
            say("  Nothing to compare - every captured call may precede the")
            say("  first reveal. Not a verdict either way.")
            return None
        say(f"  disagreement rate : {disagree/compared:.1%}")
        say("")
        if disagree > 0:
            say("  PRE-COMMIT. The hook reports what the model would say now")
            say("  about tokens it has already fixed. That is the shadow")
            say("  prediction, and Finding B transfers to Dream.")
            say("")
            say("  The port is small: wrap diffusion_generate, record the same")
            say("  four arrays logging_patch records for LLaDA, and gate it")
            say("  with its OWN reconstruction LOCK. LLaDA's gate proves")
            say("  nothing about Dream.")
        else:
            say("  POST-COMMIT, or the hook sees only masked positions. The")
            say("  argmax always agrees with what was committed, so this hook")
            say("  cannot see regret. Dream's sampler must be reimplemented")
            say("  the way LLaDA's was - the source is already saved at")
            say("  outputs/step13a_dream_sampler_source.txt.")
        return disagree / compared
    finding_b()

    # =======================================================================
    section("E. STEADY-STATE SPEED, AND THE FOUR ALGORITHMS")
    say("  13a measured 8.67 s on a single FIRST call, which includes CUDA")
    say("  warm-up. Budgeting from that would repeat the Phase C error of")
    say("  guessing 10 s/q against a measured 2.01.")
    say("")

    @guarded("timing")
    def timing():
        with torch.no_grad():                       # warm-up, not timed
            model.diffusion_generate(
                prompt_ids(kept[1]), max_new_tokens=config.GEN_LENGTH,
                steps=config.DENOISING_STEPS, temperature=0.0, alg="origin")
        say("  alg            s/question   12,317 q     cost")
        say("  -------------  ----------  ----------  -------")
        rates = {}
        for alg in ALGS:
            try:
                t = time.time()
                for rec in kept[2: 2 + N_TIMING]:
                    with torch.no_grad():
                        model.diffusion_generate(
                            prompt_ids(rec), max_new_tokens=config.GEN_LENGTH,
                            steps=config.DENOISING_STEPS, temperature=0.0,
                            alg=alg)
                r = (time.time() - t) / N_TIMING
                rates[alg] = r
                hrs = 12317 * r / 3600
                say(f"  {alg:<13}  {r:10.2f}  {hrs:9.1f}h  ${hrs*2.10:6.2f}")
            except Exception as exc:
                say(f"  {alg:<13}  unavailable: {type(exc).__name__}: "
                    f"{str(exc)[:60]}")
        return rates
    rates = timing() or {}

    # =======================================================================
    section("F. WHAT THIS MEANS FOR STEP 13c")
    if rates:
        best = min(rates, key=rates.get)
        hrs = 12317 * rates[best] / 3600
        say(f"  Fastest algorithm: {best} at {rates[best]:.2f} s/question.")
        say(f"  All three datasets at LLaDA's sizes: {hrs:.1f} GPU-hours, "
            f"${hrs*2.10:.2f}")
        say("")
        say("  Dream is a REPLICATION arm. Its job is to show the locked-in")
        say("  finding is not an artefact of one model - not to match LLaDA's")
        say("  precision. From claude/sample-size-power-decision.md the pooled")
        say("  MDE is 0.059 at 264 interleaving cases; half the data gives")
        say("  roughly 0.08, still below the 0.10 effect the thesis predicts.")
        say("")
        for frac in (1.0, 0.5, 0.25):
            h = hrs * frac
            say(f"    {frac:>4.0%} of LLaDA's sizes : {h:5.1f}h   "
                f"${h*2.10:6.2f}")
        say("")
        say("  That is a budget decision, not a technical one, and it is")
        say("  yours. Step 13c sizes from whichever fraction you pick and")
        say("  resumes, so starting small costs nothing later.")
    say("")
    say(f"  wall clock : {time.time() - t0:.0f}s")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()