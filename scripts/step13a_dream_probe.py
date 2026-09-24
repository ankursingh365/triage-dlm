#!/usr/bin/env python3
"""
Step 13a - probe Dream-7B before writing a single line of the logging port
==========================================================================

    modal run modal_app.py::run --script step13a_dream_probe.py

A100. About 10 minutes and $0.35, almost all of it the first download (~14 GB).

WHY A PROBE AND NOT THE PORT
============================
LLaDA's checkpoints ship **no sampler** - only the architecture - so
`src/logging_patch.py` is our own denoising loop, written against the reference
`generate.py` (plan steps 3-5). Dream is the opposite: its remote code ships
its own `diffusion_generate` with its own algorithms. Which means the port is
not "run the same loop against a different checkpoint". It is either

    (a) hook Dream's sampler, if it exposes per-step callbacks, or
    (b) reimplement Dream's algorithm the way we reimplemented LLaDA's

and which of those applies cannot be decided from documentation or memory. It
has to be read off the installed remote code. That is all this script does.

Writing the port blind and discovering the mismatch during a paid generation
run is the expensive order. Ten minutes here is the cheap order.

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not call `model_utils.load_model_and_tokenizer`. That helper calls
`resolve_mask_id` **before** loading the model, and `KNOWN_MASK_IDS` has no
Dream entry:

    KNOWN_MASK_IDS = {
        "GSAI-ML/LLaDA-8B-Instruct": 126336,
        "GSAI-ML/LLaDA-8B-Base":     126336,
        "GSAI-ML/LLaDA-1.5":         126336,
    }

So if Dream's tokenizer does not declare `mask_token_id`, that helper raises
before the model is ever fetched and the probe learns nothing. Part B works the
mask id out from the tokenizer directly and reports what it found, so the entry
can be added with evidence rather than a guess. A wrong mask id produces
plausible-looking output that describes nothing, which is the worst failure
mode available here.

EVERY PART IS INDEPENDENTLY GUARDED
===================================
A probe that dies on its first surprise is useless, because the surprises are
the point. Each section catches its own exceptions and prints the traceback,
then continues, so one run returns everything that could be learned.
"""

import inspect
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, data, data_quality
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

REPORT_PATH = config.OUT_DIR / "step13a_dream_probe.txt"
SOURCE_PATH = config.OUT_DIR / "step13a_dream_sampler_source.txt"
MAX_SOURCE_LINES = 400

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
    """Run a probe section, report a failure, keep going."""
    def deco(fn):
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except Exception:
                say(f"  !! {label} FAILED - continuing")
                for ln in traceback.format_exc().splitlines()[-12:]:
                    say("     " + ln)
                return None
        return wrapper
    return deco


def main() -> None:
    t0 = time.time()
    say("=" * 78)
    say("  TRIAGE - Step 13a: Dream-7B API probe")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    say(f"  model : {config.MODEL_DREAM}")

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    # =======================================================================
    section("A. TOKENIZER")
    tok = AutoTokenizer.from_pretrained(config.MODEL_DREAM,
                                        trust_remote_code=True)
    say(f"  class          : {type(tok).__name__}")
    say(f"  vocab          : {len(tok):,}")
    for attr in ("mask_token", "mask_token_id", "pad_token", "pad_token_id",
                 "eos_token", "eos_token_id", "bos_token", "bos_token_id"):
        say(f"  {attr:<15}: {getattr(tok, attr, '<absent>')!r}")
    say(f"  chat_template  : "
        f"{'present' if getattr(tok, 'chat_template', None) else 'ABSENT'}")
    extra = getattr(tok, "additional_special_tokens", None)
    say(f"  extra specials : {extra}")

    # =======================================================================
    section("B. MASK TOKEN - worked out, not guessed")

    @guarded("mask id resolution")
    def probe_mask():
        declared = getattr(tok, "mask_token_id", None)
        say(f"  tokenizer declares mask_token_id : {declared}")
        cands = []
        vocab = tok.get_vocab()
        for piece, idx in vocab.items():
            low = piece.lower()
            if "mask" in low:
                cands.append((idx, piece))
        cands.sort()
        say(f"  vocabulary entries containing 'mask': {len(cands)}")
        for idx, piece in cands[:12]:
            say(f"    {idx:>8}  {piece!r}")
        say("")
        say("  Does src/model_utils.resolve_mask_id handle this model?")
        try:
            from src import model_utils
            mid = model_utils.resolve_mask_id(tok, config.MODEL_DREAM)
            say(f"    yes -> {mid}")
        except Exception as exc:
            say(f"    NO  -> {type(exc).__name__}: {exc}")
            say("    KNOWN_MASK_IDS needs a Dream entry before step 13b.")
        return cands
    probe_mask()

    # =======================================================================
    section("C. MODEL")
    model = None

    @guarded("model load")
    def load():
        for loader, name in ((AutoModelForCausalLM, "AutoModelForCausalLM"),
                             (AutoModel, "AutoModel")):
            try:
                m = loader.from_pretrained(
                    config.MODEL_DREAM, torch_dtype=torch.bfloat16,
                    trust_remote_code=True, device_map="cuda")
                say(f"  loaded via {name}")
                return m
            except Exception as exc:
                say(f"  {name} failed: {type(exc).__name__}: "
                    f"{str(exc)[:120]}")
        raise RuntimeError("both loaders failed")
    model = load()
    if model is None:
        say("\n  Cannot continue without the model.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)

    model.eval()
    say(f"  class          : {type(model).__name__}")
    say(f"  module         : {type(model).__module__}")
    try:
        say(f"  remote code at : {inspect.getfile(type(model))}")
    except Exception:
        pass
    say(f"  dtype          : {next(model.parameters()).dtype}")
    say(f"  VRAM allocated : {torch.cuda.memory_allocated()/2**30:.2f} GiB")
    cfgo = getattr(model, "config", None)
    if cfgo is not None:
        for attr in ("model_type", "mask_token_id", "vocab_size",
                     "max_position_embeddings", "pad_token_id", "eos_token_id"):
            if hasattr(cfgo, attr):
                say(f"  config.{attr:<22}: {getattr(cfgo, attr)}")

    # =======================================================================
    section("D. GENERATION API - what is actually callable")

    @guarded("method discovery")
    def methods():
        names = [n for n in dir(model)
                 if any(k in n.lower() for k in
                        ("generate", "diffusion", "sample", "denoise", "hook"))
                 and not n.startswith("__")]
        say(f"  candidate methods: {names}")
        say("")
        for n in names:
            try:
                fn = getattr(model, n)
                if callable(fn):
                    say(f"  {n}{inspect.signature(fn)}")
                    say("")
            except (TypeError, ValueError):
                say(f"  {n}  <signature unavailable>")
        return names
    names = methods() or []

    # =======================================================================
    section("E. THE SAMPLER'S SOURCE - the reason this probe exists")

    @guarded("source extraction")
    def dump_source():
        target = None
        for pref in ("diffusion_generate", "_sample", "generate"):
            if pref in names:
                target = pref
                break
        if target is None:
            say("  No recognisable generation entry point. Dumping the whole")
            say("  remote generation module instead.")
        else:
            say(f"  entry point: {target}")
        chunks = []
        seen = set()
        for n in ([target] if target else names):
            try:
                fn = getattr(model, n)
                src = inspect.getsource(fn)
                mod = inspect.getmodule(fn)
                key = (getattr(mod, "__file__", ""), n)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(f"### {n}  from {getattr(mod,'__file__','?')}\n{src}")
            except Exception as exc:
                say(f"    {n}: source unavailable ({type(exc).__name__})")
        # the whole module too, since the sampler usually calls helpers
        try:
            gm = None
            for n in names:
                m = inspect.getmodule(getattr(model, n))
                if m and "generation" in (getattr(m, "__file__", "") or ""):
                    gm = m
                    break
            if gm is not None:
                chunks.append(f"### FULL MODULE {gm.__file__}\n"
                              + inspect.getsource(gm))
        except Exception as exc:
            say(f"    full module unavailable ({type(exc).__name__})")

        blob = "\n\n".join(chunks)
        SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SOURCE_PATH.write_text(blob, encoding="utf-8")
        say(f"  full source written to {SOURCE_PATH}  "
            f"({len(blob.splitlines()):,} lines)")
        say("")
        say(f"  first {MAX_SOURCE_LINES} lines follow; the file has the rest.")
        say("  " + "-" * 74)
        for ln in blob.splitlines()[:MAX_SOURCE_LINES]:
            say("  | " + ln[:200])
        say("  " + "-" * 74)
    dump_source()

    # =======================================================================
    section("F. CAN WE HOOK IT, OR MUST WE REIMPLEMENT IT?")

    @guarded("hook discovery")
    def hooks():
        found = {}
        for n in names:
            try:
                sig = inspect.signature(getattr(model, n))
            except (TypeError, ValueError):
                continue
            for p in sig.parameters:
                if any(k in p.lower() for k in
                       ("hook", "callback", "output_history",
                        "return_dict_in_generate", "output_logits",
                        "output_scores")):
                    found.setdefault(n, []).append(p)
        if found:
            say("  Hook-shaped parameters found:")
            for n, ps in found.items():
                say(f"    {n}: {ps}")
            say("")
            say("  If a per-step hook receives the logits BEFORE the reveal")
            say("  overwrite, option (a) is open and the port is small. Finding")
            say("  B - the raw argmax at already-committed positions - is the")
            say("  whole reason our logging exists, and a hook that fires after")
            say("  the overwrite cannot see it. Check the source above for")
            say("  where the hook is called relative to the commit.")
        else:
            say("  No hook parameters. Option (b): the sampler must be")
            say("  reimplemented the way LLaDA's was, from the source in")
            say(f"  {SOURCE_PATH}.")
        return found
    hooks()

    # =======================================================================
    section("G. ONE ANSWER, WITH WHATEVER DEFAULTS IT HAS")

    @guarded("generation")
    def generate_one():
        recs = data.load_records("triviaqa", n=8, seed=config.SEED)
        kept, _e, _r = data_quality.clean_records(recs)
        rec = kept[0]
        say(f"  question : {rec.question}")
        say(f"  gold     : {rec.gold_answers[:3]}")
        text = data.build_prompt_text(rec.question, rec.choices)
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, return_tensors="pt")
        except Exception:
            ids = tok(text, return_tensors="pt")["input_ids"]
        if hasattr(ids, "to"):
            ids = ids.to(model.device)
        say(f"  prompt tokens: {tuple(ids.shape)}")

        fn_name = "diffusion_generate" if hasattr(model, "diffusion_generate") \
            else "generate"
        fn = getattr(model, fn_name)
        say(f"  calling model.{fn_name}(...)")
        t = time.time()
        with torch.no_grad():
            out = fn(ids, max_new_tokens=config.GEN_LENGTH,
                     steps=config.DENOISING_STEPS, temperature=0.0)
        dt = time.time() - t
        seq = out.sequences if hasattr(out, "sequences") else out
        gen = seq[0][ids.shape[-1]:]
        say(f"  elapsed  : {dt:.2f}s   ({dt:.2f} s/question)")
        say(f"  output type: {type(out).__name__}")
        if hasattr(out, "keys"):
            say(f"  output keys: {list(out.keys())}")
        say(f"  answer   : {tok.decode(gen.tolist())!r}")
        say("")
        say(f"  At this rate, 12,317 questions = "
            f"{12317*dt/3600:.1f} GPU-hours, ${12317*dt/3600*2.10:.2f}")
    generate_one()

    # =======================================================================
    section("H. WHAT STEP 13b NEEDS FROM THIS")
    say("  Answer these from the output above before writing the port:")
    say("")
    say("   1. mask token id, and whether KNOWN_MASK_IDS needs an entry")
    say("   2. the sampler entry point and its real signature")
    say("   3. does it reveal a FIXED number of positions per round, like")
    say("      LLaDA's get_num_transfer_tokens, or is it adaptive? A fixed")
    say("      schedule is what makes settle-round features meaningless and")
    say("      schedule-relative measurement mandatory - if Dream is adaptive,")
    say("      that binding rule does not transfer and must be re-derived.")
    say("   4. is there a confidence freeze on revealed positions? Without one,")
    say("      'had it and lost it' means something different than in LLaDA.")
    say("   5. hook before the commit (port is small) or reimplement (port is")
    say("      the whole loop, plus its own reconstruction LOCK)")
    say("   6. seconds per question, for the step 13c budget")
    say("")
    say(f"  wall clock : {time.time() - t0:.0f}s")
    say("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    print(f"Sampler source: {SOURCE_PATH}")


if __name__ == "__main__":
    main()
