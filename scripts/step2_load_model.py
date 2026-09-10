#!/usr/bin/env python3
"""
TRIAGE / CRADLE - Step 2: load LLaDA-MoE in 4-bit and read one entropy value.

Run from inside your triage-dlm folder, with (.venv) active:

    python scripts/step2_load_model.py

What this proves:
  1. The model loads in 4-bit inside ~5 GB of usable VRAM.
  2. transformers can execute LLaDA's trust_remote_code modelling file.
  3. We can pull logits out of a forward pass and compute entropy from them.

Item 3 is a miniature of Step 4. If entropy works here on one position, the
only thing left for Step 4 is the loop over denoising rounds.

FIRST RUN DOWNLOADS ~15 GB to D:\\hf-cache. Allow 15-40 minutes on a normal
connection. Later runs load from disk in under a minute.

Memory strategy - three configurations, tried in order, stopping at the first
that fits:
  A  everything on GPU, NF4 + double quantisation
  B  same, but the output head is quantised too (frees ~0.6 GB)
  C  accelerate splits the model, spilling the overflow to system RAM (slow
     but always works)

You do not have to choose. The script reports which one succeeded.
"""

import gc
import os
import random
import sys
import time
import traceback
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
SEED = 20260909
PROMPT = "The capital city of France is"

REPORT_PATH = os.path.join("outputs", "step2_load_report.txt")

_report_lines = []


def say(text=""):
    """Print to screen and remember for the saved report."""
    print(text, flush=True)
    _report_lines.append(text)


def gib(n_bytes):
    return n_bytes / 1024 ** 3


def vram_now(tag):
    if not _torch.cuda.is_available():
        return
    free_b, total_b = _torch.cuda.mem_get_info(0)
    alloc = _torch.cuda.memory_allocated(0)
    say(f"  [{tag}] allocated {gib(alloc):5.2f} GiB   free {gib(free_b):5.2f} GiB "
        f"of {gib(total_b):.2f} GiB")


# ---------------------------------------------------------------------------

say("=" * 68)
say("  TRIAGE - Step 2: model load + first entropy reading")
say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
say("=" * 68)

# ---- imports, one at a time so a failure names itself clearly ----
try:
    import torch as _torch
    import numpy as np
except Exception:
    say("FAILED importing torch/numpy:")
    traceback.print_exc()
    sys.exit(1)

try:
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig
except Exception:
    say("FAILED importing transformers:")
    traceback.print_exc()
    sys.exit(1)

say(f"transformers  : {transformers.__version__}")
if transformers.__version__.startswith("5"):
    say("")
    say(">>> STOP. transformers 5.x breaks trust_remote_code models.")
    say(">>> Run:  pip install \"transformers<5\"   then try again.")
    say("=" * 68)
    sys.exit(1)

say(f"torch         : {_torch.__version__}")
say(f"HF_HOME       : {os.environ.get('HF_HOME', '(not set - will use C: drive)')}")

# ---- seeds, fixed everywhere from here on (charter warning #9) ----
random.seed(SEED)
np.random.seed(SEED)
_torch.manual_seed(SEED)
if _torch.cuda.is_available():
    _torch.cuda.manual_seed_all(SEED)
say(f"seed          : {SEED} (python, numpy, torch)")

if not _torch.cuda.is_available():
    say("")
    say(">>> STOP. torch cannot see your GPU. Nothing below will work.")
    sys.exit(1)

say("-" * 68)
say("MEMORY BEFORE LOADING")
vram_now("start")
free_start, _ = _torch.cuda.mem_get_info(0)
if gib(free_start) < 4.5:
    say("")
    say(f"  WARNING: only {gib(free_start):.2f} GiB free. Close Chrome and any")
    say("  other GPU application, then run this again. You want 5.0+ GiB.")
say("-" * 68)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

say("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception:
    say("FAILED loading tokenizer:")
    traceback.print_exc()
    sys.exit(1)

vocab_size = len(tokenizer)
say(f"  vocab size  : {vocab_size:,}")
say(f"  mask token  : {getattr(tokenizer, 'mask_token', None)} "
    f"(id {getattr(tokenizer, 'mask_token_id', None)})")

# ---------------------------------------------------------------------------
# Model - three configurations, first that fits wins
# ---------------------------------------------------------------------------

BASE_4BIT = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=_torch.bfloat16,
)

CONFIGS = [
    ("A  all on GPU, NF4 + double quant",
     dict(quantization_config=BitsAndBytesConfig(**BASE_4BIT),
          device_map={"": 0})),

    ("B  same, output head quantised too",
     dict(quantization_config=BitsAndBytesConfig(**BASE_4BIT,
                                                 llm_int8_skip_modules=[]),
          device_map={"": 0})),

    ("C  split across GPU and system RAM (slow fallback)",
     dict(quantization_config=BitsAndBytesConfig(**BASE_4BIT),
          device_map="auto",
          max_memory={0: "4GiB", "cpu": "12GiB"})),
]


def try_load(kwargs):
    """Return a model, or raise. Tries AutoModelForCausalLM then AutoModel."""
    from transformers import AutoModel, AutoModelForCausalLM
    last = None
    for cls in (AutoModelForCausalLM, AutoModel):
        try:
            m = cls.from_pretrained(MODEL_ID, trust_remote_code=True,
                                    dtype=_torch.bfloat16, **kwargs)
            say(f"  loaded via {cls.__name__}")
            return m
        except TypeError:
            # older transformers use torch_dtype instead of dtype
            try:
                m = cls.from_pretrained(MODEL_ID, trust_remote_code=True,
                                        torch_dtype=_torch.bfloat16, **kwargs)
                say(f"  loaded via {cls.__name__} (torch_dtype)")
                return m
            except Exception as exc:
                last = exc
        except Exception as exc:
            last = exc
    raise last


model = None
used_config = None

for label, kwargs in CONFIGS:
    say("")
    say(f"Trying config {label}")
    t0 = time.time()
    try:
        model = try_load(kwargs)
        model.eval()
        used_config = label
        say(f"  SUCCESS in {time.time() - t0:.0f}s")
        vram_now("after load")
        break
    except _torch.cuda.OutOfMemoryError:
        say("  OUT OF MEMORY - falling through to the next config")
        model = None
        gc.collect()
        _torch.cuda.empty_cache()
    except Exception as exc:
        say(f"  FAILED: {type(exc).__name__}: {exc}")
        say("")
        say("  --- full traceback ---")
        tb = traceback.format_exc()
        print(tb)
        _report_lines.append(tb)
        model = None
        gc.collect()
        _torch.cuda.empty_cache()

if model is None:
    say("")
    say(">>> All three configurations failed. Send me this whole report.")
    say("=" * 68)
    os.makedirs("outputs", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_report_lines) + "\n")
    sys.exit(1)

say("")
say("-" * 68)
say(f"CONFIG USED   : {used_config}")
n_params = sum(p.numel() for p in model.parameters())
say(f"parameters    : {n_params/1e9:.2f} B (as counted after quantisation)")

# ---------------------------------------------------------------------------
# One forward pass, one entropy value
# ---------------------------------------------------------------------------

say("-" * 68)
say(f"FORWARD PASS  : {PROMPT!r}")

try:
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    say(f"  input shape : {tuple(input_ids.shape)}")

    _torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    with _torch.no_grad():
        out = model(input_ids)
    dt = time.time() - t0

    logits = out.logits if hasattr(out, "logits") else out[0]
    say(f"  logits shape: {tuple(logits.shape)}")
    say(f"  forward time: {dt:.2f}s")

    # ENTROPY IN FP32. The laptop has bf16, Kaggle's T4 does not. Computing in
    # fp32 on both machines is what keeps dev and production numbers identical.
    last = logits[0, -1, :].float()
    probs = _torch.softmax(last, dim=-1)
    entropy = -(probs * _torch.log(probs + 1e-12)).sum().item()
    max_entropy = float(np.log(logits.shape[-1]))

    say("")
    say(f"  ENTROPY     : {entropy:.4f} nats")
    say(f"  max possible: {max_entropy:.4f} nats (= ln vocab)")
    say(f"  normalised  : {entropy / max_entropy:.4f}   (0 = certain, 1 = uniform)")

    top_p, top_i = _torch.topk(probs, 5)
    say("")
    say("  top 5 next-token predictions:")
    for rank, (p, i) in enumerate(zip(top_p.tolist(), top_i.tolist()), 1):
        tok = tokenizer.decode([i])
        say(f"    {rank}. {tok!r:<20} p = {p:.4f}   (id {i})")

    say("")
    vram_now("after forward")
    say(f"  peak allocated: {gib(_torch.cuda.max_memory_allocated(0)):.2f} GiB")

except _torch.cuda.OutOfMemoryError:
    say("  OUT OF MEMORY during the forward pass.")
    say("  The weights fit but the activations did not. Tell me and we will")
    say("  shorten the sequence or move the output projection to CPU.")
except Exception:
    say("  FORWARD PASS FAILED:")
    tb = traceback.format_exc()
    print(tb)
    _report_lines.append(tb)

say("=" * 68)

os.makedirs("outputs", exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_report_lines) + "\n")
say(f"Saved to {REPORT_PATH} - paste that file into the chat.")
