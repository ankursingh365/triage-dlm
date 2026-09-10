"""
Model loading utilities for TRIAGE.
==================================

Loading an 8B-class diffusion language model in 4-bit on a 6 GB consumer GPU is
fiddly enough that it deserves its own module rather than being copy-pasted into
every script. Steps 2, 4, 11 and 15 all call `load_model_and_tokenizer` from here.

Two problems this module solves
-------------------------------

**1. GPU memory.** LLaDA-family models have an *untied* output head, meaning the
input embedding and the output projection are two separate matrices of
`vocab_size x hidden_size`. bitsandbytes quantises neither by default, so a
large chunk of the model stays in bf16 no matter what `load_in_4bit` says. For
LLaDA-MoE that is 2 x 157,184 x 2,048 = 644M parameters = ~1.29 GB that cannot
be compressed by the standard path. We therefore try three configurations in
descending order of speed and stop at the first that fits.

**2. Host memory on Windows.** `safetensors` loads shards by memory-mapping
them. On Windows, memory-mapping an N-gigabyte file requires the system to
commit N gigabytes of virtual address space, which is bounded by
RAM + pagefile. On a 16 GB laptop with the default auto-managed pagefile,
loading a 15 GB model raises

    OSError: The paging file is too small for this operation to complete.
             (os error 1455)

This is not a bug in the model or in transformers - it is a system
configuration limit. `preflight_memory_check` warns about it *before* the
30-minute load begins, and the loader catches it afterwards and explains the
fix rather than dumping a raw traceback.

References
----------
LLaDA               Nie et al., arXiv:2502.09992 (NeurIPS 2025 Oral)
LLaDA-MoE           inclusionAI, arXiv:2509.24389
bitsandbytes NF4    Dettmers et al., QLoRA, arXiv:2305.14314
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Callable, Optional, Tuple

# `config` sets HF_HOME before any HuggingFace library is imported, so it has to
# come first. Importing it here means callers get the right cache automatically.
from . import config


# ===========================================================================
# Memory reporting
# ===========================================================================

def gib(n_bytes: float) -> float:
    """Bytes -> gibibytes. Used everywhere we print a memory figure."""
    return n_bytes / 1024 ** 3


# ===========================================================================
# Hardware adaptation
# ===========================================================================

def preferred_dtype():
    """bfloat16 where the GPU supports it, float16 otherwise.

    This project develops on an RTX 4050 (Ada, compute 8.9) which has bfloat16,
    and runs production on Kaggle's T4 (Turing, compute 7.5) which does NOT.
    Passing `torch.bfloat16` on a T4 either errors or silently falls back to a
    slow emulated path, so the dtype has to follow the hardware.

    **Entropy is computed in float32 regardless of what this returns.** That is
    the whole point: the compute dtype may differ between machines, but the
    measurement must not. A float16 softmax over a 126k-157k vocabulary loses
    enough precision in the tail to shift entropy measurably, which would make
    laptop numbers and Kaggle numbers incomparable.
    """
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def describe_dtype() -> str:
    import torch
    dt = preferred_dtype()
    name = "bfloat16" if dt is torch.bfloat16 else "float16"
    why = ("GPU supports bf16" if dt is torch.bfloat16
           else "GPU lacks bf16 (pre-Ampere) - using fp16")
    return f"{name} ({why})"


# Mask token ids for models whose tokenizer does not expose `mask_token_id`.
#
# LLaDA-8B is the important case: its mask id, 126336, is a RESERVED token that
# the tokenizer does not declare, which is why the reference `generate.py` passes
# it as a hard-coded default argument. Reading `tokenizer.mask_token_id` there
# returns None, and a None mask id produces a silently wrong run rather than a
# crash - every position stays "unmasked", nothing is ever revealed, and the
# trajectory is empty.
KNOWN_MASK_IDS = {
    "GSAI-ML/LLaDA-8B-Instruct": 126336,
    "GSAI-ML/LLaDA-8B-Base": 126336,
    "GSAI-ML/LLaDA-1.5": 126336,
}


def resolve_mask_id(tokenizer, model_id: str) -> int:
    """The mask token id for this model. Raises rather than returning None.

    Order: the tokenizer's own declaration, then the table above, then failure.
    Never guesses - a wrong mask id produces plausible-looking output that
    describes nothing, which is the worst failure mode available here.
    """
    mid = getattr(tokenizer, "mask_token_id", None)
    if mid is not None:
        return int(mid)

    for known, value in KNOWN_MASK_IDS.items():
        if known.lower() in (model_id or "").lower():
            return value

    raise RuntimeError(
        f"Cannot determine the mask token id for {model_id!r}. The tokenizer "
        f"does not declare one and it is not in KNOWN_MASK_IDS. Add it to "
        f"src/model_utils.py rather than guessing - a wrong mask id yields a "
        f"trajectory that looks fine and means nothing."
    )


def vram_snapshot() -> Optional[dict]:
    """Current GPU memory, or None if there is no CUDA device.

    `allocated` is what PyTorch is actively using. `reserved` includes blocks
    PyTorch is caching and can release under pressure - so a small `free`
    figure alongside a large `reserved` one is not necessarily a problem.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info(0)
    return {
        "allocated": gib(torch.cuda.memory_allocated(0)),
        "reserved": gib(torch.cuda.memory_reserved(0)),
        "free": gib(free_b),
        "total": gib(total_b),
    }


def preflight_memory_check(model_size_gb: float = 15.0,
                           log: Callable[[str], None] = print) -> bool:
    """Report host and device memory, and warn about known failure modes.

    Called before a load so problems surface in seconds rather than after a
    long download. Returns False if a hard blocker is found (no GPU), True
    otherwise - warnings alone do not stop the caller, because the limits are
    soft and often still work.

    Parameters
    ----------
    model_size_gb : approximate on-disk size of the checkpoint, used to judge
        whether the Windows commit limit is likely to be exceeded.
    log : where to send output. Scripts pass their own `say()` so the text
        lands in the saved report as well as on screen.
    """
    import torch

    log("-" * 70)
    log("PRE-FLIGHT MEMORY CHECK")

    # ---- GPU -------------------------------------------------------------
    if not torch.cuda.is_available():
        log("  GPU            : none visible to torch - cannot continue")
        return False

    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    log(f"  GPU            : {props.name}")
    log(f"  VRAM free/total: {gib(free_b):.2f} / {gib(total_b):.2f} GiB")

    if gib(free_b) < 4.5:
        log("  WARNING        : under 4.5 GiB free. Close Chrome and any other")
        log("                   GPU application, then run this again.")

    # ---- Host RAM and the Windows commit limit ---------------------------
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        log(f"  System RAM     : {gib(vm.available):.1f} GiB available "
            f"of {gib(vm.total):.1f} GiB")
        log(f"  Pagefile/swap  : {gib(sm.free):.1f} GiB free "
            f"of {gib(sm.total):.1f} GiB")

        # safetensors mmaps each shard, so the peak commit is roughly the
        # checkpoint size. Compare against RAM + pagefile, not RAM alone.
        commit_available = gib(vm.available) + gib(sm.free)
        log(f"  Commit headroom: {commit_available:.1f} GiB "
            f"(need roughly {model_size_gb:.0f} GiB to memory-map the shards)")

        if os.name == "nt" and commit_available < model_size_gb * 1.3:
            log("")
            log("  WARNING: not much commit headroom. On Windows this shows up")
            log("           as 'OSError 1455: the paging file is too small'.")
            log("           Fix: System Properties -> Advanced -> Performance")
            log("           -> Settings -> Advanced -> Virtual memory -> Change,")
            log("           untick automatic, set a custom size on a data drive")
            log("           (16384 MB initial / 65536 MB maximum), then reboot.")
    except ImportError:
        log("  System RAM     : install psutil to report this")

    log("-" * 70)
    return True


# ===========================================================================
# Quantisation configurations
# ===========================================================================

def _quant_configs():
    """The three 4-bit strategies, ordered fastest-first.

    A  Everything on the GPU. NF4 with double quantisation (which compresses
       the quantisation constants themselves, worth roughly 0.4 GB here).
       Fastest, and what we expect to use.

    B  Same, but `llm_int8_skip_modules=[]` stops transformers excluding the
       output head from quantisation. Frees ~0.6 GB on an untied model at some
       cost to output precision - acceptable for a debug model, and we do not
       report results from this one.

    C  `device_map="auto"` lets accelerate place whatever does not fit into
       system RAM. Always works, but every offloaded layer is copied across
       PCIe on each forward pass, so expect a large slowdown. Last resort.
    """
    from transformers import BitsAndBytesConfig

    base = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # normal-float 4, better than fp4
        bnb_4bit_use_double_quant=True,     # quantise the quantisation constants
        # Follows the hardware: bf16 on Ada, fp16 on Kaggle's T4. Hard-coding
        # bf16 here is the single most likely way to break the Kaggle run.
        bnb_4bit_compute_dtype=preferred_dtype(),
    )

    return [
        ("A  all on GPU, NF4 + double quant",
         dict(quantization_config=BitsAndBytesConfig(**base),
              device_map={"": 0})),

        ("B  as A, output head quantised too (frees ~0.6 GB)",
         dict(quantization_config=BitsAndBytesConfig(**base,
                                                     llm_int8_skip_modules=[]),
              device_map={"": 0})),

        ("C  split GPU/CPU via accelerate (slow fallback)",
         dict(quantization_config=BitsAndBytesConfig(**base),
              device_map="auto",
              max_memory={0: "4GiB", "cpu": "12GiB"})),
    ]


def _is_windows_pagefile_error(exc: BaseException) -> bool:
    """True for the Windows commit-limit error, which needs a specific fix.

    Worth detecting explicitly: the raw message mentions a paging file and
    gives no hint that the cure is a system setting rather than a code change.
    """
    if getattr(exc, "winerror", None) == 1455:
        return True
    return "paging file" in str(exc).lower()


# ===========================================================================
# The loader
# ===========================================================================

def load_model_and_tokenizer(model_id: Optional[str] = None,
                             log: Callable[[str], None] = print,
                             seed: bool = True) -> Tuple[object, object, str]:
    """Load a diffusion LM in 4-bit, trying each configuration until one fits.

    Returns
    -------
    (model, tokenizer, config_label)
        `config_label` records which strategy succeeded, so scripts can log it
        alongside their results. Reproducibility depends on knowing this:
        configuration B quantises the output head and will not produce
        bit-identical logits to A.

    Raises
    ------
    RuntimeError
        If every configuration fails. The message names the underlying cause,
        with the pagefile fix spelled out when that is what went wrong.

    Notes
    -----
    `trust_remote_code=True` is mandatory. LLaDA is not a built-in transformers
    architecture; its modelling code ships in the model repository. This is also
    why the project pins `transformers<5` - v5 removed internals that these
    remote modelling files import (huggingface/transformers issues #44561,
    #45020).

    LLaDA exposes itself through `AutoModelForCausalLM` on some checkpoints and
    only through `AutoModel` on others, so we try both before giving up.
    """
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or config.MODEL_DEBUG
    if seed:
        config.set_seeds()

    log(f"Loading {model_id}")
    log(f"  compute dtype  : {describe_dtype()}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    log(f"  tokenizer vocab: {len(tokenizer):,}")
    mask_id = resolve_mask_id(tokenizer, model_id)
    declared = getattr(tokenizer, "mask_token_id", None)
    source = "tokenizer" if declared is not None else "KNOWN_MASK_IDS table"
    log(f"  mask token id  : {mask_id}  (from {source})")

    last_error: Optional[BaseException] = None

    for label, kwargs in _quant_configs():
        log("")
        log(f"Trying config {label}")

        for cls in (AutoModelForCausalLM, AutoModel):
            try:
                # transformers renamed `torch_dtype` to `dtype`; support both
                # so the repo works across 4.4x-4.5x without pinning a patch.
                dt = preferred_dtype()
                try:
                    model = cls.from_pretrained(
                        model_id, trust_remote_code=True,
                        dtype=dt, **kwargs)
                except TypeError:
                    # transformers renamed torch_dtype -> dtype across 4.4x/4.5x
                    model = cls.from_pretrained(
                        model_id, trust_remote_code=True,
                        torch_dtype=dt, **kwargs)

                model.eval()   # no dropout, no gradient bookkeeping
                log(f"  loaded via {cls.__name__}")
                snap = vram_snapshot()
                if snap:
                    log(f"  VRAM allocated : {snap['allocated']:.2f} GiB "
                        f"(reserved {snap['reserved']:.2f})")
                return model, tokenizer, label

            except torch.cuda.OutOfMemoryError as exc:
                last_error = exc
                log("  out of GPU memory - trying the next configuration")
                break            # a different class will not use less VRAM

            except OSError as exc:
                last_error = exc
                if _is_windows_pagefile_error(exc):
                    # A system limit. Retrying with other settings is pointless.
                    raise RuntimeError(
                        "Windows could not memory-map the model shards: the "
                        "paging file is too small (OSError 1455).\n\n"
                        "This is a system setting, not a code problem. Fix it:\n"
                        "  1. Windows key -> 'advanced system settings'\n"
                        "  2. Performance -> Settings -> Advanced\n"
                        "  3. Virtual memory -> Change\n"
                        "  4. Untick 'Automatically manage...'\n"
                        "  5. Pick a data drive, Custom size,\n"
                        "     initial 16384 MB / maximum 65536 MB\n"
                        "  6. Set -> OK -> reboot\n\n"
                        "Closing other applications first may also be enough."
                    ) from exc
                log(f"  {type(exc).__name__}: {exc}")

            except Exception as exc:
                last_error = exc
                log(f"  {type(exc).__name__}: {exc}")

        # Release anything a failed attempt left behind before the next try.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raise RuntimeError(
        f"Could not load {model_id} under any configuration. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


# ===========================================================================
# Entropy
# ===========================================================================

def position_entropy(logits, index: int, batch: int = 0):
    """Shannon entropy, in nats, of the model's distribution at one position.

    Returns `(entropy, normalised_entropy, probs)` where `normalised_entropy`
    divides by ln(vocab) so 0 means completely certain and 1 means uniform.
    That normalisation is what makes entropies comparable across LLaDA
    (126,464 tokens) and Dream (different vocabulary) in later steps.

    **Always computed in float32, deliberately.** Development happens on an Ada
    laptop GPU that supports bfloat16; production runs on Kaggle's T4, which
    does not. Casting to fp32 before the softmax on both machines is what keeps
    the two sets of numbers comparable - and a bf16 softmax over a 157k
    vocabulary loses enough precision in the tail to shift entropy measurably.

    The `1e-12` guards `log(0)` for tokens the model has effectively ruled out.
    """
    import torch
    vec = logits[batch, index, :].float()
    probs = torch.softmax(vec, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    import math
    return entropy, entropy / math.log(vec.shape[-1]), probs