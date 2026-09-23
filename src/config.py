"""TRIAGE - central configuration. Import this FIRST, before anything else.

    from src import config          # sets up the cache, then everything else

Why first: HuggingFace libraries read HF_HOME at *import* time and cache the
value. If transformers is imported before this module runs, the setting is
ignored and models land on your C: drive. So this import goes above
`import torch`, `import transformers`, everything.

This file is the only place that knows about paths, seeds and model names.
Nothing else in the project should hard-code any of them.
"""

import os
import random
import sys
from pathlib import Path

# ===========================================================================
# 1. MODEL CACHE  -  must happen before any HuggingFace import
# ===========================================================================

def _detect_cache_dir() -> str:
    """Pick a sensible cache location for whatever machine we are on."""
    # Kaggle. /kaggle/working is capped at 20 GB and is persisted as the
    # notebook's output; LLaDA-8B alone is ~16 GB, so caching a model there
    # fills the quota and the run dies mid-download. /kaggle/temp is much
    # larger and is NOT persisted, which is exactly right for model weights -
    # they are re-downloadable, and only trajectories need to survive.
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists():
        for candidate in ("/kaggle/temp", "/kaggle/working"):
            if Path(candidate).exists():
                return str(Path(candidate) / "hf-cache")
        return "/kaggle/working/hf-cache"

    # Windows: prefer a big data drive over C:.
    if os.name == "nt":
        for drive in ("D:", "E:", "F:"):
            if Path(drive + "\\").exists():
                return drive + "\\hf-cache"

    # Linux / macOS / fallback.
    return str(Path.home() / "hf-cache")


# An HF_HOME already in the environment wins - that lets you override without
# editing this file. Otherwise we choose one.
HF_HOME = os.environ.get("HF_HOME") or _detect_cache_dir()
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
Path(HF_HOME).mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 2. PATHS
# ===========================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAJ_DIR = DATA_DIR / "trajectories"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

for _d in (DATA_DIR, RAW_DIR, TRAJ_DIR, OUT_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 3. SEEDS  -  charter warning #9: fix and log every seed
# ===========================================================================

SEED = 20260909


def set_seeds(seed: int = SEED) -> int:
    """Seed python, numpy and torch. Returns the seed so callers can log it."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed

# ===========================================================================
# 4. MODELS
# ===========================================================================

# Debugging only. Never used for reported results.
MODEL_DEBUG = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

# The two diffusion models the paper reports on.
MODEL_LLADA = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_DREAM = "Dream-org/Dream-v0-Instruct-7B"

# Supporting models.
MODEL_JUDGE = "Qwen/Qwen3-8B"                          # correctness labels
MODEL_NLI = "microsoft/deberta-large-mnli"             # entailment, Step 25

# ===========================================================================
# 5. GENERATION
# ===========================================================================

# TraceDet found 64 best. 128 was worse. Do not change without re-reading
# their experiments section.
GEN_LENGTH = 64
DENOISING_STEPS = 64
BLOCK_LENGTH = 32

# Cache key for generated trajectories.
#
# Trajectories are cached by question id so Phase C can resume after a crash.
# That cache is only valid for the decoder settings that produced it: on
# 2026-09-22 a config change was silently ignored because 300 files from the
# previous settings were already on the Volume, and the run reported new
# settings in its banner while re-reading old data. Including the settings in
# the path makes the cache correct by construction - change any of them and the
# old files are simply a different directory.
RUN_TAG = f"g{GEN_LENGTH}s{DENOISING_STEPS}b{BLOCK_LENGTH}"

# ===========================================================================
# 6. SELF-DESCRIPTION
# ===========================================================================

def describe() -> str:
    lines = [
        "TRIAGE configuration",
        f"  repo root   : {REPO_ROOT}",
        f"  HF_HOME     : {HF_HOME}",
        f"  seed        : {SEED}",
        f"  gen length  : {GEN_LENGTH} over {DENOISING_STEPS} denoising steps",
        f"  python      : {sys.version.split()[0]}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    free = None
    try:
        import shutil
        total, _used, free = shutil.disk_usage(HF_HOME)
        print(f"  cache drive : {free / 1024**3:.0f} GiB free "
              f"of {total / 1024**3:.0f} GiB")
    except Exception:
        pass
    print()
    print("If HF_HOME above points at your data drive, you are set.")
    print("No environment variable needed - this file handles it.")