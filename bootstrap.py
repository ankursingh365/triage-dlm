#!/usr/bin/env python3
"""
TRIAGE / CRADLE - Step 1 bootstrap.

Run this ONCE, from inside your triage-dlm folder:

    python bootstrap.py

It does two jobs:

  1. Builds the repository skeleton - all folders and placeholder files.
     Safe to run more than once. It NEVER overwrites a file that already
     exists, so you cannot lose work by re-running it.

  2. Prints a full environment report and saves it to
     outputs/step1_env_report.txt

Nothing here loads a model or touches the network. If it fails, the failure
is in your Python install, not in the project.
"""

import os
import platform
import shutil
import sys
import traceback
from datetime import datetime

# --------------------------------------------------------------------------
# PART 1 - repository skeleton
# --------------------------------------------------------------------------

FOLDERS = [
    "src",
    "scripts",
    "configs",
    "notebooks",
    "data",
    "data/raw",
    "data/trajectories",
    "outputs",
    "outputs/figures",
    "outputs/tables",
]

FILES = {
    "src/__init__.py": '"""TRIAGE - failure-mode-resolved hallucination detection for diffusion LLMs."""\n',

    "src/config.py": '''"""Central configuration and random seeds. Built in Step 5.

Every script imports from here so that seeds, paths and model names live in
exactly one place. Nothing else should hard-code a path.
"""
''',

    "src/logging_patch.py": '''"""Instrument a diffusion LLM's denoising loop. Built in Steps 4-6.

The hardest file in the project. For every denoising round and every
generated position we record exactly four things:

    - argmax token id        (int32)
    - probability of argmax  (float32)
    - entropy of the full distribution over vocab  (float32)
    - mask state: 1 if still masked, 0 if revealed  (uint8)

We NEVER log full logits. 32 rounds x 256 positions x 126,464 vocab x 4 bytes
is 4.1 GB per question. The four fields above are about 98 KB per question.
"""
''',

    "src/data.py": '''"""Dataset loaders and prompt construction. Built in Step 7.

CRITICAL: supporting context passages are withheld from the model at
generation time. With context, accuracy is too high and there are no
hallucinations to study. The passages are kept for labelling and for the
gold-evidence upper bound in Step 26 - just never shown to the model.
"""
''',

    "src/generate.py": '''"""Run instrumented generation at scale. Built in Steps 11-13.

Checkpoints every 50 questions. Colab and Kaggle both disconnect.
"""
''',

    "src/label_correctness.py": '''"""Label answers correct/incorrect with a Qwen3-8B judge. Built in Step 15.

Gate (Step 16): your manual review of 100 labels must agree with the judge
at least 90% of the time.

Note: call this "the judge" everywhere, never "the oracle".
"""
''',

    "src/label_failure_mode.py": '''"""Assign each wrong answer to one of three failure modes. Built in Step 17.

    1. INTERLEAVING   - flips between right and wrong, settles on wrong
    2. INCONSISTENT   - cycles through 3+ unrelated wrong candidates
    3. LOCKED-IN      - wrong answer fixed within the first ~30% of rounds
                        and never changes

Mode 3 is the whole point of the project. Trajectory detectors are
structurally blind to it because there is no hesitation to read.

Gate (Step 18): manual review of 100, plus a second annotator on 50 with
Cohen's kappa reported.
"""
''',

    "src/features.py": '''"""Turn a logged trajectory into roughly 25 numbers. Built in Step 19.

Exclusions that are NOT optional:
    - EOS and padding positions. LLaDA's SFT data causes early termination
      under low-confidence remasking; a settle-round feature computed over
      EOS measures that artefact, not truth.
    - Prompt positions. The prompt is never masked, so it has no trajectory.

In vanilla LLaDA a revealed position STAYS revealed (its confidence is set
to infinity). So "flips" must be measured on the argmax PREDICTION at
still-masked positions across rounds, not on revealed tokens changing.
"""
''',

    "src/baselines.py": '''"""Reproduce published detectors. Built in Steps 20-21.

    - Ave Entropy   GATE: must land at 62-65 AUROC on LLaDA.
                    TraceDet reports 62.8. If you get 50 or 75, there is a bug.
    - TRE
    - TraceDet      target ~72.0 (LLaDA) / ~80.8 (Dream)
    - DynHD         target ~84.2 - the strongest trajectory baseline

Do NOT use semantic entropy as an anchor. It transfers badly to diffusion
models and lands below chance on some settings.
"""
''',

    "src/evidence.py": '''"""External evidence stage - the CRADLE contribution. Built in Steps 24-27.

    BM25 retrieval over Wikipedia  ->  DeBERTa-large-MNLI entailment

Step 26 also runs the entailment stage on the WITHHELD gold passages to give
a perfect-evidence upper bound, so we can report what fraction of the
available headroom BM25 actually captures.
"""
''',

    "src/evaluate.py": '''"""Metrics and the key experiment. Built in Steps 22-23, 28-31.

AUROC, AUPRC and AURAC - reported overall AND separately per failure mode,
each with a 1000-resample bootstrap confidence interval.

Validation gates:
    - shuffle the labels and retrain: AUROC must fall to ~0.50 (leakage test)
    - CommonsenseQA: evidence must give ZERO gain (negative control)
"""
''',

    "scripts/.gitkeep": "",
    "configs/.gitkeep": "",
    "notebooks/.gitkeep": "",

    ".gitignore": """# data and model outputs - never commit these
data/
outputs/
*.pt
*.pth
*.safetensors
*.jsonl
*.npz

# python
.venv/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/

# editors and os
.vscode/
.idea/
.DS_Store
Thumbs.db

# secrets
.env
""",

    "README.md": """# TRIAGE

**T**rajectory **R**eliability **I**nspection **A**cross **G**enerative **E**rror-modes

Failure-mode-resolved hallucination detection for diffusion large language models.

## The problem

Diffusion LLMs write text by filling masked positions over ~32 denoising
rounds. Every published trajectory-based hallucination detector reads the
model's *hesitation* across those rounds. But one failure mode has no
hesitation to read: the model locks onto a wrong answer early and never
wavers. Trajectory detectors are structurally blind to it.

Nobody has measured detector performance separately by failure mode.

## Papers

| Method | Full form | Approach |
|---|---|---|
| **CRADLE** | **C**ross-checking **R**etrieved **A**ssertions to **D**etect **L**ocked-in **E**rrors | External evidence |
| **SABRE** | **S**parse **A**utoencoder-**B**ased **R**ecognition of **E**rrors | Internal sparse features |

## Status

Step 1 of 33. See the project charter for the full execution plan.
""",
}


def build_skeleton():
    made_dirs, made_files, skipped = [], [], []

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            made_dirs.append(folder)

    for path, content in FILES.items():
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(path):
            skipped.append(path)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            made_files.append(path)

    return made_dirs, made_files, skipped


# --------------------------------------------------------------------------
# PART 2 - environment report
# --------------------------------------------------------------------------

def env_report():
    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say("=" * 64)
    say("  TRIAGE - Step 1 environment report")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 64)
    say(f"Python        : {sys.version.split()[0]}")
    say(f"Platform      : {platform.system()} {platform.release()} ({platform.machine()})")
    say(f"Executable    : {sys.executable}")
    say("-" * 64)

    # ---- torch and GPU ----
    try:
        import torch
    except Exception as exc:
        say(f"torch         : NOT INSTALLED  ({type(exc).__name__}: {exc})")
        say("")
        say(">>> STOP HERE. Send me this output. torch must install first.")
        say("=" * 64)
        return lines

    say(f"torch         : {torch.__version__}")
    say(f"CUDA build    : {torch.version.cuda}")
    say(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        try:
            free_b, total_b = torch.cuda.mem_get_info(idx)
        except Exception:
            free_b, total_b = 0, props.total_memory
        say(f"GPU           : {props.name}")
        say(f"VRAM total    : {props.total_memory / 1024**3:.2f} GiB   <<< KEY NUMBER")
        say(f"VRAM free     : {free_b / 1024**3:.2f} GiB")
        say(f"Compute cap   : {props.major}.{props.minor}")
        try:
            say(f"bf16 support  : {torch.cuda.is_bf16_supported()}   <<< KEY NUMBER")
        except Exception as exc:
            say(f"bf16 support  : could not determine ({type(exc).__name__})")
    else:
        say("GPU           : none visible to torch")
        say("                (a CPU-only torch wheel is the usual cause)")

    # ---- supporting libraries ----
    say("-" * 64)
    for pkg in ["transformers", "accelerate", "bitsandbytes", "datasets",
                "safetensors", "huggingface_hub", "numpy", "scipy",
                "sklearn", "pandas"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            say(f"{pkg:<14}: {ver}")
        except Exception as exc:
            say(f"{pkg:<14}: NOT INSTALLED  ({type(exc).__name__})")

    # ---- bitsandbytes deeper check ----
    say("-" * 64)
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        BitsAndBytesConfig(load_in_4bit=True)
        say("4-bit config  : OK - bitsandbytes and transformers agree   <<< KEY NUMBER")
    except Exception as exc:
        say(f"4-bit config  : FAILED - {type(exc).__name__}: {exc}")
        say("                (this is what Step 2 needs; we will fix it)")

    # ---- disk and memory ----
    say("-" * 64)
    try:
        total, _used, free = shutil.disk_usage(".")
        say(f"Disk here     : {free / 1024**3:.0f} GiB free of {total / 1024**3:.0f} GiB")
    except Exception:
        say("Disk here     : could not determine")

    try:
        import psutil
        say(f"System RAM    : {psutil.virtual_memory().total / 1024**3:.1f} GiB")
    except Exception:
        say("System RAM    : (optional: pip install psutil)")

    say("=" * 64)
    return lines


# --------------------------------------------------------------------------

def main():
    print()
    print("### PART 1 - building the repository skeleton")
    print()
    try:
        dirs, files, skipped = build_skeleton()
        print(f"  folders created : {len(dirs)}")
        print(f"  files created   : {len(files)}")
        print(f"  already existed : {len(skipped)} (left untouched)")
        for f in files:
            print(f"    + {f}")
    except Exception:
        print("  SKELETON BUILD FAILED:")
        traceback.print_exc()

    print()
    print("### PART 2 - environment report")
    print()
    lines = env_report()

    try:
        os.makedirs("outputs", exist_ok=True)
        out = os.path.join("outputs", "step1_env_report.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print()
        print(f"Saved to {out} - copy that file's contents into the chat.")
    except Exception:
        print("(could not save the report to a file; copy it from above instead)")
    print()


if __name__ == "__main__":
    main()
