"""
Modal entry point for TRIAGE.
=============================

Runs the project's scripts on rented GPUs instead of the laptop. Everything in
`src/` and `scripts/` is used unchanged - this file only supplies a machine,
a filesystem, and the plumbing between them.

    modal run modal_app.py::benchmark                 # Step 8, A100 40GB
    modal run modal_app.py::benchmark --gpu L4        # cheaper, still bf16
    modal run modal_app.py::shell                     # interactive poke-around

Why Modal rather than Kaggle
----------------------------
The project measures entropy and confidence trajectories. 4-bit quantisation
perturbs the output distribution, which IS the quantity being measured. The
Step 20 gate requires reproducing Ave Entropy at 62-65 AUROC, and TraceDet's
62.8 was measured on an A40 in full precision. Run quantised and get 57, and an
implementation bug is indistinguishable from a quantiser artefact - the only
falsifiable anchor in the project stops working.

Kaggle's T4 has 16 GB and cannot hold LLaDA-8B in bf16 (~17 GB). An L4 (24 GB)
or A100 (40 GB) can. That is the whole argument; the cost is secondary.

What it costs
-------------
Modal bills per second. At the time of writing:

    T4          $0.59/hr    16 GB   cannot hold the model in bf16
    L4          $0.80/hr    24 GB   bf16, small batches
    A100 40GB   $2.10/hr    40 GB   bf16, batch 16+
    H100        $3.95/hr    80 GB   unnecessary here

Phase C is ~9,000 generations. On an A100 at batch 16 that is roughly 5
GPU-hours, about $11. The Starter plan includes $30/month of free compute, so
Phase C and an ablation both fit inside one month's allowance.

Persistence
-----------
Three Volumes, so nothing is re-downloaded or lost between runs:

    triage-hf-cache    model weights. LLaDA-8B is ~17 GB; download it once.
    triage-data        trajectories written by the generation steps
    triage-outputs     reports, tables, figures

Volumes cost $0.09/GiB/month with 1 TiB free, so this is effectively free.
Without them every run re-downloads 17 GB, which is both slow and billable.

Note `HF_HOME` is set in the container environment below. `src/config.py`
honours an existing `HF_HOME` before falling back to its own detection, so it
needs no Modal-specific changes - the same file works on Windows, Kaggle and
here.
"""

import modal

APP_NAME = "triage-cradle"
REMOTE_ROOT = "/root/triage"

# Per-second prices, converted to $/hour, used only to print a cost estimate.
# Update if Modal's pricing changes; nothing functional depends on them.
GPU_HOURLY = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Image
#
# transformers is pinned below 5: v5 removed internals that LLaDA's remote
# modelling code imports (huggingface/transformers issues #44561, #45020), and
# LLaDA requires trust_remote_code.
#
# bitsandbytes is still installed even though large GPUs load in full precision,
# because the fallback path in model_utils needs it if a smaller GPU is chosen.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers<5",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "psutil",
    )
    .env({
        # src/config.py honours an existing HF_HOME, so this is all that is
        # needed to point the cache at the persistent Volume.
        "HF_HOME": "/cache/hf",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        # Kaggle/Colab-style progress bars are noise in Modal's log stream.
        "HF_HUB_DISABLE_PROGRESS_BARS": "0",
    })
    .add_local_dir("src", f"{REMOTE_ROOT}/src")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
)

cache_vol = modal.Volume.from_name("triage-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("triage-data", create_if_missing=True)
out_vol = modal.Volume.from_name("triage-outputs", create_if_missing=True)

VOLUMES = {
    "/cache": cache_vol,
    f"{REMOTE_ROOT}/data": data_vol,
    f"{REMOTE_ROOT}/outputs": out_vol,
}


def _run_script(script: str, gpu: str) -> str:
    """Execute one of the project's scripts and stream its output.

    Runs it as a subprocess rather than importing it, so the scripts stay
    exactly as they are on the laptop - no Modal-specific branching inside the
    research code.
    """
    import subprocess
    import sys
    import time

    print("=" * 78)
    print(f"  Modal: {script} on {gpu}")
    print("=" * 78, flush=True)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, f"scripts/{script}"],
        cwd=REMOTE_ROOT,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    print(proc.stdout)
    if proc.stderr.strip():
        print("--- stderr ---")
        print(proc.stderr)

    rate = GPU_HOURLY.get(gpu, 0.0)
    print("=" * 78)
    print(f"  wall clock  : {elapsed/60:.1f} min")
    if rate:
        print(f"  GPU         : {gpu} at ${rate:.2f}/hr")
        print(f"  this run    : ${elapsed/3600*rate:.2f}")
        print(f"  exit code   : {proc.returncode}")
    print("=" * 78, flush=True)

    # Persist whatever the script wrote before the container disappears.
    data_vol.commit()
    out_vol.commit()
    cache_vol.commit()

    return proc.stdout


# ---------------------------------------------------------------------------
# Generic runners. One per GPU type, because Modal fixes the accelerator at
# decoration time - it cannot be chosen at call time. Everything else is
# parameterised, so future steps need no changes here:
#
#     modal run modal_app.py::run --script step9_base_rate.py
#     modal run modal_app.py::run --script step11_generate.py --gpu L4
#
# Timeout is 4 hours. Generation steps are long, and an uncapped run on a
# metered GPU is the one way this project can quietly cost real money.
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, gpu="A100-40GB", timeout=4 * 60 * 60)
def _run_a100(script: str):
    return _run_script(script, "A100-40GB")


@app.function(image=image, volumes=VOLUMES, gpu="L4", timeout=4 * 60 * 60)
def _run_l4(script: str):
    return _run_script(script, "L4")


# CPU-only runner. Analysis steps that read cached trajectories need no GPU,
# and at $0.0000131 per core-second they are effectively free. Never rent an
# A100 to re-score a CSV.
@app.function(image=image, volumes=VOLUMES, cpu=2.0, memory=8192,
              timeout=60 * 60)
def _run_cpu(script: str):
    return _run_script(script, "CPU")


@app.local_entrypoint()
def run_cpu(script: str):
    """Run an analysis script with no GPU at all.

        modal run modal_app.py::run_cpu --script step9b_rescore.py

    For anything that reads trajectories rather than generating them:
    re-scoring, feature screens, evaluation, figures.
    """
    _run_cpu.remote(script)


@app.local_entrypoint()
def run(script: str, gpu: str = "A100-40GB"):
    """Run any script from scripts/ on a rented GPU.

        modal run modal_app.py::run --script step9_base_rate.py
        modal run modal_app.py::run --script step9_base_rate.py --gpu L4

    The script name is relative to scripts/. Outputs land on the Volumes and
    survive the container, so fetch them afterwards with:

        modal volume get triage-outputs <filename> .
    """
    if gpu.upper() == "L4":
        _run_l4.remote(script)
    else:
        _run_a100.remote(script)


@app.local_entrypoint()
def benchmark(gpu: str = "A100-40GB"):
    """Step 8 - time LLaDA-8B and size the Phase C budget.

        modal run modal_app.py::benchmark
        modal run modal_app.py::benchmark --gpu L4

    First run downloads ~17 GB into the cache Volume and takes 10-20 minutes.
    Later runs start in about a minute.

    Run it on BOTH once. The A100 is 2.6x the hourly rate of the L4, so it only
    wins if it is more than 2.6x faster per question - which depends on batching,
    and batching is not implemented yet. The cheaper card may well be the right
    production choice, and this is a $2 experiment that settles it.
    """
    if gpu.upper() == "L4":
        _run_l4.remote("step8_kaggle_benchmark.py")
    else:
        _run_a100.remote("step8_kaggle_benchmark.py")


@app.function(image=image, volumes=VOLUMES, gpu="L4", timeout=60 * 60)
def _inspect():
    """Report what is on the Volumes, and whether the GPU sees full precision."""
    import subprocess
    import sys

    print("--- GPU ---")
    subprocess.run(["nvidia-smi"], check=False)

    print("\n--- volumes ---")
    for path in ("/cache/hf", f"{REMOTE_ROOT}/data", f"{REMOTE_ROOT}/outputs"):
        subprocess.run(["du", "-sh", path], check=False)

    print("\n--- config as seen here ---")
    subprocess.run([sys.executable, "src/config.py"], cwd=REMOTE_ROOT, check=False)

    print("\n--- load strategy this card will choose ---")
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from src import model_utils as m;"
        "print('dtype     :', m.describe_dtype());"
        "print('configs   :');"
        "[print('   ', label) for label, _ in m._quant_configs()]"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REMOTE_ROOT, check=False)


@app.local_entrypoint()
def inspect():
    """Cheap sanity check before spending money on a long run.

        modal run modal_app.py::inspect

    Confirms the code uploaded, the Volumes mounted, the cache is where it
    should be, and - the important one - that `model_utils` offers config 0
    (full precision, no quantisation) on this card. If config 0 is missing, the
    GPU is too small and the run would silently fall back to 4-bit, which is the
    thing this whole move to Modal exists to avoid.
    """
    _inspect.remote()