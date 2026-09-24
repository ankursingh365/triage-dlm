"""
Modal entry point for TRIAGE.
=============================

Runs the project's scripts on rented GPUs instead of the laptop. Everything in
`src/` and `scripts/` is used unchanged - this file only supplies a machine,
a filesystem, and the plumbing between them.

    modal run --detach modal_app.py::run --script step11_phasec_generate.py
    modal run modal_app.py::run_cpu --script step10d_fit_and_apply.py
    modal run modal_app.py::inspect

USE --detach FOR ANYTHING LONGER THAN A FEW MINUTES
---------------------------------------------------
`modal run` without it creates an EPHEMERAL app, which Modal stops the moment
the client disconnects. The client is the process on your laptop. So a closed
lid, a sleeping machine, a dropped wifi connection or a closed terminal all
kill the GPU job - and because the Volume is committed only when the job
finishes, everything it had generated is lost with it.

This is not hypothetical: a Phase C run died this way, with

    socket.gaierror: [Errno 11001] getaddrinfo failed

which is the laptop's DNS failing after it went to sleep, not a problem in the
container. From Modal's docs: "Ephemeral Apps are stopped automatically when
the calling program exits, or when the server detects that the client is no
longer connected", and "The --detach flag ensures training will continue even
if you close your terminal or turn off your computer."

**Rule for this project: any GPU run goes through `modal run --detach`.**
Watch it at modal.com/apps, or with `modal app logs <app-id>`.

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

Phase C is 11,250 generations at a measured 2.01 s/question: about 6.3
GPU-hours, roughly $13. The Starter plan includes $30/month of free compute.

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

# Modal's hard ceiling on a single Function call is 24 hours. The previous
# value here was 4, which was a guess, and it shaped a whole generation script
# around splitting work that did not need splitting. Phase C fits comfortably.
MAX_TIMEOUT = 24 * 60 * 60

# How often the Volumes are committed WHILE a script is still running.
#
# A Volume's contents are persisted on commit. `_run_script` used to commit
# only after the subprocess returned, so a crash, a timeout or a client
# disconnect discarded every file written during the run. For a six-hour
# generation job that is hours of GPU time thrown away. Committing every few
# minutes bounds the loss to the last interval.
COMMIT_EVERY_S = 5 * 60

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
        # Unbuffered, so a long run's progress lines appear as they happen
        # rather than all at once when the subprocess finally exits.
        "PYTHONUNBUFFERED": "1",
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


def _commit_all() -> None:
    """Persist all three Volumes. Safe to call repeatedly."""
    for vol in (data_vol, out_vol, cache_vol):
        try:
            vol.commit()
        except Exception as exc:          # a failed commit must not kill the run
            print(f"  [volume commit warning] {exc}", flush=True)


def _run_script(script: str, gpu: str) -> int:
    """Execute one of the project's scripts, streaming output and committing.

    Runs it as a subprocess rather than importing it, so the scripts stay
    exactly as they are on the laptop - no Modal-specific branching inside the
    research code.

    Two properties this needs that the obvious `subprocess.run(...,
    capture_output=True)` does not have:

    **Output appears as it happens.** `capture_output` buffers everything until
    the process exits, so a six-hour job shows nothing for six hours and then
    prints a wall of text. Progress lines are the only way to tell a slow run
    from a hung one.

    **The Volume is committed periodically.** Committing only at the end means a
    crash, a timeout, or a client disconnect discards every file the run
    produced. Committing every few minutes bounds that loss to one interval.
    """
    import subprocess
    import sys
    import time

    print("=" * 78)
    print(f"  Modal: {script} on {gpu}")
    print(f"  committing volumes every {COMMIT_EVERY_S // 60} min")
    print("=" * 78, flush=True)

    t0 = time.time()
    last_commit = t0

    proc = subprocess.Popen(
        [sys.executable, "-u", f"scripts/{script}"],
        cwd=REMOTE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream, committing between lines. Reading line by line keeps memory flat
    # no matter how chatty the script is.
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
        now = time.time()
        if now - last_commit >= COMMIT_EVERY_S:
            _commit_all()
            print(f"  [volumes committed at {(now - t0) / 60:.0f} min]", flush=True)
            last_commit = now

    proc.wait()
    elapsed = time.time() - t0

    # Final commit, always - including after a non-zero exit, because a script
    # that failed halfway still generated real work worth keeping.
    _commit_all()

    rate = GPU_HOURLY.get(gpu, 0.0)
    print("=" * 78)
    print(f"  wall clock  : {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)")
    if rate:
        print(f"  GPU         : {gpu} at ${rate:.2f}/hr")
        print(f"  this run    : ${elapsed / 3600 * rate:.2f}")
    print(f"  exit code   : {proc.returncode}")
    print("=" * 78, flush=True)

    return proc.returncode


# ---------------------------------------------------------------------------
# Generic runners. One per GPU type, because Modal fixes the accelerator at
# decoration time - it cannot be chosen at call time. Everything else is
# parameterised, so future steps need no changes here:
#
#     modal run --detach modal_app.py::run --script step11_phasec_generate.py
#     modal run modal_app.py::run --script step9_base_rate.py --gpu L4
# ---------------------------------------------------------------------------

@app.function(image=image, volumes=VOLUMES, gpu="A100-40GB", timeout=MAX_TIMEOUT)
def _run_a100(script: str):
    return _run_script(script, "A100-40GB")


@app.function(image=image, volumes=VOLUMES, gpu="L4", timeout=MAX_TIMEOUT)
def _run_l4(script: str):
    return _run_script(script, "L4")


# CPU-only runner. Analysis steps that read cached trajectories need no GPU,
# and at $0.0000131 per core-second they are effectively free. Never rent an
# A100 to re-score a CSV.
@app.function(image=image, volumes=VOLUMES, cpu=2.0, memory=8192,
              timeout=MAX_TIMEOUT)
def _run_cpu(script: str):
    return _run_script(script, "CPU")


@app.local_entrypoint()
def run_cpu(script: str):
    """Run an analysis script with no GPU at all.

        modal run modal_app.py::run_cpu --script step10d_fit_and_apply.py

    For anything that reads trajectories rather than generating them:
    re-scoring, feature screens, evaluation, figures.
    """
    _run_cpu.remote(script)


@app.local_entrypoint()
def run(script: str, gpu: str = "A100-40GB"):
    """Run any script from scripts/ on a rented GPU.

        modal run --detach modal_app.py::run --script step11_phasec_generate.py
        modal run modal_app.py::run --script step9_base_rate.py --gpu L4

    **Use --detach for anything longer than a few minutes.** Without it the App
    is ephemeral and dies with the client - a sleeping laptop is enough to kill
    a six-hour job and lose everything it had generated.

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

    print("\n--- trajectory counts ---")
    subprocess.run(["bash", "-c",
                    f"find {REMOTE_ROOT}/data/trajectories -name '*.npz' | "
                    f"sed 's|/[^/]*$||' | sort | uniq -c"], check=False)

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
    should be, how many trajectories exist per directory, and - the important
    one - that `model_utils` offers config 0 (full precision, no quantisation)
    on this card. If config 0 is missing, the GPU is too small and the run would
    silently fall back to 4-bit, which is the thing this whole move to Modal
    exists to avoid.
    """
    _inspect.remote()