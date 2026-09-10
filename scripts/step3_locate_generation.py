#!/usr/bin/env python3
"""
Step 3 - locate the denoising loop
==================================

    python scripts/step3_locate_generation.py

Purpose
-------
Find the exact code that has to be instrumented in Steps 4-6, and put a copy
of the reference implementation where it can be read. No model is loaded and
no GPU is used; this runs in a couple of seconds.

Three jobs:

  1. Download the reference implementation of LLaDA's sampler from the official
     repository (ML-GSAI/LLaDA) into `outputs/reference/`. This is the canonical
     description of low-confidence remasking and the thing every paper in the
     related work is describing.

  2. Find the *remote code* for the model actually on this machine. LLaDA-family
     checkpoints are not built into transformers - their modelling code ships in
     the model repository and is executed via `trust_remote_code=True`. It gets
     cached on disk, and Step 5 patches it, so its location has to be known.

  3. Scan both for the constructs that matter and print file:line references
     that can be opened directly in VS Code (Ctrl+P, paste the path, then
     Ctrl+G and the line number).

What the algorithm does
-----------------------
Understanding this is the actual point of Step 3. The sampler runs a fixed
number of rounds over a block of masked positions:

    for each denoising round:
        logits    = model(x)                     # full bidirectional pass
        x0        = argmax(logits)               # best guess at EVERY position
        x0_p      = softmax(logits).max()        # confidence in that guess
        confidence = where(mask_index, x0_p, -inf)   # <-- the key line
        pick the k highest-confidence masked positions and reveal them
        everything else stays masked and is re-predicted next round

`torch.where(mask_index, x0_p, -np.inf)` is the line the whole project turns
on. Positions already revealed are assigned **negative** infinity, so `topk`
- which selects the highest values - can never choose them again. A revealed
token is frozen for the rest of generation.

The consequence, which is charter Warning #4: a "flip" cannot be a revealed
token changing, because that is impossible by construction. A flip is the
*argmax prediction at a still-masked position* differing between consecutive
rounds. That distinction is the difference between a working feature set and
a set of features that are all identically zero.

`get_num_transfer_tokens` decides k per round - masked count divided evenly
across the remaining rounds, remainder distributed to the earliest rounds. So
the number of positions revealed per round is fixed in advance, not adaptive.
Worth knowing: it means "how many rounds did position i survive" is bounded by
a schedule, and settle-round features must be interpreted against it.

Licensing note
--------------
The reference file is downloaded to `outputs/`, which is gitignored. Check
ML-GSAI/LLaDA's LICENSE before vendoring any of it into a public release -
citing and linking is always safe, redistribution depends on the terms.
"""

import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step3_locate_generation.py")
    sys.exit(1)

REFERENCE_URL = "https://raw.githubusercontent.com/ML-GSAI/LLaDA/main/generate.py"
REFERENCE_DIR = config.OUT_DIR / "reference"
REFERENCE_FILE = REFERENCE_DIR / "llada_generate.py"
REPORT_PATH = config.OUT_DIR / "step3_locate_report.txt"

# Constructs worth finding, and why each one matters. Printed alongside every
# hit so the report explains itself rather than being a wall of line numbers.
PATTERNS = [
    (r"\bdef\s+generate\b",        "the sampling loop entry point"),
    (r"\bdef\s+\w*generate\w*\b",  "any other generation entry point"),
    (r"add_gumbel_noise",          "temperature sampling; temperature=0 makes it argmax"),
    (r"get_num_transfer_tokens",   "how many positions are revealed each round"),
    (r"low_confidence",            "the remasking strategy this project measures"),
    (r"mask_index",                "boolean map of which positions are still masked"),
    (r"-\s*np\.inf|-\s*float\(.inf.\)|-\s*torch\.inf",
                                   "freezing revealed positions - THE key line"),
    (r"\btopk\b",                  "selecting which positions to reveal"),
    (r"block_length",              "semi-autoregressive blocking"),
    (r"\bsteps\b",                 "number of denoising rounds"),
    (r"mask_id|mask_token",        "the mask token id"),
]

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def download_reference() -> bool:
    """Fetch the official sampler. Returns True if a readable copy now exists."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    if REFERENCE_FILE.exists() and REFERENCE_FILE.stat().st_size > 1000:
        say(f"  already present: {REFERENCE_FILE}")
        return True

    say(f"  downloading {REFERENCE_URL}")
    try:
        with urllib.request.urlopen(REFERENCE_URL, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        header = (
            "# Downloaded by TRIAGE Step 3 for study only.\n"
            f"# Source: {REFERENCE_URL}\n"
            "# Copyright belongs to the LLaDA authors (ML-GSAI). Check their\n"
            "# LICENSE before redistributing any part of this file.\n"
            f"# Retrieved: {datetime.now().isoformat(timespec='seconds')}\n\n"
        )
        REFERENCE_FILE.write_text(header + body, encoding="utf-8")
        say(f"  saved to {REFERENCE_FILE}  ({len(body):,} bytes)")
        return True
    except Exception as exc:
        say(f"  download failed ({type(exc).__name__}: {exc})")
        say(f"  open it in a browser instead: {REFERENCE_URL}")
        return False


def find_remote_code() -> list[Path]:
    """Locate the cached modelling code for the model on this machine.

    HuggingFace stores executed remote code under
    `$HF_HOME/modules/transformers_modules/`, and the raw repository files
    under `$HF_HOME/hub/models--*/snapshots/*/`. Both are searched, because
    which one is populated depends on the transformers version.
    """
    root = Path(config.HF_HOME)
    found: list[Path] = []

    modules = root / "modules" / "transformers_modules"
    if modules.is_dir():
        found.extend(sorted(modules.rglob("*.py")))

    hub = root / "hub"
    if hub.is_dir():
        for repo in hub.glob("models--*LLaDA*"):
            for snap in repo.glob("snapshots/*"):
                found.extend(sorted(snap.glob("*.py")))

    # Ignore trivial files - __init__.py and the like carry no logic.
    return [p for p in found if p.stat().st_size > 500]


def scan(path: Path) -> None:
    """Print every pattern hit in one file, with line numbers."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        say(f"  could not read ({type(exc).__name__}: {exc})")
        return

    lines = text.splitlines()
    say(f"  {len(lines):,} lines")

    any_hit = False
    for pattern, why in PATTERNS:
        rx = re.compile(pattern)
        hits = [(n, ln) for n, ln in enumerate(lines, 1) if rx.search(ln)]
        if not hits:
            continue
        any_hit = True
        say("")
        say(f"    {why}")
        for n, ln in hits[:4]:            # 4 is enough to locate it
            say(f"      line {n:>5} | {ln.strip()[:78]}")
        if len(hits) > 4:
            say(f"      ... and {len(hits) - 4} more")

    if not any_hit:
        say("    (no generation constructs found - probably not the sampler)")


def main() -> None:
    say("=" * 74)
    say("  TRIAGE - Step 3: locate the denoising loop")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 74)
    say(f"HF_HOME : {config.HF_HOME}")

    # ---- 1. reference implementation -------------------------------------
    say("")
    say("-" * 74)
    say("1. REFERENCE SAMPLER (ML-GSAI/LLaDA, the canonical implementation)")
    if download_reference():
        say("")
        say(f"  {REFERENCE_FILE}")
        scan(REFERENCE_FILE)

    # ---- 2. the model's own remote code -----------------------------------
    say("")
    say("-" * 74)
    say("2. REMOTE CODE FOR THE MODEL ON THIS MACHINE")
    say("   (executed because trust_remote_code=True; Step 5 patches this)")

    files = find_remote_code()
    if not files:
        say("")
        say("  None found. Run step2c once first - remote code is only cached")
        say("  after the model has been loaded at least once.")
    else:
        say(f"\n  {len(files)} candidate file(s):")
        for path in files:
            say("")
            say(f"  {path}")
            scan(path)

    # ---- 3. what to do next -----------------------------------------------
    say("")
    say("=" * 74)
    say("WHAT TO READ")
    say("")
    say("  Open the reference sampler and find these four things:")
    say("")
    say("    a) the loop over denoising rounds, and where model(x) is called")
    say("    b) x0 = argmax(logits) - the guess at EVERY position, masked or not")
    say("    c) confidence = torch.where(mask_index, x0_p, -np.inf)")
    say("       Revealed positions get NEGATIVE infinity so topk never picks")
    say("       them again. This is why a revealed token can never change, and")
    say("       why 'flips' must be measured on predictions at STILL-MASKED")
    say("       positions across rounds.")
    say("    d) get_num_transfer_tokens - the reveal schedule, fixed in advance")
    say("")
    say("  In VS Code: Ctrl+P, paste a path, Enter. Then Ctrl+G, type a line")
    say("  number, Enter.")
    say("")
    say("  Step 4 inserts logging at (a) and (b): for every round and every")
    say("  position we record argmax id, its probability, entropy, and mask")
    say("  state. Four values - never the full logits, which would be 4.1 GB")
    say("  per question instead of about 98 KB.")
    say("=" * 74)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
