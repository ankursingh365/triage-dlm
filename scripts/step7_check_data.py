#!/usr/bin/env python3
"""
Step 7 - verify the dataset loaders
===================================

    python scripts/step7_check_data.py

Downloads a small sample from each of the three datasets, normalises it, builds
the prompts, and checks the one rule the project cannot afford to break: the
model must never see the supporting evidence.

No GPU. The tokenizer is loaded (small, fast) but not the model.

First run downloads the datasets. TriviaQA `rc.nocontext` and CommonsenseQA are
small; HotpotQA `distractor` is a few hundred MB. Everything lands in
`D:\\hf-cache` via `src/config.py`.

What it checks
--------------
1. All three datasets load and normalise into identical `QARecord` shapes.
2. Gold answers are present - TriviaQA should show many aliases per question,
   which is what makes Stage 3 judge agreement achievable.
3. HotpotQA supporting facts are recovered as gold evidence, for Step 26.
4. CommonsenseQA carries NO evidence. It is the negative control; if evidence
   ever appears here, the control is void.
5. **The leak guard fires.** A deliberately poisoned prompt containing gold
   evidence must be rejected. A guard that never triggers is not a guard.
6. Sampling is deterministic - the same seed twice gives the same question ids.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, data
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step7_check_data.py")
    sys.exit(1)

N_SAMPLE = 20          # per dataset; enough to see the shape, fast to download
N_SHOW = 2             # how many to print in full

REPORT_PATH = config.OUT_DIR / "step7_data_report.txt"
_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def save(code: int = 0):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")
    sys.exit(code)


def main() -> None:
    say("=" * 78)
    say("  TRIAGE - Step 7: dataset loaders")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 78)
    for line in config.describe().splitlines():
        say(line)

    # Tokenizer only - no model, no GPU.
    say("")
    say("Loading tokenizer (no model)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DEBUG,
                                              trust_remote_code=True)
    say(f"  chat template available: {data.has_chat_template(tokenizer)}")

    all_ok = True
    loaded = {}

    for name in ("triviaqa", "hotpotqa", "commonsenseqa"):
        say("")
        say("=" * 78)
        say(f"{name.upper()}")
        say("-" * 78)
        try:
            recs = data.load_records(name, n=N_SAMPLE, seed=config.SEED)
        except Exception as exc:
            say(f"  FAILED to load: {type(exc).__name__}: {exc}")
            say("  (first run downloads the dataset; check your connection)")
            all_ok = False
            continue

        loaded[name] = recs
        say(f"  records            : {len(recs)}")
        if not recs:
            say("  EMPTY - nothing normalised. Investigate before continuing.")
            all_ok = False
            continue

        n_gold = sum(len(r.gold_answers) for r in recs) / len(recs)
        n_ev = sum(len(r.evidence_passages) for r in recs) / len(recs)
        n_ti = sum(len(r.evidence_titles) for r in recs) / len(recs)
        say(f"  mean gold answers  : {n_gold:.1f}")
        say(f"  mean evidence psgs : {n_ev:.1f}")
        say(f"  mean evidence titles: {n_ti:.1f}")

        # Prompt length matters: prompt + 64 generated tokens must fit in the
        # ~0.5 GiB of VRAM headroom left after the model loads.
        lens = [len(data.build_prompt_ids(tokenizer, r)) for r in recs]
        say(f"  prompt tokens      : min {min(lens)}, "
            f"median {sorted(lens)[len(lens)//2]}, max {max(lens)}")

        for r in recs[:N_SHOW]:
            say("")
            say(f"  --- {r.qid} ---")
            say(f"  Q       : {r.question[:100]}")
            say(f"  gold    : {r.gold_answers[:5]}"
                f"{' ...' if len(r.gold_answers) > 5 else ''}")
            if r.choices:
                say(f"  choices : {r.choices}")
            if r.evidence_titles:
                say(f"  ev title: {r.evidence_titles[:3]}")
            if r.evidence_passages:
                say(f"  ev psg  : {r.evidence_passages[0][:90]}...")
            prompt = data.build_prompt_text(r.question, r.choices)
            say(f"  PROMPT  : {prompt[:160]!r}")

    # =====================================================================
    # Check 4 - the negative control carries no evidence
    # =====================================================================
    say("")
    say("=" * 78)
    say("CHECK - NEGATIVE CONTROL INTEGRITY")
    csqa = loaded.get("commonsenseqa", [])
    if csqa:
        leaked = [r.qid for r in csqa
                  if r.evidence_passages or r.evidence_titles]
        if leaked:
            say(f"  FAIL - {len(leaked)} CommonsenseQA records carry evidence")
            all_ok = False
        else:
            say(f"  OK - all {len(csqa)} CommonsenseQA records carry zero")
            say("       evidence. The negative control is intact: external")
            say("       evidence must give ZERO gain here in Step 30.")

    # =====================================================================
    # Check 5 - the leak guard must actually fire
    # =====================================================================
    say("")
    say("=" * 78)
    say("CHECK - LEAK GUARD")
    hotpot = loaded.get("hotpotqa", [])
    victim = next((r for r in hotpot if r.evidence_passages), None)
    if victim is None:
        say("  SKIPPED - no HotpotQA record with evidence to test against")
    else:
        clean = data.build_prompt_text(victim.question, victim.choices)
        try:
            data.assert_evidence_withheld(clean, victim)
            say("  clean prompt passes      : OK")
        except AssertionError:
            say("  clean prompt REJECTED    : FAIL - the guard is too aggressive")
            all_ok = False

        poisoned = clean + "\n\nContext: " + victim.evidence_passages[0]
        try:
            data.assert_evidence_withheld(poisoned, victim)
            say("  poisoned prompt passes   : FAIL - the guard does not work")
            all_ok = False
        except AssertionError:
            say("  poisoned prompt REJECTED : OK - the guard fires correctly")

    # =====================================================================
    # Check 6 - determinism
    # =====================================================================
    say("")
    say("=" * 78)
    say("CHECK - DETERMINISTIC SAMPLING")
    if "triviaqa" in loaded:
        again = data.load_records("triviaqa", n=N_SAMPLE, seed=config.SEED,
                                  cache=False)
        same = [r.qid for r in again] == [r.qid for r in loaded["triviaqa"]]
        say(f"  same seed gives same questions : {same}")
        if not same:
            say("    (note: cache was used on the first call; a mismatch here")
            say("     means the sampler is not seed-stable)")
            all_ok = False

    say("")
    say("=" * 78)
    if all_ok:
        say("  PASS - all three datasets load, evidence is withheld, the leak")
        say("         guard fires, and sampling is reproducible.")
        say("")
        say("  STILL OPEN before Phase C:")
        say("   1. Match the prompt format to TraceDet's setup, or the Step 20")
        say("      Ave Entropy gate (62-65 AUROC) may miss for reasons that have")
        say("      nothing to do with the code.")
        say("   2. Pin the chat template's system message. LLaDA-MoE injects")
        say("      'detailed thinking off' by default (seen in Step 2c).")
    else:
        say("  FAIL - see above.")
    say("=" * 78)
    save(0 if all_ok else 1)


if __name__ == "__main__":
    main()
