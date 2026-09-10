#!/usr/bin/env python3
"""
Step 6b - screen candidate features before building features.py
================================================================

    python scripts/step6b_screen_features.py

CPU only. No model, no GPU, no generation. Reads the 50 trajectories already on
disk from Step 6 and runs every candidate feature through two tests.

Why this exists
---------------
Step 6's diagnostic summary contained two bad metrics, and they failed in two
different ways:

  * **Median settle round** was identical (15) for all 50 questions.
    `get_num_transfer_tokens` reveals exactly `gen_length / steps = 2` positions
    per round, deterministically, so the 32nd of 64 positions is always revealed
    at round 16. The statistic is a property of the schedule, not the model.

  * **Raw regret count** appeared to track wrongness (obscure 192.7 vs short 1.9)
    but was measuring answer length. Controlling for length by comparing only
    63-token answers reversed the direction: correct answers showed ~308, wrong
    ones ~258.

Both are the same underlying error - proposing a metric without checking whether
it *can* vary for reasons unrelated to what it claims to measure. Step 19 offers
about twenty-five more opportunities to make it, and there it would be buried in
a logistic regression where a dead feature merely looks like a small coefficient.

The two screens
---------------
**Screen 1 - variance.** A feature whose standard deviation is near zero across
questions carries no information. Coefficient of variation below 0.02 fails.

**Screen 2 - length confound.** A feature correlating above |0.80| with content
token count is a proxy for answer length. It may still be usable, but only with
length as an explicit covariate, and it must never be compared across groups that
differ in length.

Correctness labels
------------------
The 30 short factual questions have known answers, hand-written below. Substring
matching, case-insensitive - crude, but adequate for a sanity check and honest
about being crude. This gives a small real correctness signal instead of Step 6's
assumption that "obscure" means "wrong".

Medium and tricky answers are left unlabelled: judging them needs the Qwen3 judge
from Stage 3. Obscure questions are marked `presumed_wrong` and that assumption is
labelled as such in the output rather than hidden.

Output
------
`outputs/tables/step6b_feature_screen.csv`  - every feature, both screens
`outputs/tables/step6b_per_question.csv`    - features per question, for plotting
"""

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config, logging_patch
except ImportError as exc:
    print(f"Could not import from src/ ({exc}).")
    print("Run from the repository root: python scripts/step6b_screen_features.py")
    sys.exit(1)

TRAJ_DIR = config.TRAJ_DIR / "step6"
REPORT_PATH = config.OUT_DIR / "step6b_screen_report.txt"
FEATURE_CSV = config.TAB_DIR / "step6b_feature_screen.csv"
PERQ_CSV = config.TAB_DIR / "step6b_per_question.csv"

# Screening thresholds.
CV_FLOOR = 0.02        # coefficient of variation below this = structurally constant
CONFOUND_CEIL = 0.80   # |r| with answer length above this = length proxy

# Gold answers for the 30 short factual questions. Substring match, lower-cased.
# Alternatives separated by "|" - any one counts as correct.
GOLD = {
    "s01": "tokyo",            "s02": "shakespeare",
    "s03": "au",               "s04": "7|seven",
    "s05": "pacific",          "s06": "mars",
    "s07": "leonardo|vinci",   "s08": "everest",
    "s09": "1945",             "s10": "pound|sterling|gbp",
    "s11": "carbon dioxide|co2", "s12": "nile",
    "s13": "einstein",         "s14": "2|two",
    "s15": "france",           "s16": "diamond",
    "s17": "4|four",           "s18": "canberra",
    "s19": "hydrogen",         "s20": "achebe",
    "s21": "100",              "s22": "mediterranean",
    "s23": "blue whale|whale", "s24": "rome|roma",
    "s25": "12|twelve",        "s26": "vitamin d| d ",
    "s27": "armstrong",        "s28": "portuguese",
    "s29": "11|eleven",        "s30": "32",
}

_lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


# ===========================================================================
# Feature definitions
# ===========================================================================

def compute_features(traj: logging_patch.Trajectory) -> dict:
    """Every candidate feature for one trajectory.

    Names prefixed `BAD_` are the two known-broken metrics, kept deliberately so
    the screen can be seen catching them. If the screen does not flag those two,
    the screen itself is wrong.
    """
    steps, gen_length = traj.entropy.shape
    masked = traj.mask_state == 1                     # (steps, gen)
    content = traj.content_mask() & masked            # excludes EOS and mask ids
    revealed = traj.mask_state == 0
    revealed_content = revealed & traj.content_mask()

    ent = traj.entropy
    settle = traj.revealed_at()                        # (gen,)
    final = traj.final_ids
    is_content_pos = (final != traj.eos_id) & (final != traj.mask_id)
    n_content = int(is_content_pos.sum())

    f = {}

    # ---- entropy level ---------------------------------------------------
    f["ent_mean_all"] = float(ent[masked].mean()) if masked.any() else np.nan
    f["ent_mean_content"] = float(ent[content].mean()) if content.any() else np.nan
    f["ent_max_content"] = float(ent[content].max()) if content.any() else np.nan
    f["ent_min_content"] = float(ent[content].min()) if content.any() else np.nan
    f["ent_std_content"] = float(ent[content].std()) if content.any() else np.nan

    # ---- entropy shape over rounds ---------------------------------------
    # Step 4 showed entropy RISES for the first ~5 rounds before falling: topk
    # reveals the easy positions first, so the surviving pool gets harder before
    # context makes it easy again. Endpoint-only features miss that.
    per_round = np.array([
        ent[r][content[r]].mean() if content[r].any() else np.nan
        for r in range(steps)
    ])
    valid = per_round[~np.isnan(per_round)]
    f["ent_first_round"] = float(valid[0]) if valid.size else np.nan
    f["ent_last_round"] = float(valid[-1]) if valid.size else np.nan
    f["ent_drop"] = (float(valid[0] - valid[-1]) if valid.size > 1 else np.nan)
    f["ent_auc"] = float(valid.mean()) if valid.size else np.nan
    f["ent_peak_round_frac"] = (float(np.nanargmax(per_round) / steps)
                                if valid.size else np.nan)

    # ---- settle rounds ---------------------------------------------------
    # BAD_settle_median_all is the schedule artefact. Screen 1 must flag it.
    f["BAD_settle_median_all"] = float(np.median(settle))

    cs = settle[is_content_pos]
    f["settle_mean_content"] = float(cs.mean()) if cs.size else np.nan
    f["settle_std_content"] = float(cs.std()) if cs.size else np.nan
    f["settle_first_content"] = float(cs.min()) if cs.size else np.nan
    f["settle_last_content"] = float(cs.max()) if cs.size else np.nan
    # Fraction of the ANSWER settled in the first 30% of rounds. This is the
    # quantity the locked-in failure mode is really about, and unlike the median
    # it is not fixed by the schedule.
    f["settle_frac_early"] = (float((cs < 0.3 * steps).mean())
                              if cs.size else np.nan)

    # ---- flips (Warning #4: masked positions only) -----------------------
    flips = traj.flips_per_round()
    f["BAD_flips_total"] = float(flips.sum())
    # Normalised. The denominator must be masked CONTENT positions, not all
    # masked positions: the reveal schedule is identical for every question, so
    # dividing by total masked positions leaves the numerator's dependence on
    # answer length completely intact. Screening a first attempt at this feature
    # caught exactly that - it still correlated +1.00 with length.
    flip_opportunities = float((masked[1:] & traj.content_mask()[1:]).sum())
    f["flip_rate"] = (float(flips.sum() / flip_opportunities)
                      if flip_opportunities > 0 else np.nan)
    half = steps // 2
    f["flip_early_frac"] = (float(flips[:half].sum() / flips.sum())
                            if flips.sum() > 0 else np.nan)

    # ---- regret (Finding B) ----------------------------------------------
    regret = traj.regret_per_round()
    f["BAD_regret_total"] = float(regret.sum())
    # Normalised: opportunities = (round, revealed CONTENT position) pairs.
    # EOS padding is excluded because the model reconstructs it trivially and
    # it would otherwise dominate the denominator on short answers.
    regret_opportunities = float(revealed_content.sum())
    f["regret_rate"] = (float(regret.sum() / regret_opportunities)
                        if regret_opportunities > 0 else np.nan)
    f["regret_final_rate"] = (float(regret[-1] / revealed_content[-1].sum())
                              if revealed_content[-1].any() else np.nan)

    # ---- confidence ------------------------------------------------------
    f["prob_mean_content"] = (float(traj.pred_probs[content].mean())
                              if content.any() else np.nan)
    f["prob_min_content"] = (float(traj.pred_probs[content].min())
                             if content.any() else np.nan)

    # ---- the confounder itself, so correlations can be measured against it
    f["n_content_tokens"] = float(n_content)

    return f


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, ignoring pairs where either value is NaN."""
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ===========================================================================

def main() -> None:
    say("=" * 80)
    say("  TRIAGE - Step 6b: candidate feature screening")
    say("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    say("=" * 80)
    say(f"reading {TRAJ_DIR}")

    files = sorted(TRAJ_DIR.glob("*.npz"))
    if not files:
        say("")
        say(f"No trajectories found in {TRAJ_DIR}.")
        say("Run scripts/step6_reconstruct_50.py first.")
        sys.exit(1)

    rows = []
    for path in files:
        traj = logging_patch.Trajectory.load(path)
        qid = traj.question_id or path.stem
        cat = ("short" if qid.startswith("s") else
               "medium" if qid.startswith("m") else
               "tricky" if qid.startswith("t") else "obscure")

        feats = compute_features(traj)

        # ---- correctness, where we can judge it -------------------------
        # Judging needs the decoded answer TEXT, which this module has no
        # tokenizer for. It is joined in below from the Step 6 CSV, which
        # already stored it. Until then the label stays pending.
        correct = None
        label_source = "unlabelled"
        if qid in GOLD:
            label_source = "gold-pending"
        elif cat == "obscure":
            correct = False
            label_source = "presumed_wrong"

        rows.append(dict(id=qid, category=cat, correct=correct,
                         label_source=label_source, **feats))

    # Correctness for the short questions needs the decoded answer text, which
    # lives in the Step 6 CSV. Join on id if that file exists.
    step6_csv = config.TAB_DIR / "step6_per_question.csv"
    answers = {}
    if step6_csv.exists():
        with open(step6_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                answers[r["id"]] = r.get("answer", "")
    for row in rows:
        if row["id"] in GOLD and row["id"] in answers:
            text = answers[row["id"]].lower()
            row["correct"] = any(alt in text for alt in GOLD[row["id"]].split("|"))
            row["label_source"] = "gold"

    n_labelled = sum(1 for r in rows if r["correct"] is not None
                     and r["label_source"] == "gold")
    n_correct = sum(1 for r in rows if r["correct"] is True)
    say(f"{len(rows)} trajectories, {n_labelled} with gold labels "
        f"({n_correct} correct)")

    feature_names = [k for k in rows[0] if k not in
                     ("id", "category", "correct", "label_source")]

    # =====================================================================
    # Screen 1 and 2
    # =====================================================================
    length = np.array([r["n_content_tokens"] for r in rows], dtype=float)

    say("")
    say("=" * 80)
    say("FEATURE SCREEN")
    say("")
    say("  feature                    mean      std       CV     r(len)  verdict")
    say("  -------------------------  --------  --------  -----  ------  -------")

    screen_rows = []
    for name in feature_names:
        vals = np.array([r[name] for r in rows], dtype=float)
        finite = vals[~np.isnan(vals)]
        if finite.size < 3:
            continue

        mean, std = float(finite.mean()), float(finite.std())
        cv = abs(std / mean) if mean != 0 else (0.0 if std == 0 else float("inf"))
        r_len = pearson(vals, length) if name != "n_content_tokens" else 1.0

        verdicts = []
        if cv < CV_FLOOR:
            verdicts.append("CONSTANT")
        if not math.isnan(r_len) and abs(r_len) > CONFOUND_CEIL \
                and name != "n_content_tokens":
            verdicts.append("LENGTH")
        verdict = "+".join(verdicts) if verdicts else "ok"

        say(f"  {name:<25}  {mean:8.3f}  {std:8.3f}  {cv:5.3f}  "
            f"{r_len:+6.2f}  {verdict}")
        screen_rows.append(dict(feature=name, mean=mean, std=std, cv=cv,
                                r_with_length=r_len, verdict=verdict))

    # =====================================================================
    # Did the screen catch the two known-bad metrics?
    # =====================================================================
    say("")
    say("-" * 80)
    say("VALIDATION - the screen must catch the two metrics we already know are bad")
    for bad, why in [("BAD_settle_median_all", "schedule-determined, expect CONSTANT"),
                     ("BAD_regret_total", "length proxy, expect LENGTH"),
                     ("BAD_flips_total", "length proxy, expect LENGTH")]:
        hit = next((s for s in screen_rows if s["feature"] == bad), None)
        if hit:
            ok = hit["verdict"] != "ok"
            say(f"  {bad:<25} {hit['verdict']:<12} "
                f"{'CAUGHT' if ok else 'MISSED'}   ({why})")

    say("")
    say("  If any of the three says MISSED, the screen thresholds are too loose")
    say("  and the surviving features cannot be trusted either.")

    # =====================================================================
    # Fixed versions, and whether they separate correct from incorrect
    # =====================================================================
    say("")
    say("-" * 80)
    say("FIXED METRICS vs their broken originals")
    for bad, good in [("BAD_regret_total", "regret_rate"),
                      ("BAD_flips_total", "flip_rate"),
                      ("BAD_settle_median_all", "settle_frac_early")]:
        b = next((s for s in screen_rows if s["feature"] == bad), None)
        g = next((s for s in screen_rows if s["feature"] == good), None)
        if b and g:
            say(f"  {bad:<25} r(len) {b['r_with_length']:+.2f}  ->  "
                f"{good:<20} r(len) {g['r_with_length']:+.2f}   "
                f"CV {g['cv']:.3f}  [{g['verdict']}]")

    # ---- correctness separation on the gold-labelled subset --------------
    gold_rows = [r for r in rows if r["label_source"] == "gold"]
    if len(gold_rows) >= 10:
        n_wrong = sum(1 for r in gold_rows if r["correct"] is False)
        say("")
        say("-" * 80)
        say(f"SIGNAL CHECK on {len(gold_rows)} gold-labelled short questions "
            f"({n_wrong} wrong)")
        if n_wrong < 3:
            say("  Too few wrong answers to compare. Expected - these are easy")
            say("  factual questions. A real separation test needs Stage 3 labels")
            say("  on a dataset where the model fails more often.")
        else:
            say("")
            say("  feature                    correct    wrong    difference")
            say("  -------------------------  ---------  -------  ----------")
            for name in ["ent_mean_content", "ent_max_content", "regret_rate",
                         "flip_rate", "settle_frac_early", "prob_mean_content"]:
                c = [r[name] for r in gold_rows if r["correct"] is True
                     and not math.isnan(r[name])]
                w = [r[name] for r in gold_rows if r["correct"] is False
                     and not math.isnan(r[name])]
                if c and w:
                    say(f"  {name:<25}  {sum(c)/len(c):9.3f}  "
                        f"{sum(w)/len(w):7.3f}  {sum(w)/len(w) - sum(c)/len(c):+10.3f}")
            say("")
            say("  n is small and these are not significance tests. Directional only.")

    # =====================================================================
    # Output
    # =====================================================================
    FEATURE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(screen_rows[0].keys()))
        w.writeheader()
        w.writerows(screen_rows)

    with open(PERQ_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    survivors = [s["feature"] for s in screen_rows
                 if s["verdict"] == "ok" and s["feature"] != "n_content_tokens"]
    say("")
    say("=" * 80)
    say(f"SURVIVORS - {len(survivors)} features pass both screens")
    for name in survivors:
        say(f"    {name}")
    say("")
    say("  These are the candidates features.py should be built from in Step 19.")
    say("  Features flagged LENGTH are not banned - they are usable with answer")
    say("  length as an explicit covariate, and must never be compared across")
    say("  groups of differing length.")
    say("")
    say(f"  {FEATURE_CSV}")
    say(f"  {PERQ_CSV}")
    say("=" * 80)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
