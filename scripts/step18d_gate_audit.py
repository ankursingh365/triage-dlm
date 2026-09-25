#!/usr/bin/env python3
"""
Step 18d - audit the two gates UPSTREAM of the cuts, before refitting anything
==============================================================================

    modal run modal_app.py::run_cpu --script step18d_gate_audit.py

CPU only, seconds, free. Reads CSVs. No model, no trajectories, no GPU.

WHY THIS COMES BEFORE THE REFIT
===============================
Step 18c settled the Dream hand labels (kappa 0.797, 60/60) and the
population-weighted comparison came out:

    class            hand labels          classifier
    locked_in        43.2% [27.4, 58.7]      71.0%    <- outside
    interleaving     10.8% [ 3.1, 21.5]       3.1%    <- at the bound
    inconsistent     35.6% [22.5, 49.6]      20.3%    <- outside
    neither          10.4% [ 1.9, 21.0]       5.6%

The obvious next move is to refit `s_cut` and `c_cut` so the locked_in share
comes down. **That would be wrong, and it would bake an error in.** Two of the
classifier's gates run BEFORE the cuts are ever consulted, and the per-stratum
table says both are misfiring:

    classifier said   n   consensus said
    untestable        4   L 3, I 1        <- agreed on ZERO
    interleaving     12   I 11, X 1       <- 92% precision, but the arm share
                                             is 3.1% against 10.8% by hand

`classify()` reads, in order:

    truncated            -> degenerate
    echo rule            -> echo
    NOT gold_testable    -> untestable          (1)
    gold_ever            -> interleaving        (2)
    stable/cands cuts    -> locked_in | inconsistent

(1) **returns early.** An item whose gold cannot be tested for interleaving is
    removed from the analysis altogether, as though it were not a
    hallucination. But "we cannot test whether the right answer appeared" is
    not the same claim as "this is not a wrong answer". The hand labels say so
    directly: three of the four untestable items on the sheet are locked-in.

(2) fires on a whole-string match against the gold. Item #33's gold is
    `Makoto, born July 15, 1980` and its trace holds `Makoto Kinaka`; item
    #48's gold is `924`, three characters, below the worksheet's
    MIN_GOLD_CHARS. Both are interleaving by hand and neither fired. High
    precision with a three-fold volume shortfall is the signature of a RECALL
    problem, which is exactly what a too-strict match produces.

Fitting cuts on top of (1) and (2) would tune `s_cut` and `c_cut` to absorb
mistakes that belong to the gates, and the fitted numbers would then be wrong
for the right reasons - undiagnosable later.

WHAT THIS SCRIPT DOES
=====================
It measures how big each gate error is, on BOTH arms, and validates the
candidate fix against hand labels that were made blind before any of this was
visible. **It changes no rule, refits no cut, and writes no new mode column.**
Step 18e does the refit, once the gates are known to be right.

It opens with a LOCK: `classify()` is restated here from
`step17d_dream_labels.py` and must reproduce the stored `mode` column for every
row of every arm. If it cannot, this script's understanding of the rule is
wrong and nothing below it means anything, so it stops.

ON CIRCULARITY - stated before the numbers
==========================================
Part B tests the gold matcher against the 60 consensus labels. Those labels are
a fair external test with two exceptions, and they are named here rather than
buried: items **#33 and #48** are the two where the annotator could SEE that
the worksheet's marker had missed a gold, and reasoned partly from that. Part B
therefore reports recall twice, with and without them. If the conclusion only
survives with them included, it is not a conclusion.
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from src import config
except ImportError as exc:                                  # pragma: no cover
    print(f"Could not import from src/ ({exc}).")
    sys.exit(1)

# The rule as shipped. Restated, not imported, so the LOCK below is a real
# test: two independent expressions of the same rule must agree.
CUTS = dict(s_cut=0.26, c_cut=3.4, e_cut=0.75, r_cut=6)

DREAM_CSV = config.TAB_DIR / "step17d_dream_modes.csv"
LLADA_CSVS = [config.TAB_DIR / "step12c_final_modes.csv",
              config.TAB_DIR / "step12b_csqa_modes.csv"]
CONSENSUS_CSV = config.TAB_DIR / "step18c_consensus.csv"
LLADA_KEY_CSV = config.TAB_DIR / "step10c_answer_key.csv"
LLADA_CAL_CSV = config.TAB_DIR / "step10d_final_modes.csv"
REPORT_PATH = config.OUT_DIR / "step18d_gate_audit.txt"

# LLaDA's 60-item round, consensus of two blind passes, kappa 0.882.
# Copied verbatim from step10d_fit_and_apply.py. "." is item #11, dropped
# there because the worksheet truncated its answer beyond reading.
LLADA_HAND = ("L C C X C X C I X C . X X X C C I L C C I L L I L C C C C C "
              "L C L X X X X X X C C C I X X L C C C L L L X I X X X X L C").split()

MODES = ("locked_in", "interleaving", "inconsistent", "echo", "untestable",
         "degenerate")
MODE_TO_LETTER = {"locked_in": "L", "interleaving": "I", "inconsistent": "C",
                  "echo": "X", "untestable": "X", "degenerate": "X"}
LABELS = ("L", "I", "C", "X")
LONG = {"L": "locked_in", "I": "interleaving", "C": "inconsistent",
        "X": "neither"}

# Items whose hand label is NOT independent evidence about the gold matcher.
#
# Both #33 and #48 are worksheet marker misses that Claude noticed and wrote
# into the sealed file. Only ONE of them is circular:
#
#   #48  Ankur labelled it I independently, from the trace alone - he read
#        "P Porsche 924" and saw it dropped. Both annotators agreed, so the
#        consensus came from agreement, not from Claude's reasoning. Legitimate.
#   #33  Ankur labelled it C. Claude labelled it I, arguing partly FROM the
#        marker's absence, and the adjudication adopted Claude's call. That one
#        is circular: the matcher is being tested against a label the matcher's
#        own failure helped produce.
#
# So only #33 is excluded. The first version excluded both, which threw away
# real evidence and understated the gate's recall.
NOT_INDEPENDENT = {33}

NUMERIC = ("stable_top3", "cands_top3", "q_overlap", "copy_run")
BOOLEAN = ("correct", "truncated", "gold_ever", "gold_testable")

_lines: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def rule(char: str = "=") -> None:
    say(char * 78)


# ---------------------------------------------------------------------------
#  A loader that REFUSES to invent data
# ---------------------------------------------------------------------------

FULL_NEED = {"qid", "mode", *NUMERIC, *BOOLEAN}
GATE_NEED = {"qid", "mode", "gold_ever", "gold_testable"}


def load(paths, label: str, need=None, fatal: bool = True) -> list:
    """Rows from one or more mode CSVs. Never substitutes a default.

    step19a's loader defaulted a missing column to 0.0. `n_content` was never
    written by step12c, so LLaDA read 0.00 for every row and the script then
    reported "largest drift: n_content (+4.43 sd)" as a finding. Nothing
    crashed; the answer was simply false.

    So the header is validated before a single row is read. But `need` is the
    caller's business, not this function's: a part of the report that only
    looks at two flags must not demand the columns it never touches. The first
    version demanded the full set everywhere, and a pilot CSV without
    `copy_run` killed the whole run at Part C - refusing to invent data is
    right, refusing to run because of a column nobody asked for is not.

    With `fatal=False` a CSV that cannot satisfy `need` is reported and
    skipped, and the caller says what has been lost.
    """
    need = FULL_NEED if need is None else need
    out = []
    for p in ([paths] if isinstance(paths, Path) else paths):
        if not p.exists():
            say(f"  {label}: {p.name} not found - skipped.")
            continue
        with open(p, encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            have = set(rd.fieldnames or ())
            missing = sorted(need - have)
            if missing:
                say("")
                say(f"  {'FATAL' if fatal else 'SKIPPED'} - {p.name} is missing "
                    f"{len(missing)} column(s) this step needs:")
                for c in missing:
                    say(f"    {c}")
                say("  No default is substituted. A rate computed over a column")
                say("  that does not exist is not a weaker result, it is a")
                say("  fabricated one.")
                if not fatal:
                    say("  Continuing without this file.")
                    continue
                say("  Regenerate the CSV, or fix the writer's `keep` list.")
                REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                REPORT_PATH.write_text("\n".join(_lines) + "\n",
                                       encoding="utf-8")
                sys.exit(1)
            for r in rd:
                row = dict(qid=r["qid"], mode=(r["mode"] or "").strip(),
                           dataset=r.get("dataset", ""), source=p.name)
                for c in NUMERIC:
                    try:
                        row[c] = float(r[c])
                    except (TypeError, ValueError, KeyError):
                        row[c] = None            # kept as None, never 0.0
                for c in BOOLEAN:
                    row[c] = str(r.get(c, "")).strip().lower() == "true"
                out.append(row)
    return out


def wrong(rows: list) -> list:
    """Only the rows a failure mode was assigned to."""
    return [r for r in rows if r["mode"] in MODE_TO_LETTER]


# ---------------------------------------------------------------------------
#  The rule, and the rule with gate (1) repaired
# ---------------------------------------------------------------------------

def classify(r, cuts=CUTS, untestable_falls_through: bool = False) -> str:
    """step17d_dream_labels.py's classify(), ORDER 'A'.

    With `untestable_falls_through`, an item whose gold cannot be tested skips
    the interleaving test - which genuinely cannot be run on it - and continues
    to the locked_in / inconsistent decision instead of returning early. That
    is the only change; no cut moves.
    """
    if r["truncated"]:
        return "degenerate"
    if r["q_overlap"] is None or r["copy_row_missing"]:
        return "unknown"
    echo = r["q_overlap"] >= cuts["e_cut"] and r["copy_run"] >= cuts["r_cut"]
    if echo:
        return "echo"
    if not r["gold_testable"]:
        if not untestable_falls_through:
            return "untestable"
    elif r["gold_ever"]:
        return "interleaving"
    if r["stable_top3"] >= cuts["s_cut"] and r["cands_top3"] <= cuts["c_cut"]:
        return "locked_in"
    return "inconsistent"


def prep(rows: list) -> None:
    """Flag rows whose numeric columns are unusable, instead of coercing them."""
    for r in rows:
        r["copy_row_missing"] = any(r[c] is None for c in NUMERIC)


def shares(rows: list, key="mode") -> dict:
    n = len(rows)
    c = Counter(r[key] for r in rows)
    return {m: c[m] / n for m in MODES} if n else {}


# ---------------------------------------------------------------------------

def main() -> None:
    rule()
    say("  TRIAGE - Step 18d: audit the gates upstream of the cuts")
    say(f"  {datetime.now():%Y-%m-%d %H:%M}")
    rule()
    say("")
    say("  Nothing is refitted here. This script measures two gate errors and")
    say("  tests one candidate fix against hand labels. Step 18e refits.")
    say("")

    arms = {}
    for name, paths in (("Dream", DREAM_CSV), ("LLaDA", LLADA_CSVS)):
        rows = load(paths, name)
        if rows:
            prep(rows)
            arms[name] = rows
    if "Dream" not in arms:
        say("\n  Dream's mode CSV is required. Run step17d first.")
        sys.exit(1)
    say("")
    for name, rows in arms.items():
        w = wrong(rows)
        say(f"  {name:<6} {len(rows):6d} rows, {len(w):6d} with a failure mode")
    say("")

    # --- LOCK: can this script reproduce the stored mode column? ---------
    rule("-")
    say("  LOCK - reproduce the stored `mode` column from the raw columns")
    rule("-")
    say("")
    say("  classify() is restated in this file rather than imported, so this")
    say("  is two independent expressions of one rule being compared. A")
    say("  mismatch means this script does not understand the rule, and")
    say("  everything below it would be built on that misunderstanding.")
    say("")
    lock_ok = True
    for name, rows in arms.items():
        w = wrong(rows)
        bad = [r for r in w if classify(r) != r["mode"]]
        say(f"    {name:<6} {len(w)-len(bad):6d} / {len(w):<6d} reproduced"
            f"{'' if not bad else f'   {len(bad)} MISMATCH'}")
        if bad:
            lock_ok = False
            seen = Counter((r["mode"], classify(r)) for r in bad)
            for (stored, got), c in seen.most_common(8):
                say(f"      stored {stored:<13} -> recomputed {got:<13} {c:5d}")
    say("")
    if not lock_ok:
        rule()
        say("  LOCK FAILED. Stopping. Do not refit anything.")
        say("  Either CUTS here differ from the run that wrote the CSV, or")
        say("  classify() has changed since. Reconcile them first.")
        rule()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        sys.exit(1)
    say("    LOCK PASSED - the rule is understood. Continuing.")
    say("")

    # --- Part A: gate (1), the untestable early return -------------------
    rule("-")
    say("  PART A - gate (1): `untestable` returns early")
    rule("-")
    say("")
    say("  What changes if an untestable item skips the interleaving test it")
    say("  cannot run, and continues to the locked_in / inconsistent")
    say("  decision? No cut moves. Only the early return is removed.")
    say("")
    for name, rows in arms.items():
        w = wrong(rows)
        before = shares(w)
        for r in w:
            r["mode_ft"] = classify(r, untestable_falls_through=True)
        after = shares(w, "mode_ft")
        n_un = sum(1 for r in w if r["mode"] == "untestable")
        say(f"    {name} - {n_un} untestable ({n_un/len(w):.1%} of wrong "
            f"answers)")
        say("")
        say("      mode            as shipped   with fall-through      diff")
        for m in MODES:
            say(f"      {m:<15} {before.get(m,0):9.1%} {after.get(m,0):17.1%}"
                f"  {after.get(m,0)-before.get(m,0):+9.1%}")
        redist = Counter(r["mode_ft"] for r in w if r["mode"] == "untestable")
        if redist:
            say("")
            say("      the untestable rows land in: "
                + ", ".join(f"{m} {c}" for m, c in redist.most_common()))
        say("")

    # --- Part B: gate (2), gold_ever recall, against hand labels ---------
    rule("-")
    say("  PART B - gate (2): does `gold_ever` catch the interleaving cases?")
    rule("-")
    # All three stay empty / None if Part B cannot run, so Part D degrades
    # instead of raising.
    measured_recall, joined, missed = None, [], []
    if not CONSENSUS_CSV.exists():
        say("")
        say(f"  {CONSENSUS_CSV.name} not found. Run step18c first.")
    else:
        cons = {}
        with open(CONSENSUS_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("consensus") and r.get("qid"):
                    cons[r["qid"]] = (int(r["n"]), r["consensus"])
        by_qid = {r["qid"]: r for r in wrong(arms["Dream"])}
        joined = [(n, lab, by_qid[q]) for q, (n, lab) in cons.items()
                  if q in by_qid]
        say("")
        say(f"  {len(joined)}/{len(cons)} consensus items joined to a Dream row.")
        if len(joined) < len(cons):
            say("  A partial join means the consensus CSV and step17d were not")
            say("  built from the same run. Fix that before trusting Part B.")
        say("")
        say("  Hand label vs the two gate flags:")
        say("")
        say("    consensus      n   gold_testable   gold_ever   both")
        for l in LABELS:
            sub = [(n, r) for n, lab, r in joined if lab == l]
            if not sub:
                continue
            t = sum(1 for _n, r in sub if r["gold_testable"])
            e = sum(1 for _n, r in sub if r["gold_ever"])
            b = sum(1 for _n, r in sub if r["gold_testable"] and r["gold_ever"])
            say(f"    {l} {LONG[l]:<12} {len(sub):3d}   {t:11d}   {e:9d}   "
                f"{b:4d}")
        say("")

        # ------------------------------------------------------------------
        #  Recall, and why the raw sheet number is not it.
        #
        #  The sheet over-samples the classifier's `interleaving` stratum:
        #  12 of 60 items against 3.1% of the arm. Every row in that stratum
        #  has gold_ever = True by construction, so the sheet's I items are
        #  enriched with gate-positive cases and the raw recall is biased
        #  UPWARD. Reweighting by the arm's mode shares removes the bias:
        #
        #      recall = sum_m p_m * n(I and fired | m) / sum_m p_m * n(I | m)
        #
        #  This is the third place in step 18 where a share read straight off
        #  the stratified sheet would have been wrong. Both numbers are
        #  printed so the size of the bias is visible rather than asserted.
        # ------------------------------------------------------------------
        pop = Counter(r["mode"] for r in wrong(arms["Dream"]))
        n_arm = sum(pop.values())
        sheet_n = Counter(r["mode"] for _n, _lab, r in joined)

        def recall_pair(items):
            """(raw sheet recall, hits, recall reweighted to the arm)."""
            hits = sum(1 for _n, r in items if r["gold_ever"])
            raw = hits / len(items) if items else None
            num = den = 0.0
            for mode, count in pop.items():
                if not sheet_n.get(mode):
                    continue
                w = (count / n_arm) / sheet_n[mode]
                sub = [r for _n, r in items if r["mode"] == mode]
                den += w * len(sub)
                num += w * sum(1 for r in sub if r["gold_ever"])
            return raw, hits, (num / den if den else None)

        i_items = [(n, r) for n, lab, r in joined if lab == "I"]
        say("    recall of interleaving - the fraction of hand-labelled I")
        say("    items the gate actually fires on:")
        say("")
        say("      subset                  raw on sheet   reweighted to arm")
        recall = {}
        for tag, keep in (("all I items", i_items),
                          ("excluding #33, #48",
                           [(n, r) for n, r in i_items
                            if n not in NOT_INDEPENDENT])):
            if not keep:
                continue
            raw, hits, wtd = recall_pair(keep)
            recall[tag] = wtd if wtd is not None else raw
            wtd_s = "n/a" if wtd is None else f"{wtd:.1%}"
            say(f"      {tag:<22}  {hits}/{len(keep)} = {raw:5.1%}"
                f"   {wtd_s:>16}")
        say("")
        say("      The raw column is biased UPWARD: the sheet over-samples the")
        say("      classifier's own interleaving stratum, every row of which")
        say("      has gold_ever set by construction. The reweighted column is")
        say("      the one that describes the arm.")
        measured_recall = recall.get("excluding #33, #48",
                                     recall.get("all I items"))
        say("")
        missed = [(n, r) for n, r in i_items if not r["gold_ever"]]
        if missed:
            say("    items the hand labels call interleaving and the gate "
                "missed,")
            say("    and the mode the classifier gave them instead:")
            for n, r in sorted(missed):
                why = ("gold not testable" if not r["gold_testable"]
                       else "testable, but no match in the trace")
                say(f"      #{n:<4} filed as {r['mode']:<13} ({why})")
            say("")
            say("    This is where the lost interleaving cases go. It is a")
            say("    direct measurement, not an inference: every one of these")
            say("    is currently inflating the bucket named beside it.")
            say("")
            say("      lost to: " + ", ".join(
                f"{m} {c}" for m, c in
                Counter(r["mode"] for _n, r in missed).most_common()))
            say("")
        # The other direction: does the gate fire where the hand labels say no?
        fired_not_i = sorted(n for n, lab, r in joined
                             if r["gold_ever"] and lab != "I")
        say(f"    gate fired on {len(fired_not_i)} item(s) the hand labels do "
            f"NOT call interleaving"
            + (": " + ", ".join("#" + str(n) for n in fired_not_i)
               if fired_not_i else ""))
        say("    Precision matters as much as recall - a looser matcher that")
        say("    buys recall by firing on non-interleaving cases is not a fix.")
        say("")

    # --- Part C: the same gates on LLaDA's own 59 labels ------------------
    rule("-")
    say("  PART C - the same test on LLaDA's 59 labels (free - they exist)")
    rule("-")
    say("")
    say("  If these gates are broken they are broken in BOTH arms, and the")
    say("  LLaDA labels from the kappa-0.882 round cost nothing to re-use.")
    say("  A gate error that shows up only on Dream is a Dream story; one")
    say("  that shows up on both is a rule story, and changes step 23.")
    say("")
    if not LLADA_KEY_CSV.exists():
        say(f"  {LLADA_KEY_CSV.name} not found - cannot map worksheet position")
        say("  to qid, so LLaDA's labels cannot be attached. This part is")
        say("  OWED, not answered. Part A's LLaDA column is unaffected.")
    else:
        pos2qid = {}
        with open(LLADA_KEY_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    pos2qid[int(r["n"])] = r.get("qid", "")
                except (KeyError, TypeError, ValueError):
                    continue
        # Part C reads two boolean flags and a mode. It asks for nothing else,
        # so a pilot CSV without `copy_run` is perfectly usable here.
        pool = {}
        for src in (*LLADA_CSVS, LLADA_CAL_CSV):
            for r in wrong(load(src, "LLaDA-labels", need=GATE_NEED,
                                fatal=False)):
                pool.setdefault(r["qid"], r)
        pairs = [(i, LLADA_HAND[i - 1], pool[pos2qid[i]])
                 for i in range(1, 61)
                 if LLADA_HAND[i - 1] in LABELS
                 and pos2qid.get(i) in pool]
        say(f"  {len(pairs)}/59 LLaDA hand labels joined to a row.")
        if not pairs:
            say("  No join. The pilot qids are not in any LLaDA mode CSV on")
            say("  the volume. OWED.")
        else:
            say("")
            say("    consensus      n   gold_testable   gold_ever")
            for l in LABELS:
                sub = [p for p in pairs if p[1] == l]
                if not sub:
                    continue
                t = sum(1 for _i, _l, r in sub if r["gold_testable"])
                e = sum(1 for _i, _l, r in sub if r["gold_ever"])
                say(f"    {l} {LONG[l]:<12} {len(sub):3d}   {t:11d}   {e:9d}")
            i_sub = [p for p in pairs if p[1] == "I"]
            if i_sub:
                hit = sum(1 for _i, _l, r in i_sub if r["gold_ever"])
                say("")
                say(f"    recall of interleaving on LLaDA: {hit}/{len(i_sub)}"
                    f" = {hit/len(i_sub):.1%}")
            un = [p for p in pairs if p[2]["mode"] == "untestable"]
            if un:
                say("")
                say(f"    LLaDA rows the rule calls untestable: {len(un)}; "
                    "hand labels say "
                    + ", ".join(f"{k} {v}" for k, v in
                                Counter(p[1] for p in un).most_common()))
    say("")

    # --- Part D: what it implies, stated as an implication ----------------
    rule("-")
    say("  PART D - decomposing the locked_in gap: gates or cuts?")
    rule("-")
    say("")
    say("  Step 18c measured a 27.8-point gap on locked_in. The point of this")
    say("  step is to find out how much of it the two gates can explain, so")
    say("  the refit is asked to close the remainder and nothing more. A cut")
    say("  tuned to swallow a gate error is wrong for a reason no later reader")
    say("  could recover.")
    say("")
    dream_w = wrong(arms["Dream"])
    n_d = len(dream_w)
    s_lock = sum(1 for r in dream_w if r["mode"] == "locked_in") / n_d
    s_int = sum(1 for r in dream_w if r["mode"] == "interleaving") / n_d
    s_lock_ft = sum(1 for r in dream_w if r["mode_ft"] == "locked_in") / n_d

    # The hand-label target, recomputed here by the same stratified estimator
    # step18c used, rather than copied in - two scripts agreeing is a test.
    hand_lock = None
    if measured_recall is not None:
        pop_d = Counter(r["mode"] for r in dream_w)
        sheet_by_mode = defaultdict(list)
        for _n, lab, r in joined:
            sheet_by_mode[r["mode"]].append(lab)
        num = den = 0.0
        for mode, count in pop_d.items():
            obs = sheet_by_mode.get(mode)
            if not obs:
                continue
            w = count / n_d
            den += w
            num += w * (obs.count("L") / len(obs))
        hand_lock = num / den if den else None

    say("    step  what changes                          locked_in    change")
    say(f"    0     as shipped                              {s_lock:7.1%}")
    say(f"    1     untestable falls through to L/C         {s_lock_ft:7.1%}"
        f"   {s_lock_ft-s_lock:+7.1%}")

    gate2_shift = None
    if measured_recall and measured_recall > 0 and missed:
        true_int = min(1.0, s_int / measured_recall)
        to_move = max(0.0, true_int - s_int)
        share_from_lock = sum(1 for _n, r in missed
                              if r["mode"] == "locked_in") / len(missed)
        gate2_shift = to_move * share_from_lock
        say(f"    2     gold match at recall {measured_recall:.0%}         "
            f"        {s_lock_ft-gate2_shift:7.1%}   {-gate2_shift:+7.1%}")
        say(f"            interleaving {s_int:.1%} -> {true_int:.1%}, and "
            f"{share_from_lock:.0%} of the missed")
        say(f"            cases were filed as locked_in")
    say("")

    after = s_lock_ft - (gate2_shift or 0.0)
    if hand_lock is not None:
        say(f"    hand labels, same stratified estimator     {hand_lock:7.1%}")
        say(f"    still unexplained after both gates        "
            f"{after-hand_lock:+9.1%}")
        say("")
        if abs(after - hand_lock) <= 0.05:
            say("    The gates account for essentially all of it. Fix them and")
            say("    re-measure BEFORE touching a single cut - the refit may")
            say("    turn out to be unnecessary.")
        else:
            dropped = ", ".join(
                f"{sum(1 for r in wrong(rows) if r['mode'] == 'untestable')} "
                f"{name}" for name, rows in arms.items())
            say("    The gates do NOT account for most of the gap. Both fixes")
            say(f"    are still correct and still worth making - {dropped} rows")
            say("    are currently dropped from the analysis altogether - but")
            say("    the remainder is the CUTS' doing, and the refit has to")
            say("    close it.")
            say("")
            say("    Note the direction of step 1: repairing the untestable")
            say("    early return pushes locked_in UP, away from the hand")
            say("    labels. A fix that makes the headline number worse is the")
            say("    kind that gets quietly dropped. It is correct, so it stays.")
    else:
        say("    Hand-label target could not be recomputed here, so no")
        say("    decomposition is drawn.")
    say("")

    rule()
    say("  NOTHING WAS REFITTED, AND NO MODE COLUMN WAS REWRITTEN.")
    say("")
    say("  Read Part A and Part B, then step 18e does, in this order:")
    say("    1. remove the untestable early return, keeping a flag that says")
    say("       interleaving could not be tested on that row")
    say("    2. widen the gold match ONLY as far as Part B's precision column")
    say("       allows, and re-validate on both arms' hand labels")
    say("    3. ONLY THEN refit Dream's s_cut and c_cut, on the exact decimal")
    say("       grids, with a population-weighted objective declared first")
    say("  LLaDA's cuts are not touched. Its GATES are shared, so any gate")
    say("  change is re-validated on its 59 labels before it is adopted.")
    rule()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT_PATH}")


# ---------------------------------------------------------------------------
#  Regression tests:  python step18d_gate_audit.py --test
# ---------------------------------------------------------------------------

def _test() -> None:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {got!r} (want {want!r})")

    def row(**kw):
        r = dict(stable_top3=0.5, cands_top3=2.0, q_overlap=0.1, copy_run=1,
                 correct=False, truncated=False, gold_ever=False,
                 gold_testable=True, copy_row_missing=False)
        r.update(kw)
        return r

    check("truncated wins over everything",
          classify(row(truncated=True, gold_ever=True)), "degenerate")
    check("echo needs BOTH overlap and run",
          classify(row(q_overlap=0.9, copy_run=2)), "locked_in")
    check("echo fires when both clear",
          classify(row(q_overlap=0.9, copy_run=9)), "echo")
    check("gold_ever -> interleaving",
          classify(row(gold_ever=True)), "interleaving")
    check("untestable returns early as shipped",
          classify(row(gold_testable=False, stable_top3=0.5, cands_top3=2.0)),
          "untestable")
    check("untestable falls through to locked_in when asked",
          classify(row(gold_testable=False, stable_top3=0.5, cands_top3=2.0),
                   untestable_falls_through=True), "locked_in")
    check("untestable falls through to inconsistent when the cuts say so",
          classify(row(gold_testable=False, stable_top3=0.1, cands_top3=9.0),
                   untestable_falls_through=True), "inconsistent")
    # The fall-through must NOT smuggle in an interleaving verdict: the test
    # cannot be run on these rows, so it must not be run.
    check("fall-through never returns interleaving",
          classify(row(gold_testable=False, gold_ever=True),
                   untestable_falls_through=True), "locked_in")
    # Cut boundaries are >= and <= exactly, on exact decimals.
    check("s_cut boundary is inclusive",
          classify(row(stable_top3=0.26, cands_top3=3.4)), "locked_in")
    check("just under s_cut is inconsistent",
          classify(row(stable_top3=0.25, cands_top3=3.4)), "inconsistent")
    check("just over c_cut is inconsistent",
          classify(row(stable_top3=0.26, cands_top3=3.5)), "inconsistent")
    # A row with an unusable numeric column must be named, never guessed.
    check("missing numeric column is reported, not defaulted",
          classify(row(copy_row_missing=True)), "unknown")

    check("LLaDA hand labels are 60 long with one drop",
          (len(LLADA_HAND), LLADA_HAND.count(".")), (60, 1))

    print("\n  " + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    main()