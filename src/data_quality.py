"""
Dataset quality flagging.
=========================

The three benchmarks all contain defects that would corrupt this project's
positive class. They were found by printing raw records in Step 7 - a good
argument for always inspecting a loader's output before trusting it.

Why flag rather than silently drop
-----------------------------------
Every function here *labels* a record and leaves the decision to the caller.
Silent filtering hides how much data was removed and makes the paper impossible
to reproduce. Flagging makes the exclusion rate measurable and reportable:
"we excluded N% of HotpotQA for truncated questions" is a sentence a reviewer can
check. "We used HotpotQA" is not.

Which direction of error to prefer
-----------------------------------
This project's positive class is hallucinations. Two ways to corrupt it:

  * Mark a WRONG answer correct  -> shrinks the positive class
  * Mark a CORRECT answer wrong  -> fills the positive class with non-hallucinations

The second is worse. A detector trained on a positive class polluted with correct
answers is learning noise, and the per-failure-mode analysis - which already
splits a modest sample three ways - cannot absorb it. So every judgement call
below leans towards keeping data and keeping aliases.

That principle is why the first version of this module was wrong. It capped
TriviaQA aliases at 12 to suppress mis-linked entities, and in doing so dropped
`Belgie`, `Koninkrijk Belgie` and `Cockpit of Europe` - all valid names for
Belgium. Mean aliases fell 18.0 to 8.1. The cap is gone.

The reasoning that replaced it: aliases are never the sole correctness criterion.
TraceDet's 90% human agreement came from an LLM judge, not string matching, and
Step 16 verifies 100 labels by hand. A mis-linked alias like `Sjakk` (Norwegian
for chess, attached to an answer of "Chess Records") is inert - it can only cause
a false positive if the model actually produces it, which will not happen in
English QA. A missing alias is not inert at all.

The defects
-----------

**1. Duplicate choices (CommonsenseQA), 2.0% observed.**
`['A. clumsy', 'B. ineffectual', 'C. dull', 'D. clumsy', 'E. stupid']` - A and D
identical. This is the NEGATIVE CONTROL; an unanswerable question there breaks
the control before it is run. Excluded.

**2. Wikipedia artefacts in TriviaQA aliases.** Disambiguation stubs and ISO
codes appear verbatim. Note they appear in BOTH forms: `Rudolph (disambiguation)`
from the `aliases` field and `rudolph disambiguation` from `normalized_aliases`,
which strips punctuation. The first version of this filter only caught the
parenthesised form.

**3. Truncated questions (HotpotQA).** `"When was the American actor, film
director which Dana Brunetti is the president of his "` - ends mid-clause.
Excluded.

**4. Missing punctuation (HotpotQA), 3.7% observed.** `"Which came out first,
Dinosaur or McFarland, USA"` is a perfectly good question that merely lacks a
question mark. INFORMATIONAL only - excluding on this alone discarded usable
data in the first version.

**5. Missing entity titles (TriviaQA).** Not a defect, a scope decision:
`rc.nocontext` strips `entity_pages`, so there is no gold evidence for TriviaQA.
Step 26's upper bound runs on HotpotQA only, where `supporting_facts` is real
gold evidence and where retrieval was already expected to be the weak link.
"""

from __future__ import annotations

import re

MANUAL_CHECK_NOTE = (
    "Heuristics cannot reliably detect malformed multi-hop questions. "
    "Spot-check 50 HotpotQA questions by hand at Step 10 and report the "
    "observed malformation rate alongside the automatic flag rate."
)

# ---------------------------------------------------------------------------
# Wikipedia bookkeeping that appears verbatim in TriviaQA alias lists.
#
# Two forms, because TriviaQA supplies both `aliases` (punctuation intact) and
# `normalized_aliases` (punctuation stripped, lower-cased):
#
#     "Rudolph (disambiguation)"   ->  parenthesised
#     "rudolph disambiguation"     ->  normalised
#
# The first version matched only the parenthesised form, so every normalised
# artefact survived. Both are matched now.
# ---------------------------------------------------------------------------

_ARTEFACT_WORDS = r"(?:disambiguation|disambig|surname|given\s+name|name\s+list)"

_WIKI_ARTEFACT_PATTERNS = [
    re.compile(rf"\(\s*{_ARTEFACT_WORDS}\s*\)\s*$", re.IGNORECASE),  # parenthesised
    re.compile(rf"\s{_ARTEFACT_WORDS}\s*$", re.IGNORECASE),          # normalised
    re.compile(r"^ISO\s*\d{3,4}[-\s]?\d?\s*:", re.IGNORECASE),       # ISO 3166-1:BE
    re.compile(r"^(?:list\s+of|index\s+of|outline\s+of)\s", re.IGNORECASE),
    re.compile(r"^\W+$"),                                            # punctuation only
]


def is_wiki_artefact(text: str) -> bool:
    """True if this alias is Wikipedia bookkeeping rather than a real answer."""
    t = str(text).strip()
    return any(p.search(t) for p in _WIKI_ARTEFACT_PATTERNS)


# Function words that should never end a well-formed question. A question ending
# in one is almost certainly truncated mid-clause.
_TRAILING_FUNCTION_WORDS = {
    "of", "the", "a", "an", "his", "her", "its", "their", "in", "on", "at",
    "to", "for", "with", "by", "and", "or", "is", "was", "were", "are",
    "that", "which", "who", "from", "as", "into", "about", "than",
}

_QUESTION_OPENERS = (
    "who", "what", "when", "where", "which", "why", "how", "whose", "whom",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "has", "have", "had", "name", "in", "on", "at", "the",
)


# ===========================================================================
# 1. Alias cleaning  (TriviaQA)
# ===========================================================================

def clean_aliases(golds: list[str],
                  max_keep: int | None = None) -> tuple[list[str], list[str]]:
    """Remove Wikipedia artefacts from an alias list. Returns `(kept, dropped)`.

    `max_keep` defaults to **None - no cap**. The first version capped at 12 and
    destroyed valid alternate names (`Belgie`, `Koninkrijk Belgie`, `Cockpit of
    Europe`), which makes the judge mark correct answers wrong and inflates the
    hallucination rate. See the module docstring on which direction of error to
    prefer.

    Pass an integer only when bounding a prompt's length, and use
    `aliases_for_prompt` for that rather than truncating the stored record - the
    record should keep everything.

    `golds[0]` is TriviaQA's `answer.value`, the canonical answer, and is always
    kept regardless of what it looks like.
    """
    kept, dropped = [], []
    for i, g in enumerate(golds):
        text = str(g).strip()
        if not text:
            continue
        if i == 0:
            kept.append(text)
            continue
        if is_wiki_artefact(text):
            dropped.append(text)
            continue
        kept.append(text)

    if max_keep is not None and len(kept) > max_keep:
        dropped.extend(kept[max_keep:])
        kept = kept[:max_keep]
    return kept, dropped


def aliases_for_prompt(golds: list[str], limit: int = 15) -> list[str]:
    """A bounded alias list for the Stage 3 judge prompt.

    Storage and use are separate concerns: the record keeps every alias, and only
    the prompt is truncated, so nothing is ever lost from the data itself.
    Aliases are ordered roughly by prominence, so the head is the useful part.
    """
    return list(golds[:limit])


# ===========================================================================
# 2. Per-dataset flags
# ===========================================================================

def flag_commonsenseqa(record) -> list[str]:
    """Duplicate or missing options make a multiple-choice question unanswerable."""
    flags = []
    if not record.choices:
        return ["no_choices"]

    texts = [c.split(". ", 1)[-1].strip().lower() for c in record.choices]
    if len(set(texts)) != len(texts):
        flags.append("duplicate_choices")
    if len(texts) < 3:
        flags.append("too_few_choices")
    if not record.gold_answers:
        flags.append("no_gold")
    return flags


def flag_hotpotqa(record) -> list[str]:
    """Heuristics for broken bridge questions.

    Deliberately conservative, and revised after the first version excluded
    `"Which came out first, Dinosaur or McFarland, USA"` - a valid question
    missing only its punctuation. Missing punctuation is now informational;
    truncation and spliced titles still exclude.
    """
    flags = []
    q = record.question.strip()
    words = q.split()

    if len(words) < 5:
        flags.append("very_short_question")

    # INFORMATIONAL. 3.7% of HotpotQA, and most are answerable.
    if not q.endswith("?"):
        flags.append("no_question_mark")

    # EXCLUDES. A question ending on a function word stopped mid-clause:
    # "...is the president of his " - nothing follows, so nothing can answer it.
    #
    # Requires the question mark to be ABSENT as well. A question that
    # terminates properly is not truncated whatever its final word looks like -
    # testing caught "...Badly Drawn Boy or Wolf A?" being flagged because "a"
    # is a function word. Truncation means the sentence stopped without
    # terminating; a "?" is evidence it did terminate.
    tail = q.split()
    if not q.endswith("?") and tail and tail[-1].lower() in _TRAILING_FUNCTION_WORDS:
        flags.append("truncated_question")

    # EXCLUDES. A title fragment spliced mid-clause where the question also
    # failed to terminate: "of The Hard Easy" with no question mark.
    if re.search(r"\b(of|by|in)\s+The\s+[A-Z]", q) and not q.endswith("?"):
        flags.append("spliced_title")

    # INFORMATIONAL. Fires on ~27% - many valid questions are declarative.
    if words and words[0].lower() not in _QUESTION_OPENERS:
        flags.append("odd_opening")

    # INFORMATIONAL. Only limits Step 26's gold-evidence upper bound.
    if not record.evidence_passages:
        flags.append("no_supporting_facts")

    return flags


def flag_triviaqa(record) -> list[str]:
    flags = []
    if not record.gold_answers:
        flags.append("no_gold")
    if len(record.question.split()) < 4:
        flags.append("very_short_question")
    q = record.question.strip()
    tail = q.split()
    if not q.endswith("?") and tail and tail[-1].lower() in _TRAILING_FUNCTION_WORDS:
        flags.append("truncated_question")
    return flags


_FLAGGERS = {
    "commonsenseqa": flag_commonsenseqa,
    "hotpotqa": flag_hotpotqa,
    "triviaqa": flag_triviaqa,
}


def flag_record(record) -> list[str]:
    """All quality flags for one record. Empty list means clean."""
    fn = _FLAGGERS.get(record.dataset)
    return fn(record) if fn else []


# ===========================================================================
# 3. Filtering, with an audit trail
# ===========================================================================

# Flags severe enough to exclude. `no_question_mark`, `odd_opening` and
# `no_supporting_facts` are deliberately NOT here - see flag_hotpotqa.
EXCLUDE_BY_DEFAULT = {
    "duplicate_choices",
    "too_few_choices",
    "no_choices",
    "no_gold",
    "very_short_question",
    "truncated_question",
    "spliced_title",
}

INFORMATIONAL = {
    "no_question_mark",
    "odd_opening",
    "no_supporting_facts",
}


def partition(records, exclude: set[str] | None = None):
    """Split records into `(kept, excluded, counts)`.

    `counts` maps each flag to how many records carried it - the number that goes
    in the paper's data section. Report it whether or not a reviewer asks.
    """
    exclude = EXCLUDE_BY_DEFAULT if exclude is None else exclude
    kept, excluded, counts = [], [], {}

    for rec in records:
        flags = flag_record(rec)
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
        if any(f in exclude for f in flags):
            excluded.append((rec, flags))
        else:
            kept.append(rec)

    return kept, excluded, counts


def clean_records(records, exclude: set[str] | None = None,
                  max_aliases: int | None = None):
    """Clean aliases, then partition. The one call every later step should use.

    Returns `(kept, excluded, report)` where `report` carries everything the
    paper's data section needs: counts in and out, which flags fired and how
    often, alias statistics, and examples of what was dropped.

    `max_aliases` defaults to None - no cap. Do not set it without reading the
    module docstring first.

    `src/data.py` is untouched by design: it loads and normalises, this module
    judges. Keeping the loader defect-agnostic means it stays correct if a
    dataset is fixed upstream.
    """
    aliases_before = aliases_after = 0
    alias_examples: list[str] = []

    for rec in records:
        if rec.dataset == "triviaqa" and rec.gold_answers:
            aliases_before += len(rec.gold_answers)
            kept_a, dropped_a = clean_aliases(rec.gold_answers, max_keep=max_aliases)
            rec.gold_answers = kept_a
            aliases_after += len(kept_a)
            for d in dropped_a:
                if len(alias_examples) < 12:
                    alias_examples.append(d)

    kept, excluded, counts = partition(records, exclude)

    return kept, excluded, {
        "n_in": len(records),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "flag_counts": counts,
        "aliases_before": aliases_before,
        "aliases_after": aliases_after,
        "alias_examples": alias_examples,
    }