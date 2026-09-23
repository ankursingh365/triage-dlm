"""
Dataset loading and prompt construction.
========================================

Three datasets, one normalised record type, and one rule that the whole project
depends on: **the model never sees the supporting evidence.**

The withholding rule
--------------------
TriviaQA and HotpotQA both ship supporting passages. If those are placed in the
prompt, accuracy rises to the point where there are almost no hallucinations left
to study - the task becomes reading comprehension rather than recall. So the
passages are loaded, kept, and deliberately withheld at generation time.

They are needed later for two things:

  * **Stage 3 labelling** - deciding whether an answer was correct.
  * **Step 26, the gold-evidence upper bound** - running the entailment stage on
    perfect evidence instead of BM25 results, which yields "retrieval captures X
    of the Y points available from perfect evidence".

This module enforces the rule structurally rather than by convention:
`build_prompt` receives only the question and choices. It has no parameter
through which evidence could be passed, so no future edit can leak it by
accident.

The three datasets
------------------
**TriviaQA** (`mandarjoshi/trivia_qa`, config `rc.nocontext`) - the main dataset.
Short factual answers, verifiable in Wikipedia. The `nocontext` config is used
because we withhold context anyway; it downloads in minutes rather than hours.
It retains `entity_pages.title`, which names the Wikipedia articles the answer
came from - enough to fetch gold evidence at Step 26 without carrying gigabytes
of passage text through Phase C.

**HotpotQA** (`hotpotqa/hotpot_qa`, config `distractor`) - harder, multi-hop.
Retrieval is expected to be the weak link here, and the `distractor` config
provides `supporting_facts`, genuine gold evidence, which is what makes the
Step 26 headroom analysis diagnostic rather than decorative.

**CommonsenseQA** (`tau/commonsense_qa`) - the NEGATIVE CONTROL. Its answers are
not in Wikipedia, so external evidence should give ZERO improvement. If evidence
helps here, there is a bug - most likely evidence leaking into the prompt, or a
retrieval index accidentally containing the questions. It carries no evidence
passages by design, and `evidence_passages` is empty for every record.

Splits
------
Validation splits throughout. Test splits of TriviaQA and HotpotQA have hidden
answers, and CommonsenseQA's test set has no `answerKey`.

Sampling is seeded and deterministic: the same seed and count always produce the
same questions, in the same order, so Phase C is reproducible and resumable.

Prompt format
-------------
The format below has NOT been checked against TraceDet's. Before Step 20, read
their experimental setup and match it - the Ave Entropy sanity gate expects
62-65 AUROC, and a different prompt format is a plausible reason to miss that
band while every line of code is correct.

Also unresolved: the LLaDA-MoE chat template injects
`<role>SYSTEM</role>detailed thinking off<|role_end|>` by default. That was
observed in Step 2c. It must be pinned explicitly before Phase C, and recorded
in the paper, or the runs are not reproducible by anyone else.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from . import config


# ===========================================================================
# The normalised record
# ===========================================================================

@dataclass
class QARecord:
    """One question, in a form identical across all three datasets.

    `generate.py` sees only `question` and `choices`. Everything else exists for
    labelling and for the gold-evidence upper bound, and must never reach the
    model at generation time.
    """

    dataset: str                    # "triviaqa" | "hotpotqa" | "commonsenseqa"
    qid: str                        # stable id, used as the trajectory filename
    question: str

    # Every acceptable surface form. TriviaQA supplies extensive aliases
    # ("JFK", "John F. Kennedy", "Kennedy, John Fitzgerald"), which matters a
    # great deal for judge agreement in Stage 3.
    gold_answers: list[str] = field(default_factory=list)

    # CommonsenseQA only. Multiple choice, so the options must be shown or the
    # question is unanswerable.
    choices: Optional[list[str]] = None

    # WITHHELD FROM THE MODEL. For Stage 3 labelling and Step 26 only.
    evidence_titles: list[str] = field(default_factory=list)     # TriviaQA
    evidence_passages: list[str] = field(default_factory=list)   # HotpotQA

    split: str = "validation"

    def to_json(self) -> dict:
        return asdict(self)


# ===========================================================================
# Per-dataset normalisers
# ===========================================================================

def _norm_triviaqa(row) -> QARecord:
    ans = row.get("answer") or {}
    golds = []
    if ans.get("value"):
        golds.append(ans["value"])
    for key in ("aliases", "normalized_aliases"):
        golds.extend(ans.get(key) or [])

    pages = row.get("entity_pages") or {}
    titles = list(pages.get("title") or [])

    return QARecord(
        dataset="triviaqa",
        qid=str(row.get("question_id") or _hash_id(row["question"])),
        question=row["question"].strip(),
        gold_answers=_dedupe(golds),
        evidence_titles=titles,
    )


def _norm_hotpotqa(row) -> QARecord:
    # `context` is {"title": [...], "sentences": [[...], ...]} and
    # `supporting_facts` is {"title": [...], "sent_id": [...]}. Join them to
    # recover only the sentences actually needed to answer - that is the gold
    # evidence, not the whole distractor set.
    ctx = row.get("context") or {}
    titles = list(ctx.get("title") or [])
    sentences = list(ctx.get("sentences") or [])
    by_title = {t: s for t, s in zip(titles, sentences)}

    sup = row.get("supporting_facts") or {}
    passages = []
    for t, sid in zip(sup.get("title") or [], sup.get("sent_id") or []):
        sents = by_title.get(t)
        if sents and 0 <= sid < len(sents):
            passages.append(f"{t}: {sents[sid].strip()}")

    return QARecord(
        dataset="hotpotqa",
        qid=str(row.get("id") or _hash_id(row["question"])),
        question=row["question"].strip(),
        gold_answers=[row["answer"]] if row.get("answer") else [],
        evidence_titles=_dedupe(list(sup.get("title") or [])),
        evidence_passages=passages,
    )


def _norm_commonsenseqa(row) -> QARecord:
    ch = row.get("choices") or {}
    labels = list(ch.get("label") or [])
    texts = list(ch.get("text") or [])
    key = row.get("answerKey") or ""

    gold = []
    if key and key in labels:
        gold = [texts[labels.index(key)], key]

    return QARecord(
        dataset="commonsenseqa",
        qid=str(row.get("id") or _hash_id(row["question"])),
        question=row["question"].strip(),
        gold_answers=_dedupe(gold),
        choices=[f"{l}. {t}" for l, t in zip(labels, texts)],
        # Deliberately empty. This is the negative control: there is no evidence
        # to find, so evidence must not help.
        evidence_titles=[],
        evidence_passages=[],
    )


SPECS = {
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", _norm_triviaqa),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", _norm_hotpotqa),
    "commonsenseqa": ("tau/commonsense_qa", None, _norm_commonsenseqa),
}


def _dedupe(items) -> list[str]:
    """Order-preserving dedupe, case-insensitive, dropping blanks."""
    seen, out = set(), []
    for x in items:
        if not x:
            continue
        k = str(x).strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(str(x).strip())
    return out


def _hash_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# Loading
# ===========================================================================

def load_records(name: str, n: Optional[int] = None,
                 seed: Optional[int] = None,
                 split: str = "validation",
                 cache: bool = True) -> list[QARecord]:
    """Load and normalise `n` questions from one dataset.

    Sampling is seeded and deterministic: the same `(name, n, seed)` always
    yields the same questions in the same order, so Phase C can be interrupted
    and resumed, and anyone re-running the project gets identical data.

    Sampling happens on the *index list*, not the rows, so it does not depend on
    how the datasets library orders or shards anything internally.

    With `cache=True` the normalised records are written to
    `data/raw/<name>_<split>_<n>_<seed>.json`, so later steps do not re-download
    or re-normalise.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(SPECS)}")

    seed = config.SEED if seed is None else seed
    path, cfg, normalise = SPECS[name]

    cache_file = config.RAW_DIR / f"{name}_{split}_{n or 'all'}_{seed}.json"
    if cache and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [QARecord(**r) for r in raw]

    from datasets import load_dataset
    ds = (load_dataset(path, cfg, split=split) if cfg
          else load_dataset(path, split=split))

    idx = list(range(len(ds)))
    if n is not None and n < len(idx):
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])          # sort back so disk access stays sequential

    records = []
    for i in idx:
        try:
            rec = normalise(ds[i])
        except Exception:
            continue                    # skip malformed rows rather than abort
        if rec.question and rec.gold_answers:
            rec.split = split
            records.append(rec)

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps([r.to_json() for r in records], ensure_ascii=False, indent=1),
            encoding="utf-8")

    return records


# ===========================================================================
# Prompt construction
# ===========================================================================

# Note the signature: `question` and `choices` only. There is no parameter
# through which evidence could be passed, which is how the withholding rule is
# enforced - by the shape of the function rather than by a comment asking future
# edits to behave.

def build_prompt_text(question: str, choices: Optional[list[str]] = None) -> str:
    """The user turn, before any chat template is applied.

    Short answers are requested explicitly. Generation length is fixed at 64
    tokens (TraceDet found 64 best, 128 worse), so a discursive answer would be
    truncated mid-sentence and the trailing positions would record a truncation
    artefact rather than the model's behaviour.
    """
    if choices:
        options = "\n".join(choices)
        return (f"{question}\n\n{options}\n\n"
                f"Answer with the single best option.")
    return f"{question}\n\nAnswer as briefly as possible."


def build_prompt_ids(tokenizer, record: QARecord) -> list[int]:
    """Tokenise one record's prompt, applying the model's chat template.

    Falls back to a plain format if the tokenizer has no template. The fallback
    is off-distribution for an instruct checkpoint and Step 2c showed the
    difference is large, so callers should check `has_chat_template` and record
    which path was used.
    """
    text = build_prompt_text(record.question, record.choices)
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    except Exception:
        return tokenizer(f"Question: {text}\nAnswer:",
                         add_special_tokens=True)["input_ids"]


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


# ===========================================================================
# Guard
# ===========================================================================

def assert_evidence_withheld(prompt_text: str, record: QARecord,
                             min_overlap: int = 8) -> None:
    """Raise if gold evidence was ADDED to the prompt.

    Cheap insurance against the single mistake that would invalidate the whole
    project. Leakage produces *better* results and therefore no symptom that
    would prompt anyone to go looking, so it has to be caught mechanically.

    The subtlety: **HotpotQA questions restate their own supporting passages.**
    Bridge questions are constructed by stitching facts out of the context, so
    a question like

        "Roger O. Egeberg was Assistant Secretary for Health and Scientific
         Affairs during the administration of a president that served during
         what years?"

    shares long verbatim spans with its gold evidence by design. A naive n-gram
    check against the whole prompt fires on ~every HotpotQA record - the first
    version of this function killed a Step 9 run that way.

    So the test is not "does evidence appear in the prompt" but "does evidence
    appear in the prompt in a way the QUESTION does not already explain". A span
    present in the question is the dataset's own phrasing; a span present in the
    prompt but absent from the question came from somewhere else, and somewhere
    else is a leak.
    """
    if not record.evidence_passages:
        return

    def norm(text: str) -> str:
        return " ".join(str(text).lower().split())

    hay = norm(prompt_text)
    question = norm(record.question)

    for passage in record.evidence_passages:
        words = norm(passage).split()
        for i in range(0, max(0, len(words) - min_overlap) + 1):
            span = " ".join(words[i:i + min_overlap])
            if span in hay and span not in question:
                raise AssertionError(
                    f"EVIDENCE LEAK in {record.dataset}/{record.qid}: the span\n"
                    f"    {span!r}\n"
                    f"appears in the prompt but not in the question, so it did "
                    f"not come from the dataset's own phrasing. Generation must "
                    f"never see supporting passages."
                )