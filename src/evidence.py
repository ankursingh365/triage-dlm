"""External evidence stage - the CRADLE contribution. Built in Steps 24-27.

    BM25 retrieval over Wikipedia  ->  DeBERTa-large-MNLI entailment

Step 26 also runs the entailment stage on the WITHHELD gold passages to give
a perfect-evidence upper bound, so we can report what fraction of the
available headroom BM25 actually captures.
"""
