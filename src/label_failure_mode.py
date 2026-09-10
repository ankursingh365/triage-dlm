"""Assign each wrong answer to one of three failure modes. Built in Step 17.

    1. INTERLEAVING   - flips between right and wrong, settles on wrong
    2. INCONSISTENT   - cycles through 3+ unrelated wrong candidates
    3. LOCKED-IN      - wrong answer fixed within the first ~30% of rounds
                        and never changes

Mode 3 is the whole point of the project. Trajectory detectors are
structurally blind to it because there is no hesitation to read.

Gate (Step 18): manual review of 100, plus a second annotator on 50 with
Cohen's kappa reported.
"""
