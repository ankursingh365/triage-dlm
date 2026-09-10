"""Turn a logged trajectory into roughly 25 numbers. Built in Step 19.

Exclusions that are NOT optional:
    - EOS and padding positions. LLaDA's SFT data causes early termination
      under low-confidence remasking; a settle-round feature computed over
      EOS measures that artefact, not truth.
    - Prompt positions. The prompt is never masked, so it has no trajectory.

In vanilla LLaDA a revealed position STAYS revealed (its confidence is set
to infinity). So "flips" must be measured on the argmax PREDICTION at
still-masked positions across rounds, not on revealed tokens changing.
"""
