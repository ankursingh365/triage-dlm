"""Metrics and the key experiment. Built in Steps 22-23, 28-31.

AUROC, AUPRC and AURAC - reported overall AND separately per failure mode,
each with a 1000-resample bootstrap confidence interval.

Validation gates:
    - shuffle the labels and retrain: AUROC must fall to ~0.50 (leakage test)
    - CommonsenseQA: evidence must give ZERO gain (negative control)
"""
