"""Reproduce published detectors. Built in Steps 20-21.

    - Ave Entropy   GATE: must land at 62-65 AUROC on LLaDA.
                    TraceDet reports 62.8. If you get 50 or 75, there is a bug.
    - TRE
    - TraceDet      target ~72.0 (LLaDA) / ~80.8 (Dream)
    - DynHD         target ~84.2 - the strongest trajectory baseline

Do NOT use semantic entropy as an anchor. It transfers badly to diffusion
models and lands below chance on some settings.
"""
