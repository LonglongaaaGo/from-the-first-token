# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 10.02: Measured scaling and budget choices.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Scaling studies help allocate limited compute across parameter count and training tokens. Fit measured controlled runs, include uncertainty and test held-out budget predictions. The fixed-floor log fit is pedagogical: real loss floors, data reuse and system efficiency can change conclusions. Never extrapolate a few toy runs straight to a large training commitment.

INPUTS AND RETURN
compute: at least2 distinct positive budgets; losses: same-length measured held-out losses, each>floor; floor: assumed fixed irreducible-loss estimate. Return coefficient a,exponent b for loss≈floor+a*C^(-b) using log-space least squares. This is a toy fit, not a universal law.

WORKED EXAMPLE
C=[1,4,16], L=[4,2,1], floor0. Doubling sqrt(C) halves loss, so a=4,b=.5. Logs form a line with slope-.5. Predicted C=64 loss=.5. Repeat with floor1 and L=[5,3,2]: subtracting floor yields the same a,b, and C64 predicts1.5.

BOUNDARIES
Comparing models with different data quality or tokenizers confounds the fit. An estimated improvement must be checked at a new budget.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def fit_scaling(compute: list[float], losses: list[float], floor: float = 0.) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Convert lists to NumPy float64, x=log(compute), y=log(losses-floor).
    # STEP 2: Center x,y; compute covariance sum divided by squared x sum for slope.
    # STEP 3: intercept=y.mean()-slope*x.mean(); return exp(intercept),-slope in named dict.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement fit_scaling for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=fit_scaling([1,4,16],[4,2,1]);b=fit_scaling([1,4,16],[5,3,2],1.)
    assert math.isclose(a['coefficient'],4) and math.isclose(a['exponent'],.5)
    assert math.isclose(b['coefficient'],4) and math.isclose(b['exponent'],.5)
    c=fit_scaling([2,8,32],[2,1,.5]);assert math.isclose(c['exponent'],.5)

if __name__=="__main__":
    run('10.02',run_checks)
