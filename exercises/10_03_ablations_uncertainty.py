# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 10.03: Original ideas, ablations and uncertainty.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
An original contribution needs a falsifiable hypothesis and controlled evidence. Change one mechanism first, match data/tokenizer and budget, run paired seeds, inspect failure cases and remove the proposed component as an ablation. A positive average alone can reflect noise, unequal compute or selection over many trials. Record negative results and limits.

INPUTS AND RETURN
baseline,variant: equal nonempty paired held-out losses under matched seeds/budgets; seed: local bootstrap RNG seed; bootstrap: positive replicates. Return mean_improvement,ci_low,ci_high,count. Improvement=baseline-variant; percentile bootstrap interval95%, exploratory for small samples.

WORKED EXAMPLE
Baseline losses [3,4,5], variant[2,3,4] yield differences[1,1,1], mean1 and every bootstrap mean1, so interval[1,1]. Differences[-1,0,1] have mean0 and an interval spanning0; that does not establish a reliable gain. Three seeds are weak evidence even when the toy interval degenerates.

BOUNDARIES
Do not report only the best seed or claim significance from a tiny bootstrap. Predefine the primary metric; multiple variants require honest selection reporting and final independent evaluation.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def paired_effect(baseline: list[float], variant: list[float], seed: int, bootstrap: int = 1000) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Convert paired arrays and subtract variant from baseline.
    # STEP 2: Use np.random.default_rng(seed), sample indices shape[bootstrap,n] with replacement.
    # STEP 3: Mean sampled differences along axis1, use np.quantile at [.025,.975].
    # STEP 4: Return actual mean and interval/count, without changing global NumPy RNG.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement paired_effect for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=paired_effect([3,4,5],[2,3,4],5);assert a=={'mean_improvement':1.,'ci_low':1.,'ci_high':1.,'count':3}
    b=paired_effect([1,2,3],[2,2,2],7,2000);assert b['mean_improvement']==0 and b['ci_low']<0<b['ci_high']
    assert b==paired_effect([1,2,3],[2,2,2],7,2000)
    state=np.random.get_state();paired_effect([1],[1],2);after=np.random.get_state();assert np.array_equal(state[1],after[1])

if __name__=="__main__":
    run('10.03',run_checks)
