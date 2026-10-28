# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 09.01: Memory, compute and profiling.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Separate persistent model state from activations, temporary workspaces and cache. 6PN is a dense-Transformer training estimate, not exact attention or optimizer cost. Measure warmup-separated iterations and synchronize accelerator timing. Profile before optimizing: data loading, forward, backward and optimizer can dominate different runs.

INPUTS AND RETURN
parameters,tokens: positive counts; seconds>0 measured synchronized training duration; parameter_bytes: positive bytes per parameter. Return parameter_memory,gradient_memory,optimizer_memory bytes, approximate_flops, tokens_per_second. Gradient/moment baseline uses float32.

WORKED EXAMPLE
P=1000,N=2000,time2s,fp32 parameters: weights4000 bytes, grads4000,moments8000,total16000 before activations. Estimated FLOPs=12,000,000, throughput1000 tokens/s. If forward=.2s, backward=.5s, data=.3s, reducing only data to0 cannot improve more than1/.7≈1.43x for that measured iteration.

BOUNDARIES
A half-precision parameter scheme may need extra master weights; standard autocast often keeps parameters fp32. Estimated state memory is not peak memory.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def training_budget(parameters: int, tokens: int, seconds: float, parameter_bytes: int = 4) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Multiply parameters by parameter_bytes,4 and8 for the three state terms.
    # STEP 2: Estimate6*parameters*tokens FLOPs; divide tokens by measured seconds.
    # STEP 3: Return named fields. Report missing activation/master-weight costs separately in experiments.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement training_budget for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=training_budget(1000,2000,2.);assert a=={'parameter_memory':4000,'gradient_memory':4000,'optimizer_memory':8000,'approximate_flops':12000000,'tokens_per_second':1000.}
    b=training_budget(1000,2000,2.,2);assert b['parameter_memory']==2000 and b['optimizer_memory']==8000

if __name__=="__main__":
    run('09.01',run_checks)
