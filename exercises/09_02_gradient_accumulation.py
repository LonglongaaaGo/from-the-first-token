# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 09.02: Position-weighted gradient accumulation.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Accumulation trades more sequential microbatches for lower activation memory at one optimizer update. Normalize by the total valid target count, including an incomplete final microbatch. Each backward frees its graph while gradients add into buffers. The learning-rate schedule advances once per optimizer update, not once per backward.

INPUTS AND RETURN
batches: nonempty (X,Y,mask) list with at least one target total; model returns logits, dropout off for equality tests; optimizer owns model parameters. Return pre-update global target-mean loss float; zero grads once, backward per microbatch, step once.

WORKED EXAMPLE
MicrobatchA has2 targets and mean loss1; B has4 and mean loss2. Backpropagate contributions2/6*1 and4/6*2, total5/3. Equal dividing by2 gives1.5 and a different gradient. With deterministic layers, concatenated full-batch SGD and accumulated SGD must produce matching parameters.

BOUNDARIES
Accumulation does not make BatchNorm or stochastic dropout identical to a full batch. Clipping should follow all accumulation and AMP unscaling.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def accumulated_step(model: nn.Module, batches: list[tuple], optimizer: torch.optim.Optimizer) -> float:
    # ===== START LEARNER CODE =====
    # STEP 1: Count all valid masks before forward; zero optimizer gradients once.
    # STEP 2: For each microbatch compute unreduced CE, select valid values and sum divided by global count.
    # STEP 3: Accumulate contribution.item into report, contribution.backward() each time.
    # STEP 4: optimizer.step once after loop; return the report.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement accumulated_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(42);a=nn.Sequential(nn.Embedding(4,3),nn.Linear(3,4)).double();b=copy.deepcopy(a)
    x=torch.tensor([[0,1],[1,2],[2,3]]);y=(x+1)%4;mask=torch.tensor([[1,0],[1,1],[1,1]],dtype=torch.bool)
    oa=torch.optim.SGD(a.parameters(),lr=.1);ob=torch.optim.SGD(b.parameters(),lr=.1)
    u=accumulated_step(a,[(x[:1],y[:1],mask[:1]),(x[1:],y[1:],mask[1:])],oa)
    ob.zero_grad();z=b(x);per=F.cross_entropy(z.reshape(-1,4),y.reshape(-1),reduction='none').reshape_as(y);l=per[mask].mean();l.backward();ob.step()
    assert math.isclose(u,l.item(),abs_tol=1e-10)
    for p,q in zip(a.parameters(),b.parameters()):close(p,q)

if __name__=="__main__":
    run('09.02',run_checks)
