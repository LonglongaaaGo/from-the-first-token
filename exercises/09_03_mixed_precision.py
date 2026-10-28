# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 09.03: Autocast and gradient scaling.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Autocast selects lower precision for eligible operations and retains safer precision where appropriate. FP16 has a narrow exponent range; scaling helps avoid tiny-gradient underflow. BF16 shares FP32 exponent range but has fewer precision bits. CPU BF16 checks teach operation order; CUDA overflow/skip behavior needs an accelerated validation run.

INPUTS AND RETURN
model: module; x,y: long[B,T]; optimizer: model optimizer; scaler: None for CPU BF16 or caller-owned CUDA GradScaler for CUDA FP16. Return pre-update loss float. Clip unscaled global grad norm to1 before step. Existing fp32 model parameters remain fp32.

WORKED EXAMPLE
Loss=.01, true grad=.000001, scale1024: scaled loss10.24 and grad.001024. unscale restores.000001; clipping before unscale would operate on the wrong norm. In CPU fixture the embedding/linear model stays fp32 while linear output may be BF16 and CE is computed safely. Save scaler state with checkpoints.

BOUNDARIES
Do not cast all parameters to half just because autocast is used. For real CUDA accumulation unscale/step/update once per optimizer update.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def mixed_precision_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer, scaler=None) -> float:
    # ===== START LEARNER CODE =====
    # STEP 1: zero_grad; choose device_type=x.device.type and dtype BF16 for CPU or FP16 for CUDA.
    # STEP 2: Under torch.autocast(device_type=...,dtype=...), forward and CE; leave context before backward.
    # STEP 3: If scaler is None backward, clip_grad_norm_, step; else scale/backward, unscale_, clip, scaler.step and update.
    # STEP 4: Return loss.item; leave optimizer/scaler ownership with caller.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement mixed_precision_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(43);m=nn.Sequential(nn.Embedding(3,4),nn.Linear(4,3));x=torch.tensor([[0,1,2]]);y=torch.tensor([[1,2,0]])
    before=[p.detach().clone() for p in m.parameters()];o=torch.optim.SGD(m.parameters(),lr=.05)
    l=mixed_precision_step(m,x,y,o);assert math.isfinite(l) and l>0
    assert all(p.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(not torch.equal(a,p) for a,p in zip(before,m.parameters()))
    norm=math.sqrt(sum(float(p.grad.square().sum()) for p in m.parameters()));assert norm<=1.00001

if __name__=="__main__":
    run('09.03',run_checks)
