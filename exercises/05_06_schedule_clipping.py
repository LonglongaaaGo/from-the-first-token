# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.06: Learning-rate schedules and gradient clipping.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Warmup increases step size while moments and activation scales stabilize. Cosine decay reduces it near the budget endpoint. Global norm clipping scales all parameter gradients by a single factor and preserves their direction. It bounds the gradient norm, not necessarily the AdamW parameter displacement.

INPUTS AND RETURN
parameters: tensors with optional .grad; step: 0-based update index; 1<=warmup<total, 0<=step<=total; peak>=minimum>=0; max_norm>0. Return (lr,original_global_grad_norm), and scale existing gradients in place. No parameter value update.

WORKED EXAMPLE
warmup=2,total=6,peak=.1,min=.01: step0 lr=.05, step1=.1, step2=.1, step4=.055, step6=.01. Gradients [3,4] have norm5; max_norm=2 yields factor.4 and [1.2,1.6]. A missing grad contributes nothing. Zero gradients need no scaling and no division by0.

BOUNDARIES
Clip after unscaling AMP gradients and before optimizer.step. Per-parameter clipping changes the combined direction.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def scheduled_clip(parameters: list[torch.Tensor], step: int, warmup: int, total: int, peak: float, minimum: float, max_norm: float) -> tuple[float, float]:
    """Return (learning rate, pre-clipping global norm); mutate only .grad.

    parameters: list of distinct floating tensors, with matching dense real
    .grad tensors or None. Shapes may differ; no overlapping gradient storage.
    Finite inputs whose squared sum is representable in float64, on a backend
    supporting these statistics. Native float32/float64 cases are checked.
    step=s: zero-based update index, 0<=s<=total; warmup=w and total=S
    satisfy 1<=w<S. peak>=minimum>=0; max_norm=c>0. Return two Python floats.
    Preserve parameters, gradient objects/shapes/dtypes/devices and None.
    No backward, optimizer update, moment change or optimizer-rate assignment.

    EXACT SCHEDULE
      s<w:  lr=peak*(s+1)/w
      s>=w: progress=(s-w)/(S-w)
             lr=minimum+.5*(peak-minimum)*(1+cos(pi*progress))
    total is an endpoint INDEX: total=6 permits 0..6, seven sample points.
    A loop range(total) uses only 0..5. Do not add an extra training step
    merely to reach the endpoint. Peak occurs at both w-1 and w by design.
    AdamW in05.05 uses one-based t; with one update each round, t=s+1.

    GRADIENT LEDGER
      collect only existing .grad tensors, without creating missing ones
      sum squared entries in float64, converting BEFORE square
      original_norm=sqrt(sum), a Python float; empty/all-zero gives0
      if original_norm>c: factor=c/original_norm; scale ALL grads in place
      return lr,original_norm; never replace it with the clipped norm
    No epsilon is needed in this branch: norm>c>0 guarantees a nonzero divisor.

    ORIGINAL FIXTURE
      p=[1,1],p.grad=[3,4],shape(2,); q=[1],q.grad=None,shape(1,).
      s=4,w=2,S=6,peak=.1,minimum=.01,c=2.
      progress=.5,cos(pi/2)=0,lr=.055; square sum25,norm5,factor.4.
      New p.grad=[1.2,1.6],norm2; p/q and q.grad=None remain unchanged.
      Return (.055,5.0). Schedule s=0,1,2,3,4,5,6 gives
      [.05,.1,.1,.08681980515,.055,.02318019485,.01].
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces use `if step < warmup:`. At EIGHT spaces
    # save lr=peak*(step+1)/warmup. s=0 gives .05, s=1 gives .1.
    # STEP 2: At FOUR spaces use `else:`. At EIGHT spaces compute progress
    # (step-warmup)/(total-warmup), then math.cos(math.pi*progress) in the
    # exact schedule above. s=4 gives .055; do not change < to <=.
    # STEP 3: Back at FOUR spaces make an empty gradient list and a `for
    # parameter in parameters:` loop. At EIGHT spaces check `if
    # parameter.grad is None:`; at TWELVE spaces `continue` skips that
    # parameter. Back at EIGHT append the existing .grad. None is absent,
    # unlike a tensor containing zeros. Do not create gradients or copies.
    # STEP 4: At FOUR spaces set squared_norm=0.0 and loop over collected
    # gradients. At EIGHT spaces detach the statistics read, convert to
    # torch.float64 BEFORE square(), sum() all entries, then use .item()
    # to add the scalar. Original fixture: 9+16=25. Do not square parameter
    # values, average entries, or directly add separate tensor norms.
    # STEP 5: At FOUR spaces save original_norm=math.sqrt(squared_norm).
    # This is a Python float, 5.0 in the fixture or0 for no/zero gradients.
    # Retain it unchanged as the diagnostic returned after clipping.
    # STEP 6: At FOUR spaces test `if original_norm > max_norm:`. At EIGHT
    # compute factor=max_norm/original_norm, .4 here. This branch guarantees
    # a nonzero divisor; small/zero gradients require no scaling or division.
    # STEP 7: Inside that if, `with torch.no_grad():` uses EIGHT spaces,
    # its gradient loop TWELVE, and gradient.mul_(factor) SIXTEEN. The
    # underscore changes existing buffers. Rebinding gradient=... does not.
    # Use the SAME factor for all tensors, retaining parameter values/None.
    # STEP 8: After ALL branches/loops return lr,original_norm at FOUR
    # spaces. Fixture (.055,5.0), not (.055,2.0); no optimizer.step here.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement scheduled_clip for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====






# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    p=torch.ones(2,requires_grad=True);p.grad=torch.tensor([3.,4.]);q=torch.ones(1,requires_grad=True)
    lr,n=scheduled_clip([p,q],4,2,6,.1,.01,2.);assert math.isclose(lr,.055) and math.isclose(n,5.);close(p.grad,torch.tensor([1.2,1.6]));close(p,torch.ones(2));assert q.grad is None
    assert math.isclose(scheduled_clip([],0,2,6,.1,.01,2.)[0],.05)
    assert math.isclose(scheduled_clip([],6,2,6,.1,.01,2.)[0],.01)

if __name__=="__main__":
    run('05.06',run_checks)
