# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.05: AdamW, initialization and decay.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Adam adapts each coordinate using running moments with startup bias correction. AdamW decays parameter values separately from the gradient moments. Initialization sets early activation scales; the paired notes derive fan-in and an optional residual-depth approximation. This assignment implements only the update, not model initialization. Excluding normalization gains and biases from decay is a common configuration choice, not a required universal policy.

INPUTS AND RETURN
parameter,gradient,first,second: matching float tensors; first/second are mutable moment buffers; step: positive 1-based index; learning_rate>=0; betas in[0,1); eps>0; decay>=0. Update parameter and moments in place under no_grad, return None. gradient stays unchanged.

WORKED EXAMPLE
p=1,g=2,m=v=0,step1,beta1=.9,beta2=.99,eta=.1,eps=1e-8,decay=.1. m=.2,v=.04; corrected m=2,v=4. Decay yields .99; gradient step is about.1, final p≈.8900000005. Plain SGD gives .8 before decay, so the two algorithms differ.

BOUNDARIES
L2 added into g changes the moments and differs from decoupled decay. step=0 causes zero correction denominators.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def adamw_update(parameter: torch.Tensor, gradient: torch.Tensor, first: torch.Tensor, second: torch.Tensor, step: int, learning_rate: float, beta1: float, beta2: float, eps: float, decay: float) -> None:
    """Mutate one dense parameter and its two moment buffers; return None.

    parameter/gradient/first/second: matching nonempty real float tensors,
    dtype/device/shape aligned and with separate, non-overlapping storage.
    This exercise checks float32/float64. Finite inputs; second>=0;
    moment buffers are ordinary non-differentiable state, initially zero.
    step: positive integer count of updates to this parameter, starting at1.
    learning_rate>=0, beta1/beta2 in [0,1), eps>0, decay>=0.
    Change parameter/first/second in place under no_grad. Preserve gradient,
    parameter.grad, object identity, shape, dtype/device and requires_grad.
    No backward, gradient clearing, model reset, schedule or parameter groups.

    SHAPE AND STATE LEDGER
      first <- beta1*old_first + (1-beta1)*gradient: same parameter shape
      second <- beta2*old_second + (1-beta2)*gradient**2: same shape
      first_hat = first/(1-beta1**step): temporary, same shape
      second_hat = second/(1-beta2**step): temporary, same shape
      denominator = sqrt(second_hat)+eps: same shape, eps OUTSIDE root
      parameter <- parameter*(1-learning_rate*decay): in place
      parameter <- parameter-learning_rate*first_hat/denominator: in place
      return None; keep UNCORRECTED first/second for the next call.

    Scalar slice: p=1,g=2,m=v=0,step1,lr=.1,betas(.9,.99),eps1e-8,
    decay=.1 -> m=.2,v=.04,m_hat=2,v_hat=4,decayed=.99,
    adaptive=.0999999995,new p=.8900000005. Shape (1,) throughout.
    Original full fixture: p=[1,-2], g1=[2,-1],g2=[-.5,3],shape(2,).
    After step1: p~[.8900000005,-1.880000001], m=[.2,-.1],v=[.04,.01].
    Step2 uses denominators .19,.0199, NOT the step1 denominators:
    m=[.13,.21],v=[.0421,.0999],p~[.83405914204,-1.91052981590].
    First moment can retain its sign when the current gradient reverses.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces enter `with torch.no_grad():`. Its body uses
    # EIGHT spaces. `with` limits this mode to its indented region and then
    # restores the previous mode; existing gradients are not erased.
    # STEP 2: Inside, update first in place: mul_ keeps beta1 times its old
    # contents; add_ adds a scaled gradient. Generic syntax is
    # buffer.add_(other,alpha=coefficient). Use coefficient (1-beta1).
    # Scalar first becomes .2. Local rebinding first=... is insufficient.
    # STEP 3: Keep beta2 times old second with mul_, then add the gradient
    # square scaled by (1-beta2). Generic buffer.addcmul_(a,b,value=c)
    # means buffer += c*a*b. Use gradient for BOTH factors, not first.
    # Scalar second becomes .04. No mean/sum across parameter coordinates.
    # STEP 4: Form NEW first_hat/second_hat tensors with ordinary division
    # by 1-beta1**step and 1-beta2**step. ** is power; ^ is not power.
    # Do not div_ either stored buffer. Scalar corrected values are 2,4;
    # stored first/second must remain .2,.04 for the next recurrence.
    # STEP 5: Build denominator from second_hat.sqrt(), THEN add eps.
    # Keep the same shape; scalar value 2.00000001. No eps inside root.
    # STEP 6: Decay parameter in place with mul_ using scalar multiplier
    # 1-learning_rate*decay. Scalar parameter becomes .99. Do this before
    # subtracting the adaptive term; do not add decay into gradient/moments.
    # STEP 7: Subtract the adaptive term in place. Generic
    # buffer.addcdiv_(a,b,value=c) means buffer += c*a/b. Use first_hat,
    # denominator and NEGATIVE learning_rate, or form a change and sub_ it.
    # Scalar parameter becomes .8900000005; negative changes get subtracted.
    # STEP 8: Outside `with`, return to FOUR spaces and return None, or
    # let the function end. Do not return a new parameter/moment tuple.
    # Caller retains updated parameter/first/second; gradient and .grad stay
    # unchanged. Step2 reuses these buffers, not corrected or zeroed copies.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement adamw_update for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====




# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    p=torch.tensor([1.,-2.],dtype=torch.float64,requires_grad=True);r=nn.Parameter(p.detach().clone());m=torch.zeros_like(p);v=torch.zeros_like(p)
    opt=torch.optim.AdamW([r],lr=.1,betas=(.9,.99),eps=1e-8,weight_decay=.1)
    for step,g in enumerate((torch.tensor([2.,-1.],dtype=torch.float64),torch.tensor([-.5,3.],dtype=torch.float64)),1):
        g0=g.clone();assert adamw_update(p,g,m,v,step,.1,.9,.99,1e-8,.1) is None
        r.grad=g.clone();opt.step();close(p,r);close(g,g0)
    close(m,opt.state[r]['exp_avg']);close(v,opt.state[r]['exp_avg_sq'])

if __name__=="__main__":
    run('05.05',run_checks)
