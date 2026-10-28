# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.08: Residuals, normalization and feed-forward layers.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Normalization controls feature scale per token. The feed-forward layer changes features independently at each position; attention is the component that exchanges information between positions. A residual adds an identity path, so gradients have a direct route through deep blocks. Pre-norm normalizes the branch input rather than the final sum.

INPUTS AND RETURN
x: float [B,T,D]; norm: supplied LayerNorm(D); up: Linear(D,F); down: Linear(F,D). Return x+down(gelu(up(norm(x)))) [B,T,D]. GELU uses the PyTorch default approximate="none".

WORKED EXAMPLE
x=[1,3], mean=2, population variance=1. With eps=0 and unit gain/zero bias, LN(x)=[-1,1]. If up/down weights and biases are zero, branch=0 and output=x exactly. Its Jacobian with respect to x is identity. F can differ from D, but down must restore D before addition.

BOUNDARIES
torch.var defaults may use sample variance; LayerNorm uses population variance. Normalizing over time mixes sequence positions.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def pre_norm_ffn(x: torch.Tensor, norm: nn.Module, up: nn.Module, down: nn.Module) -> torch.Tensor:
    """Return a pre-norm FFN residual, shape (B,T,D).

    x: floating (B,T,D), existing hidden features.
    norm: supplied nn.LayerNorm(D); statistics only over each row's D features.
    up: supplied nn.Linear(D,Dff); down: supplied nn.Linear(Dff,D).
    Dff is a width; F in F.gelu is the imported functional module, not a width.
    All objects use compatible dtypes/devices. Reuse modules; do not initialize
    or update their parameters. Preserve x, parameter values and all gradients.

    SHAPE LEDGER
      normalized = norm applied to x:       (B,T,D)
      expanded = up applied to normalized: (B,T,Dff)
      activated = elementwise GELU:        (B,T,Dff)
      branch = down applied to activated:  (B,T,D)
      original x plus branch, returned:    (B,T,D)
    nn.Linear stores weight as (out,in) and applies @ weight.T + bias itself.
    Keep norm inside the branch. Do not normalize the final residual sum.
    Primary fixture: B=1,T=1,D=2,Dff=5, x=[[[1,3]]], eps=0,
    gamma=1,beta=0; normalized=[[[-1,1]]]. All up/down weights and biases
    are zero, so expanded/activated have five zeros, branch=[[[0,0]]],
    output=[[[1,3]]]. Its input gradient under output.sum() is [[[1,1]]].
    """
    # ===== START LEARNER CODE =====
    # Keep each statement inside this function with four-space indentation.
    # STEP 1: Call the supplied norm on x, saving normalized. This is a module
    # call, not construction. Shape (B,T,D); fixture [[[-1,1]]]. Keep x unchanged.
    # STEP 2: Call up on normalized, saving expanded. The module handles its
    # transpose/bias; shape (B,T,Dff), fixture (1,1,5) with five zero features.
    # STEP 3: Call F.gelu on expanded, saving activated. Use its default
    # approximate="none"; shape unchanged and GELU(0)=0. No softmax here.
    # STEP 4: Call down on activated, saving branch. Shape returns to (B,T,D),
    # fixture [[[0,0]]]. Restoring D makes residual addition possible.
    # STEP 5: Add branch to the ORIGINAL x using ordinary +, and return one
    # new tensor at four-space indentation. Fixture [[[1,3]]], shape (1,1,2).
    # Do not add to normalized, use +=, detach, no-grad, or create new modules.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement pre_norm_ffn for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.tensor([[[1.,3.]]],requires_grad=True);n=nn.LayerNorm(2,eps=0.);u=nn.Linear(2,5);d=nn.Linear(5,2)
    close(n(x),torch.tensor([[[-1.,1.]]]))
    with torch.no_grad():
        u.weight.zero_();u.bias.zero_();d.weight.zero_();d.bias.zero_()
    y=pre_norm_ffn(x,n,u,d);close(y,x);y.sum().backward();close(x.grad,torch.ones_like(x))
    torch.manual_seed(8);n=nn.LayerNorm(3);u=nn.Linear(3,7);d=nn.Linear(7,3);z=torch.randn(2,4,3)
    close(pre_norm_ffn(z,n,u,d),z+d(F.gelu(u(n(z)))))

if __name__=="__main__":
    run('04.08',run_checks)
