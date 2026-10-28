# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.03: Gated feed-forward networks.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
The gate controls a separate content branch coordinate by coordinate. A smooth gate allows small negative values as well as positive values; it is not a hard boolean switch. Three matrices use about3DF parameters; a conventional two-matrix FFN uses2DF, so compare similar budgets when claiming improvements.

INPUTS AND RETURN
x: float [...,D]; gate,up: [D,F]; down: [F,D]. Return [...,D], bias-free. F is selected from the parameter budget, not automatically4D.

WORKED EXAMPLE
D=F=1, x=1, gate=2, up=3, down=4. Gate preactivation=2; sigmoid2≈.880797; SiLU2≈1.761594. Content=3; gated content≈5.284782; output≈21.13913. If gate matrix=0, SiLU0=0 and output0 even with a nonzero content branch.

BOUNDARIES
Using @ between gate/content is not elementwise gating. F widths must match in both branches.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Return a bias-free gated FFN output with the same leading axes as x.

    x: floating (...,D); gate/up: (D,F); down: (F,D), positive D,F.
    Inputs share dtype/device on a backend supporting these operations;
    the checks use float32/float64. Return (...,D), preserving input values
    and gradients for x and all three supplied weights. Create no parameter,
    optimizer, residual connection or normalization inside this function.

    VARIABLES AND SHAPES
      gate_pre = x projected by gate: (...,F)
      content = x projected by up:    (...,F)
      gate_values = SiLU(gate_pre):   (...,F), a*sigmoid(a)
      gated = gate_values*content:    (...,F), elementwise product
      output = gated projected down: (...,D)
    Shape symbol F denotes intermediate width; F.silu uses the imported
    torch.nn.functional namespace. Do not assign an integer to Python F.
    @ projects/reduces a feature axis; * multiplies matching coordinates.

    ORIGINAL SCALAR FIXTURE, ALL TENSORS SHAPE (1,1)
    x=1, gate=2, up=3, down=4 -> gate_pre=2, content=3.
    sigmoid(2)=0.88079708; SiLU(2)=1.76159416;
    gated=5.28478247; output=21.13912987. SiLU is not sigmoid alone.
    Separate D=2,F=3 example and full matrices are in the paired notes.
    Gate values can exceed1 or be negative, so they are not probabilities.
    With gate matrix zero, output is zero, but gate gradients may be nonzero.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, use matrix multiplication @ to project x
    # through gate; save gate_pre. (...,D)@(D,F)->(...,F).
    # Scalar fixture gate_pre=[[2.]]; use the supplied tensor, not a new one.
    # STEP 2: At four spaces, separately project the SAME original x
    # through up; save content. Same shape; scalar fixture [[3.]].
    # Do not project gate_pre or the activated branch through up.
    # STEP 3: Apply the existing F.silu function to gate_pre, saving
    # gate_values. It computes a*sigmoid(a), not just sigmoid(a).
    # Shape stays (...,F); fixture [[1.76159416]]. F is the imported module.
    # STEP 4: Multiply gate_values and content using *, saving gated.
    # This keeps (...,F); fixture [[5.28478247]]. @ here would mix axes.
    # STEP 5: Use @ down to map (...,F)@(F,D)->(...,D), then return
    # that tensor, still at four spaces. Fixture [[21.13912987]].
    # All five steps are in the function body, without for/if/no_grad,
    # detach/item, in-place changes, a softmax or an added residual.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement swiglu for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.ones(1,1,requires_grad=True)
    y=swiglu(x,torch.tensor([[2.]]),torch.tensor([[3.]]),torch.tensor([[4.]]));close(y,torch.tensor([[24/(1+math.exp(-2))]]))
    y.sum().backward();assert x.grad.item()>0
    z=torch.randn(2,3,4);close(swiglu(z,torch.zeros(4,7),torch.randn(4,7),torch.randn(7,4)),torch.zeros_like(z))

if __name__=="__main__":
    run('05.03',run_checks)
