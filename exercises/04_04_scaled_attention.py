# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.04: Scaled dot-product attention.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Dot products score compatibility. Dividing by sqrt(d) controls score variance when coordinates have similar scale; it is not a probability normalization. Softmax across key positions produces a convex combination of value vectors for each query.

INPUTS AND RETURN
q: float [...,Tq,d]; k: [...,Tk,d]; values: [...,Tk,dv]. Return output [...,Tq,dv] and weights [...,Tq,Tk]. Leading batch axes match; no mask in this lesson.

WORKED EXAMPLE
q=[[[1,0]]], k=[[[0,0],[sqrt(2)*ln(3),0]]], values=[[[2],[6]]]. d=2, scores=[0,ln(3)], exp scores=[1,3], weights=[.25,.75], output=.25*2+.75*6=5. A row sums to one; output width is dv=1, not Tk=2.

BOUNDARIES
Normalizing over queries assigns the wrong meaning to each row. Uniform weights are correct when every score in a row is equal.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def attention(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unmasked attention; the caller supplies this axis layout.

    INPUT CONTRACT (torch.Tensor alone does not specify dimensions)
      q:      (..., Tq, d)   Tq query rows, d coordinates per row.
      k:      (..., Tk, d)   Tk key rows, d coordinates per row.
      values: (..., Tk, dv)  Tk value rows, dv coordinates per row.
    The ... axes match; think B first, or B,H when heads are present.

    WHY TRANSPOSE K?
      A matrix product (m,n) @ (n,p) sums over the matching n axes.
      q @ k would try (Tq,d) @ (Tk,d): key vectors are still rows.
      Turn each key row into a column: k.transpose(-2,-1) -> (...,d,Tk).
      Then (Tq,d) @ (d,Tk) -> (Tq,Tk): one score per query/key pair.
      Example: (3,4) @ (5,4) cannot multiply; transpose K to (4,5).
      When Tk=d=2, sizes happen to match without transposing, but the
      row/column meanings and computed entries are still wrong.

    INTERMEDIATE AND RETURN SHAPES
      k_transposed: (...,d,Tk); scores and weights: (...,Tq,Tk).
      output: (...,Tq,dv), from (Tq,Tk) @ (Tk,dv).
      Return (output, weights), preserving inputs and the autograd graph.
      Original fixture: q=(1,1,2), k=(1,2,2), values=(1,2,1),
      output=(1,1,1), weights=(1,1,2). No causal mask in this function.
    """
    # ===== START LEARNER CODE =====
    # Write each statement at this four-space indentation, inside this function.
    # STEP 1: Assign k.transpose(-2, -1) to k_transposed.
    # The final axes change from (Tk,d) to (d,Tk); keep all leading axes.
    # STEP 2: Assign the scaled dot products to scores: use @, then divide
    # by math.sqrt(q.shape[-1]). The matching d axes are summed, producing
    # shape (...,Tq,Tk). In the example the contents are [[[0,ln(3)]]].
    # STEP 3: Assign torch.softmax(scores, dim=-1) to weights.
    # Normalize over Tk, keeping shape (...,Tq,Tk): [[[0.25,0.75]]].
    # STEP 4: Assign weights @ values to output. Contract the Tk axes;
    # shape (...,Tq,dv), with contents [[[5]]] in the worked example.
    # STEP 5: Return output, weights in that order, at the same indentation.
    # Keep the autograd graph: do not detach, rebuild tensors, or change inputs.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement attention for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    q=torch.tensor([[[1.,0.]]],requires_grad=True);k=torch.tensor([[[0.,0.],[math.sqrt(2)*math.log(3),0.]]]);v=torch.tensor([[[2.],[6.]]])
    o,a=attention(q,k,v);close(o,torch.tensor([[[5.]]]));close(a,torch.tensor([[[.25,.75]]]))
    o.sum().backward();assert q.grad is not None and torch.isfinite(q.grad).all()
    o,a=attention(torch.zeros(2,3,4),torch.zeros(2,5,4),torch.ones(2,5,7))
    assert o.shape==(2,3,7);close(a.sum(-1),torch.ones(2,3));close(o,torch.ones_like(o))

if __name__=="__main__":
    run('04.04',run_checks)
