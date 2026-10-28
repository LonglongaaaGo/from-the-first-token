# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.01: Why context matters.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
The Chapter 03 model reads only E[x_t] at each position, even when T > 1.
Compute h_t = (E[x_0] + ... + E[x_t]) / (t + 1), including the current input.
Earlier positions now influence h_t; future positions must not influence it.
This fixed average adds no parameters. Later attention learns content-dependent
aggregation, combined with position representations for order information.

INPUTS AND RETURN
embedding_table: floating [V,D], V token rows with D coordinates per row.
input_ids: torch.long [B,T], B samples with T positions each.
Return floating [B,T,D], one visible-prefix mean per sample and position.
Assume T > 0, valid IDs, and compatible devices. Preserve input values and the
autograd path to embedding_table; keep its floating dtype and device.

WORKED EXAMPLE
E=[[0,0],[2,0],[0,4]], X=[[0,1],[2,1]]. Lookup gives [[0,0],[2,0]] versus [[0,4],[2,0]]. At t=0 outputs are [0,0] and [0,4]; at t=1 means are [1,0] and [1,2]. Both end in ID1, yet their contextual states differ. Changing the second token must leave the first output unchanged.

BOUNDARIES
A mean of the whole sequence leaks future tokens. A mean across dim=-1 destroys feature width.
Axes for [B,T,D]: dim=0 samples, dim=1 positions, dim=2 feature coordinates.
Here B=T=D=2 only by choice; the counts [1,2] must align with positions.
Reshape counts to [1,T,1]. Raw [T] divisors would align with the feature axis.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def prefix_mean(embedding_table: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # Use four spaces per statement. This tensor route needs no loop or branch.
    # STEP 1: Name the lookup result embedded. Index with input_ids, not a mean
    # of the IDs. Expected shape: [B,T,D]; sample 1 is [[0,4],[2,0]].
    # STEP 2: Build prefix_sums using .cumsum(dim=1) on those vectors.
    # Keep [B,T,D]; sample 1 becomes [[0,4],[2,4]].
    # STEP 3: Read T from input_ids.shape[1]. Build counts with torch.arange:
    # start at 1, stop before T+1, and match embedded.device / embedded.dtype.
    # Expected counts here: floating [1,2], with shape [T].
    # STEP 4: Reshape counts to [1,T,1], optionally naming it divisors.
    # A shared batch axis and feature axis let each position use one divisor.
    # STEP 5: Divide prefix_sums by those divisors and return [B,T,D].
    # Keep the graph connected; do not detach or recreate result tensors.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement prefix_mean for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    E=torch.tensor([[0.,0.],[2.,0.],[0.,4.]],requires_grad=True)
    X=torch.tensor([[0,1],[2,1]])
    h=prefix_mean(E,X); close(h,torch.tensor([[[0.,0.],[1.,0.]],[[0.,4.],[1.,2.]]]))
    h.sum().backward(); assert E.grad is not None
    Z=X.clone();Z[:,1]=0;close(prefix_mean(E,Z)[:,0],h[:,0])
    close(prefix_mean(E,torch.tensor([[1]])),E[1].view(1,1,2))

if __name__=="__main__":
    run('04.01',run_checks)
