# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.06: Multiple attention heads.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Each head sees a different projected feature slice and computes its own key distribution. Heads are not repeated batches or extra sequence positions. Concatenating heads restores total width D, then output projection lets their information interact.

INPUTS AND RETURN
x: float [B,T,D]; wq,wk,wv,wo: float [D,D]; heads: positive integer H
dividing D. All floating inputs use compatible dtypes/devices. Return one tensor
[B,T,D], not a tuple. Reuse causal_attention from 04.05; preserve inputs and graph.

WORKED EXAMPLE
B=1,T=3,D=4,H=2,dh=2; all four weights are I4.
x=[[[1,0,0,1],[0,1,0,1],[1,1,1,0]]]. Reshape produces (1,3,2,2),
then transpose(1,2) produces (1,2,3,2). Head 0 reads rows [1,0],[0,1],[1,1];
head 1 reads [0,1],[0,1],[1,0]. Merged output is approximately
[[[1,0,0,1],[0.330238,0.669762,0,1],[0.751745,0.751745,0.503490,0.496510]]].
The first row is unchanged; H=1 must match the single-head causal result.

BOUNDARIES
Using sqrt(D) instead of sqrt(dh) changes attention temperature. Reshaping without axis permutation mixes tokens with heads.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

causal_attention=load('04_05_causal_mask.py','causal_attention')



def multihead(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor, heads: int) -> torch.Tensor:
    """Project, split, attend independently per head, concatenate, and project.

    INPUT CONTRACT
      x: (B,T,D), rows are positions, final coordinates have total width D.
      wq,wk,wv,wo: (D,D), independent right-multiplication weights.
      heads: integer H>0 dividing D; head_width=dh=D//H.
      Here each head's Q/K/V widths are all dh; this is this exercise's design.

    SHAPE LEDGER -- read this before choosing any reshape or transpose
      x @ wq, x @ wk, x @ wv:              (B,T,D)
      reshape each with (B,T,H,dh):         (B,T,H,dh)
      transpose axes 1 and 2 (T and H):     (B,H,T,dh)
      causal_attention receives these Q/K/V tensors, with leading axes B,H.
      Inside that helper K becomes (B,H,dh,T); scores/weights are (B,H,T,T).
      Helper output and its returned weights: (B,H,T,dh), (B,H,T,T).
      Transpose helper output axes 1 and 2: (B,T,H,dh)
      Reshape to concatenate H*dh=D:        (B,T,D)
      Right-multiply by wo=(D,D):           (B,T,D), the single return tensor.

    T=3,H=2 example: (1,3,4)->(1,3,2,2)->(1,2,3,2), then reverse to merge.
    transpose(1,2) groups positions by head; it is different from the helper's
    K.transpose(-2,-1), which turns key vectors into matrix-product columns.
    causal_attention already scales by sqrt(dh) and masks; do not repeat them.
    """
    # ===== START LEARNER CODE =====
    # Keep each statement inside this function with four-space indentation.
    # STEP 1: Unpack x.shape into batch,length,width. Read head_width using
    # width // heads (integer division). Fixture: 1,3,4 and head_width=2.
    # STEP 2: Project x independently with @ and wq,wk,wv. Suggested names:
    # q_flat,k_flat,v_flat; each has shape (batch,length,width)=(1,3,4).
    # STEP 3: Reshape each to (batch,length,heads,head_width), retaining the
    # per-position order. Suggested q_split,k_split,v_split: (1,3,2,2).
    # STEP 4: Use .transpose(1,2) on each split tensor, giving q_heads,
    # k_heads,v_heads of shape (batch,heads,length,head_width)=(1,2,3,2).
    # Each head keeps all T positions; this does not transpose K for scoring.
    # STEP 5: Call causal_attention with those three head tensors.
    # Unpack its two returns: head_output (B,H,T,dh), head_weights (B,H,T,T).
    # You may ignore head_weights; do not normalize, scale or mask again.
    # STEP 6: Transpose head_output axes 1,2 into by_position: (B,T,H,dh).
    # STEP 7: Reshape by_position to (batch,length,width), naming it merged.
    # This concatenates head coordinates per position. reshape can copy when
    # necessary after a transpose; do not assume view has the same behavior.
    # STEP 8: Multiply merged by wo and return that single (B,T,D) tensor.
    # Preserve all inputs, weight objects and autograd connections.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement multihead for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(6);x=torch.randn(2,3,4,requires_grad=True);I=torch.eye(4)
    y=multihead(x,I,I,I,I,2);assert y.shape==x.shape;close(y[:,0],x[:,0])
    close(multihead(x,I,I,I,I,1),causal_attention(x,x,x)[0]);y.sum().backward();assert x.grad is not None
    z=x.detach().clone();z[:,2]+=50;close(multihead(z,I,I,I,I,2)[:,:2],y[:,:2])

if __name__=="__main__":
    run('04.06',run_checks)
