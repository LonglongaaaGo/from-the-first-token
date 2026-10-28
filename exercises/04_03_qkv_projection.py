# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.03: Query, key and value projections.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
A query describes what a position seeks, a key describes how it can be matched, and a value is the content retrieved. These are learned linear views of the same hidden states; they are not predefined linguistic labels. Separate projections let matching and transmitted content differ.

INPUTS AND RETURN
x: float [B,T,D]; wq,wk: float [D,d]; wv: float [D,dv]. Return Q,K [B,T,d] and Vvalues [B,T,dv]. Same dtype/device; preserve autograd and inputs. Vvalues is a tensor, V in earlier lessons denotes vocabulary size.

WORKED EXAMPLE
X=[[[1,2],[3,4]]], Wq=[[1],[0]], Wk=[[0],[1]], Wv=identity. Q=[[[1],[3]]], K=[[[2],[4]]], values=X. QK^T before scaling is [[2,4],[6,12]]. Row i belongs to query i and column j to key j; projecting alone has not mixed positions.

BOUNDARIES
Do not transpose x or mix time with width. Equal dimensions in one example do not mean the axes are interchangeable.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def project_qkv(x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ===== START LEARNER CODE =====
    # Use four-space indentation for every statement in this function.
    # STEP 1: Compute the query projection with x @ wq; name it q if useful.
    # [B,T,D] @ [D,d] gives [B,T,d]; the fixture gives [[[1],[3]]].
    # @ contracts feature coordinates, preserving B and T.
    # STEP 2: Use the SAME x with wk and with wv in two independent products.
    # k is [[[2],[4]]] here; values is [[[1,2],[3,4]]].
    # Q/K have width d; values may have a different width dv.
    # These are right-multiply matrices, not nn.Linear's stored weight layout.
    # STEP 3: Return the three tensors as a tuple in query, key, value order.
    # The caller uses q,k,v = project_qkv(...). Keep the graph connected.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement project_qkv for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
    q,k,v=project_qkv(x,torch.tensor([[1.],[0.]]),torch.tensor([[0.],[1.]]),torch.eye(2))
    close(q,torch.tensor([[[1.],[3.]]]));close(k,torch.tensor([[[2.],[4.]]]));close(v,x)
    (q.sum()+k.sum()+v.sum()).backward();close(x.grad,torch.full_like(x,2.))
    q,k,v=project_qkv(torch.randn(2,3,5),torch.randn(5,2),torch.randn(5,2),torch.randn(5,4))
    assert q.shape==(2,3,2) and v.shape==(2,3,4)

if __name__=="__main__":
    run('04.03',run_checks)
