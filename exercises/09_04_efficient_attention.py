# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 09.04: Efficient attention with equivalence checks.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Memory-efficient attention avoids materializing the full score/weight matrices in some kernels by tiling and online softmax. The mathematical objective stays the same. PyTorch chooses implementations based on input/device; the CPU check validates semantics, not a speedup. Compare outputs and gradients before profiling actual accelerated kernels.

INPUTS AND RETURN
q,k,values: float[B,H,T,d] for causal default; explicit allowed bool mask may support rectangular keys, True means visible. Return attention output, dropout0. Use PyTorch scaled_dot_product_attention when available, matching default scale.

WORKED EXAMPLE
T=3 and zero scores with values2,4,8 must still output2,3,14/3. In SDPA boolean masks True means allowed; in04.05 the mask True meant blocked. For cached P=2,S=1, explicit allowed shape[1,3] is allTrue. Setting is_causal=True instead may use an unsuitable upper-left alignment.

BOUNDARIES
Kernel numerical results can differ slightly; compare dtype-appropriate tolerances. Explicit masks and dropout semantics must be preserved in both implementations.
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



def efficient_attention(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # STEP 1: If allowed is None, call F.scaled_dot_product_attention with dropout_p=0.0,is_causal=True.
    # STEP 2: Otherwise pass attn_mask=allowed and is_causal=False, retaining dropout0.
    # STEP 3: Return the tensor without requesting attention weights, which optimized kernels may not expose.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement efficient_attention for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(44);q=torch.randn(1,2,3,4,dtype=torch.float64,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
    a=efficient_attention(q,k,v);b=causal_attention(q,k,v)[0];close(a,b)
    ga=torch.autograd.grad(a.square().sum(),(q,k,v),retain_graph=True);gb=torch.autograd.grad(b.square().sum(),(q,k,v))
    for u,w in zip(ga,gb):close(u,w)
    qr=torch.zeros(1,1,1,2);kr=torch.zeros(1,1,3,2);vr=torch.tensor([[[[2.],[4.],[8.]]]])
    o=efficient_attention(qr,kr,vr,torch.ones(1,3,dtype=torch.bool));close(o,torch.tensor([[[[14/3]]]]))

if __name__=="__main__":
    run('09.04',run_checks)
