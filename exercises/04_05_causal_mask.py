# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.05: Causal visibility and isolation.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Training predicts many next tokens in parallel because the mask gives each position only its observed prefix. Teacher-forced future targets must never enter earlier hidden states. Mask scores before softmax: exp(-infinity)=0, so allowed entries renormalize to sum one.

INPUTS AND RETURN
q,k: float [...,T,d]; values: [...,T,dv], equal positive T and aligned sequence starts.
Matching leading axes may be B or B,H; widths d and dv may differ. Use finite
inputs with compatible floating dtypes/devices. Return output [...,T,dv] and
weights [...,T,T], with future-key weights exactly zero. Preserve inputs and graph.

WORKED EXAMPLE
With zero q,k and values [2,4,8], unmasked output at every row is 14/3. Causal weights are [[1,0,0],[1/2,1/2,0],[1/3,1/3,1/3]], giving outputs [2,3,14/3]. Changing value8 to800 affects only row2, while rows0 and1 stay2 and3. The diagonal stays visible because x_t predicts x_(t+1).

BOUNDARIES
True means BLOCKED for the mask passed to masked_fill, not allowed.
Multiplying future scores by zero leaves exp(0)=1 and leaks information.
Masking an entire row yields NaNs; retain the diagonal. This lesson assumes a
square, aligned self-attention layout, without padding or cached offsets.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def causal_attention(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal output (...,T,dv) and weights (...,T,T), in that order.

    q and k: (...,T,d); values: (...,T,dv). Each final pair means
    (position rows, coordinates per row); leading axes match, e.g. B or B,H.
    Transpose only K's last two axes: (...,T,d) -> (...,d,T), so
    (T,d) @ (d,T) -> (T,T). Mask future j>i before key-axis softmax.
    The boolean (T,T) mask broadcasts over leading axes; True means blocked.
    Then (T,T) @ (T,dv) -> (T,dv). Equal positive T, aligned starts,
    finite compatible floating inputs; preserve inputs and autograd paths.
    Fixture: q,k=(1,3,2); values=(1,3,1); output=(1,3,1).
    """
    # ===== START LEARNER CODE =====
    # Each statement belongs inside this function, with four-space indentation.
    # STEP 1: Compute scaled scores as in 04.04, using q, k.transpose(-2,-1),
    # @, and math.sqrt(q.shape[-1]). Save scores, shape (...,T,T); fixture: zeros.
    # Put q on the LEFT: (T,d) @ (d,T) -> (T,T). Reversing the operands
    # gives (d,T) @ (T,d) -> (d,d), a legal product with the wrong meaning.
    # Fixture: q=(1,3,2), transposed k=(1,2,3), scores must be (1,3,3).
    # STEP 2: Read the position count from q.shape[-2], naming it length.
    # The final axis is d; the next-to-last is T. Fixture: length is 3.
    # STEP 3: Create a (length,length) table with torch.ones, dtype=torch.bool,
    # device=q.device; apply .triu(1), saving future_mask. True means BLOCKED.
    # Keep the mask two-dimensional; leading batch/head axes broadcast.
    # A mask mismatch can come from an earlier wrong scores product.
    # triu(1) keeps entries strictly above the main diagonal:
    # [[False,True,True],[False,False,True],[False,False,False]].
    # STEP 4: Use scores.masked_fill(future_mask, float("-inf")), saving
    # scores_masked. Shape stays (...,T,T); do not use an in-place operation.
    # The (T,T) mask broadcasts over every leading batch/head axis.
    # STEP 5: Apply torch.softmax(..., dim=-1) to scores_masked, saving weights.
    # Fixture rows: [1,0,0], [0.5,0.5,0], [1/3,1/3,1/3].
    # STEP 6: Compute output with weights @ values, shape (...,T,dv).
    # Return output, weights in that order, keeping both statements at four spaces.
    # Fixture output: [[[2],[3],[14/3]]]. Keep inputs and autograd graph intact.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement causal_attention for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    q=torch.zeros(1,3,2);v=torch.tensor([[[2.],[4.],[8.]]])
    o,a=causal_attention(q,q,v);close(o,torch.tensor([[[2.],[3.],[14/3]]]))
    assert torch.equal(a.triu(1),torch.zeros_like(a));close(a.sum(-1),torch.ones(1,3))
    v2=v.clone();v2[:,2]=800;close(causal_attention(q,q,v2)[0][:,:2],o[:,:2])
    torch.manual_seed(5);q=torch.randn(2,4,3,requires_grad=True);k=torch.randn_like(q);v=torch.randn(2,4,5,requires_grad=True)
    o,_=causal_attention(q,k,v);o[:,0].sum().backward();assert torch.count_nonzero(v.grad[:,1:])==0

if __name__=="__main__":
    run('04.05',run_checks)
