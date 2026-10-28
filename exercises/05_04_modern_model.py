# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.04: Compose the modern baseline.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
A modern decoder can replace absolute positions, LayerNorm and GELU with RoPE, RMSNorm and SwiGLU while retaining the causal residual structure. Compose validated pieces first. Weight tying uses the embedding matrix for the vocabulary projection and reduces parameters, but it is a modeling choice, not a mandatory improvement.

INPUTS AND RETURN
x: float [B,T,D]; constructor width=D, heads=H divides D with even D/H, hidden=F. Return [B,T,D]. Supplied ModernLM stacks this block with token embedding, final RMSNorm and optionally tied head.

WORKED EXAMPLE
D=8,H=2,dh=4,F=16: RMS input [B,T,8], projected heads [B,2,T,4], rotate Q/K at positions0..T-1, causal attention, concatenate [B,T,8], output projection and residual. A second RMS feeds two [B,T,16] branches and restores8. Setting every attention/FFN matrix to0 makes the block exactly identity.

BOUNDARIES
Do not add learned absolute position vectors as well in this chosen baseline. Preserve both residual paths and cache-compatible RoPE convention.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

rmsnorm=load('05_01_rmsnorm.py','rmsnorm')
apply_rope=load('05_02_rotary_positions.py','apply_rope')
swiglu=load('05_03_swiglu.py','swiglu')
causal_attention=load('04_05_causal_mask.py','causal_attention')

class ModernLM(nn.Module):
    def __init__(self,vocab_size,width,heads,hidden,layers,tied=False):
        super().__init__();self.embedding=nn.Embedding(vocab_size,width)
        self.blocks=nn.ModuleList([ModernBlock(width,heads,hidden) for _ in range(layers)])
        self.gain=nn.Parameter(torch.ones(width));self.head=nn.Linear(width,vocab_size,bias=False)
        if tied:self.head.weight=self.embedding.weight
    def forward(self,input_ids):
        x=self.embedding(input_ids)
        for block in self.blocks:x=block(x)
        return self.head(rmsnorm(x,self.gain))

class ModernBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__();self.heads=heads
        self.gain1=nn.Parameter(torch.ones(width));self.gain2=nn.Parameter(torch.ones(width))
        for name in ("q","k","v","o"):
            setattr(self,name,nn.Parameter(torch.randn(width,width)/math.sqrt(width)))
        self.gate=nn.Parameter(torch.randn(width,hidden)/math.sqrt(width))
        self.up=nn.Parameter(torch.randn(width,hidden)/math.sqrt(width))
        self.down=nn.Parameter(torch.randn(hidden,width)/math.sqrt(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compose two pre-norm residual branches without changing (B,T,D).

        x: floating (B,T,D), positive B/T; D divisible by self.heads=H,
        with positive even per-head width d=D//H. Compatible parameter
        dtype/device; this lesson checks float32/float64. hidden=F may differ
        from D. Return one tensor (B,T,D), preserving x and autograd.
        Supplied attributes q/k/v/o are raw (D,D) Parameters, used with @,
        not Linear modules. gain1/gain2 are separate (D,) Parameters.

        SHAPE LEDGER
          Z=rmsnorm(x,gain1):              (B,T,D)
          Z@q, Z@k, Z@v:                  (B,T,D), each independently
          reshape(B,T,H,d), transpose:    (B,H,T,d)
          positions=0..T-1:               (T,), shared across B/H
          apply_rope on q/k only:         (B,H,T,d)
          causal_attention context:       (B,H,T,d)
          causal_attention weights:       (B,H,T,T)
          transpose then merge context:   (B,T,D)
          U=x+merged@o:                    (B,T,D)
          R=rmsnorm(U,gain2):              (B,T,D)
          swiglu(R,gate,up,down):          (B,T,D), internally (B,T,F)
          U+FFN change:                    (B,T,D)

        Numeric fixture in the paired notes: B=1,T=2,D=4,H=2,d=2,F=3;
        X rows [1,0,0,1] and [0,1,1,0], unit gains, q=k=v=I4,o=.5*I4.
        At row0, Z is [c,0,0,c], c=1/sqrt(.500001). Attention sees only
        itself, so U=[1.70710607,0,0,1.70710607]. With the specified FFN
        matrices, its change is [.32177162,0,0,.32177162] and final row0
        is [2.02887770,0,0,2.02887770]. Row1 uses nontrivial prefix weights.
        This differs from the supplied random driver; all weights and both
        rows are traced in the notes. No padding, cache or offsets here.
        """
        # ===== START LEARNER CODE =====
        # All ordinary method statements use EIGHT spaces. No new loop or
        # parameter creation is needed; keep original x for the first bypass.
        # STEP 1: Read B,T,D from x.shape, H from self.heads, d via D//H.
        # // is integer division. Call rmsnorm(x,self.gain1), saving Z.
        # Same shape (B,T,D); numeric row0 becomes [c,0,0,c].
        # STEP 2: Independently project the SAME Z using @ self.q/k/v.
        # Each projected (B,T,D) uses .reshape(B,T,H,d).transpose(1,2).
        # Save local q,k,v of shape (B,H,T,d). self.q is a raw Parameter,
        # not a callable Linear module; do not transpose the weight here.
        # STEP 3: torch.arange(T,device=x.device) gives positions (T,).
        # Call apply_rope separately for q and k with these positions.
        # Keep v unrotated; base defaults match and positions start at0.
        # STEP 4: Call causal_attention(q,k,v) and UNPACK its two returns:
        # context (B,H,T,d), weights (B,H,T,T). Unused weights can be named
        # _, an ordinary name indicating deliberate non-use. The helper
        # already scales, masks and softmaxes; do not repeat those operations.
        # STEP 5: context.transpose(1,2) is (B,T,H,d). Then reshape(B,T,D)
        # merges heads in order. Use @ self.o to make the attention change.
        # If using view after transpose, make contiguous first instead.
        # STEP 6: Save U as original x plus that change. Do not add Z.
        # U keeps (B,T,D); numeric row0 [1.70710607,0,0,1.70710607].
        # STEP 7: Normalize U using self.gain2, saving R, then call swiglu
        # with R,self.gate,self.up,self.down. Its return is an FFN CHANGE
        # (B,T,D); numeric row0 [.32177162,0,0,.32177162], not yet final Y.
        # STEP 8: At eight spaces return U plus the FFN change. Add U,
        # not old x or normalized R. Keep all rows and the gradient graph;
        # no detach, input overwrite, optimizer update, token IDs or softmax.
        # TODO: implement the contract using the worked example and numbered hints.
        raise NotImplementedError("Implement ModernBlock.forward for this exercise")
        # ===== END LEARNER CODE =====
        # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(14);b=ModernBlock(8,2,16);x=torch.randn(2,3,8,requires_grad=True)
    y=b(x);assert y.shape==x.shape;z=x.detach().clone();z[:,2]+=20;close(b(z)[:,:2],y[:,:2])
    y.sum().backward();assert all(p.grad is not None for p in b.parameters())
    with torch.no_grad():
        for name in ('q','k','v','o','gate','up','down'):getattr(b,name).zero_()
    close(b(x),x)
    m=ModernLM(7,8,2,16,2,tied=True);assert m.head.weight is m.embedding.weight
    assert m(torch.tensor([[1,2,3]])).shape==(1,3,7)

if __name__=="__main__":
    run('05.04',run_checks)
