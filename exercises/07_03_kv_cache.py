# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 07.03: KV cache and position alignment.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND CONNECTION
07.02 scores fixed targets. 07.03 makes repeated generation forwards cheaper by
retaining this layer's earlier keys/values. A causal layer's past rows cannot
depend on future tokens, so an unchanged prefix and fixed eval-mode weights
produce the same past states. New queries still attend to old keys/values.
This helper receives already projected Q/K/V, not token IDs or a model.
Each layer and each request owns a separate cache. 07.04 supplies the loop.
No new environment, data download, model fitting or BPB calculation is needed.

INPUTS AND RETURN
q,k: floating [B,H,S,d], positive even d; raw new projected queries/keys.
values: floating [B,H,S,dv], new values; dv can differ from d.
B=batch rows,H=heads,S=NEW positions,d=matching width,dv=value width.
cache: None on the first call, else exactly 'keys'/'values' tensors with
shapes[B,H,P,d]/[B,H,P,dv]. Cached keys ALREADY include RoPE; values do not.
All batch/head/feature sizes, dtypes and devices are compatible. S is positive.
Here S counts new positions; it is unrelated to 07.01's summed-NLL symbol S.
Return output[B,H,S,dv] and a NEW cache dict with shapes[B,H,P+S,d]/
[B,H,P+S,dv]. Keep input/old-cache tensors unchanged. Inference-only results
should not retain autograd history; clone values inside no_grad on first use.
The supplied05.02 apply_rope accepts(x,positions,base=10000), with positions
an integer vector[S] on x.device. Its existing half-dtype follow-up is separate;
the original07.03 driver uses float32 and the worked reference also uses float64.

FORMULAS AND SHAPES
P=0 without a cache, else cache['keys'].shape[-2]. New positions=P..P+S-1.
Qrot=RoPE(q,positions); Knew=RoPE(k,positions). Do not rotate cached keys again.
Concatenate only on dim=-2 (time): Kall[P+S,d], Vall[P+S,dv] per batch/head.
R=Qrot@Kall.transpose(-2,-1)/sqrt(d), shape[B,H,S,P+S].
Query i has absolute position P+i. Key j is future iff j>P+i.
key_positions[None,:] has shape[1,P+S]; positions[:,None] has[S,1].
Their comparison broadcasts to[S,P+S], then across B/H for masked_fill.
future_mask True means BLOCK, unlike07.02's score-mask True meaning COUNT.
Replace future scores with -inf, softmax over the final key axis, multiply
Vall: [B,H,S,P+S]@[B,H,P+S,dv] -> output[B,H,S,dv].

WORKED EXAMPLE
B=H=1,P=S=d=2,dv=1. Past/new q/k are zero. Past values[2,4], new[8,10].
New positions[2,3]. Future mask[[False,False,False,True],[False]*4].
All raw scores0; masked rows[0,0,0,-inf] and[0,0,0,0].
Weights[[1/3,1/3,1/3,0],[1/4,1/4,1/4,1/4]]. New outputs[14/3,6],
shape(1,1,2,1). Full-prefix outputs[2,3,14/3,6] match prefill2 then append2.
The new cache has4 positions; the passed old cache remains length2.
Separate nonzero RoPE illustration, d2: raw[1,0] at positions2/3 becomes
[-.416147,.909297]/[-.989992,.141120]. Resetting new positions to0/1 is wrong.

BOUNDARIES
KV cache is an inference optimization, not a longer context guarantee. Cache memory grows with layers, sequence length and head dimensions.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

apply_rope=load('05_02_rotary_positions.py','apply_rope')



def cached_attention(q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, cache: dict | None) -> tuple[torch.Tensor, dict]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces open `with torch.no_grad():`; its body uses EIGHT.
    #         q/k/values are tensors, so this helper has no model.eval() call.
    # STEP 2: At EIGHT read S=q.shape[-2], d=q.shape[-1]. Determine P with
    #         `if cache is None:` (EIGHT), then each branch body uses TWELVE.
    #         None means no cache yet: P=0. Otherwise read the old key time axis.
    # STEP 3: Back at EIGHT, torch.arange from P to P+S (exclusive), on q.device,
    #         yields integer positions[S]. Main append example: [2,3].
    # STEP 4: Call supplied apply_rope separately on q and k with these positions.
    #         Each remains[B,H,S,d]. Do not rotate values or old cached keys.
    # STEP 5: At EIGHT branch again on cache is None; branch bodies TWELVE.
    #         First call: new rotated keys become all_keys; clone values inside
    #         no_grad for all_values. clone makes a fresh tensor without a graph.
    #         Later call: torch.cat accepts a tuple/list of OLD then NEW tensors.
    #         Use dim=-2 for both keys and values. Toy shapes: keys(1,1,4,2),
    #         values(1,1,4,1). Do not assign into the passed cache dictionary.
    # STEP 6: Back at EIGHT, transpose all_keys' LAST TWO axes. Multiply rotated
    #         q by that transpose and divide by math.sqrt(d). The contracted
    #         dimension is d, not S/dv. Scores toy shape(1,1,2,4), all zeros.
    # STEP 7: Make key positions0..P+S-1 on q.device. Compare key_positions[None,:]
    #         > positions[:,None]. Shapes(1,4)>(2,1) -> future_mask(2,4).
    #         Toy row0 blocks only key3; row1 blocks nothing. True means BLOCK.
    # STEP 8: Use masked_fill(future_mask,float('-inf')), then softmax(dim=-1).
    #         Do not replace blocked scores with0: exp(0)=1 still gets weight.
    #         Toy weights rows[1/3,1/3,1/3,0] and[1/4,1/4,1/4,1/4].
    # STEP 9: Multiply weights by all_values: [B,H,S,P+S]@[B,H,P+S,dv].
    #         Toy output(1,1,2,1) contains14/3 then6. Keep all B/H axes.
    # STEP 10: Return output and a NEW {'keys':all_keys,'values':all_values} dict,
    #          at EIGHT inside with (or FOUR after the block). Context exit
    #          restores the caller's grad mode. Do not cache Q or output O.
    # STEP 11: Check first-call P0, append P2 and one-new-token S1 reasoning.
    #          Original randomized driver has B2,H3,total length4,d6,dv5;
    #          its two append calls each have S2. Values width need not equal d.
    # STEP 12: Replace only the placeholder raise; preserve signature/imports,
    #          markers and original driver. No loop/optimizer/backward is needed.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement cached_attention for this exercise")
    # ===== END LEARNER CODE =====

    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(33);q=torch.randn(2,3,4,6);k=torch.randn_like(q);v=torch.randn(2,3,4,5)
    full,c=cached_attention(q,k,v,None);a,old=cached_attention(q[:,:,:2],k[:,:,:2],v[:,:,:2],None)
    saved=old['keys'].clone();b,new=cached_attention(q[:,:,2:],k[:,:,2:],v[:,:,2:],old)
    close(torch.cat((a,b),-2),full);close(old['keys'],saved);assert old['keys'].shape[-2]==2 and new['keys'].shape[-2]==4
    close(c['keys'],new['keys']);close(c['values'],new['values']);assert not full.requires_grad

if __name__=="__main__":
    run('07.03',run_checks)
