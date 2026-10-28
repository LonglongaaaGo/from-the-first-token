# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 07.04: Cached inference interface.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

CONNECTION AND PURPOSE
07.03 computes one layer's new attention outputs and K/V state. The supplied
CachedModernLM calls it at every layer, then returns vocabulary logits. Write
only generate_cached: prefill the prompt once, select a token, and feed only
that token into the NEXT call. A selected token is not cached until processed.
The outer no_grad also covers embedding, FFN and head outside07.03's guard.
No environment setup, trained checkpoint, tokenizer or web server is needed.

EXACT INPUTS AND OUTPUT
model: supplied parameter-owning CachedModernLM. forward(ids,cache) returns
logits[B,S,V] and a new Python list of per-layer {'keys','values'} dictionaries.
Fixture: B1,V5,D8,H2,d4,L2. Initial ids[[0,1]] has shape[1,2]; later[[last_id]]
has[1,1]. Prefill logits[1,2,5]; subsequent logits[1,1,5]. Select[0,-1,:]->[5].
Each layer's cache grows[1,2,P,4]->[1,2,P+S,4]; equal shapes across layers do
not imply equal vector contents. Do not stack layers along the time axis.
prompt: nonempty list[int], preserve it. budget: nonnegative NEW ID count.
eos_id: int or None; EOS0 is valid. temperature>=0; zero uses argmax.
generator: caller-owned CPU torch.Generator for this CPU fixture; do not reseed.
Return a FRESH full prompt+generated list, including a newly selected EOS.
Preserve model weights; use eval/no_grad and restore the entry training flag
on success or exception. Uniform model modes are the assignment's convention.

FORMULAS AND COMPLETE SCRIPTED EXAMPLE
(Z_s,C_next)=model(U_s,C_old); U_0=prompt, U_s=[previous_selected_id] thereafter.
z=Z_s[0,-1,:]; positive tau samples softmax(z/tau), zero uses argmax(z).
Teaching labels: IDs0,1,2,3,4 mean I,like,tea,.,EOS. These labels are not real BPE.
prompt[0,1],budget3,eos4,tau0:
 call0 input[[0,1]], cache None->length2, z=[0,ln2,ln4,ln2,0], choose2.
       output list[0,1,2]; cache covers only[0,1].
 call1 input[[2]], old length2->3, z=[ln2,0,0,ln4,0], choose3.
       output list[0,1,2,3]; cache covers[0,1,2].
 call2 input[[3]], old length3->4, z=[0,0,0,0,ln8], choose4 then stop.
       return[0,1,2,3,4]; EOS is selected but never forwarded.
Input lengths2,1,1, not2,3,4. Scripted logits differ from the random driver.
At tau1, first-row probabilities[.1,.2,.4,.2,.1]; at tau.5 they are
[1,4,16,4,1]/26. Budget0 returns a fresh prompt copy without forward/sampling.
Do not decode reserved EOS as ordinary bytes in a later text-display layer.

BOUNDARIES
One unpadded CPU sequence. No top-k, beam search, context eviction or network
service here. The wrapper has no trained-context-limit enforcement; production
limits and request-owned caches remain necessary. The standalone workbench's
text stream excludes EOS; this exercise's returned ID list includes it.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

ModernLM=load('05_04_modern_model.py','ModernLM')
rmsnorm=load('05_01_rmsnorm.py','rmsnorm')
swiglu=load('05_03_swiglu.py','swiglu')
cached_attention=load('07_03_kv_cache.py','cached_attention')

class CachedModernLM(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
    def forward(self,ids,cache=None):
        x=self.base.embedding(ids);new=[]
        if cache is None:cache=[None]*len(self.base.blocks)
        for block,past in zip(self.base.blocks,cache):
            B,T,D=x.shape;H=block.heads;dh=D//H;n=rmsnorm(x,block.gain1)
            q,k,v=[(n@w).reshape(B,T,H,dh).transpose(1,2) for w in (block.q,block.k,block.v)]
            o,state=cached_attention(q,k,v,past);new.append(state)
            x=x+o.transpose(1,2).reshape(B,T,D)@block.o
            x=x+swiglu(rmsnorm(x,block.gain2),block.gate,block.up,block.down)
        return self.base.head(rmsnorm(x,self.base.gain)),new

def generate_cached(model: nn.Module, prompt: list[int], budget: int, eos_id: int | None, temperature: float, generator: torch.Generator) -> list[int]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces copy prompt with list(prompt) into sequence and
    #         save model.training as old_mode. Never append into prompt itself.
    # STEP 2: At FOUR initialize cache=None. next(model.parameters()).device
    #         gives the registered parameter device for the new ID tensor.
    # STEP 3: At FOUR open try:, then EIGHT call model.eval(). This controls
    #         module behavior; no_grad separately controls autograd recording.
    # STEP 4: At EIGHT open with torch.no_grad():. At TWELVE open
    #         for _ in range(budget):; its repeated body uses SIXTEEN spaces.
    #         range(0) runs no iterations; underscore is an unused loop index.
    # STEP 5: At SIXTEEN branch on cache is None; branch bodies TWENTY.
    #         First current_ids is sequence; later it is [sequence[-1]].
    #         Example lists: [0,1], then [2], then [3]. Do not resend the prefix.
    # STEP 6: Back at SIXTEEN torch.tensor([current_ids],dtype=torch.long,
    #         device=device) adds the batch axis: shapes[1,2], then[1,1].
    #         A scalar ID or flat[S] tensor is not the wrapper's input shape.
    # STEP 7: Call model(input_ids,cache), unpack logits AND the next cache.
    #         Layers update their own K/V; do not concatenate different layers.
    # STEP 8: logits[0,-1,:] selects one batch, the last new position and all V
    #         candidates. Its shape is[V]; do not average prompt positions.
    # STEP 9: At SIXTEEN branch on temperature==0. Greedy branch uses argmax;
    #         positive branch softmaxes logits/temperature over dim=-1.
    #         Never divide by zero and never normalize over input positions.
    # STEP 10: Convert the chosen scalar to a Python int. Sampling uses
    #          torch.multinomial(probabilities,1,generator=generator).item().
    #          Keep the caller's generator; do not construct/reseed inside loop.
    # STEP 11: Back at SIXTEEN sequence.append(token), THEN test
    #          eos_id is not None and token==eos_id. EOS0 must not be skipped.
    # STEP 12: Inside that condition use break (TWENTY). It exits the loop;
    #          the appended EOS remains. Do not return after one ordinary token.
    # STEP 13: At FOUR finally:, then EIGHT model.train(old_mode). It restores
    #          the entry mode even on forward failure. Never return from finally.
    # STEP 14: At FOUR after cleanup return sequence, outside the loop. This
    #          handles budget0 with a fresh copy and no forward/random draw.
    #          Preserve supplied wrapper/signature/imports/driver; replace raise.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement generate_cached for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(34);base=ModernLM(5,8,2,16,2);m=CachedModernLM(base);x=torch.tensor([[0,1,2]])
    with torch.no_grad():
        z,c=m(x,None);close(z,base(x));u,c=m(torch.tensor([[3]]),c);close(u,base(torch.tensor([[0,1,2,3]]))[:,-1:])
    p=[0,1];g=torch.Generator().manual_seed(9);a=generate_cached(m,p,3,None,0.,g);assert len(a)==5 and p==[0,1] and m.training
    ref=list(p)
    with torch.no_grad():
        for _ in range(3):ref.append(int(base(torch.tensor([ref]))[0,-1].argmax()))
    assert a==ref and generate_cached(m,p,0,None,0.,g)==p
    b=generate_cached(m,p,8,a[2],0.,g);assert b==p+[a[2]]
    a=generate_cached(m,p,3,None,.7,torch.Generator().manual_seed(4));b=generate_cached(m,p,3,None,.7,torch.Generator().manual_seed(4));assert a==b

if __name__=="__main__":
    run('07.04',run_checks)
