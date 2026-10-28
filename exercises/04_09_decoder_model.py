# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.09: Assemble a causal decoder.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
A decoder stacks the same causal operation at increasing representation depth. Every block preserves [B,T,D]; only the vocabulary head changes D to V. ModuleList makes all blocks trainable and serializable. These supplied blocks expose the assembly task without repeating earlier core implementations.

INPUTS AND RETURN
Constructor integers: V,D,H,L,M; D divisible by H, L>=1. input_ids: long [B,T], 1<=T<=M. Return raw logits [B,T,V]. Supplied DecoderBlock reuses 04.06/04.08 and has registered parameters.

WORKED EXAMPLE
V=5,D=4,H=2,L=2,M=8,X=[[0,1,2]]: lookup [1,3,4], positions [3,4], two block outputs [1,3,4], final logits [1,3,5]. Change final ID2 to4: first two logit rows must stay unchanged. Unlike Chapter03, changing ID0 can change the final row. Test both isolation and contextual sensitivity.

BOUNDARIES
Returning inside the block loop skips later blocks. Final softmax is inappropriate before F.cross_entropy.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

multihead=load('04_06_multihead_attention.py','multihead')
add_positions=load('04_07_position_embeddings.py','add_positions')
pre_norm_ffn=load('04_08_residual_norm_ffn.py','pre_norm_ffn')

class DecoderBlock(nn.Module):
    """Supplied pre-norm block: (B,T,D) -> (B,T,D), with both residuals.

    Each instance owns its parameters. Attention uses heads groups of D/heads
    coordinates and a causal mask; FFN uses D -> 4D -> D at each position.
    Do not edit this supplied block or add another residual around its return.
    """
    def __init__(self,width,heads):
        super().__init__();self.heads=heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.q=nn.Parameter(torch.randn(width,width)/math.sqrt(width))
        self.k=nn.Parameter(torch.randn(width,width)/math.sqrt(width))
        self.v=nn.Parameter(torch.randn(width,width)/math.sqrt(width))
        self.o=nn.Parameter(torch.randn(width,width)/math.sqrt(width))
        self.up=nn.Linear(width,4*width);self.down=nn.Linear(4*width,width)
    def forward(self,x):
        h=x+multihead(self.norm1(x),self.q,self.k,self.v,self.o,self.heads)
        return pre_norm_ffn(h,self.norm2,self.up,self.down)

class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, width: int, heads: int, layers: int, max_length: int):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,width)
        self.positions=nn.Parameter(torch.zeros(max_length,width))
        self.blocks=nn.ModuleList([DecoderBlock(width,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,vocab_size,bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw next-token logits for every supplied input position.

        input_ids: torch.long (B,T), B>=1, 1<=T<=max_length, IDs in [0,V).
        Reuse all modules from __init__; require compatible input/model device.
        Output: floating (B,T,V), not probabilities or selected integer IDs.
        This interface has no cache/offset/padding mask; positions start at 0.

        SHAPE LEDGER
          self.embedding(input_ids):        (B,T,D)
          add_positions with self.positions:(B,T,D), using position rows 0:T
          each block, sequentially:         (B,T,D)
          self.norm after all blocks:       (B,T,D)
          self.head:                        (B,T,V)
        Primary dimensions: V=5,D=4,heads=2,layers=2,max_length=8;
        input [[0,1,2]] gives (1,3,4) hidden states then (1,3,5) logits.
        Full deterministic numeric example and both matrix-product dimensions
        are in the paired lesson; its special weights are not the driver weights.
        blocks already implement attention/FFN residuals. Return all T rows.
        """
        # ===== START LEARNER CODE =====
        # This is a class method: ordinary body statements have eight spaces.
        # STEP 1: Call the existing self.embedding module with input_ids.
        # Use module(argument) syntax, saving hidden: (B,T)->(B,T,D).
        # Do not construct another Embedding or convert the IDs to floats.
        # STEP 2: Pass hidden and self.positions to add_positions, replacing
        # hidden with its returned tensor. Use the default offset=0. The helper
        # selects rows 0:T, adds across B, and preserves (B,T,D). This is H0.
        # STEP 3: Use `for block in self.blocks:` at eight spaces. Its body
        # uses twelve spaces: call this block on the latest hidden and assign
        # the result back to hidden. Two iterations produce H1 then H2.
        # The loop runs over depth, not token positions. Each block keeps T,D.
        # Do not ignore its return, restart from H0, or add another residual.
        # STEP 4: After the loop, return to eight spaces. Apply self.norm to
        # the latest hidden; save the normalized tensor, still (B,T,D).
        # STEP 5: Apply self.head to the normalized tensor, saving logits.
        # The existing Linear handles its weight transpose: (B,T,D)->(B,T,V).
        # Do not flatten, softmax, argmax, sample, detach or update weights.
        # STEP 6: Return all logits at eight spaces, outside the loop.
        # The single-sample fixture gives (1,3,5); the driver expects (2,3,5).
        # A return inside the loop would stop after only the first block.
        # TODO: implement the contract using the worked example and numbered hints.
        raise NotImplementedError("Implement TinyDecoder.forward for this exercise")
        # ===== END LEARNER CODE =====
        # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(9);m=TinyDecoder(5,4,2,2,8);x=torch.tensor([[0,1,2],[2,1,0]])
    z=m(x);assert z.shape==(2,3,5)
    x2=x.clone();x2[:,-1]=4;close(m(x2)[:,:2],z[:,:2])
    x3=x.clone();x3[:,0]=3;assert not torch.allclose(m(x3)[:,-1],z[:,-1])
    F.cross_entropy(z.reshape(-1,5),torch.tensor([1,2,3,1,0,2])).backward()
    assert all(p.grad is not None for p in m.parameters())
    n=TinyDecoder(5,4,2,2,8);n.load_state_dict(m.state_dict());close(n(x),z)

if __name__=="__main__":
    run('04.09',run_checks)
