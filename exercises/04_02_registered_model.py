# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.02: Registered parameters and modules.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
nn.Module organizes trainable state, device moves and named state dictionaries.
Assigning a submodule to self registers its parameters. This familiar
position-wise lookup/head model teaches organization; it has no prefix mean.
Forward computes outputs; backward computes gradients; an optimizer updates
parameters. nn.Linear stores [out_features,in_features], here [V,D].

INPUTS AND RETURN
vocab_size: positive V; width: positive D, used by supplied constructor. input_ids: long [B,T]. Return logits [B,T,V]. The supplied embedding/head are registered modules; forward must preserve gradient flow.

WORKED EXAMPLE
V=3,D=2: embedding has 6 values, head weight has 6 and bias has 3, totaling15. Set embedding=[[1,0],[0,1],[1,1]], head weight=[[1,0],[0,1],[1,1]], bias=0. X=[[0,2]] produces logits [[[1,0,1],[1,1,2]]], shape [1,2,3]. module.parameters() yields three parameter objects; state_dict has three named tensors.

BOUNDARIES
A plain tensor attribute is not automatically a Parameter. ModuleList registers a list of blocks; an ordinary Python list does not. model.eval() changes layer mode, while no_grad changes graph recording.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





class TokenModel(nn.Module):
    def __init__(self, vocab_size: int, width: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # ===== START LEARNER CODE =====
        # STEP 1: With eight-space indentation, call self.embedding(input_ids).
        # This looks up self.embedding.weight; [B,T] becomes [B,T,D].
        # Fixture result: [[[1,0],[1,1]]], shape [1,2,2].
        # STEP 2: At the same indentation, pass these vectors to self.head.
        # It maps the final D axis to V: [[[1,0,1],[1,1,2]]], shape [1,2,3].
        # STEP 3: Return raw logits, preserving the tensor and gradient graph.
        # The constructor already created parameters. Keep those layers;
        # do not apply softmax, call backward, or update them inside forward.
        # TODO: implement the contract using the worked example and numbered hints.
        raise NotImplementedError("Implement TokenModel.forward for this exercise")
        # ===== END LEARNER CODE =====
        # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    m=TokenModel(3,2)
    with torch.no_grad():
        m.embedding.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]))
        m.head.weight.copy_(m.embedding.weight);m.head.bias.zero_()
    z=m(torch.tensor([[0,2]]));close(z,torch.tensor([[[1.,0.,1.],[1.,1.,2.]]]))
    assert sum(p.numel() for p in m.parameters())==15
    z.sum().backward();assert all(p.grad is not None for p in m.parameters())
    n=TokenModel(3,2);n.load_state_dict(m.state_dict());close(n(torch.tensor([[0,2]])),z)

if __name__=="__main__":
    run('04.02',run_checks)
