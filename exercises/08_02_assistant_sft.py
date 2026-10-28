# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 08.02: Assistant-only supervised fine-tuning.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
SFT maximizes probability of demonstrated assistant answers given the conversation. User text receives no direct target loss but contributes context and may influence parameter gradients through attention. Fine-tuning updates a pretrained model with a lower, measured learning rate and separate instruction validation; it does not automatically create new reliable knowledge.

INPUTS AND RETURN
logits: float [B,T,V] from full conversation; ids: long[B,T]; learn_mask: bool[B,T], at least one true entry after position0. Return connected scalar mean over shifted assistant targets. Padding mask must already be combined into learn_mask.

WORKED EXAMPLE
Use the six IDs from08.01, V=259. Shift logits to positions0..4, targets to IDs1..5, mask to [0,0,0,1,1]. Only predictions for b and EOS contribute. Uniform logits give loss ln259≈5.556828; gradients at logit positions0,1,2,5 are zero, positions3,4 are active.

BOUNDARIES
Masking logits by zero is not masking the loss. Do not shift twice when a data loader already returns aligned X/Y.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def assistant_loss(logits: torch.Tensor, ids: torch.Tensor, learn_mask: torch.Tensor) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # STEP 1: Slice logits[:,:-1], ids[:,1:], learn_mask[:,1:] with matching time length.
    # STEP 2: Flatten scores/targets, compute CE reduction="none", reshape to shifted target shape.
    # STEP 3: Select the shifted mask and mean; return tensor without item or detach.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement assistant_loss for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    z=torch.zeros(1,6,259,requires_grad=True);ids=torch.tensor([[256,97,258,257,98,258]]);m=torch.tensor([[0,0,0,0,1,1]],dtype=torch.bool)
    loss=assistant_loss(z,ids,m);assert math.isclose(loss.item(),math.log(259),rel_tol=1e-6)
    loss.backward();assert torch.count_nonzero(z.grad[:,:3])==0 and torch.count_nonzero(z.grad[:,5:])==0 and torch.count_nonzero(z.grad[:,3:5])>0
    changed=z.detach().clone();changed[:,:3]=torch.randn_like(changed[:,:3])*30;close(assistant_loss(changed,ids,m),loss)

if __name__=="__main__":
    run('08.02',run_checks)
