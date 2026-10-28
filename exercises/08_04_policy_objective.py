# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 08.04: Rewards, advantages and clipped policy objectives.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
A reward judges sampled outcomes; a baseline converts rewards to advantages to reduce variance. REINFORCE weights log-probability gradients by advantage. PPO uses a ratio to an old sampling policy and a clipped surrogate. A language-model RL system additionally needs rollout generation, verified rewards, reference KL, value or group baselines, masking and repeated evaluation.

INPUTS AND RETURN
new_logp,old_logp,advantage: float[B] action log probabilities and precomputed advantages; clip in(0,1). Return scalar negative clipped surrogate mean. Detach old_logp and advantage; no reward-model gradient. This is the policy term only, not a complete PPO trainer.

WORKED EXAMPLE
clip=.2, ratios=[1.5,.5], advantages=[1,-1]. Unclipped terms=[1.5,-.5]; clipped=[1.2,-.8]; elementwise min=[1.2,-.8], mean=.2, loss=-.2. For positive advantage excessive probability increase is capped; for negative advantage excessive decrease is capped. At ratio1 the gradient sign follows advantage.

BOUNDARIES
Maximizing an unreliable reward can teach reward hacking. A complete algorithm also handles KL, value estimates and sampling mismatch; this small loss check certifies none of those systems.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def clipped_policy_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # STEP 1: Compute ratio=exp(new_logp-old_logp.detach()).
    # STEP 2: Detach advantage, form ratio*A and ratio.clamp(1-clip,1+clip)*A.
    # STEP 3: Use torch.minimum elementwise, then negative mean.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement clipped_policy_loss for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    new=torch.tensor([math.log(1.5),math.log(.5)],requires_grad=True);old=torch.zeros(2,requires_grad=True);adv=torch.tensor([1.,-1.],requires_grad=True)
    l=clipped_policy_loss(new,old,adv,.2);close(l,torch.tensor(-.2));l.backward();close(new.grad,torch.zeros(2));assert old.grad is None and adv.grad is None
    new=torch.zeros(2,requires_grad=True);clipped_policy_loss(new,old,adv,.2).backward();close(new.grad,torch.tensor([-.5,.5]))

if __name__=="__main__":
    run('08.04',run_checks)
