# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 08.03: Direct preference optimization.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Preference pairs express relative quality under the same prompt. DPO increases the chosen-versus-rejected policy log-ratio relative to a frozen reference, using a stable logistic loss. Pair quality and prompt matching matter; poor preferences can teach verbosity, artifacts or incorrect answers. This objective is derived in the primary DPO paper, not an assurance that all alignment goals are solved.

INPUTS AND RETURN
four tensors: float[B] summed assistant-response log probabilities for matched prompts; chosen/rejected: trainable policy, reference tensors frozen; beta>0. Return scalar mean DPO loss; detach reference inputs. Sequence log probabilities are sums, not token means.

WORKED EXAMPLE
Policy and reference both have chosen=-2,rejected=-3; beta=.2. Relative margin0, loss ln2=.693147. Derivative wrt chosen=-.1, rejected=+.1, so gradient descent increases chosen log probability and decreases rejected. If policy chosen improves to-1 while rejected stays-3, scaled margin=.2 and loss≈.598139.

BOUNDARIES
A frozen reference must not be updated by the optimizer. Comparing responses from different prompts breaks the intended ratio.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def dpo_loss(chosen: torch.Tensor, rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # STEP 1: Compute policy chosen-minus-rejected, reference difference with both ref tensors detached.
    # STEP 2: Subtract reference difference, multiply beta, use -F.logsigmoid for stability.
    # STEP 3: Mean across preference pairs and return connected scalar.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement dpo_loss for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=torch.tensor([-2.],requires_grad=True);b=torch.tensor([-3.],requires_grad=True);r=torch.tensor([-2.],requires_grad=True);s=torch.tensor([-3.],requires_grad=True)
    l=dpo_loss(a,b,r,s,.2);close(l,torch.tensor(math.log(2)));l.backward();close(a.grad,torch.tensor([-.1]));close(b.grad,torch.tensor([.1]));assert r.grad is None and s.grad is None
    assert dpo_loss(torch.tensor([-1.]),b.detach(),r,s,.2)<l
    assert torch.isfinite(dpo_loss(torch.tensor([-10000.]),torch.tensor([0.]),torch.tensor([0.]),torch.tensor([0.]),1.))

if __name__=="__main__":
    run('08.03',run_checks)
