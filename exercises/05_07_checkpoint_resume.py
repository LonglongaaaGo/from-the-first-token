# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.07: Exact checkpoint and resume.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Lessons 05.05-05.06 explain AdamW memory and update-indexed learning rates.
To reproduce the next update after stopping, capture current model values,
optimizer memories/settings, completed-update count and both CPU RNG streams.
This standalone assignment uses supplied PyTorch objects, not unfinished course
helpers. It creates an in-memory snapshot; writing a disk file is optional.

EXACT INPUTS
model: nn.Module, the existing live model. Read model.state_dict(), containing
named parameter/persistent-buffer tensors with their individual shapes/dtypes.
optimizer: torch.optim.Optimizer attached to model.parameters(). Its state_dict
contains nested state and param_groups, including memories, counters and settings.
step: nonnegative Python int, number of completed updates. Preserve it unchanged.
At step=1 the next global update is update2; the next 05.06 schedule index is1
when each iteration performs one update. A fresh AdamW may have empty state.
sampler: caller-owned CPU torch.Generator, independent of the default CPU stream.

RETURN CONTRACT
Return one dict with exactly these five keys:
- model: model.state_dict() values, independently snapshotted.
- optimizer: optimizer.state_dict(), including nested state and param_groups.
- step: the supplied completed-update count.
- torch_rng: torch.get_rng_state(), the current default CPU generator state.
- sampler_rng: sampler.get_state(), the current separate generator state.
RNG states are uint8 tensors encoding internal state, not seeds or random draws.
Recursively copy the assembled dict so no mutable tensor storage is shared with
original live model/optimizer state. Ordinary state_dict tensors are detached,
but detach and a shallow dict copy do not by themselves copy tensor storage.
Do not copy the live Module or Optimizer objects.

COMPLETE ORIGINAL EXAMPLE
The supplied driver initializes nn.Linear(1,1) with seed17. Input has shape(1,1)
and contents[[1]], target=2. There are two parameters: weight shape(1,1) and bias
shape(1,). Loss=(w+b-2)^2, so both gradients equal 2*(w+b-2) for this input.
AdamW uses lr=.02, default betas=(.9,.999), eps=1e-8, weight_decay=.01.
Existing PyTorch2.0 CPU float32 trace, rounded for reading:
  Initial: w=-.13151753, b=.07021916; loss=4.24895096, gradient=-4.12259674.
  Update1: w=-.11149123, b=.09020511; each m=-.41225967, v=.01699581, count=1.
  Capture independent C1 with step=1.
  Update2: gradient=-4.04257202; w=-.09148012, b=.11017589;
           each m=-.77529091, v=.03332120, count=2.
Recreate model, optimizer on NEW model parameters, and sampler; supplied
restore_state loads C1. Repeat update2 and match both parameters and subsequent
random draws. The saved model values must still equal the update1 values.
Saving only weights and using a fresh AdamW instead gives w=-.09146895 and
b=.11018707: same next gradient, different memory/counter and next update.
These numbers explain the driver; do not hard-code them in the function.

STATE AND ORDER BOUNDARIES
Capture only at an optimizer boundary without pending gradient accumulation.
Do not call forward/backward/step/zero_grad, reseed, draw random numbers or mutate
inputs here. Default CPU RNG and sampler RNG must be read separately. A seed is
a starting point; state also tracks how far the generator has advanced.
Create replacement objects before restoring RNG: model initialization consumes
the default stream. The supplied next-update helper clears stale gradients.
The return dict is not a disk file or a saved autograd graph. Real training may
also need data cursor/order, architecture/configuration, train/eval mode,
scheduler/scaler state, used accelerator/Python/NumPy RNGs and portable provenance.
CPU tolerance checks do not certify exactness across devices or software versions.

SYNTAX AND OWNERSHIP
Edit only the marked learner region. A function body uses four spaces; dictionary
entries can use eight for readability, then close the brace/return at four.
Dictionary entries use 'key': expression, with commas. Parentheses call methods;
without them, model.state_dict is a method object, not the exported values.
copy is already imported; copy.deepcopy recursively copies the complete bundle.
No loop or branch is required. Keep supplied restore_state and run_checks intact.
The numbered hints guide your implementation rather than supply a filled core.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run



def restore_state(state,model,optimizer,sampler):
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng']);sampler.set_state(state['sampler_rng'])
    return state['step']

def checkpoint_state(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, sampler: torch.Generator) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: At four-space function indentation, start a temporary dictionary.
    #         Syntax: local_name = { ... }; each entry uses 'key': value,.
    # STEP 2: Add key 'model' using model.state_dict() WITH parentheses.
    #         Expect separate weight/bias tensors, not the live Module object.
    # STEP 3: Add key 'optimizer' using optimizer.state_dict(). Keep the whole
    #         nested result: both state and param_groups, not just m and v.
    # STEP 4: Add key 'step' using the supplied integer unchanged (including0).
    #         Do not add1 or call optimizer.step(), which would train the model.
    # STEP 5: Add key 'torch_rng' using torch.get_rng_state(). Read current
    #         default CPU state; do not draw numbers or reset the seed.
    # STEP 6: Add key 'sampler_rng' using sampler.get_state(). This separate
    #         stream needs its own state. Close the dictionary at four spaces.
    # STEP 7: Return copy.deepcopy applied to the ENTIRE temporary dictionary.
    #         An outer .copy() leaves nested tensors shared with live state.
    # STEP 8: Remove the placeholder raise. Keep restore_state/run_checks intact;
    #         the driver owns all updates, object recreation and random draws.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement checkpoint_state for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(17);m=nn.Linear(1,1);o=torch.optim.AdamW(m.parameters(),lr=.02);g=torch.Generator().manual_seed(7)
    def update(m,o):
        o.zero_grad();loss=(m(torch.ones(1,1))-2).square().mean();loss.backward();o.step()
    update(m,o);c=checkpoint_state(m,o,1,g);assert c['step']==1
    saved=copy.deepcopy(c['model']);update(m,o);expected=copy.deepcopy(m.state_dict());draw=torch.rand(3,generator=g);rng_draw=torch.rand(3)
    n=nn.Linear(1,1);p=torch.optim.AdamW(n.parameters(),lr=.02);h=torch.Generator()
    restore_state(c,n,p,h);update(n,p)
    for k in expected:close(n.state_dict()[k],expected[k]);close(c['model'][k],saved[k])
    close(torch.rand(3,generator=h),draw);close(torch.rand(3),rng_draw)

if __name__=="__main__":
    run('05.07',run_checks)
