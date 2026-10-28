# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.10: Train and generate with the tiny Transformer.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
A tiny-batch overfit diagnoses wiring before data scale. If it cannot memorize a tiny consistent mapping, investigate targets, masks, registration and gradients. This is not held-out generalization. Adam manages moments for every registered parameter; recreate neither model nor optimizer each step.

INPUTS AND RETURN
model: TinyDecoder-compatible module; input_ids,target_ids: long [B,T], already shifted; steps: nonnegative int; learning_rate: positive float. Return pre-update losses list, length steps, modifying the same model. Use Adam with no weight decay for this overfit diagnostic.

WORKED EXAMPLE
X=[[0,1,0,1],[1,0,1,0]], Y=[[1,0,1,0],[0,1,0,1]], V=2. At uniform predictions CE=ln2=.693147. After80 steps on this one batch, require loss<.15 in the supplied deterministic fixture. Greedy generation calls the full growing prefix, selects logits[0,-1].argmax(), appends one ID, and repeats while inside max_length.

BOUNDARIES
Teacher forcing supplies true preceding inputs, not the model samples. One overfit success does not establish a useful language model.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

TinyDecoder=load('04_09_decoder_model.py','TinyDecoder')



def fit_tiny(model: nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor, steps: int, learning_rate: float) -> list[float]:
    """Fit one aligned batch; update the passed model and return loss floats.

    model: existing registered module, returning logits (B,T,V).
    input_ids/target_ids: torch.long (B,T), B,T>=1, valid IDs in [0,V),
    compatible devices, T within capacity. Targets are already shifted.
    steps: nonnegative Python int; learning_rate: positive float.
    Return history: list[float], exactly steps PRE-UPDATE scalar mean losses.
    Keep model/parameter objects and input/target values. Set training mode.
    Zero steps returns [] with unchanged parameters; the mode flag may change.

    SHAPE LEDGER
      model(input_ids):             (B,T,V)
      logits.reshape(-1,V):         (B*T,V)
      target_ids.reshape(-1):       (B*T,)
      F.cross_entropy:              scalar tensor
      loss.item():                  Python float, appended to history
    X=[[0,1,0,1],[1,0,1,0]], Y=1-X, V=2 gives (8,2) scores and targets
    [1,0,1,0,0,1,0,1]. Uniform probabilities imply loss ln(2); that is not
    necessarily the randomly initialized model's first measured loss.
    Use Adam defaults, including zero weight decay; one optimizer per call.
    Preserve its moments across iterations, not across separate fit calls.
    Generation is supplied check code, not another core to implement here.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, create optimizer using torch.optim.Adam,
    # model.parameters() and lr=learning_rate; call model.train(); create
    # empty history. Do not rebuild the model/optimizer per update.
    # STEP 2: Use `for _ in range(steps):` at four spaces; its body uses
    # eight spaces. range(0) is empty; _ is an unused counter name.
    # STEP 3: Inside, optimizer.zero_grad(set_to_none=True) clears old
    # parameter gradients without deleting Adam's running moments.
    # STEP 4: Call model(input_ids), saving logits (B,T,V). Read V from
    # its last axis. Reshape scores to (-1,V) and targets to (-1): the
    # fixture gives (8,2) and (8,). Keep candidates together, no new shift.
    # STEP 5: F.cross_entropy takes these raw logits and labels. Save
    # scalar loss, then append loss.item() before updating. .item() makes
    # a Python float; keep tensor loss for the following backward call.
    # STEP 6: Call backward once on loss, at eight spaces. This calculates
    # gradients without itself changing parameter values.
    # STEP 7: Call optimizer.step() after backward, still inside the loop.
    # It updates the same model from current gradients and stored moments.
    # STEP 8: Return history at four spaces, outside the loop: steps=0
    # returns []; steps=80 returns 80 pre-update float losses. history[79]
    # precedes update 80; no final reevaluation is required here.
    # No softmax, detached training loss, no_grad training, sampling,
    # parameter replacement or input/target modification belongs here.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement fit_tiny for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====



# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(10);model=TinyDecoder(2,8,2,1,16)
    x=torch.tensor([[0,1,0,1],[1,0,1,0]]);y=1-x
    before=[p.detach().clone() for p in model.parameters()]
    assert fit_tiny(model,x,y,0,.03)==[]
    assert all(torch.equal(a,b) for a,b in zip(before,model.parameters()))
    h=fit_tiny(model,x,y,80,.03);assert len(h)==80 and h[-1]<.15 and h[-1]<h[0]
    model.eval()
    with torch.no_grad():
        prefix=[0]
        for _ in range(5):
            prefix.append(int(model(torch.tensor([prefix]))[0,-1].argmax()))
    assert prefix==[0,1,0,1,0,1]

if __name__=="__main__":
    run('04.10',run_checks)
