# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 04.07: Position information.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Attention without positional information is permutation equivariant before masking; a causal boundary supplies visibility but does not provide a rich distance representation. Learned absolute positions give each slot an identity. Offset matters during continuation and cached inference. Chapter05 replaces these additions with rotary query/key positions.

INPUTS AND RETURN
x: float [B,T,D]; position_table: float [M,D]; offset: nonnegative start index, offset+T<=M. Return [B,T,D] with learned absolute vectors added; preserve gradients for both.

WORKED EXAMPLE
x=[[[1,2],[1,2]]], P=[[0,0],[10,20],[30,40]], offset=1. Select P[1:3]=[[10,20],[30,40]], broadcast across batch, and obtain [[[11,22],[31,42]]]. Even identical token vectors differ by position. P rows outside the selected interval get zero gradient.

BOUNDARIES
Adding the same position row everywhere does not encode order. Learned tables do not inherently extrapolate past their allocated length.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def add_positions(x: torch.Tensor, position_table: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Add learned absolute position vectors; return one tensor (B,T,D).

    x: floating (B,T,D), already embedded vectors, not token IDs.
    position_table: floating (M,D), one D-vector per available position.
    offset: nonnegative Python integer, absolute position of local row t=0.
    Require offset+T<=M and compatible floating dtypes/devices. All batch
    samples share this offset. Preserve both inputs and their gradients.

    SHAPE LEDGER
      selected table rows [offset:offset+T]: (T,D)
      selected.unsqueeze(0):                (1,T,D)
      x + shared position addend:           (B,T,D)
    Slices include their start and exclude their stop. Insert axis 0 for
    batch sharing; unsqueeze adds no values, tokens or trainable parameters.
    Fixture: x=(1,2,2), table=(3,2), offset=1; select rows 1,2.
    selected=[[10,20],[30,40]], return=[[[11,22],[31,42]]].
    This function selects and adds an existing table; it does not train it.
    """
    # ===== START LEARNER CODE =====
    # Each statement is inside this function at four-space indentation.
    # STEP 1: Read x.shape[1] into length, the number of positions T.
    # Use position_table[start:stop] with start=offset, stop=offset+length,
    # saving selected. The stop is excluded: offset=1,T=2 selects rows 1,2.
    # selected has shape (T,D)=(2,2), contents [[10,20],[30,40]].
    # Keep all D coordinates; do not select a single row or use D as length.
    # STEP 2: Apply .unsqueeze(0) to selected, saving position_batched.
    # Shape: (T,D)->(1,T,D)=(1,2,2); values and their order do not change.
    # The new axis is a shared batch axis, not another position or feature.
    # STEP 3: Use ordinary elementwise + with x and position_batched.
    # Broadcasting shares positions across B samples. Return one (B,T,D)
    # tensor at four-space indentation; fixture: [[[11,22],[31,42]]].
    # No loop, in-place +=, detach, manual repeat or parameter update needed.
    # Keep the original x/table values and autograd connections intact.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement add_positions for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.tensor([[[1.,2.],[1.,2.]]],requires_grad=True);p=torch.tensor([[0.,0.],[10.,20.],[30.,40.]],requires_grad=True)
    y=add_positions(x,p,1);close(y,torch.tensor([[[11.,22.],[31.,42.]]]))
    y.sum().backward();close(p.grad[0],torch.zeros(2));close(x.grad,torch.ones_like(x))
    assert add_positions(x.expand(3,-1,-1),p,1).shape==(3,2,2)

if __name__=="__main__":
    run('04.07',run_checks)
