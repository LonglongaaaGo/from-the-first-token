# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.02: Rotary query and key positions.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
RoPE rotates two-dimensional feature planes by position-dependent angles. Rotation preserves pair norm; the dot product of rotated query/key depends on relative position through R(p)^T R(q)=R(q-p). Different pair frequencies represent different distance scales. Positions are absolute indices even when a cache holds only recent tensors.

INPUTS AND RETURN
x: float [B,H,T,d], positive even d; positions: integer [T] absolute positions, same device; base: positive float. Return same shape/dtype. Adjacent coordinate pairs (0,1),(2,3) are the fixed layout convention. Apply to queries and keys, not values.

WORKED EXAMPLE
For d=2 there is one frequency omega=1. Position0 leaves [1,0] unchanged. Position1 rotates [1,0] to [cos1,sin1]≈[.540302,.841471]. Both have squared norm1. Rotated q at p=1 and k at p=3 have dot cos2≈-.416147. Shift both positions by5 and the dot stays cos2.

BOUNDARIES
Do not mix adjacent-pair and half-split conventions between training and inference. Resetting cached positions to0 changes predictions.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.) -> torch.Tensor:
    """Rotate adjacent feature pairs at supplied absolute positions.

    x: floating (B,H,T,d), positive even d; float16/bfloat16/float32/float64.
    positions: integer (T,), same device as x, shared across B and H.
    base: positive Python float; default 10000. Angles are in radians.
    Return (B,H,T,d), original dtype/device, preserving x/positions and
    autograd. Apply separately to projected Q and K, not to V in this design.
    This standalone function does not depend on the RMSNorm learner core.

    SHAPE AND VALUE LEDGER
      pair_coords = 0,2,...,d-2: (d/2,), the first coordinate of each pair
      inv_freq = base**(-pair_coords/d): (d/2,)
      angles = positions[:,None] * inv_freq[None,:]: (T,d/2)
      cos/sin: (T,d/2), broadcast across the two leading B,H axes
      even/odd slices of working x: each (B,H,T,d/2)
      rotated_even/rotated_odd: each (B,H,T,d/2)
      stack(...,dim=-1): (B,H,T,d/2,2)
      flatten(start_dim=-2): (B,H,T,d), then restore x.dtype
    Use float64 work for float64 x; float32 for the other supported dtypes.
    Convert positions before angles, and x before products; retain gradients.
    Adjacent-pair convention is fixed: (0,1),(2,3), not half-split layout.

    First fixture: x (1,1,2,2) has two [1,0] rows; positions=[0,1].
    inv_freq=[1], angles=[[0],[1]], output rows [1,0] and
    [0.54030231,0.84147098]. No factor of 2*pi is added to radian angles.
    Wider example d=4: inv_freq=[1,0.01]. At position1, [1,0,2,0]
    becomes [0.54030231,0.84147098,1.99990000,0.01999967].

    PLAIN-LANGUAGE FOLLOW-UP
    A feature pair is just two adjacent coordinates, not a new tensor type.
    For one row [1,2,3,4], actual pairs are [1,2] and [3,4]. Slices
    even=[1,3], odd=[2,4] match by column: (even[0],odd[0]), then
    (even[1],odd[1]). The entries within even are not rotated together.
    omega_i is the Greek omega with a pair index, meaning radians per
    position step; it is not a W weight matrix. base=10000 is fixed here.
    d=4 gives omega0=1, omega1=1/sqrt(10000)=0.01; at position p=3,
    angles are 3*1=3 and 3*0.01=0.03. See the opening paired notes before
    implementing if the symbols/pairing are unclear.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces read d from x.shape[-1]. Select work_dtype:
    # if x.dtype == torch.float64: choose float64; else choose float32.
    # if/else are at four spaces; each branch body uses eight spaces.
    # All remaining steps return to four spaces; no loop is necessary.
    # STEP 2: Cast x with .to(work_dtype), saving work. Use torch.arange
    # with start=0, end=d, step=2, device=x.device and dtype=work_dtype
    # to build pair_coords. For d=4 this is [0,2], shape (2,), not [0,1].
    # STEP 3: Compute inverse frequencies using base**(-pair_coords/d).
    # Python ** means power. For d=4, base=10000: inv_freq=[1,0.01].
    # STEP 4: Cast positions to work_dtype, then use [:,None] for (T,1).
    # Use inv_freq[None,:] for (1,d/2); their product angles is (T,d/2).
    # None inserts a singleton axis. For positions=[0,1], d=2: [[0],[1]].
    # STEP 5: Use angles.cos() and angles.sin(); both remain (T,d/2).
    # Take work[...,0::2] as even and work[...,1::2] as odd. The ...
    # preserves B,H,T; ::2 selects every second coordinate on the last axis.
    # STEP 6: Compute two NEW tensors, preserving both original slices:
    # rotated_even = even*cos - odd*sin; rotated_odd = even*sin + odd*cos.
    # Each shape is (B,H,T,d/2); (T,d/2) broadcasts across B,H.
    # First fixture at p=1: rotated_even=cos(1), rotated_odd=sin(1).
    # STEP 7: torch.stack the two tensors, even result first, on dim=-1.
    # Shape becomes (B,H,T,d/2,2), with each pair adjacent. Flatten only
    # the last TWO axes using .flatten(start_dim=-2), restoring (B,H,T,d).
    # torch.cat on the last axis would incorrectly group all even then odd.
    # STEP 8: Cast this result back to x.dtype and return one tensor.
    # Keep the device, input values and gradient path; no detach/item,
    # in-place write, token reorder, attention mask or parameter update.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement apply_rope for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.tensor([[[[1.,0.],[1.,0.]]]],requires_grad=True);p=torch.tensor([0,1])
    y=apply_rope(x,p);close(y,torch.tensor([[[[1.,0.],[math.cos(1),math.sin(1)]]]]));close(y.square().sum(-1),x.square().sum(-1))
    y.sum().backward();assert x.grad is not None
    z=torch.randn(2,3,4,6,dtype=torch.float64);p=torch.arange(4)
    close(apply_rope(z,p).square().sum(-1),z.square().sum(-1))
    q=torch.tensor([[[[1.,0.]]]],dtype=torch.float64)
    dot=lambda a,b:(apply_rope(q,torch.tensor([a]))*apply_rope(q,torch.tensor([b]))).sum()
    close(dot(1,3),dot(6,8));close(dot(1,3),torch.tensor(math.cos(2),dtype=torch.float64))

if __name__=="__main__":
    run('05.02',run_checks)
