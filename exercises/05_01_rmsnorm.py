# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 05.01: RMS normalization.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
RMSNorm rescales magnitude without subtracting the mean. Learned gain adjusts individual features. Low-precision squares can overflow, so statistics use a wider accumulator and then return to the input dtype. This protects the reduction, not every operation in a network.

INPUTS AND RETURN
x: float [...,D]; gain: float [D], matching x dtype/device; eps: positive float. Return same shape and dtype. Compute mean-square in float32 for float16/bfloat16; float64 remains float64.

WORKED EXAMPLE
x=[3,4], mean-square=12.5, RMS=sqrt(12.5)=3.535534. With gain=[1,2], eps=1e-6, output≈[.848528,2.262742]. For x=0 denominator=sqrt(eps), output=0 rather than NaN. Scaling x by2 leaves the normalized direction almost unchanged when eps is negligible.

BOUNDARIES
RMSNorm and LayerNorm differ on nonzero means. Adding eps outside the square root defines a different function.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def rmsnorm(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return same-shape/dtype RMS-normalized features with supplied gain.

    x: floating (...,D), D>=1; float16/bfloat16/float32/float64.
    gain: floating (D,), same dtype/device as x, shared over leading axes.
    eps: positive Python float, inside the root; default 1e-6.
    Return one tensor with x's shape/dtype/device. Preserve input/gain values
    and all autograd connections. This function creates no learned parameter.

    SHAPE AND PRECISION LEDGER
      work = widened x:          (...,D), float32 for half/bfloat16
      mean_square over last D:  (...,1), retain this axis with keepdim=True
      inverse_rms:              (...,1), rsqrt(mean_square+eps)
      normalized:               (...,D), work*inverse_rms, then cast to x.dtype
      returned normalized*gain: (...,D), gain broadcasts across leading axes
    Other inputs retain x.dtype, including float64. Widen BEFORE the square:
    float16 can hold 1000 but not 1000**2. Casting infinity later cannot fix it.
    Fixture x=[[3,4]], gain=[1,2] gives mean_square=[[12.5]], inverse_rms
    about [[0.28284270]], normalized about [[0.84852810,1.13137080]],
    output about [[0.84852810,2.26274161]]. No mean subtraction or bias.
    Matching gain dtype is required for the promised returned dtype.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces use if x.dtype in (torch.float16, torch.bfloat16):
    # to choose work_dtype. `in` tests tuple membership. Its eight-space
    # body selects torch.float32. A four-space else: has an eight-space body
    # selecting x.dtype. Float64 statistics must not be reduced to float32.
    # STEP 2: Back at four spaces, use x.to(work_dtype), saving work.
    # Shape remains (...,D); cast before square, keep original x unchanged.
    # STEP 3: Use work.square() and .mean(dim=-1,keepdim=True), saving
    # mean_square. Fixture: (1,2)->(1,1), value [[12.5]]. Reduce features D,
    # not T or the whole batch. No centering or sample-variance correction.
    # STEP 4: Add eps to mean_square, then torch.rsqrt to get inverse_rms.
    # torch.rsqrt(z)=1/sqrt(z); fixture [[0.28284270]], shape (...,1).
    # Do not put eps outside rsqrt or cast this statistic back to float16.
    # STEP 5: Multiply work by inverse_rms; broadcasting restores (...,D).
    # Save normalized, then cast normalized to the original x.dtype.
    # Fixture before gain: [[0.84852810,1.13137080]]. Keep gradients intact.
    # STEP 6: At four spaces, multiply by the supplied gain and return one
    # tensor. Fixture [[0.84852810,2.26274161]], original shape and dtype.
    # Gain has one value per feature, shared over all leading rows. No new
    # parameter, loop over rows, in-place change, detach or optimizer needed.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement rmsnorm for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x=torch.tensor([[3.,4.]],dtype=torch.float64,requires_grad=True);g=torch.tensor([1.,2.],dtype=torch.float64,requires_grad=True)
    close(rmsnorm(x,g),torch.tensor([[3/math.sqrt(12.500001),8/math.sqrt(12.500001)]],dtype=torch.float64))
    rmsnorm(x,g).sum().backward();assert x.grad is not None and g.grad is not None
    close(rmsnorm(torch.zeros(2,3),torch.ones(3)),torch.zeros(2,3))
    z=torch.full((2,4),1000.,dtype=torch.float16);o=rmsnorm(z,torch.ones(4,dtype=z.dtype));assert o.dtype==z.dtype and torch.isfinite(o).all();close(o.float(),torch.ones(2,4))

if __name__=="__main__":
    run('05.01',run_checks)
