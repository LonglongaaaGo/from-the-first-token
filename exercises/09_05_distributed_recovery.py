# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 09.05: Distributed gradients and recovery.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
Data parallel ranks hold replicas and process distinct batches, then synchronize gradients. DDP usually averages rank gradients; unequal valid counts require scaling local objectives so the result equals a global target mean. Sharding splits optimizer/parameter storage and adds communication/checkpoint complexity. Restore model, optimizer, RNG and data cursor per rank at one shared update boundary.

INPUTS AND RETURN
local_gradients: nonempty same-shape float tensors, each a rank-local valid-target mean gradient; local_counts: matching positive target counts. Return global count-weighted mean gradient without modifying inputs. This CPU exercise models the reduction contract; supplied distributed_lab.py verifies actual DDP synchronization.

WORKED EXAMPLE
Rank0 count2 gradient[1,3]; rank1 count4 gradient[4,0]. Global=(2*[1,3]+4*[4,0])/6=[3,1]. Naive rank mean=[2.5,1.5]. For R=2, scale local mean losses by2*2/6 and2*4/6 before DDP averages. Shuffled samples must be partitioned without unintended duplication; set epoch consistently.

BOUNDARIES
All ranks must enter collectives in the same order. A successful CPU reduction does not establish multi-device throughput or sharded recovery; run the actual two-process lab and a kill/resume experiment before scaling.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def distributed_gradient(local_gradients: list[torch.Tensor], local_counts: list[int]) -> torch.Tensor:
    # ===== START LEARNER CODE =====
    # STEP 1: Sum local_counts for global count and start a zeros_like first gradient.
    # STEP 2: Loop with zip(local_gradients,local_counts), add gradient*count.
    # STEP 3: Divide by global count and return; do not pretend this executes network collectives.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement distributed_gradient for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    g=[torch.tensor([1.,3.]),torch.tensor([4.,0.])];before=[x.clone() for x in g]
    close(distributed_gradient(g,[2,4]),torch.tensor([3.,1.]));close(distributed_gradient(g,[1,1]),torch.tensor([2.5,1.5]))
    close(distributed_gradient([g[0]],[7]),g[0])
    for a,b in zip(g,before):close(a,b)

if __name__=="__main__":
    run('09.05',run_checks)
