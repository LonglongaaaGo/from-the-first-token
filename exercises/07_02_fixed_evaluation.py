# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 07.02: Fixed evaluation protocols.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND CONNECTION
06.03 records -> caller extracts texts ->06.04 token lists ->06.05 x/y/mask.
06.06 trains the passed model. This evaluator scores an existing model on a
fixed list of those aligned tensor tuples, without a training update. It returns
NLL sum/count so07.01-style reporting can normalize correctly. This exercise
calls no unfinished learner helper; the supplied driver gives tensors directly.
No tokenizer fitting, model construction, sampling, download or setup is needed.

FORMULAS BEFORE CODE
For batch k: logits=model(x), shape(B_k,T_k,V); ell=-ln p(correct y|context).
S_k = sum(ell selected by mask), K_k = mask.sum().
Global S=sum(S_k), K=sum(K_k)>0, loss=S/K. All logs are natural (nats).
Accumulate TOTAL selected losses/counts, not an equal mean of batch means/PPLs.
Targets are already shifted. A loss mask does not remove input context or a
vocabulary column. Use raw logits with F.cross_entropy(reduction='none').

EXACT INPUTS AND RETURN
model: existing nn.Module returning floating logits(B,T,V).
batches: nonempty list of(x,y,mask) TUPLES, not document dictionaries.
x/y: torch.long(B,T), aligned legal input/target IDs; mask: torch.bool(B,T).
Each batch's shapes/devices/model capacity are compatible; B/T can differ
between batches. ALL evaluated targets, including excluded entries, are legal
classes0..V-1. No negative ignore-label extension or validation task is assigned.
An individual batch may select zero targets; the full list selects at least one.
Return a NEW dict: 'nll_sum':float, 'tokens':int, 'loss':float.
Preserve input tensors, parameter values/identities and existing .grad objects
AND contents, including None. Restore the saved root model.training Boolean on
success or evaluation failure. No optimizer, backward, step or zero_grad here.

COMPLETE ORIGINAL DRIVER AND ANALYTIC BASELINE
Driver model: Sequential(Embedding(3,4),Dropout(.8),Linear(4,3)).
x=[[0,1],[1,2]], y=[[1,2],[2,0]], mask=[[True,False],[True,True]].
B2,T2,V3,K3. Selected slots:(0,0),(1,0),(1,1); target0 remains valid.
If every logit is zero, target probability1/3 -> each lossln3 -> S=3ln3,
K=3,loss=ln3≈1.098612289. Random driver weights do NOT promise zero logits.
Its check compares the same rows in one batch versus two one-row batches,
checks both initial modes, state_dict values and old gradient identities.

GUIDED SESSION WITH THE SAME INPUT LAYOUT
Separate fixed probabilities: input0 -> [.25,.5,.25]; input1/2 -> [.5,.25,.25].
Natural logs of these probabilities are compatible raw logits. This is not a
numeric prediction of the random original model. Position losses are
[[ln2,2ln2],[2ln2,ln2]]; mask selects [ln2,2ln2,ln2]. S=4ln2,K=3,
loss≈.924196241. Row-split running totals: (ln2,1), then(4ln2,3).
07.01 consumes these totals: PPL≈2.519842100; BPB additionally requires matching
source-byte coverage M. These synthetic IDs do not determine M or numeric BPB.
The paired lesson's guided trace and localized diagram explain every position.

SEPARATE NONUNIFORM EXAMPLE
Fixed two-class lookup logits W=[[-1,ln(1-exp(-1))],
                               [ln(1-exp(-2)),-2]]. Each row exponentiates to1.
A:x=[[0,0,0]],y=[[0,0,1]],mask=[[True,True,False]]: losses selected[1,1].
B:x=[[1,1,1],[1,0,0]],y=[[1,1,1],[1,1,1]],
  mask=[[True,True,True],[True,False,False]]: selected[2,2,2,2].
Running totals after A:S2,K2; after B:S10,K6; result loss5/3≈1.666667.
Mean batch losses(1+2)/2=1.5 is wrong. Empty selected batch adds0/0 to TOTALS;
do not divide within that empty batch or take an empty mean.

TWO CONTROLS AND CLEANUP
model.eval() changes relevant module behavior, such as ordinary dropout.
It does not disable autograd. with torch.no_grad() temporarily avoids ordinary
new reverse-mode graph recording; it does not switch off dropout, clear old
.grad or change requires_grad. Together they are this evaluator's controls,
not a universal guarantee against custom randomness or nondeterministic kernels.
Save old_mode BEFORE eval. try/finally restores model.train(old_mode), including
on a later forward/CE exception. Let the error propagate; don't return in finally.
with restores the caller's preceding grad context on exit. train(old_mode) sets
mode only; it performs no parameter update. A saved root flag does not recreate
a heterogeneous mixture of child-module flags. Ordinary model behavior is assumed;
this helper cannot sandbox arbitrary custom forward parameter/buffer mutations.

SHAPES, INDENTATION AND PROTOCOL
Logits(B,T,V)->(B*T,V); y(B,T)->(B*T,); CE(none)->(B*T,)->(B,T).
Select with mask ->(K_k,), sum ->scalar tensor, item ->Python float.
mask.sum().item() is a Python int count. Use reshape for non-contiguous inputs.
FOUR spaces: setup,try,finally,final return. EIGHT: eval,with,restore.
TWELVE: for x,y,mask in batches. SIXTEEN: forward/CE/selected totals.
Freeze dataset/tokenizer/contexts/coverage/special-target policy in the caller.
Regrouping preserves scores only with the same rows/prefixes/masks and a
row-independent eval model. Cutting/repacking context is a different protocol.
This evaluator sums whatever mask selects, possibly including EOS.07.01's BPB
requires its text-only selection plus matching source UTF-8 bytes, which this
interface cannot infer. Loss is not a capability or truthfulness benchmark.
The numbered hints guide the core; preserve the supplied driver.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def evaluate_fixed(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces save old_mode=model.training BEFORE changing it.
    #         Initialize a float NLL total (0.0) and integer target total (0)
    #         OUTSIDE the loop. These totals describe the whole batches list.
    # STEP 2: At FOUR spaces open `try:`. At EIGHT call model.eval().
    #         Keep the existing model; no optimizer or training clone is needed.
    # STEP 3: At EIGHT spaces open `with torch.no_grad():`. Its body starts
    #         at TWELVE. This controls graph recording, independently of eval.
    # STEP 4: At TWELVE use `for x, y, mask in batches:` to unpack each tuple.
    #         Its body is SIXTEEN spaces. Do not shift y or encode records here.
    # STEP 5: At SIXTEEN compute logits=model(x); read V=logits.shape[-1].
    #         Original shapes: (2,2,3), or (1,2,3) for each split row batch.
    # STEP 6: Reshape logits to(-1,V), y to(-1); compute F.cross_entropy with
    #         reduction='none'; restore losses with reshape_as(y). Original
    #         shapes:(4,3),(4,),then(2,2). Keep individual target losses.
    # STEP 7: At SIXTEEN select using mask, sum, then item(). Add that float
    #         to the NLL total with +=; do NOT average batch means. Separate
    #         unequal-count fixture: running S=2 after A,10 after B.
    #         Guided original-layout fixture: S=ln2 after row0,4ln2 after row1.
    # STEP 8: At SIXTEEN add mask.sum().item() to the integer count. In that
    #         fixture K=2 then6. An empty selection contributes sum0,count0.
    #         Guided original-layout fixture: K=1 after row0,3 after row1.
    # STEP 9: Finish ALL batches before division/return. Keep accumulation in
    #         the loop; with restores the previous grad context when it exits.
    #         Do not take a mean over an empty selection or return early.
    # STEP 10: At FOUR spaces write `finally:` aligned with try. At EIGHT
    #          restore model.train(old_mode). This runs on success OR failure;
    #          no unconditional train(),zero_grad or return belongs here.
    # STEP 11: AFTER finally, at FOUR spaces return a NEW dict with exactly
    #          'nll_sum':total float,'tokens':total int,'loss':total/count.
    #          Unequal example expects approximately10.,6,1.666667. An error
    #          propagates after cleanup, so this return runs only on success.
    # REVIEW FOLLOW-UP: Boolean-index the losses before summing. Multiplying
    # by a zero mask is unsafe when an excluded loss is inf: 0*inf is NaN.
    # A selected finite loss ln2 and an excluded inf should still report ln2.
    # The positive-total contract does not assign a mean loss to total_tokens=0.
    # STEP 12: Replace only the placeholder raise; preserve markers/signature,
    #          imports and supplied run_checks. No backward/step/new setup.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement evaluate_fixed for this exercise")
    # ===== END LEARNER CODE =====

    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    m=nn.Sequential(nn.Embedding(3,4),nn.Dropout(.8),nn.Linear(4,3));m.train()
    x=torch.tensor([[0,1],[1,2]]);y=torch.tensor([[1,2],[2,0]]);mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool)
    before=copy.deepcopy(m.state_dict())
    for p in m.parameters():p.grad=torch.ones_like(p)
    grads=[p.grad for p in m.parameters()]
    a=evaluate_fixed(m,[(x,y,mask)]);b=evaluate_fixed(m,[(x[:1],y[:1],mask[:1]),(x[1:],y[1:],mask[1:])])
    assert a['tokens']==3 and math.isclose(a['loss'],b['loss'],abs_tol=1e-6) and m.training
    assert all(p.grad is g for p,g in zip(m.parameters(),grads))
    for k,v in before.items():close(m.state_dict()[k],v)
    m.eval();evaluate_fixed(m,[(x,y,mask)]);assert not m.training

if __name__=="__main__":
    run('07.02',run_checks)
