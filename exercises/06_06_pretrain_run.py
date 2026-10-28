# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.06: A complete pretraining experiment.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect already aligned x/y/mask to a model, masked objective and AdamW loop.
Reuse04.10's loop,05.05's optimizer concepts and06.05's valid-target mean.
The supplied driver loads the actual04.09 TinyDecoder; no unfinished06.04/06.05
helper is called because the driver supplies tensors directly. Use the existing
environment. Do not construct a new model, fit a tokenizer or fetch data here.
Pretraining learns next-token prediction with targets supplied by text shifting;
this fixed-batch primitive is one part of a complete recorded corpus experiment.

CONTINUOUS CONNECTION TO06.05 (SEPARATE FROM THE ORIGINAL DRIVER)
Clean records A='abab',B='ac' -> extracted texts -> fixed ab->256 BPE plusEOS257
-> token_documents=[[256,256,257],[97,99,257]]. Explicitly reservePAD258; model
capacity must cover every ID, so V>=259. With T3, actual pack_tokens returns
x=[[256,256,257],[97,99,258]], y=[[256,257,97],[99,257,258]],
mask=[[True,True,True],[True,True,False]]. Shapes(2,3),K5. pretrain receives
these tensors, not records/texts; caller constructs the compatible model.
At exactly zero logits(2,3,259), every class probability1/259; select five
ln259 losses and mean -> ln259=5.556828062. Zeroing then averaging six gives
4.630690051, the wrong denominator. This is an analytic baseline, not a model
measurement. Original driver below separately usesV2,K3; infer dimensions.
PAD class258 still participates in selected positions' softmax; only its padded
POSITION is excluded. No new tokenization, EOS insertion or shifting here.

FORMULAS BEFORE CODE
logits = model(x), shape(B,T,V); probabilities are conceptual softmax(logits).
Position loss ell[b,t] = -log(probability of already aligned y[b,t]).
K = mask.sum() > 0; L = sum(ell at mask-True positions)/K.
F.cross_entropy accepts RAW logits and reduction='none'; no manual softmax.
Flatten positions only: logits(B,T,V)->(B*T,V), y(B,T)->(B*T,).
Restore losses(B*T,)->(B,T), select mask, then mean. Never mean all padded slots.
Direct logit gradient at a selected position is (p-one_hot(y))/K; excluded
outputs have zero direct gradient. This does not block attention to their inputs.

EXACT INPUTS AND RETURN
model: existing registered causal nn.Module returning floating logits(B,T,V).
x,y: torch.long(B,T), aligned next-token IDs, all IDs legal for the vocabulary.
mask: torch.bool(B,T), at least oneTrue; may exclude real tokens, not only padding.
Model/input devices are compatible and T fits capacity. No new validation task.
updates: nonnegative int; learning_rate: positive float.
Return a NEW list[float] of exactly updates PRE-UPDATE losses. Modify the passed
model using one AdamW per call, lr=learning_rate, explicit weight_decay=.01;
use its ordinary betas(.9,.999),eps1e-8 defaults. Preserve inputs and parameter
objects. Set model.train() and leave training mode. Zero updates returns[]
without forward/backward/step; mode may change, weights do not.

COMPLETE ORIGINAL FIXTURE AND UNIFORM BASELINE
x=[[0,1,0],[1,0,1]], y=1-x=[[1,0,1],[0,1,0]],
mask=[[True,True,True],[False,False,False]]. B2,T3,V2,K3; y is ALREADY shifted.
The second row contains real token IDs, deliberately excluded; it is not PAD.
If all logits are[0,0], each probability is[.5,.5], CE=ln2=.693147...
Selected three sum to3ln2, then /3 givesln2. Zeroing excluded losses then
averaging six cells instead givesln2/2, the wrong denominator. Random TinyDecoder
initial logits need not be zero; uniform values are an analytic baseline only.
The actual driver uses seed26,TinyDecoder(V2,width8,heads2,layers1,max_length8),
80 updates,lr.03. It compares initial clones with changes only to excluded y;
pre-update histories/final parameters must match and last history loss<.15.
It also checks updates0. No language-quality or held-out claim follows.

SEPARATE ONE-UPDATE ARITHMETIC MODEL
A diagnostic W(2,2) directly supplies logits W[x]; rows input IDs, columns output
classes. W starts0; use the same x/y/mask. Gradient is[[1/3,-1/3],[-1/6,1/6]]:
input0 occurs twice,input1 once, all divided byK3. First AdamW update at lr.03
produces about[[-.03,+.03],[+.03,-.03]] with eps1e-8. Initial decay contributes0
because W0=0. Loss changes fromln2 toabout.663597114, but history[0] recordsln2.
This is not TinyDecoder's embedding or a replacement assignment model.

LIFETIME, LOGGING AND EXPERIMENT BOUNDARIES
Create optimizer once OUTSIDE the loop; its moments persist within this call.
zero_grad clears parameter gradients, not optimizer moments. A new call creates
new moments, so repeated calls are not exact resume from model weights alone.
model.train() sets mode; it does not enable autograd or perform an update.
Use ordinary grad-enabled execution. item() is for logging; keep tensor loss for
backward. history[79] precedes update80; model ends at parameters after80 updates.
The function returns no optimizer/checkpoint/evaluation/sampler. The later lab
also needs07.02 evaluation and07.04 inference dependencies. Its model.pt saves
weights/config/update count, not the full optimizer/RNG/sampler resume bundle.
A real experiment records full code commit,data/tokenizer versions,portable
config,budget,held-out protocol,artifacts/checksums and conclusions. Large data
and weights stay outside Git; use no personal infrastructure fields in records.
The numbered hints guide the core; leave the supplied driver intact.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

TinyDecoder=load('04_09_decoder_model.py','TinyDecoder')



def pretrain(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, updates: int, learning_rate: float) -> list[float]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create ONE torch.optim.AdamW using
    #         model.parameters(),lr=learning_rate,weight_decay=.01. It belongs
    #         outside the update loop. Do not create/replace the passed model.
    # STEP 2: At FOUR spaces call model.train(); create history=[]. The mode
    #         changes in place. History will hold Python floats, not tensors.
    # STEP 3: At FOUR spaces use `for _ in range(updates):`. _ is an unused
    #         counter; range(0) is empty. Per-update body uses EIGHT spaces.
    # STEP 4: Inside the loop call optimizer.zero_grad(set_to_none=True).
    #         Clear old .grad without resetting parameters/optimizer moments.
    # STEP 5: At EIGHT spaces call model(x), saving logits(B,T,V); read V from
    #         logits.shape[-1]. Original fixture gives(2,3,2). Do not shift y
    #         again, detach the graph, softmax, sample or recreate the model.
    # STEP 6: Reshape logits to(-1,V) and y to(-1). Call F.cross_entropy with
    #         reduction='none'; reshape returned losses using reshape_as(y).
    #         Expected original shapes:(6,2),(6,),then losses(2,3).
    #         BPE connection instead has(6,259),(6,),then losses(2,3).
    #         -1 infers position count; reshape_as(y) uses y's SHAPE only.
    #         Select only AFTER individual losses exist; keep class columns.
    # STEP 7: Select losses using mask, then .mean() for scalar tensor loss.
    #         Uniform example selects three ln2 values, givingln2, notln2/2.
    #         BPE connection selects five ln259 values, givingln259.
    #         Read valid count from selection, not a hard-coded3 or5.
    #         Do not derive mask from token0 or mean all allocated positions.
    # STEP 8: At EIGHT spaces append loss.item() to history BEFORE updating.
    #         item gives a float; retain tensor loss for backward. append
    #         returns None; do not overwrite history with that return value.
    # STEP 9: Still at EIGHT spaces call loss.backward(), then optimizer.step().
    #         backward computes gradients; step changes existing parameters.
    #         No no_grad around training; no optimizer reconstruction per step.
    # STEP 10: AFTER the loop dedent to FOUR spaces; return history. updates80
    #          gives80 PRE-UPDATE floats,0 gives[]. Do not append a final extra
    #          evaluation. Remove raise; preserve markers and supplied checks.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement pretrain for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    torch.manual_seed(26);m=TinyDecoder(2,8,2,1,8);x=torch.tensor([[0,1,0],[1,0,1]]);y=1-x;mask=torch.tensor([[1,1,1],[0,0,0]],dtype=torch.bool)
    a=copy.deepcopy(m);b=copy.deepcopy(m);y2=y.clone();y2[~mask]=x[~mask]
    h=pretrain(a,x,y,mask,80,.03);assert len(h)==80 and h[-1]<.15
    k=pretrain(b,x,y2,mask,80,.03);close(torch.tensor(h),torch.tensor(k))
    for p,q in zip(a.parameters(),b.parameters()):close(p,q)
    assert pretrain(m,x,y,mask,0,.03)==[]

if __name__=="__main__":
    run('06.06',run_checks)
