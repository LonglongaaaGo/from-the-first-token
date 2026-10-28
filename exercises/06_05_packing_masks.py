# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.05: Packing and supervision masks.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Turn06.04's document token lists into rectangular next-token tensors. Reuse
Python lists/slices/for loops, shifted targets and tensor shapes/dtypes. No model
helper, new environment, download or tokenizer training is required.
This explicit first policy concatenates documents WITH existing EOS and permits
cross-document context inside each row. Pack train/validation separately.
EOS does not mask attention. Rows are independent contexts; no cache carries over.

REPRESENTATION HANDOFF BEFORE PACKING
Earlier06.03 documents are dict records; this function receives encoded lists.
Caller: records -> texts=[record['text'] for record in records]
       -> encode_documents(texts,merges,eos_id) -> list[list[int]] -> pack_tokens.
Here each loop variable document is ONE list[int], not a dict or raw text.
For token_documents=[[256,256,257],[97,99,257]], successive extend calls give
[256,256,257] then [256,256,257,97,99,257]. Variable names do not set types.
Direct extend({'id':'A','text':'abab'}) adds the KEYS ['id','text'], violating
this contract. Do not change the valid packing loop to document['text'].
Caller metadata stays in records/manifests; this function returns no ID mapping.
The original EOS4/PAD0 fixture below is separate from this BPE example.

FORMULAS BEFORE CODE
stream = concatenation of documents; N = len(stream); P = N-1 valid source pairs.
T = context_length; B = ceil(P/T) = (P+T-1)//T for positive integers.
For row b and slot t, j=b*T+t. If j<P: X[b,t]=stream[j], Y[b,t]=stream[j+1],
mask[b,t]=True. Otherwise fill X,Y with pad_id and mask with False.
For start=b*T, n=min(T,P-start); fill only the first n slots of that row.
Each source adjacent pair appears exactly once. Never create a final EOS->PAD pair.
Later loss = sum(valid position losses)/number of valid targets, not B*T.

EXACT INPUTS AND RETURN
documents: ordered list[list[int]] with EOS already included; total N>=2.
context_length: positive int T, number of input/target positions per row.
pad_id: int fill ID, normally reserved; synthetic tests allow a real matching ID.
Return (X,Y,mask), three NEW independent CPU tensors, each shape(B,T), dtypes
(torch.long,torch.long,torch.bool). Preserve input lists/order/occurrences.
B counts output rows, not documents. No embedding dimension exists here.
No added EOS, cleaning, deduplication or input-validation task is required.

COMPLETE ORIGINAL EXAMPLE
documents=[[1,2,4],[3,4]], T=3, pad_id=0. This toy version uses EOS4, not EOS257.
stream=[1,2,4,3,4]; N=5; P=4; B=(4+3-1)//3=2.
Pairs indexed j=0..3: 1->2,2->4,4->3,3->4. The final4 is only a target.
Row0: start0,n3; X prefix[1,2,4], Y prefix[2,4,3], all three flagsTrue.
Row1: start3,n1; X prefix[3], Y prefix[4], only first flagTrue.
Return X=[[1,2,4],[3,0,0]], Y=[[2,4,3],[4,0,0]],
mask=[[True,True,True],[True,False,False]]. Shapes all(2,3), valid count4.
4->3 crosses a document boundary under this stated policy. Predicting3 in row0
can use prefix[1,2,4]. Predicting4 in row1 uses[3], with no row0 history.
T=1 gives B=4 for the SAME two documents; T=8 gives one row and four pad slots.

POSITIONAL VALIDITY AND LOSS BOUNDARIES
Separate synthetic collision: documents=[[2,0,3]],T=3,pad0 gives X=[[2,0,0]],
Y=[[0,3,0]],mask=[[True,True,False]]. Both 2->0 and0->3 are real pairs.
Neither X!=pad_id nor Y!=pad_id is the validity rule. Derive it from n/positions.
Supervision mask(B,T) means True=include target loss. A causal attention mask
controls query/key visibility, often(T,T); boolean conventions depend on the API.
Not calculating one target's loss does not block attention to that input.
Chosen loss fixture [[.2,.4,.6],[.8,9,9]] has valid mean2/4=.5. Zeroing padding
then averaging all six gives2/6=1/3, still wrong. Select before averaging.
If computing CE on all cells, pad targets must be legal classes or explicitly
ignored. Negative ignore labels are not valid input embedding indices.

SYNTAX AND OWNERSHIP
Use FOUR spaces in the function, EIGHT inside each for. extend adds individual
IDs to a fresh stream; append would keep nested document lists. // is integer
floor division; the adjusted numerator computes ceiling. range(0,P,T) visits
starts0,3 in this fixture; enumerate adds row counters0,1. Slices exclude the
right endpoint. tensor[b,:n] selects a row prefix; assignment preserves padding
outside it. Allocate X and Y separately. Return AFTER the loop at FOUR spaces.
The numbered hints guide your implementation; the supplied driver stays intact.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def pack_tokens(documents: list[list[int]], context_length: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create stream=[]. Loop over documents at FOUR
    #         spaces; at EIGHT use stream.extend(document). Expect[1,2,4,3,4].
    #         Here document is list[int] from06.04, NOT a record dict.
    #         Caller extracts record['text'], then encodes BEFORE this call.
    #         extend adds integers; append would leave a list of lists.
    # STEP 2: AFTER flattening, at FOUR spaces compute N=len(stream), P=N-1,
    #         T=context_length. Expect N5,P4,T3; include the existing EOS IDs.
    # STEP 3: At FOUR spaces calculate B=(P+T-1)//T. Expect2, not P//T=1.
    #         // floors the adjusted numerator. B counts rows, not documents.
    # STEP 4: At FOUR spaces allocate X and Y SEPARATELY with torch.full:
    #         shape tuple(B,T), fill value pad_id, dtype=torch.long. Both begin
    #         [[0,0,0],[0,0,0]]. Never bind both names to one tensor object.
    # STEP 5: At FOUR spaces use torch.zeros for mask, same shape, bool dtype.
    #         Initially allFalse. Validity is not a test of token identity.
    # STEP 6: At FOUR spaces enumerate range(0,P,T), receiving row b and start.
    #         range excludes P; enumerate adds indices. Example pairs(b,start):
    #         (0,0),(1,3). The repeated row body uses EIGHT spaces.
    # STEP 7: Inside that loop compute n=min(T,P-start): first3, then1.
    #         This counts source pairs, not all remaining tokens.
    # STEP 8: At EIGHT spaces write converted long tensors into X[b,:n] and
    #         Y[b,:n]. Input slice: stream[start:start+n]. Shifted target slice:
    #         stream[start+1:start+n+1]. Last row receives[3] and[4]. Preserve
    #         both padded suffixes; don't pad the stream before shifting.
    # STEP 9: Still at EIGHT spaces set mask[b,:n]=True. Rows become
    #         [True,True,True] and[True,False,False]. Do not compare IDs to PAD.
    # STEP 10: AFTER the loop dedent to FOUR spaces; return X,Y,mask in order.
    #          Remove the placeholder raise; preserve markers, inputs and
    #          supplied run_checks. Model execution/loss comes in06.06.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement pack_tokens for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====




# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    x,y,m=pack_tokens([[1,2,4],[3,4]],3,0)
    assert torch.equal(x,torch.tensor([[1,2,4],[3,0,0]])) and torch.equal(y,torch.tensor([[2,4,3],[4,0,0]]))
    assert torch.equal(m,torch.tensor([[True,True,True],[True,False,False]]))
    x,y,m=pack_tokens([[0,1,2,3,4]],2,0);assert m.all() and m.sum()==4 and x.shape==(2,2)
    assert x.dtype==y.dtype==torch.long and m.dtype==torch.bool

if __name__=="__main__":
    run('06.05',run_checks)
