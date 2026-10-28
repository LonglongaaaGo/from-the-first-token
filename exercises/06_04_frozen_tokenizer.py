# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.04: Frozen tokenizer and special IDs.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse the completed02.05 encode_bpe helper and02.06 byte decoding concepts.
Tokenizer training learns rules; encoding applies existing rules only. Freeze the
vocabulary/merge order/preprocessing/special mapping across train, validation and
inference. This does not freeze trainable embedding vectors. No new environment,
BPE training, download, tensor conversion, padding or model update is needed.

FORMULA BEFORE CODE
ordinary = encode_bpe(text, merges); document_ids = ordinary concatenated [eos_id].
Each document length increases by1, including empty text. eos_id is reserved
outside ALL ordinary byte/merge IDs, so the ordinary body cannot contain it.
Model vocabulary capacity >= 1 + max(all allocated ordinary and reserved IDs).
A token ID is a row index, not vocabulary size, sequence length or vector width.

EXACT INPUTS AND RETURN
texts: ordered list[str], each element one ordinary Unicode document, possibly ''.
merges: dict[tuple[int,int],int], fixed learned ordinary pair-to-new-ID rules;
the supplied encoder applies them in their learned priority. Preserve this dict.
eos_id: int, a reserved structural end-of-sequence ID outside ordinary vocabulary.
Return a NEW list[list[int]]: one fresh inner list per document, in input order,
ordinary BPE IDs then exactly one eos_id. Lengths may differ; no tensor shape yet.
Preserve texts and merges. Do not clean/sort/deduplicate/flatten documents here.
Valid inputs are supplied; no extra validation branch is required.

COMPLETE ORIGINAL EXAMPLE
texts=['abab','ac',''], merges={(97,98):256}, eos_id=257.
This fixture has ONE rule. The Chapter02 two-rule example allocating257 is a
DIFFERENT tokenizer; if257 is learned there, EOS must use another reserved ID.
'abab': UTF-8 [97,98,97,98] -> ordinary [256,256] -> [256,256,257].
'ac': bytes [97,99], no learned rule -> [97,99,257].
'': bytes [], ordinary [] -> [257].
Return [[256,256,257],[97,99,257],[257]], preserving three document boundaries.
texts=[] instead returns []: no documents. texts=[''] returns [[257]]: one document.
Literal '<eos>' is five ordinary characters -> [60,101,111,115,62,257].
There is no special-string parser in encode_bpe. The last257 is appended explicitly.
'你' -> [228,189,160,257]. No new BPE rule is created for unseen content.

MODEL AND DECODING BOUNDARIES
Ordinary IDs0..256 plus EOS257 need at least258 embedding rows/output classes.
An illustrative width D=4 gives E shape(258,4); lookup [256,256,257] gives(3,4).
If future PAD258 is also allocated, capacity becomes at least259; D may stay4.
PAD/BOS/role IDs are extensions, not inserted by this function.
Existing decode_bpe has no byte entry for EOS257: interpret the boundary before
ordinary decoding, then join vocabulary byte chunks and UTF-8 decode. Do not call
bytes([256,256,257]); these are vocabulary IDs, not raw byte values0..255.
For z=[256,256,257], shifted X=[256,256], Y=[256,257]: learn to predict EOS.
EOS alone does not reset attention/cache or stop a generation loop. Those are
explicit downstream policies.06.05 packing allows cross-document context.

SYNTAX AND OWNERSHIP
The supplied encode_bpe(text,merges) returns a fresh list. A function body uses
FOUR spaces; a for body uses EIGHT. Each inner append adds one integer; outer
append adds one whole document list. Outer extend would flatten the documents.
append changes the list and returns None; do not assign that return or nest the
two append calls. Return at FOUR spaces after the loop, including zero iterations.
The numbered hints guide your implementation; keep the supplied driver intact.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

encode_bpe=load('02_05_bpe_encode.py','encode_bpe')



def encode_documents(texts: list[str], merges: dict[tuple[int, int], int], eos_id: int) -> list[list[int]]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create a NEW empty outer list, e.g. result = [].
    #         This holds document lists, not the input strings. Initial length0.
    # STEP 2: At FOUR spaces use `for text in texts:`. The colon starts a body
    #         at EIGHT spaces. text is one str; empty texts means zero iterations.
    # STEP 3: Inside the loop call encode_bpe(text, merges), storing the fresh
    #         list, e.g. ids. For 'abab', intermediate ids is [256,256]. Reuse
    #         the supplied helper; do not train rules or change merges.
    # STEP 4: Still at EIGHT spaces, append the INTEGER eos_id to ids. For the
    #         first document ids becomes [256,256,257]; empty text becomes [257].
    #         Do not look for the spelling '<eos>' or append it as a string.
    # STEP 5: At EIGHT spaces append the WHOLE ids list to result. After one
    #         iteration result is [[256,256,257]], not [256,256,257]. Outer
    #         append preserves grouping; outer extend would flatten the IDs.
    # STEP 6: AFTER the loop dedent to FOUR spaces and return result. Expected
    #         three-document value: [[256,256,257],[97,99,257],[257]]. The return
    #         must also execute for texts=[], giving [], not None.
    # STEP 7: Both append calls are separate statements returning None. Do not
    #         assign their returns or nest result.append(ids.append(eos_id)).
    #         Only the new inner/outer lists change; inputs stay unchanged.
    # STEP 8: Remove the placeholder raise; preserve the START/END markers and
    #         run_checks. No new if/continue/break or tensor operation is needed.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement encode_documents for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    m={(97,98):256};before=m.copy()
    assert encode_documents(['abab','ac',''],m,257)==[[256,256,257],[97,99,257],[257]] and m==before
    r=encode_documents(['<eos>'],m,257)[0];assert r[-1]==257 and len(r)>1 and 257 not in r[:-1]
    assert encode_documents([],m,257)==[]

if __name__=="__main__":
    run('06.04',run_checks)
