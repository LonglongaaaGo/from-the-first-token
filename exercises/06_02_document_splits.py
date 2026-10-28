# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.02: Split by document before windows.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse UTF-8/hashlib from06.01 and next-token windows from earlier chapters.
Split whole documents before windowing, so overlapping passages are not placed
in train and validation. Content-based assignment keeps exact duplicate text on
the same side even under different IDs. This helper partitions dictionaries;
it does not fit a tokenizer, create windows, deduplicate or train a model.

FORMULA BEFORE CODE
payload = UTF8(seed) + one zero byte + UTF8(text)
hex_digest = SHA256(payload) displayed in hexadecimal
integer = int(hex_digest,16); u = integer / 2**256
Mathematically0<=u<1. Choose validation when u<validation_fraction, otherwise
train. Use the ENTIRE digest and strict<. ** is exponentiation; ^ is not.
Preserve fraction1 explicitly: floating division can round a maximal ratio to1.
Routing can use: fraction==1.0 OR u<fraction. At fraction0 all documents train.

EXACT INPUTS AND RETURN
documents: ordered list of dictionaries with unique id:str and text:str;
identical text with different IDs is allowed. validation_fraction: float in[0,1],
a threshold rather than a guaranteed count. seed: fixed public string; for this
zero-delimited protocol use a seed without embedded NUL. No new validation task.
Return (train,validation), two separate NEW lists, preserving relative input
order. Append ORIGINAL document dictionaries, not IDs/copies/manifest entries.
Do not mutate the input list or dictionaries, sort, drop duplicates or clean text.
This is a shallow partition: output entries still reference the input documents.
No random draw, RNG reseeding or device/tensor operation belongs in this helper.

COMPLETE ORIGINAL EXAMPLE
Input IDs/text: d1:'a', d2:'b', d9:'a'; seed='v1', fraction=.5.
Seed bytes[118,49], separator[0], text'a' bytes[97] -> payload[118,49,0,97].
The separator is b'\x00', NOT b'0' (byte48). Hashing this payload gives
'ad9799e8f62da609cdf1fe8e7dfaead172cf5825aefbbc1fabb4343ab6cd2db7'.
Parse all64 digits in base16 and divide by2**256 -> u=.6780945008996954.
This is not below.5: append d1 to train. For'b', u=.8114345740135137,
so append d2 to train. d9 has the same text as d1, hence exactly the same
payload/hash/score; append d9 to train. Return (original three documents,[]).
All three training at fraction.5 is CORRECT; counts need not match the fraction.
Extended illustration: append d3:'e',d4:'f'. Scores.27013664867714454 and
.08541676979045831 enter validation. Train IDs[d1,d2,d9], validation[d3,d4].
These values illustrate the protocol; do not hard-code assignments or hashes.

BOUNDARIES AND ORDER
Empty input returns two different empty lists. Fractions0 and1 force all train
and all validation respectively. Strict comparison puts equality at an interior
threshold in train. Reordering inputs preserves membership but changes relative
output order accordingly. Adding new text does not move existing members.
Increasing the fraction with seed/content fixed can only add validation members.
Seed is hashed as bytes, not passed to a random generator. IDs/indices never enter
this content hash. Do not replace exact text with strip/normalization here.
Exact duplicates stay together but are not deleted. Near duplicates, related
sources and later cleaning need a broader grouping policy;06.03 develops cleaning.
Fit BPE on train only, then encode both splits with the SAME frozen tokenizer.

SYNTAX AND OWNERSHIP
Function body: four spaces; for body: eight; if/else bodies inside it: twelve.
`for document in documents:` visits each original dict. `+` concatenates bytes;
all operands must be bytes. int(hex_digest,16) parses base16. `or` is true if
either condition is true and short-circuits. append modifies a list and returns
None: never reassign the list to append's return. Final return is AFTER the loop,
at four spaces. hashlib is already imported. Preserve the supplied check driver.
The numbered hints guide implementation; the learner writes the core.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def split_documents(documents: list[dict], validation_fraction: float, seed: str) -> tuple[list[dict], list[dict]]:
    # ===== START LEARNER CODE =====
    # REVIEW FOLLOW-UP: The current body passes the original driver. Its u<1
    # mathematical property can round to u==1.0 in float arithmetic. Preserve
    # the documented fraction1 contract with an explicit fraction==1 condition
    # OR the existing strict u<fraction comparison. See the paired explanation.
    # Your submitted computation below is preserved; adjust this branch yourself.
    # STEP 1: At FOUR spaces create two SEPARATE empty lists, train/validation.
    #         Encode seed as UTF-8 once outside the loop; it never changes here.
    # STEP 2: At four spaces use `for document in documents:`. Its repeated body
    #         has EIGHT spaces. Original iteration order is d1,d2,d9.
    # STEP 3: Encode document['text'] as UTF-8. Concatenate encoded seed,
    #         b'\x00', encoded text, in that order. For'a': [118,49,0,97].
    #         The zero separator is one byte; no ID/index enters the payload.
    # STEP 4: hashlib.sha256(payload).hexdigest() gives the FULL64 hex digits.
    #         Do not reuse06.01's unseeded content hash or a short prefix.
    # STEP 5: Parse the digest using int(hex_digest,16), not int(hex_digest).
    #         Base16 syntax: int('ff',16)==255. The real digest is longer.
    # STEP 6: Divide that integer by2**256. ** means power, ^ does not.
    #         For'a' the score is about.6780945009; for'b' about.8114345740.
    # STEP 7: At EIGHT spaces choose validation when validation_fraction==1.0
    #         OR u<validation_fraction; otherwise choose train. Python `or`
    #         handles the explicit endpoint even if a ratio rounds to1.0.
    #         At TWELVE spaces append(document) to the selected list. Append
    #         returns None. At fraction.5 original docs all train; extra'e'
    #         validates. Keep exact duplicates and original dictionary objects.
    # STEP 8: AFTER the loop dedent to FOUR spaces and return train,validation.
    #         Empty input must reach this return. Remove the placeholder raise,
    #         keep run_checks intact and never mutate/sort input documents.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement split_documents for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    docs=[{'id':'d1','text':'a'},{'id':'d2','text':'b'},{'id':'d9','text':'a'}]
    assert split_documents(docs,0.,'v1')==(docs,[]);assert split_documents(docs,1.,'v1')==([],docs)
    a,b=split_documents(docs,.5,'v1');assert len(a)+len(b)==3
    assert ('d1' in [d['id'] for d in b])==('d9' in [d['id'] for d in b])
    c,d=split_documents(list(reversed(docs)),.5,'v1');assert {x['id'] for x in b}=={x['id'] for x in d}

if __name__=="__main__":
    run('06.02',run_checks)
