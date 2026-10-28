# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.03: Cleaning and deduplication.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse Unicode/text identity from06.01-06.02, dictionaries/lists and for loops.
Specify a prose cleaning policy, then keep the first representative of each
remaining canonical string. This helper uses Python's unicodedata and containers;
no model, tokenizer, RNG or training operation is needed. Unicode normalization,
whitespace collapse and duplicate filtering are three separate operations.

FORMULA BEFORE CODE
normalized = NFC(text)
cleaned = single_space.join(normalized.split())
keep iff len(cleaned)>=min_characters AND cleaned is not in retained seen strings.
If kept: add cleaned to seen; append a NEW {'id':original_id,'text':cleaned}.
Otherwise: skip this document and keep scanning. Measure length AFTER cleaning.
An exact-threshold length passes. A string's len counts Unicode code points,
not UTF-8 bytes, BPE tokens or general visible grapheme clusters.

EXACT INPUTS AND RETURN
documents: ordered list of dictionaries containing id:str and text:str;
IDs are stable document labels. min_characters: positive Python int, default1.
Return a NEW list of NEW dicts containing exactly id and text. Preserve original
inputs and extra input metadata; copy only the two requested fields to output.
Duplicate identity uses cleaned TEXT, not ID. Preserve first-retained input order
and first ID. Output dictionaries must not alias input dictionaries; this differs
from06.02's shallow partition. Strings are immutable; no deepcopy is needed.

COMPLETE ORIGINAL EXAMPLE
Input IDs0,1,2,3 have texts '  cafe\u0301  ', 'café', 'a\n\tb', ' '.
The first decomposed accent uses e U+0065 plus U+0301. NFC composes it into
é U+00E9 but leaves edge spaces: '  café  '. split() gives ['café']; joining
with one space gives 'café', length4 code points and5 UTF-8 bytes.
At minimum1:
  ID0: cleaned='café', len4, seen=set() -> keep; seen={'café'}.
  ID1: cleaned='café', len4, already seen -> skip; first ID0 remains.
  ID2: split ['a','b'], cleaned='a b', len3, new -> keep; seen adds 'a b'.
  ID3: split [], cleaned='', len0 -> too short, skip without adding to seen.
Return [{'id':'0','text':'café'},{'id':'2','text':'a b'}].
At minimum4 return only [{'id':'0','text':'café'}]; length exactly4 passes.
Empty input or all-whitespace input gives[]. Do not hard-code these records.

POLICY BOUNDARIES
NFC preserves case/accents here: café, cafe and CAFÉ stay different. Do not add
lowercasing, accent removal or NFKC. Use no-argument split(), not split(' '):
whitespace runs collapse, edges vanish; ' '.join inserts one ASCII space.
This policy can destroy code indentation, tables and paragraph layout. It is not
a universal quality filter. Exact canonical equality does not find all near
copies or paraphrases. Version the policy and audit representatives/removals.
For a new split, establish canonical content groups before partitioning. Already
frozen splits require a documented cross-split conflict policy, not silently
changing held-out membership. Two per-split seen sets cannot detect cross-overlap.
Keep raw/cleaned fingerprints and data artifacts separately; this return value
is not a complete provenance/removal manifest.

SYNTAX AND OWNERSHIP
Function body uses four spaces, for body eight, conditional body twelve.
seen=set() creates an EMPTY SET; {} creates an empty dict. Create seen/result
outside the loop. `cleaned in seen` tests membership. seen.add changes the set;
result.append changes the list; both return None, so do not rebind to that return.
`continue` skips only the current iteration; break would stop scanning later
valid documents. Final return is after the loop at four spaces. unicodedata is
already imported. Preserve the supplied driver; the learner writes the core.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def clean_deduplicate(documents: list[dict], min_characters: int = 1) -> list[dict]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create seen=set() and result=[]. Empty {} is a
    #         dictionary, not a set. Keep both containers OUTSIDE the loop.
    # STEP 2: At four spaces use `for document in documents:`. Its body uses
    #         EIGHT spaces. Original IDs arrive in order '0','1','2','3'.
    # STEP 3: Read document['text']; call unicodedata.normalize('NFC',text).
    #         Store the normalized STRING. First row becomes '  café  ';
    #         the combining accent composes, edge spaces still remain.
    # STEP 4: Use normalized.split() without a separator, then ' '.join(...).
    #         Store cleaned as a STRING. Examples: 'café', 'a b', ''.
    # STEP 5: At EIGHT spaces test len(cleaned)<min_characters; at TWELVE
    #         continue skips this document. Count cleaned code points, not raw
    #         characters/UTF-8 bytes; length equal to the minimum must pass.
    # STEP 6: Back at eight spaces test `cleaned in seen`; a twelve-space
    #         continue skips duplicates. ID1 is skipped; ID2 is still reached.
    # STEP 7: Back at eight spaces seen.add(cleaned), then append a NEW dict
    #         with exactly 'id' from document['id'] and 'text' as cleaned.
    #         Do not edit/append the input dictionary. add/append return None;
    #         never assign that return value back to seen/result.
    # STEP 8: AFTER the loop dedent to FOUR spaces and return result. Empty
    #         input must reach the return. Remove the placeholder raise and
    #         preserve run_checks. Do not sort/lowercase/change the policy.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement clean_deduplicate for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    docs=[{'id':str(i),'text':t} for i,t in enumerate(('  café  ','café','a\n\tb',' '))];before=copy.deepcopy(docs)
    r=clean_deduplicate(docs);assert r==[{'id':'0','text':'café'},{'id':'2','text':'a b'}] and docs==before
    assert clean_deduplicate(docs,4)==[{'id':'0','text':'café'}];assert clean_deduplicate([])==[]

if __name__=="__main__":
    run('06.03',run_checks)
