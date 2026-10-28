# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 06.01: Data provenance and fingerprints.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Chapter05 ends at05.07; this is the next scheduled lesson. After recovering a
training state, identify the exact data version. Reuse Chapter01's UTF-8 bytes,
Python dictionaries/lists and a for loop. hashlib is a standard-library module;
the supplied common course imports use the existing environment. No new setup,
model training, tensor operations or data download is required.

FORMULA BEFORE CODE
raw = UTF8(text); byte_count = len(raw); fingerprint = SHA256(raw) in hex.
Both count and hash use the SAME encoded bytes. The count is not len(text) or
sum(raw). A SHA-256 digest has256 bits=32 bytes; hexdigest displays64 hex
characters. This256-bit output width differs from256 possible values of a byte.
A hash is a content fingerprint, not a reversible encoding or token sequence.

EXACT INPUTS AND RETURN
documents: ordered list of dicts with id:str and text:str; IDs are unique portable
labels. source_id and license_id: nonempty neutral metadata strings, preserved
unchanged. Return a NEW dict with exactly source_id, license_id, documents.
Each returned documents entry is a NEW dict with exactly:
  id: original document ID, str;
  sha256: full lowercase64-character fingerprint, str;
  bytes: encoded byte count, Python int.
The key 'bytes' is a field name, not a constructor call. Preserve input order and
all input objects/content; do not sort, deduplicate, strip or normalize text.
Only export the specified fields: omit raw text, paths and private mappings.
The supplied conditions require no new validation branches.

COMPLETE ORIGINAL EXAMPLE
Input documents=[{'id':'d1','text':'a'},{'id':'d2','text':'你'}],
source_id='toy-v1', license_id='self-authored'.
First document: 'a'.encode('utf-8') -> b'a', byte values[97], byte_count=1,
sha256='ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb'.
Append a fresh entry with id='d1', that full sha256, bytes=1; list length becomes1.
Second document: '你'.encode('utf-8') -> byte values[228,189,160], byte_count=3,
sha256='a0c7716669b5ded0d8051abfc232b828d489f7d889b2928eefcf43f0b8f495d6'.
Append a second fresh entry with id='d2', that full sha256, bytes=3; length becomes2.
Return both entries under documents with the unchanged outer source/license IDs.
All strings/hashes here explain the fixture; do not hard-code them in the core.

BOUNDARIES AND DATA IDENTITY
An empty input list returns documents=[] with outer metadata intact. One document
with empty text instead produces ONE entry with bytes=0 and a valid hash.
Adding a newline to 'a' produces bytes[97,10], count2 and a different fingerprint.
Two different IDs with identical text get the same content hash but remain two
entries. Changing order changes output order, not individual content hashes.
The content hash excludes document ID and source/license metadata. Do not hash
str(document). This helper hashes UTF-8 of received text, not original file bytes
lost through earlier decoding/cleaning. Version transformations separately.
This is not a corpus-wide/order hash, license verification, quality score or
anonymization. Record portable provenance and keep actual corpus data outside Git.

SYNTAX AND OWNERSHIP
A function body uses four spaces; a for body uses eight. Dictionary fields may
use another four spaces for readability. document['text'] reads a key without
removing it. .encode('utf-8') returns bytes. hashlib.sha256(raw) creates a hash
object; .hexdigest() returns its hex string. entries.append(entry) changes the
list and returns None: do not assign that return value to entries.
Dedent the final return to FOUR spaces, after the loop, so empty input works.
The numbered hints guide your implementation; the supplied driver stays intact.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def data_manifest(documents: list[dict], source_id: str, license_id: str) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create an empty result list, e.g. entries = [].
    #         Its initial length is0; never reuse the input documents list.
    # STEP 2: Use `for document in documents:` at four spaces. The colon starts
    #         the repeated body, which uses EIGHT spaces. Do not sort the input.
    # STEP 3: Inside the loop read document['text'] and encode it as 'utf-8'
    #         once, saving the bytes as e.g. raw. Example values: [97], then
    #         [228,189,160]. Encoding returns bytes, not a list or tensor.
    # STEP 4: Compute len(raw), saving a Python int such as byte_count.
    #         Expected1 then3. len(text) counts code points; sum(raw) adds values.
    # STEP 5: Compute hashlib.sha256(raw).hexdigest(). Keep all64 characters.
    #         Hash raw text bytes only, not ID, metadata or str(document).
    # STEP 6: Build a NEW dictionary INSIDE the loop with keys 'id', 'sha256',
    #         'bytes'; use original ID, computed hash and byte_count. Append it
    #         to entries. append returns None; do not reassign entries to it.
    # STEP 7: AFTER the loop, dedent to FOUR spaces and return a new outer dict
    #         with 'source_id', 'license_id', 'documents'. Preserve the supplied
    #         metadata and use entries as the documents value. Empty input must
    #         still return this dict because the return is outside the loop.
    # STEP 8: Remove the placeholder raise; keep run_checks unchanged. Never
    #         mutate inputs or add raw text/private paths to the result.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement data_manifest for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    docs=[{'id':'d1','text':'a'},{'id':'d2','text':'你'}];before=copy.deepcopy(docs)
    m=data_manifest(docs,'toy-v1','self-authored');assert docs==before and m['documents'][0]['bytes']==1 and m['documents'][1]['bytes']==3
    assert m['documents'][0]['sha256']==hashlib.sha256(b'a').hexdigest();assert m['source_id']=='toy-v1'
    assert data_manifest([],'toy-v1','self-authored')['documents']==[]
    assert data_manifest([{'id':'d1','text':'a\n'}],'toy-v1','self-authored')['documents'][0]['sha256']!=m['documents'][0]['sha256']

if __name__=="__main__":
    run('06.01',run_checks)
