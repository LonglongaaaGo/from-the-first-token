# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 10.01: Reproducible experiment records.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
An experiment is interpretable only when code, data and evaluation identity travel with the result. A config hash identifies exact portable settings; artifact checksums identify saved weights without storing weights in Git. Record failures and exclusions as well as successes. Separate measured results from the explanation and next hypothesis.

INPUTS AND RETURN
code_commit: full40-character hexadecimal revision; data_id,tokenizer_id,protocol_id: versioned portable strings; config,budget,results: JSON-compatible dicts; conclusion: evidence-limited text. Return deep snapshot with config_sha256 from canonical JSON plus all supplied fields.

WORKED EXAMPLE
Config {width:8,heads:2} and the same dict inserted in reversed key order produce identical canonical bytes and digest. Changing width to16 changes the digest. Record an actual full commit, data-v1/tokenizer-v1/protocol-v1, updates100 and supervised_tokens800. A snapshot must remain unchanged if the caller later edits config.

BOUNDARIES
Hashing a config does not remove sensitive fields. Use neutral artifact IDs; actual compute mappings, logs and weights belong in ignored locations.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def experiment_record(code_commit: str, data_id: str, tokenizer_id: str, config: dict, budget: dict, protocol_id: str, results: dict, conclusion: str) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Validate code_commit using re.fullmatch(r"[0-9a-fA-F]{40}",code_commit); raise ValueError for a short revision.
    # STEP 2: json.dumps(config,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8") gives canonical bytes.
    # STEP 3: Hash bytes, build named record and deepcopy before returning.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement experiment_record for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    cfg={'width':8,'heads':2};a=experiment_record('a'*40,'data-v1','tok-v1',cfg,{'updates':100},'eval-v1',{'loss':1.},'Toy measurement only')
    b=experiment_record('a'*40,'data-v1','tok-v1',{'heads':2,'width':8},{},'eval-v1',{},'')
    assert a['config_sha256']==b['config_sha256'] and len(a['config_sha256'])==64
    cfg['width']=16;assert a['config']['width']==8
    try:experiment_record('short','d','t',{}, {},'e',{},'')
    except ValueError:pass
    else:raise AssertionError('Short revisions must fail')

if __name__=="__main__":
    run('10.01',run_checks)
