# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 07.01: Token-aware language metrics.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Report accumulated next-token probability scores in explicit units. Reuse
cross-entropy, natural logs, UTF-8 bytes and frozen-tokenizer concepts. This
standalone scalar helper needs no model, evaluator, tensor shape, loop, data
download or new environment. math is already imported.

FORMULAS BEFORE CODE
p_i is the model probability of observed correct target i under allowed context.
S = nll_sum = sum(-ln(p_i)), measured in nats (natural-log units).
N = tokens: number of those scored ordinary-text targets, not vocabulary size.
M = utf8_bytes: original UTF-8 bytes covered by those same targets.
loss = S/N (nats/token); ppl = exp(loss) = 1/geometric_mean(p_i).
bits_per_byte = (S/ln(2))/M. Divide by ln(2) to convert nats to bits.
PPL is neither mean(1/p_i) nor 1/mean(p_i); average logs first.

EXACT INPUTS AND RETURN
nll_sum: float, finite nonnegative SUM of scored negative log-likelihoods.
tokens: int > 0, corresponding scored text-token count.
utf8_bytes: int > 0, corresponding original UTF-8 byte count, not characters,
prompt/padding storage or the byte size of tensor IDs. Inputs satisfy this
contract; no new validation/exception branches are required.
Return a NEW dict with exactly 'loss', 'ppl', 'bits_per_byte', all float values.
The helper trusts totals; it cannot detect a mismatched evaluation protocol.

COMPLETE ORIGINAL EXAMPLE
Self-authored scored text 'abcd', M=4, split into chunks b'ab', b'cd' (contents,
not numeric IDs). Constructed correct-target probabilities .5 and .125 give
product1/16, S=ln2+3ln2=4ln2, N=2. loss=2ln2≈1.386294361; ppl=4.
Total ideal bits=S/ln2=4; bits_per_byte=4/4=1. This is not a compressed file.
Second constructed setup: chunks b'a',b'b',b'c',b'd', each probability .5.
Same product1/16, S=4ln2 and M=4, but N=4: loss=ln2, ppl=2, BPB=1.
Actual retokenization does NOT guarantee equal S. At fixed S, larger N lowers
PPL; vocabulary size alone implies no direction or language-quality improvement.
Original zero-S call with N=3,M=8 returns loss0.,ppl1.,bits_per_byte0.

COVERAGE AND AGGREGATION
This lesson uses text-only scoring: exclude structural EOS/BOS/role and PAD
TARGETS from S,N consistently; keep only matching ordinary bytes in M. They may
still be conditioning inputs. Literal '<eos>' text is ordinary, unlike EOS ID.
A later evaluator sums what its mask selects; align that policy before using
its totals. This does not change earlier training masks/objectives.
For whole scored text, len(text.encode('utf-8')) counts bytes: '你a' has2 code
points but4 bytes [228,189,160,97]. For partial spans retain byte coverage.
Exclude unscored prompt/first-token bytes; do not double-count overlapping
window targets while dividing by unique bytes. Match context/reset/stride rules.
Accumulate S,N,M across batches, then normalize ONCE. Example A:S=ln2,N=M=1;
B:S=9ln2,N=M=3. Combined loss=2.5ln2,ppl=sqrt32≈5.656854,BPB=2.5.
Mean batch PPL=5 and token-weighted linear mean=6.5 are both wrong.
If cross-entropy gives mean L over N selected targets, recover S=L*N.

BOUNDARIES AND SYNTAX
All quantities are scalars; / is true division, math.log(2) is natural log,
math.exp(loss) computes e**loss. Use a dict literal with 'key': value entries.
Function statements use FOUR spaces; no loop or conditional is needed.
At S=0: loss=0,PPL=1,BPB=0. PPL can exceed vocabulary size and BPB can exceed8
when correct targets receive small probabilities. Lower held-out NLL alone does
not prove truthfulness or general reasoning.
exp(1000) overflows a float; real reports retain log-PPL=loss. Supplied fixtures
avoid overflow; do not clip or add a new exception-handling task here.
The numbered hints guide your implementation, not a supplied complete answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run





def language_metrics(nll_sum: float, tokens: int, utf8_bytes: int) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Read S=nll_sum, N=tokens, M=utf8_bytes as existing totals.
    #         Original call: S=4*math.log(2), N=2, M=4. No tensor/encoding loop.
    # STEP 2: At FOUR spaces use / to divide nll_sum by tokens; store loss.
    #         Expected intermediate loss=2*ln2≈1.386294361 (nats/token).
    # STEP 3: At FOUR spaces call math.exp on that MEAN, storing ppl≈4.
    #         exp(nll_sum) would give16: inverse product without the Nth root.
    # STEP 4: Convert TOTAL nats to bits by dividing by math.log(2).
    #         An optional local total_bits has value4 in the original fixture.
    # STEP 5: Divide total bits by utf8_bytes, storing bits_per_byte≈1.
    #         Equivalent denominator: (utf8_bytes * math.log(2)). Do not use
    #         loss as this numerator: loss was already divided by token count.
    # STEP 6: Return a NEW dictionary with exactly 'loss', 'ppl' and
    #         'bits_per_byte'. A dict field uses 'key': value; separate fields
    #         with commas. Use computed float values. return is at FOUR spaces
    #         after the calculations; brace-continuation lines can indent more.
    # STEP 7: Replace only the placeholder raise, which currently stops the
    #         function. Preserve signature/imports/markers and supplied driver.
    #         No new for/if/try or validation branch is needed for valid inputs.
    # STEP 8: Predict the other calls: same S,M with N=4 gives loss=ln2,
    #         ppl=2,BPB=1; zero S with N=3,M=8 gives 0.,1.,0. Then run checks.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement language_metrics for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=language_metrics(4*math.log(2),2,4);b=language_metrics(4*math.log(2),4,4)
    assert math.isclose(a['ppl'],4) and math.isclose(b['ppl'],2) and math.isclose(a['bits_per_byte'],1) and math.isclose(b['bits_per_byte'],1)
    assert language_metrics(0.,3,8)=={'loss':0.,'ppl':1.,'bits_per_byte':0.}

if __name__=="__main__":
    run('07.01',run_checks)
