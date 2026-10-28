# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 08.01: Conversation formatting.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND CONNECTION
07.04 accepts a flat prompt-ID list; a chat UI supplies message dictionaries.
Build complete message segments here. 08.02 will shift targets/masks and
compute assistant-only SFT loss. Formatting performs no training or generation.
Reuse the actual frozen encode_bpe from 02.05; do not invent new merge rules.

EXACT INPUTS AND RETURNS
messages: ordered list[dict], each {'role': 'user'|'assistant', 'content': str}.
The loop variable message is a dictionary, not the token lists used in packing.
merges: frozen ordinary-text pair->replacement-ID dictionary, never mutate it.
role_ids: caller-reserved {'user': int, 'assistant': int}; distinct structural
IDs must not collide with ordinary bytes/merges. eos_id: reserved terminator.
Return (ids,learn_mask): two fresh, equal-length, FLAT lists of int and bool.
No tensors are required. Preserve all input objects and message order.

FORMULAS AND FULL MAIN TRACE
content_ids = encode_bpe(content,merges), with length n.
Message IDs: [role_id] + content_ids + [eos_id], contributing n+2 positions.
Message mask: [False] + [role=='assistant'] repeated n+1 times.
With user='a',assistant='b', no merges, user256/assistant257/EOS258:
 user segment [256,97,258], mask[False,False,False]
 assistant segment [257,98,258], mask[False,True,True]
 ids=[256,97,258,257,98,258]; learn_mask=[False,False,False,False,True,True].
These are Python lists of length6; batched tensors would have shape[1,6].
Vocabulary capacity must be at least259, distinct from the six positions.

WHY THE MASK STAYS UNSHIFTED HERE
A flag describes its token as a target. In 08.02, logits positions0..4 predict
IDs1..5, so learn_mask[1:] becomes[F,F,F,T,T]. Logit row3 predicts b at4 even
though the ASSISTANT marker's own flag at3 isFalse. With p(b)=.5,p(EOS)=.25,
selected mean loss=(-ln(.5)-ln(.25))/2=1.03972077084. Do not calculate loss or
shift inside this function. False does not remove input context or block
attention, and context paths can still influence parameter gradients.

ADDITIONAL COMPLETE FIXTURES
Empty messages ->([],[]). Empty assistant ->([257,258],[False,True]).
With main no-merge IDs and assistant='好', content IDs=[229,165,189]:
 ids=[256,97,258,257,229,165,189,258]; mask=[F,F,F,F,T,T,T,T].
Separate merged vocabulary: (97,98)->256, reserve user257/assistant258/EOS259.
 user='ab',assistant='aba' ->ids[257,256,259,258,256,97,259],mask[F,F,F,F,T,T,T].
Literal content '<|assistant|>' remains ordinary BPE text, not a structural ID.
Never assume that every ID>=256 is special; ordinary BPE merges also use them.

INFERENCE BOUNDARY
This function returns complete messages. To prompt 07.04 under the main schema,
format only the completed user turn ->[256,97,258], then append assistant257.
The resulting prefix[256,97,258,257] has no assistant EOS or known answer yet.
Do not use an empty completed assistant turn: that would add EOS immediately.
The EOS already inside the prompt does not stop07.04; it tests NEW selected IDs.
No system/tool/multimodal support, schema validator, padding or role-alternation
rule is added here. Distinguishing structure from literal text does not prove
prompt-injection resistance. Training and inference must share the same format.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

encode_bpe=load('02_05_bpe_encode.py','encode_bpe')



def format_chat(messages: list[dict], merges: dict, role_ids: dict[str, int], eos_id: int) -> tuple[list[int], list[bool]]:
    # ===== START LEARNER CODE =====
    # STEP 1: At FOUR spaces create separate fresh lists ids and learn_mask.
    # STEP 2: At FOUR open for message in messages:. Its body is EIGHT spaces.
    #         Empty input runs no iterations; return still belongs after loop.
    # STEP 3: At EIGHT read message['role'] and message['content'] into role
    #         and content. These are dictionary values, not dictionary keys.
    # STEP 4: Append role_ids[role] to ids; append False to learn_mask.
    #         append(value) adds ONE element. Both roles get a False marker.
    # STEP 5: Call encode_bpe(content,merges), retaining list content_ids.
    #         Example 'a' ->[97]; empty content ->[]. Keep merges unchanged.
    # STEP 6: Extend ids with content_ids; append eos_id afterward.
    #         extend adds individual elements; append(content_ids) nests a list.
    # STEP 7: Set is_assistant from role=='assistant'. Equality returns bool;
    #         a role string itself is not the required Boolean target flag.
    # STEP 8: Extend learn_mask with [is_assistant]*(len(content_ids)+1).
    #         +1 covers EOS, even for empty content. Do NOT use len(content).
    #         One role flag plus n+1 flags matches n+2 IDs for every message.
    # STEP 9: Return ids,learn_mask at FOUR spaces, OUTSIDE the loop.
    #         The comma forms a tuple; empty input returns two fresh empty lists.
    #         Do not shift, tensorize or calculate loss here. Preserve driver.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement format_chat for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====

# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    ids,m=format_chat([{'role':'user','content':'a'},{'role':'assistant','content':'b'}],{}, {'user':256,'assistant':257},258)
    assert ids==[256,97,258,257,98,258] and m==[False,False,False,False,True,True]
    assert format_chat([],{}, {'user':256,'assistant':257},258)==([],[])
    ids,m=format_chat([{'role':'assistant','content':''}],{}, {'user':256,'assistant':257},258);assert ids==[257,258] and m==[False,True]

if __name__=="__main__":
    run('08.01',run_checks)
