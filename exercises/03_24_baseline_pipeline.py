# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.24: Connect the small baseline.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE
An integration experiment connects boundaries already learned. Freeze tokenization, build shifted windows, update the same tensors, then evaluate at the final state. Here the byte tokenizer has 256 entries; BPE can replace it later without changing the model contract.

INPUTS AND RETURN
text: nonempty ASCII fixture, str; schedules: nonempty lists of valid window starts; context_length: positive T; learning_rate: nonnegative float; updates: nonnegative integer schedule-repeat count. Actual optimizer steps equal updates*len(train_starts). Return dict with history (list[float]), eval_loss (float), continuation (str). Create fresh initial parameters per run; keep the same tensors for training, evaluation and generation within that run. Use the frozen byte vocabulary.

WORKED EXAMPLE
Use text="ababab", T=2, schedule=[[0,2]], updates=2. Bytes are [97,98,97,98,97,98]. One batch is X=[[97,98],[97,98]], Y=[[98,97],[98,97]]. Initial logits are zero, so each target probability is 1/256 and loss is ln(256)=5.5451774445. There are two updates and eight supervised positions. Generate four NEW IDs from a one-dimensional prompt tensor [97]. The generator returns all five IDs; remove the prompt IDs first, then .tolist() -> bytes -> UTF-8 decode(errors="replace"). These samples show wiring, not language ability; evaluation starts may overlap only in this arithmetic fixture.

BOUNDARIES
Do not fit a tokenizer on validation text. A low training loss on repeated "ab" is memorization, not evidence of a capable model.
The specified initialization has identical embedding rows. It raises a/b marginal probabilities but does not reliably learn alternation; greedy tie choices can depend on rounding.
No outer no_grad around training. Each new run starts fresh; the same tensors flow through all stages inside one run.
The numbered hints guide your implementation, not a supplied answer.
"""
from __future__ import annotations
import copy, hashlib, json, math, re, unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from _course_support import load, close, run

train_batches=load('03_22_batch_training_loop.py','train_batches')
evaluate_batches=load('03_23_evaluate_batches.py','evaluate_batches')
generate_token_ids=load('03_14_autoregressive_generation.py','generate_token_ids')

def initial_parameters():
    return (torch.ones(256,1,dtype=torch.float64,requires_grad=True),
            torch.zeros(1,256,dtype=torch.float64,requires_grad=True),
            torch.zeros(256,dtype=torch.float64,requires_grad=True))

def run_baseline(text: str, train_starts: list[list[int]], eval_starts: list[list[int]], context_length: int, learning_rate: float, updates: int) -> dict:
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: encode text with .encode("utf-8"), convert bytes
    # to a list of integers, then torch.tensor(...,dtype=torch.long).
    # Save token_ids: one-dimensional [N], example [97,98,97,98,97,98].
    # The byte IDs are 0..255; torch.long is the storage type for indices.
    #
    # STEP 2: Four spaces: call supplied initial_parameters() ONCE and unpack
    # embedding_table,output_weights,output_bias. Shapes [256,1],[1,256],[256].
    # These are new leaves per run. Keep these SAME objects for every stage.
    #
    # STEP 3: Four spaces: form training_schedule by train_starts * updates.
    # [[0,2]] * 2 gives [[0,2],[0,2]]. This repeats a LIST, not its ID values.
    # updates counts repetitions; actual batch updates are len(training_schedule).
    # For updates=0, this is []; train_batches already handles that case.
    #
    # STEP 4: Four spaces: call train_batches once, saving history.
    # Argument order: embedding_table,token_ids,output_weights,output_bias,
    # training_schedule,context_length,learning_rate. It builds/updates batches.
    # Reuse it; do not add another loop/backward/update or outer no_grad.
    # Example history approximately [5.5451774445,5.3469056593].
    #
    # STEP 5: Four spaces: call evaluate_batches on the resulting SAME tensors,
    # using token_ids,eval_starts,context_length. Save eval_loss (Python float).
    # Argument order: embedding_table,token_ids,output_weights,output_bias,
    # eval_starts,context_length. No learning_rate argument. Example 5.1473975533.
    # Do not reinitialize, update or clear gradients here.
    #
    # STEP 6: Four spaces: take context_ids from token_ids[:1], shape [1].
    # token_ids[0] would be a scalar; a Python list is not the required tensor.
    # Call generate_token_ids with embedding_table,context_ids,output_weights,
    # output_bias,4. Save generated_ids: prompt plus FOUR new IDs, shape [5].
    #
    # STEP 7: Four spaces: slice generated_ids after context_ids.numel() prompt
    # IDs, saving continuation_ids [4]. .tolist() turns IDs into list[int].
    # bytes(...) reconstructs byte values; .decode("utf-8",errors="replace")
    # produces continuation str. The decode policy concerns display, not training.
    # Direct bytes conversion is valid here because every ID is a base byte ID.
    # Merged BPE IDs would require vocabulary lookup before joining bytes.
    #
    # STEP 8: Four spaces: return a dict with keys "history", "eval_loss",
    # "continuation", mapped to the three saved results. Dictionary syntax:
    # {"name": value}. Return reports only; the core remains yours to write.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement run_baseline for this exercise")
    # ===== END LEARNER CODE =====


    
    


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def run_checks():
    a=run_baseline('ababab',[[0,2]],[[1,3]],2,.2,2)
    assert isinstance(a,dict) and {'history','eval_loss','continuation'} <= a.keys()
    assert isinstance(a['history'],list) and all(isinstance(v,float) for v in a['history'])
    assert len(a['history'])==2
    for actual,expected in zip(a['history'],[5.545177444479562,5.346905659303794]):
        assert math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10)
    assert isinstance(a['eval_loss'],float) and math.isclose(a['eval_loss'],5.147397553316883,rel_tol=1e-10,abs_tol=1e-10)
    assert isinstance(a['continuation'],str) and len(a['continuation'])==4
    print('PASS: pre-update history, final-state evaluation and continuation-only return')
    b=run_baseline('ababab',[[0]],[[1]],2,0.,0)
    assert b['history']==[] and math.isclose(b['eval_loss'],math.log(256),abs_tol=1e-8)
    assert isinstance(b['continuation'],str) and len(b['continuation'])==4
    print('PASS: zero schedule repetitions still evaluate and generate from fresh parameters')

    # Verify inter-stage contracts with real learner helpers, not an answer key.
    global initial_parameters,train_batches,evaluate_batches,generate_token_ids
    old=(initial_parameters,train_batches,evaluate_batches,generate_token_ids)
    calls=[];state={};source=torch.tensor(list(b'abbaabba'),dtype=torch.long)
    def same_parameters(table,weights,bias):
        assert table is state['parameters'][0]
        assert weights is state['parameters'][1] and bias is state['parameters'][2]
    def guarded_init():
        assert calls==[];calls.append('initialize')
        state['parameters']=old[0]()
        return state['parameters']
    def guarded_train(table,ids,weights,bias,schedule,length,rate):
        assert calls==['initialize'];same_parameters(table,weights,bias)
        assert ids.dtype==torch.long and torch.equal(ids,source)
        assert schedule==[[0],[1,4],[0],[1,4]] and length==2 and rate==.2
        assert torch.is_grad_enabled()
        calls.append('train');result=old[1](table,ids,weights,bias,schedule,length,rate)
        state['final_values']=[p.detach().clone() for p in state['parameters']]
        return result
    def guarded_eval(table,ids,weights,bias,schedule,length):
        assert calls==['initialize','train'];same_parameters(table,weights,bias)
        assert torch.equal(ids,source) and schedule==[[2,5]] and length==2
        assert all(torch.equal(p,v) for p,v in zip(state['parameters'],state['final_values']))
        calls.append('evaluate');return old[2](table,ids,weights,bias,schedule,length)
    def guarded_generate(table,context_ids,weights,bias,budget):
        assert calls==['initialize','train','evaluate'];same_parameters(table,weights,bias)
        assert context_ids.dtype==torch.long and torch.equal(context_ids,source[:1])
        assert budget==4 and all(torch.equal(p,v) for p,v in zip(state['parameters'],state['final_values']))
        calls.append('generate');return old[3](table,context_ids,weights,bias,budget)
    initial_parameters,train_batches,evaluate_batches,generate_token_ids=guarded_init,guarded_train,guarded_eval,guarded_generate
    try:
        train=[[0],[1,4]];evaluation=[[2,5]]
        c=run_baseline('abbaabba',train,evaluation,2,.2,2)
        assert calls==['initialize','train','evaluate','generate']
        assert len(c['history'])==4 and len(c['continuation'])==4
        assert train==[[0],[1,4]] and evaluation==[[2,5]]
        assert all(torch.equal(p,v) for p,v in zip(state['parameters'],state['final_values']))
    finally:
        initial_parameters,train_batches,evaluate_batches,generate_token_ids=old
    print('PASS: fresh initialization, repeated multi-batch schedule, real helper order and shared final parameters')

if __name__=="__main__":
    run('03.24',run_checks)
