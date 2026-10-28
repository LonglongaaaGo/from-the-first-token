# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.23: fixed-parameter, position-weighted batch loss evaluation.

Run from the repository root after the walkthrough:
    python exercises/03_23_evaluate_batches.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
03.22's history mixes changing batches and changing parameter states.
Here hold parameters fixed across an explicit evaluation schedule, return
one mean over all scheduled supervision positions, and preserve all state.
Reuse make_next_token_batch, compute_batch_logits, and batch_cross_entropy.
The arithmetic fixture reuses familiar source IDs; it is not held-out data.

EXACT INPUTS
embedding_table: floating torch.Tensor [V,D], supplied token-vector table.
token_ids: fixed torch.long tensor [N], supplied evaluation source sequence.
output_weights: floating tensor [D,V], supplied vocabulary projection.
output_bias: floating tensor [V], supplied candidate biases.
batch_starts: nonempty list[list[int]], one nonempty start list per batch.
context_length: positive int T, supervised positions per sample.
No learning_rate: this function reads parameters and performs no updates.
Assume valid IDs, N>=T+1, starts 0<=start<=N-T-1, compatible dtype/device,
finite scores in the checks' supported numerical range, and ordinary tensor
parameters. Batch sizes may differ. Duplicate starts count each occurrence.
Input validation and an empty evaluation policy are not learner assignments.

RETURN AND STATE
Return a Python float: total position loss divided by total position count.
Preserve parameter values, storage, dtype/device, trainability, existing .grad
objects/values, source IDs, nested schedule, caller gradient mode, and random
state. Forward uses torch.no_grad(). Do not backward, clear gradients, call
a training helper, update parameters, or change requires_grad flags.

FORMULAS
    theta = (E,W,beta) remains fixed across all evaluation batches.
    n_k = B_k*T = target_ids.numel()
    mean_k = mean(-log true_target_probability) in batch k
    loss_sum = sum(n_k*mean_k); total_positions = sum(n_k)
    evaluation_loss = loss_sum / total_positions
Simple averaging of batch means is generally different for unequal n_k.
All positions are included equally; no padding, ignore IDs, or class weights.

COMPLETE EXAMPLE: fixed-eval-six-pairs-v1 (hand-picked)
    token_ids=[0,1,0,1,1,0,1]; batch_starts=[[0],[3,4]]; T=2
    E=[[1],[2]]; W=[[1,0]]; beta=[0,0]; D=1,V=2
    Batch A: X=[[0,1]], Y=[[1,0]], n=2, mean=0.7200948493
             position losses [1.3132616875,0.1269280110]
             loss_sum=1.4401896986, total_positions=2
    Batch B: X=[[1,1],[1,0]], Y=[[1,0],[0,1]], n=4, mean=0.9235114302
             position losses [2.1269280110,0.1269280110,0.1269280110,1.3132616875]
             loss_sum=5.1342354192, total_positions=6
    Return 5.1342354192/6 = 0.8557059032; every parameter stays unchanged.
    Unweighted mean of the two batch means is 0.8218031397, a different metric.

NEW SYNTAX AND INDENTATION
target_ids.numel() returns the number of elements, B_k*T, not a sum of IDs.
loss.item() extracts the mean tensor's Python float for numeric aggregation.
Use four spaces for totals, the with header, and final return; eight for the
for header inside with; twelve for the batch-building/score/loss/totals body.
Only one no-grad context surrounds the evaluation loop. It restores the
caller's previous gradient mode on exit and does not clear old .grad buffers.
The later nn.Module.eval() setting handles certain layer behaviors; it is
separate from gradient tracking. This exercise has plain tensor parameters.
"""

from __future__ import annotations

import copy
import importlib.util
from math import isclose, isfinite
from pathlib import Path

import torch
import torch.nn.functional as F


def evaluate_batches(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    batch_starts: list[list[int]],
    context_length: int,
) -> float:
    """Return the mean over all scheduled positions, with fixed parameter state."""
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: initialize loss_sum to 0.0 and total_positions to 0.
    # These are Python totals for position-loss sum and supervised-position count.
    #
    # STEP 2: Four spaces: begin with torch.no_grad():. Evaluation needs scores
    # and loss values, not an autograd graph. Preserve requires_grad and .grad.
    #
    # STEP 3: Eight spaces INSIDE the context: for start_positions in batch_starts:.
    # The nested loop body uses twelve spaces. Example visits [0], then [3,4].
    #
    # STEP 4: Twelve spaces: call make_next_token_batch(token_ids,start_positions,
    # context_length) once, unpacking input_ids,target_ids. First shapes [1,2];
    # second shapes [2,2]. Use existing next-token alignment without shifting again.
    #
    # STEP 5: Twelve spaces: call compute_batch_logits(embedding_table,input_ids,
    # output_weights,output_bias) once, saving logits. First [1,2,2], second [2,2,2].
    # All batches read the SAME unchanged parameter values; no training helper.
    #
    # STEP 6: Twelve spaces: call batch_cross_entropy(logits,target_ids) once,
    # saving loss. This scalar tensor is already the mean within this batch.
    # First mean 0.7200948493; second 0.9235114302 at the same initial state.
    #
    # STEP 7: Twelve spaces: save batch_positions from target_ids.numel().
    # Increase loss_sum by loss.item()*batch_positions, and total_positions
    # by batch_positions. Example totals: (1.4401896986,2), then (5.1342354192,6).
    # Multiplication reconstructs a position sum; do not average means equally.
    #
    # STEP 8: Back at four spaces, OUTSIDE loop/context: return loss_sum divided
    # by total_positions. Example Python float 0.8557059032. Nonempty evaluation
    # is assumed, so denominator is positive. No backward, reset, or update.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement evaluate_batches for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_module(filename, name):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


make_next_token_batch = _load_module("03_19_next_token_batch.py", "fixed_eval_builder").make_next_token_batch
_forward_module = _load_module("03_20_batched_logits_and_loss.py", "fixed_eval_forward_loss")
compute_batch_logits = _forward_module.compute_batch_logits
batch_cross_entropy = _forward_module.batch_cross_entropy


def _parameter(values, dtype=torch.float64, requires_grad=True):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def _fixture():
    return (_parameter([[1.], [2.]]), torch.tensor([0, 1, 0, 1, 1, 0, 1]),
            _parameter([[1., 0.]]), _parameter([0., 0.]), [[0], [3, 4]], 2)


def _reference(arguments):
    table, source, weights, bias, schedule, length = arguments
    with torch.no_grad():
        starts = [start for batch in schedule for start in batch]
        positions = torch.tensor(starts, dtype=torch.long, device=source.device)[:, None]
        positions = positions+torch.arange(length, device=source.device)[None, :]
        inputs, targets = source[positions], source[positions+1]
        logits = table[inputs] @ weights+bias
        return F.cross_entropy(logits.reshape(-1, bias.numel()), targets.reshape(-1)).item()


def _snapshot(tensors):
    return [(p.detach().clone(), p.data_ptr(), p.dtype, p.device, p.requires_grad,
             p.grad, None if p.grad is None else p.grad.clone()) for p in tensors]


def _preserved(tensors, snapshots):
    for p, (value, pointer, dtype, device, flag, grad, grad_value) in zip(tensors, snapshots):
        assert torch.equal(p.detach(), value) and p.data_ptr() == pointer
        assert p.dtype == dtype and p.device == device and p.requires_grad == flag
        assert p.grad is grad
        if grad is not None:
            assert torch.equal(p.grad, grad_value)


def _checked_evaluation(arguments):
    table, source, weights, bias, schedule, length = arguments
    parameters = (table, weights, bias)
    snapshot = _snapshot(parameters)
    source_value, source_pointer = source.clone(), source.data_ptr()
    schedule_value = copy.deepcopy(schedule)
    rng, mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    expected = _reference(arguments)
    actual = evaluate_batches(table, source, weights, bias, schedule, length)
    tolerance = 3e-6 if table.dtype == torch.float32 else 1e-10
    assert type(actual) is float and isfinite(actual)
    assert isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)
    _preserved(parameters, snapshot)
    assert torch.equal(source, source_value) and source.data_ptr() == source_pointer
    assert schedule == schedule_value
    assert torch.equal(torch.get_rng_state(), rng) and torch.is_grad_enabled() == mode
    return actual


def check_evaluation():
    fixture = _fixture()
    result = _checked_evaluation(fixture)
    assert abs(result-.8557059032013895) < 1e-10
    assert abs(result-.8218031397211916) > .03
    print("PASS: six-position mean weights unequal batch sizes and returns one Python float")

    values = []
    for schedule in ([[0], [3, 4]], [[0, 3, 4]], [[4], [3], [0]]):
        fixture = list(_fixture()); fixture[4] = schedule
        values.append(_checked_evaluation(fixture))
    assert all(isclose(value, values[0], rel_tol=0, abs_tol=1e-12) for value in values)
    fixture = list(_fixture()); fixture[4] = [[0, 0], [3]]
    _checked_evaluation(fixture)
    print("PASS: grouping/order invariance for the same positions and deliberate duplicate-start counting")

    global make_next_token_batch, compute_batch_logits, batch_cross_entropy
    original_builder, original_forward, original_loss = make_next_token_batch, compute_batch_logits, batch_cross_entropy
    fixture = _fixture(); parameters = (fixture[0], fixture[2], fixture[3])
    events, built, scored = [], [], []
    def guarded_builder(source, starts, length):
        index = len(built)
        assert events == [event for _ in range(index) for event in ("build", "scores", "loss")]
        assert not torch.is_grad_enabled() and source is fixture[1]
        assert starts is fixture[4][index] and length == fixture[5]
        result = original_builder(source, starts, length)
        events.append("build"); built.append(result)
        return result
    def guarded_forward(table, inputs, weights, bias):
        assert not torch.is_grad_enabled() and events[-1] == "build"
        assert all(actual is expected for actual, expected in zip((table, weights, bias), parameters))
        assert inputs is built[-1][0]
        result = original_forward(table, inputs, weights, bias)
        assert not result.requires_grad and result.grad_fn is None
        events.append("scores"); scored.append(result)
        return result
    def guarded_loss(scores, targets):
        assert not torch.is_grad_enabled() and events[-1] == "scores"
        assert scores is scored[-1] and targets is built[-1][1]
        result = original_loss(scores, targets)
        assert not result.requires_grad and result.grad_fn is None
        events.append("loss")
        return result
    make_next_token_batch, compute_batch_logits, batch_cross_entropy = guarded_builder, guarded_forward, guarded_loss
    try:
        _checked_evaluation(fixture)
        assert events == ["build", "scores", "loss", "build", "scores", "loss"]
    finally:
        make_next_token_batch, compute_batch_logits, batch_cross_entropy = original_builder, original_forward, original_loss
    print("PASS: each completed helper reused once per batch under no-grad without score/loss graphs")

    fixture = _fixture()
    for p in (fixture[0], fixture[2], fixture[3]):
        p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)+7.
    _checked_evaluation(fixture)
    print("PASS: fixed parameter values/storage/trainability and existing gradient objects/values preserved")

    fixture = _fixture()
    first = _checked_evaluation(fixture); second = _checked_evaluation(fixture)
    assert first == second
    with torch.no_grad():
        fixture[3][1] += .2
    third = _checked_evaluation(fixture)
    assert not isclose(first, third, rel_tol=0, abs_tol=1e-8)
    print("PASS: repeated fixed-state results agree and changed supplied parameters are reevaluated")

    single = (_parameter([[3., 2.]]), torch.tensor([0, 0, 0]),
              _parameter([[1.], [2.]]), _parameter([0.]), [[0]], 2)
    assert _checked_evaluation(single) == 0.
    extreme = (_parameter([[1000.], [-1000.]], torch.float32), torch.tensor([0, 1, 0]),
               _parameter([[1., -1.]], torch.float32), _parameter([0., 0.], torch.float32), [[0], [1]], 1)
    assert _checked_evaluation(extreme) == 2000.
    print("PASS: one-candidate zero loss and stable finite loss for extreme raw scores")

    V, D = 258, 2
    table = _parameter(torch.arange(V*D).reshape(V, D).div(1000).tolist(), torch.float32, False)
    weights = _parameter(torch.arange(D*V).reshape(D, V).div(2000).tolist(), torch.float32, False)
    bias = _parameter(torch.zeros(V).tolist(), torch.float32, False)
    source = torch.tensor([0, 99, 256, 99, 257, 99, 1, 99, 0, 99, 256, 99])[::2]
    assert not source.is_contiguous()
    arguments = (table, source, weights, bias, [[0], [1, 3]], 2)
    _checked_evaluation(arguments)
    with torch.no_grad():
        _checked_evaluation(arguments)
        assert not torch.is_grad_enabled()
    assert torch.is_grad_enabled()
    print("PASS: strided source, merged IDs, floating dtypes, untrainable tensors, and caller-mode restoration")
    print("Lesson 03.23 checks passed.")


if __name__ == "__main__":
    check_evaluation()
