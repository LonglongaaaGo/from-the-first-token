# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.20: batched lookup/projection and aligned mean token loss.

Run in the environment described in README.md after the walkthrough:
    python exercises/03_20_batched_logits_and_loss.py --check logits
    python exercises/03_20_batched_logits_and_loss.py --check loss
    python exercises/03_20_batched_logits_and_loss.py --check all

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect the [B,T] window shapes from 03.19 to completed 03.09 compute_logits
and 03.10 cross_entropy_loss. Stage A returns [B,T,V] raw scores; Stage B
matches each score row to one target and returns one connected scalar.
Keep the earlier helpers' two-dimensional contracts by flattening positions
in the SAME logical order. This remains the independent lookup-plus-head
baseline. No attention, mask, backward, parameter update, or sampling here.

EXACT INPUTS
embedding_table: floating torch.Tensor [V,D], token vectors.
input_ids: torch.long tensor [B,T], observed input IDs.
output_weights: floating tensor [D,V], vocabulary projection.
output_bias: floating tensor [V], candidate biases.
logits: floating tensor [B,T,V], Stage A output or supplied Stage B scores.
target_ids: torch.long tensor [B,T], already shifted correct next IDs.
Assume B,T,V,D>0, valid IDs, compatible devices/dtypes, finite scores and
the supported numerical range used in the checks. Validation is not assigned.

RETURNS AND STATE
compute_batch_logits returns connected raw scores [B,T,V].
batch_cross_entropy returns a scalar tensor, mean over B*T true-target losses.
Preserve all inputs, parameter values/storage/trainability/stored gradients,
caller gradient mode, and global random state. Preserve autograd when enabled.
Do not call backward, clear gradients, detach, use no_grad, update parameters,
or extract a Python float inside either learner function.
No ignored positions, padding weights, class weights, or label smoothing.

FORMULAS
    H[b,t,:] = E[input_ids[b,t]], shape [B,T,D]
    Z[b,t,v] = sum(H[b,t,d]*W[d,v] over d) + output_bias[v]
    q = b*T+t; flat_logits[q,v] = Z[b,t,v]
    flat_targets[q] = target_ids[b,t]
    loss = -(1/(B*T))*sum(log_softmax(Z[b,t,:])[target_ids[b,t]])
Only B and T merge; D or V stays as the final feature/candidate dimension.
Targets are already shifted by data construction; do not shift them again.

COMPLETE WORKED EXAMPLE: batch-three-id-v1 (hand-picked, not trained)
Source IDs [0,1,2,0,1], starts [0,2], T=2 give:
    input_ids = [[0,1],[2,0]]; target_ids = [[1,2],[0,1]]
    E = [[1,2,3],[2,1,3],[3,2,1]]; W = identity(3); bias = [0,0,0]
    logits = [[[1,2,3],[2,1,3]], [[3,2,1],[1,2,3]]], shape [2,2,3]
Flat position order is (0,0),(0,1),(1,0),(1,1):
    flat_logits = [[1,2,3],[2,1,3],[3,2,1],[1,2,3]], shape [4,3]
    flat_targets = [1,2,0,1], shape [4]
Each score row has maximum 3 and shifted exponential sum
    1 + exp(-1) + exp(-2) = 1.5032147244
True-target probabilities are approximately [0.2447284711,0.6652409558,
0.6652409558,0.2447284711]. Their negative log-probabilities are
[1.4076059644,0.4076059644,0.4076059644,1.4076059644].
Mean loss = 3.6304238578 / 4 = 0.9076059644. Use true targets, not argmax.

NEW SYNTAX BEFORE IMPLEMENTATION
batch_size, context_length = input_ids.shape unpacks its two dimensions.
embedding_table[input_ids] looks up one D-vector per ID and keeps [B,T].
.reshape(-1,D) infers B*T while preserving the final D dimension.
Only one reshape dimension can be -1. Reshape preserves logical element
order; it may use a view or copy, and it keeps autograd connected.
Do not swap axes of scores or targets independently before flattening.
Every learner statement uses four spaces; neither stage requires a loop.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_batch_logits(
    embedding_table: torch.Tensor,
    input_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Return connected [B,T,V] scores through the completed 2D projection."""
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: unpack input_ids.shape into batch_size and
    # context_length. Example B=2,T=2. These are dimensions, not token values.
    #
    # STEP 2: Four spaces: obtain hidden_vectors by embedding_table[input_ids].
    # Example shape [2,2,3]; repeated ID 0 selects the same table row twice.
    #
    # STEP 3: Four spaces: obtain embedding_dim from embedding_table.shape[1].
    # Reshape hidden_vectors to (-1,embedding_dim), saving flat_hidden.
    # Example [4,3], rows in order (0,0),(0,1),(1,0),(1,1).
    #
    # STEP 4: Four spaces: call compute_logits(flat_hidden, output_weights,
    # output_bias), saving flat_logits. Reuse the completed helper once.
    # Example [4,3]. Keep the graph and parameter state unchanged.
    #
    # STEP 5: Four spaces: reshape flat_logits to (batch_size,context_length,-1),
    # saving logits. The inferred last dimension is V, giving [2,2,3].
    #
    # STEP 6: Four spaces: return logits as raw connected scores.
    # Probabilities and sampled IDs are not this function's returned value.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement compute_batch_logits for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE A =====


def batch_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return connected scalar mean over all B*T aligned true targets."""
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: obtain vocab_size from logits.shape[-1].
    # Example V=3. Keep every position's three candidate scores together.
    #
    # STEP 2: Four spaces: reshape logits to (-1,vocab_size), storing flat_logits.
    # Example [4,3]. Merge B and T, preserving their logical element order.
    #
    # STEP 3: Four spaces: reshape target_ids to (-1), storing flat_targets.
    # Example [1,2,0,1], shape [4]. Do not transpose or shift the targets.
    #
    # STEP 4: Four spaces: call cross_entropy_loss(flat_logits,flat_targets),
    # saving loss. The completed helper uses stable true-target log-probabilities
    # and already averages its four rows. Do not divide by B, T, or V again.
    #
    # STEP 5: Four spaces: return the scalar loss tensor, preserving its graph.
    # Example approximately 0.9076059644. Backward belongs to a later caller.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement batch_cross_entropy for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE B =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_module(filename, module_name):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


compute_logits = _load_module("03_09_logits_and_softmax.py", "completed_batched_projection").compute_logits
cross_entropy_loss = _load_module("03_10_cross_entropy_loss.py", "completed_batched_cross_entropy").cross_entropy_loss


def _float(values, *, requires_grad=False, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def _fixture():
    table = _float([[1., 2., 3.], [2., 1., 3.], [3., 2., 1.]], requires_grad=True)
    weights = torch.eye(3, dtype=torch.float64, requires_grad=True)
    bias = _float([0., 0., 0.], requires_grad=True)
    inputs, targets = torch.tensor([[0, 1], [2, 0]]), torch.tensor([[1, 2], [0, 1]])
    return table, inputs, weights, bias, targets


def _snapshot(tensors):
    return [(x.detach().clone(), x.data_ptr(), x.requires_grad, x.grad,
             None if x.grad is None else x.grad.clone()) for x in tensors]


def _preserved(tensors, snapshot):
    for x, (value, pointer, flag, grad, grad_value) in zip(tensors, snapshot):
        assert torch.equal(x.detach(), value) and x.data_ptr() == pointer and x.requires_grad == flag
        assert x.grad is grad
        if grad_value is not None:
            assert torch.equal(x.grad, grad_value)


def _close(actual, expected):
    tolerance = 1e-5 if actual.dtype == torch.float32 else 1e-10
    assert torch.allclose(actual, expected, rtol=tolerance, atol=tolerance), (actual, expected)


def _check_logits(table, inputs, weights, bias):
    tensors = (table, inputs, weights, bias)
    snapshot, mode, rng = _snapshot(tensors), torch.is_grad_enabled(), torch.get_rng_state().clone()
    actual = compute_batch_logits(table, inputs, weights, bias)
    expected = torch.einsum("btd,dv->btv", table[inputs], weights) + bias
    assert actual.shape == (*inputs.shape, bias.numel())
    assert actual.dtype == table.dtype and actual.device == table.device
    assert actual.requires_grad == (mode and any(p.requires_grad for p in (table, weights, bias)))
    _close(actual, expected)
    _preserved(tensors, snapshot)
    assert torch.is_grad_enabled() == mode and torch.equal(torch.get_rng_state(), rng)
    return actual


def _check_loss(scores, targets):
    snapshot, mode, rng = _snapshot((scores, targets)), torch.is_grad_enabled(), torch.get_rng_state().clone()
    actual = batch_cross_entropy(scores, targets)
    expected = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1))
    assert isinstance(actual, torch.Tensor) and actual.shape == torch.Size([])
    assert actual.dtype == scores.dtype and actual.device == scores.device and torch.isfinite(actual)
    assert actual.requires_grad == (mode and scores.requires_grad)
    _close(actual, expected)
    _preserved((scores, targets), snapshot)
    assert torch.is_grad_enabled() == mode and torch.equal(torch.get_rng_state(), rng)
    return actual


def check_logits():
    global compute_logits
    table, inputs, weights, bias, _ = _fixture()
    actual = _check_logits(table, inputs, weights, bias)
    _close(actual, _float([[[1., 2., 3.], [2., 1., 3.]], [[3., 2., 1.], [1., 2., 3.]]]))
    print("PASS: worked [B,T] lookup and restored [2,2,3] raw scores")

    table2 = _float([[0., 1.], [2., -1.], [1., 3.], [-2., 0.], [3., 1.]], requires_grad=True)
    weights2 = _float([[1., 0., -1., 2., 0.5], [0., 2., 1., -1., 0.]], requires_grad=True)
    bias2 = _float([0.1, -0.2, 0.3, 0., 1.], requires_grad=True)
    for ids in ([[0]], [[1, 2, 0]], [[0, 4], [3, 2], [1, 0]]):
        _check_logits(table2, torch.tensor(ids), weights2, bias2)
    wide_table = torch.arange(258*2, dtype=torch.float64).reshape(258, 2)
    _check_logits(wide_table, torch.tensor([[256, 257]]), torch.zeros(2, 258, dtype=torch.float64),
                  torch.arange(258, dtype=torch.float64))
    print("PASS: nonsquare batch shapes, D different from V, bias, singleton axes, and merged IDs")

    real_projection, calls = compute_logits, []
    def projection_spy(vectors, supplied_weights, supplied_bias):
        assert vectors.shape == (4, 3) and supplied_weights is weights and supplied_bias is bias
        _close(vectors, table[inputs].reshape(4, 3))
        calls.append(vectors)
        return real_projection(vectors, supplied_weights, supplied_bias)
    compute_logits = projection_spy
    try:
        _check_logits(table, inputs, weights, bias)
    finally:
        compute_logits = real_projection
    assert len(calls) == 1
    strided = torch.tensor([[0, 9, 1, 9], [2, 9, 0, 9]])[:, ::2]
    _check_logits(table, strided, weights, bias)
    for p in (table, weights, bias):
        p.grad = torch.full_like(p, 99.)
    with torch.no_grad():
        _check_logits(table, inputs, weights, bias)
    _check_logits(table.detach().float().requires_grad_(), inputs,
                  weights.detach().float().requires_grad_(), bias.detach().float().requires_grad_())
    print("PASS: 2D helper reuse, logical position order, strided IDs, fixed state, and caller mode")


def check_loss():
    global cross_entropy_loss
    scores = _float([[[1., 2., 3.], [2., 1., 3.]], [[3., 2., 1.], [1., 2., 3.]]], requires_grad=True)
    targets = torch.tensor([[1, 2], [0, 1]])
    loss = _check_loss(scores, targets)
    _close(loss, _float(0.9076059644443804))
    print("PASS: four true targets selected in matching order and averaged to one scalar")

    varied = torch.arange(2*3*5, dtype=torch.float64).reshape(2, 3, 5) / 7.
    varied_targets = torch.tensor([[4, 0, 2], [1, 3, 0]])
    _check_loss(varied, varied_targets)
    _check_loss(varied.float(), varied_targets)
    _check_loss(torch.zeros(1, 1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.long))
    extreme = _check_loss(_float([[[1000., 0., -1000.]]]), torch.tensor([[2]]))
    _close(extreme, _float(2000.))
    offsets = _float([[[100.], [-30.], [0.]], [[10.], [-200.], [3.]]])
    _close(_check_loss(varied + offsets, varied_targets), _check_loss(varied, varied_targets))
    _check_loss(torch.zeros(1, 2, 258, dtype=torch.float64), torch.tensor([[256, 257]]))
    print("PASS: varied shapes/dtypes, row shifts, single candidate, large IDs, and stable extreme loss")

    real_loss, calls = cross_entropy_loss, []
    def loss_spy(flat_scores, flat_targets):
        assert flat_scores.shape == (4, 3) and flat_targets.tolist() == [1, 2, 0, 1]
        _close(flat_scores, scores.reshape(4, 3))
        calls.append(flat_targets)
        return real_loss(flat_scores, flat_targets)
    cross_entropy_loss = loss_spy
    try:
        _check_loss(scores, targets)
    finally:
        cross_entropy_loss = real_loss
    assert len(calls) == 1
    strided_scores = torch.arange(3*2*5, dtype=torch.float64).reshape(3, 2, 5).transpose(0, 1)
    strided_targets = torch.tensor([[0, 3], [2, 1], [4, 0]]).transpose(0, 1)
    _check_loss(strided_scores, strided_targets)
    with torch.no_grad():
        _check_loss(scores, targets)
    print("PASS: 2D stable helper reuse, aligned flattening of strided tensors, and preserved mode")

    scores.grad = torch.full_like(scores, 99.)
    loss = _check_loss(scores, targets)
    actual_gradient = torch.autograd.grad(loss, scores)[0]
    expected_gradient = (scores.softmax(dim=-1) - F.one_hot(targets, 3)) / targets.numel()
    _close(actual_gradient, expected_gradient)
    assert torch.equal(scores.grad, torch.full_like(scores, 99.))
    assert torch.autograd.gradcheck(batch_cross_entropy, (scores.detach().clone().requires_grad_(), targets))
    print("PASS: connected analytic gradients scaled by B*T, finite differences, and existing buffers")


def check_pipeline():
    table, inputs, weights, bias, targets = _fixture()
    values = [p.detach().clone() for p in (table, weights, bias)]
    scores = compute_batch_logits(table, inputs, weights, bias)
    loss = batch_cross_entropy(scores, targets)
    loss.backward()
    gradient_scores = (scores.detach().softmax(-1) - F.one_hot(targets, 3)) / 4
    expected_table = torch.zeros_like(table)
    expected_table.index_add_(0, inputs.reshape(-1), (gradient_scores @ weights.detach().T).reshape(-1, 3))
    _close(table.grad, expected_table)
    _close(weights.grad, table.detach()[inputs].reshape(-1, 3).T @ gradient_scores.reshape(-1, 3))
    _close(bias.grad, gradient_scores.sum(dim=(0, 1)))
    assert all(torch.equal(p.detach(), value) for p, value in zip((table, weights, bias), values))
    print("PASS: full batch forward/loss reaches all parameters with repeated-ID accumulation and no update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("logits", "loss", "all"), default="all")
    choice = parser.parse_args().check
    if choice in ("logits", "all"):
        check_logits()
    if choice in ("loss", "all"):
        check_loss()
    if choice == "all":
        check_pipeline()
        print("Lesson 03.20 checks passed.")
    else:
        print(f"Lesson 03.20 {choice} checks passed.")


if __name__ == "__main__":
    main()
