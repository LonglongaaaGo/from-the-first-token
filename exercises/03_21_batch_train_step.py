# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.21: one joint update from a batch of next-token targets.

Run from the repository root in the environment described in README.md:
    python exercises/03_21_batch_train_step.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse 03.20 compute_batch_logits and batch_cross_entropy. Extend 03.11's
single joint SGD update to inputs/targets [B,T]. One mean loss differentiates
through the table, projection weights, and bias; then update all three.
This is still the independent lookup/head baseline, before attention.

EXACT INPUTS
embedding_table: trainable floating leaf torch.Tensor [V,D].
input_ids: fixed torch.long tensor [B,T], observed input IDs.
output_weights: trainable floating leaf tensor [D,V].
output_bias: trainable floating leaf tensor [V].
target_ids: fixed torch.long tensor [B,T], already shifted next-token IDs.
learning_rate: nonnegative float, the common update multiplier.
Assume B,T,D,V>0, valid IDs, matching floating dtype/device, finite values,
normal gradient mode, and separately stored parameters. No validation TODO.

RETURN AND SIDE EFFECTS
Return (loss_value, gradient_table, gradient_weights, gradient_bias).
loss_value: Python float for the mean loss BEFORE updating.
Gradients: independent tensor snapshots [V,D], [D,V], [V], all from that
same pre-update state. Update the ORIGINAL parameters in place, preserving
their storage, dtype/device, and trainable-leaf identity. Preserve both ID
tensors and random state. Leave current .grad buffers; clear them next call.

FORMULAS
    H[b,t,:] = E[input_ids[b,t]]; Z[b,t,:] = H[b,t,:] @ W + beta
    L = -(1/(B*T))*sum(log P[b,t,target_ids[b,t]])
    G_Z = (P - one_hot(target_ids))/(B*T)
    G_W = flat_H.T @ flat_G_Z; G_beta = sum(flat_G_Z, axis=0)
    G_E[i,:] = sum(G_Z[b,t,:] @ W.T for positions with input ID i)
    E_new = E - learning_rate*G_E; similarly for W and beta.
The helper already averages positions. Do not divide a second time.

COMPLETE SMALL EXAMPLE: batch-two-id-sgd-v1 (hand-picked)
    source = [0,1,0,1]; starts = [0,1]; T=2
    input_ids = [[0,1],[1,0]]; target_ids = [[1,0],[0,1]]
    B=2,T=2,D=1,V=2; E=[[1],[2]]; W=[[1,0]]; beta=[0,0]; lr=0.1
    flat scores = [[1,0],[2,0],[2,0],[1,0]]
    position losses = [1.3132616875,0.1269280110,0.1269280110,1.3132616875]
    loss_value = 0.7200948493
    gradient_table = [[0.3655292893],[-0.0596014610]]
    gradient_weights = [[0.2463263673,-0.2463263673]]
    gradient_bias = [0.3059278283,-0.3059278283]
    new E = [[0.9634470711],[2.0059601461]]
    new W = [[0.9753673633,0.0246326367]]
    new beta = [-0.0305927828,0.0305927828]
    Recomputed loss = 0.6779073607, a separate after-update observation.

NEW SYNTAX
A tuple groups parameter references, without copying them. A for-loop body
uses eight spaces inside this function. parameter.grad = None clears old
derivatives, not parameter values. Ordinary backward accumulates gradients.
loss.backward() fills all three buffers once. .clone() saves separate values.
with torch.no_grad(): disables tracking only within its indented update block;
explicit parameter -= ... statements use eight spaces. Forward/backward and
the final return stay at four spaces outside that block. Do not use .data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def batch_train_step(
    embedding_table: torch.Tensor,
    input_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    target_ids: torch.Tensor,
    learning_rate: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update original parameters; return pre-update mean loss and gradients."""
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: group the three parameter references in parameters.
    # Use (embedding_table, output_weights, output_bias), excluding IDs.
    # Four spaces: for parameter in parameters:. Eight spaces: set its .grad
    # to None. This clears previous derivatives while preserving all values.
    #
    # STEP 2: Back at four spaces: call compute_batch_logits(embedding_table,
    # input_ids, output_weights, output_bias) once, saving logits [B,T,V].
    # Example [[[1,0],[2,0]],[[2,0],[1,0]]]. Keep the connected graph.
    #
    # STEP 3: Four spaces: call batch_cross_entropy(logits,target_ids) once,
    # saving the scalar tensor loss, approximately 0.7200948493.
    # Targets are already shifted; this helper already averages B*T positions.
    #
    # STEP 4: Four spaces: call loss.backward() once, before any updates.
    # All three .grad buffers now describe the SAME pre-update parameters.
    # Example table gradient [[0.3655292893],[-0.0596014610]].
    #
    # STEP 5: Four spaces: save loss_value from loss.item(). Preserve loss as
    # the tensor; loss_value is only a Python float for logging/return.
    #
    # STEP 6: Four spaces: save gradient_table, gradient_weights, gradient_bias
    # by .clone() on the corresponding .grad buffers. Independent snapshots
    # must survive later gradient clearing or changes to live buffers.
    #
    # STEP 7: Four spaces: enter with torch.no_grad():. Inside at eight spaces,
    # use three explicit -= updates, each subtracting learning_rate times that
    # parameter's own .grad. Example table[0,0] becomes 0.9634470711.
    # Keep original parameter objects; do not rebind with parameter = ... .
    #
    # STEP 8: Back at four spaces: return loss_value, gradient_table,
    # gradient_weights, gradient_bias in that order. Return BEFORE-update
    # observations; recomputing the new loss belongs to the supplied driver.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement batch_train_step for this exercise")
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


_batch_helpers = _load_module("03_20_batched_logits_and_loss.py", "completed_batch_forward_loss")
compute_batch_logits = _batch_helpers.compute_batch_logits
batch_cross_entropy = _batch_helpers.batch_cross_entropy


def _parameter(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=True)


def _fixture():
    return (_parameter([[1.], [2.]]), torch.tensor([[0, 1], [1, 0]]),
            _parameter([[1., 0.]]), _parameter([0., 0.]),
            torch.tensor([[1, 0], [0, 1]]))


def _reference(parameters, inputs, targets):
    table, weights, bias = parameters
    scores = table[inputs] @ weights + bias
    return F.cross_entropy(scores.reshape(-1, bias.numel()), targets.reshape(-1))


def _checked_step(arguments, learning_rate):
    table, inputs, weights, bias, targets = arguments
    parameters = (table, weights, bias)
    originals = tuple(p.detach().clone() for p in parameters)
    reference_parameters = tuple(p.detach().clone().requires_grad_() for p in parameters)
    reference_loss = _reference(reference_parameters, inputs, targets)
    reference_gradients = torch.autograd.grad(reference_loss, reference_parameters)
    pointers = tuple(p.data_ptr() for p in parameters)
    fixed_ids = (inputs.clone(), targets.clone())
    rng = torch.get_rng_state().clone()
    mode = torch.is_grad_enabled()
    backward_counts = [0, 0, 0]
    def count_hook(index):
        def hook(gradient):
            backward_counts[index] += 1
        return hook
    handles = [p.register_hook(count_hook(i)) for i, p in enumerate(parameters)]
    try:
        result = batch_train_step(table, inputs, weights, bias, targets, learning_rate)
    finally:
        for handle in handles:
            handle.remove()
    assert isinstance(result, tuple) and len(result) == 4
    loss_value, *snapshots = result
    assert type(loss_value) is float
    tolerance = 2e-6 if table.dtype == torch.float32 else 1e-10
    assert abs(loss_value-reference_loss.item()) <= tolerance*max(1., abs(loss_value))
    assert backward_counts == [1, 1, 1], "Differentiate one connected loss exactly once"
    for p, old, pointer, expected, snapshot in zip(
            parameters, originals, pointers, reference_gradients, snapshots):
        assert p.is_leaf and p.requires_grad and p.data_ptr() == pointer
        assert isinstance(snapshot, torch.Tensor) and snapshot.shape == p.shape
        assert snapshot.dtype == p.dtype and snapshot.device == p.device
        assert not snapshot.requires_grad and snapshot.grad_fn is None
        assert snapshot.data_ptr() != p.grad.data_ptr()
        torch.testing.assert_close(snapshot, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(p.grad, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(p.detach(), old-learning_rate*expected,
                                   rtol=tolerance, atol=tolerance)
    assert torch.equal(inputs, fixed_ids[0]) and torch.equal(targets, fixed_ids[1])
    assert torch.equal(torch.get_rng_state(), rng) and torch.is_grad_enabled() == mode
    return result


def check_training():
    fixture = _fixture()
    result = _checked_step(fixture, .1)
    expected = ([[.36552928931500245], [-.05960146101105882]],
                [[.2463263672928848, -.2463263672928848]],
                [.3059278283039436, -.3059278283039436])
    for actual, values in zip(result[1:], expected):
        torch.testing.assert_close(actual, torch.tensor(values, dtype=torch.float64))
    after = _reference((fixture[0], fixture[2], fixture[3]), fixture[1], fixture[4]).item()
    assert abs(result[0]-.7200948492805977) < 1e-10
    assert abs(after-.677907360692) < 1e-9 and after < result[0]
    print("PASS: four-position mean, all three gradients, in-place update, and pre-update return")

    for B, T, D, V in [(1, 3, 2, 4), (3, 1, 4, 2), (2, 3, 2, 5), (1, 1, 2, 1)]:
        table = _parameter(torch.arange(V*D).reshape(V, D).div(9).tolist())
        weights = _parameter(torch.arange(D*V).reshape(D, V).div(13).tolist())
        bias = _parameter(torch.arange(V).div(17).tolist())
        inputs = torch.arange(B*T).reshape(B, T) % V
        targets = (inputs+1) % V
        _checked_step((table, inputs, weights, bias, targets), .03)
    print("PASS: varied B/T/D/V, distinct targets, singleton axes, and one-candidate zero gradients")

    global compute_batch_logits, batch_cross_entropy
    original_forward, original_loss = compute_batch_logits, batch_cross_entropy
    fixture = _fixture()
    parameters = (fixture[0], fixture[2], fixture[3])
    old = tuple(p.detach().clone() for p in parameters)
    calls = []
    def guarded_forward(table, inputs, weights, bias):
        assert torch.is_grad_enabled()
        assert (table, weights, bias) == parameters
        assert all(p.grad is None for p in parameters)
        calls.append("forward")
        return original_forward(table, inputs, weights, bias)
    def guarded_loss(scores, targets):
        assert torch.is_grad_enabled() and calls == ["forward"]
        assert all(torch.equal(p.detach(), value) for p, value in zip(parameters, old))
        assert targets is fixture[4] and scores.shape == (2, 2, 2)
        calls.append("loss")
        return original_loss(scores, targets)
    compute_batch_logits, batch_cross_entropy = guarded_forward, guarded_loss
    try:
        _checked_step(fixture, .1)
        assert calls == ["forward", "loss"]
    finally:
        compute_batch_logits, batch_cross_entropy = original_forward, original_loss
    print("PASS: completed batch helpers called once with unchanged old parameters before backward")

    fixture = _fixture()
    parameters = (fixture[0], fixture[2], fixture[3])
    for p in parameters:
        p.grad = torch.full_like(p, 7.)
    first = _checked_step(fixture, .1)
    saved = tuple(g.clone() for g in first[1:])
    _checked_step(fixture, .1)
    for p, snapshot, values in zip(parameters, first[1:], saved):
        p.grad.fill_(91.)
        assert torch.equal(snapshot, values)
    print("PASS: stale gradients cleared, repeated calls use current parameters, and snapshots stay independent")

    fixture = _fixture()
    old = tuple(p.detach().clone() for p in (fixture[0], fixture[2], fixture[3]))
    _checked_step(fixture, 0.)
    assert all(torch.equal(p.detach(), value) for p, value in zip((fixture[0], fixture[2], fixture[3]), old))
    fixture = _fixture()
    before = _checked_step(fixture, 20.)[0]
    assert _reference((fixture[0], fixture[2], fixture[3]), fixture[1], fixture[4]).item() > before
    print("PASS: zero learning rate still computes gradients; a large step can increase loss")

    V, D = 258, 2
    table = _parameter(torch.arange(V*D).reshape(V, D).div(1000).tolist(), torch.float32)
    weights = _parameter(torch.arange(D*V).reshape(D, V).div(2000).tolist(), torch.float32)
    bias = _parameter(torch.zeros(V).tolist(), torch.float32)
    inputs = torch.tensor([[0, 99, 256, 99], [257, 99, 1, 99]])[:, ::2]
    targets = torch.tensor([[256, 99, 257, 99], [1, 99, 0, 99]])[:, ::2]
    assert not inputs.is_contiguous() and not targets.is_contiguous()
    _checked_step((table, inputs, weights, bias, targets), .05)
    print("PASS: strided ID tensors, IDs above 255, dtype preservation, and fixed input/random state")

    builder = _load_module("03_19_next_token_batch.py", "completed_batch_builder").make_next_token_batch
    inputs, targets = builder(torch.tensor([0, 1, 0, 1]), [0, 1], 2)
    table, _, weights, bias, _ = _fixture()
    fixture = (table, inputs, weights, bias, targets)
    history = [_checked_step(fixture, .1)[0] for _ in range(3)]
    assert history[2] < history[1] < history[0]
    print("PASS: actual window builder and batch helpers connect across three toy updates")
    print("Lesson 03.21 checks passed.")


if __name__ == "__main__":
    check_training()
