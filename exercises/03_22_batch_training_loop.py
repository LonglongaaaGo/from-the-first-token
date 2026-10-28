# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.22: ordered batch construction and persistent training.

Run from the repository root after completing Lesson 03.21:
    python exercises/03_22_batch_training_loop.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse 03.19 make_next_token_batch and 03.21 batch_train_step. An explicit
schedule lists the source-window starts for each update. Build a new batch,
update the SAME parameter tensors once, and record its pre-update mean loss.
Review 03.12's persistent loop, then read the paired formulas and full trace.

EXACT INPUTS
embedding_table: trainable floating leaf torch.Tensor [V,D].
token_ids: fixed torch.long tensor [N], the source sequence.
output_weights: trainable floating leaf tensor [D,V].
output_bias: trainable floating leaf tensor [V].
batch_starts: list[list[int]]; outer order is update order. Each inner list
contains valid source starts for ONE batch, with B_k=len(batch_starts[k]).
Different updates may have different B_k; repeated starts are permitted.
context_length: positive Python int T, supervised positions per sample.
learning_rate: nonnegative float, passed unchanged to every step.
Assume N>=T+1, nonempty inner lists, 0<=start<=N-T-1, valid IDs, compatible
floating dtype/device, separate parameter storage, and normal gradient mode.
The OUTER schedule may be empty. No learner validation TODO is assigned.

RETURN AND SIDE EFFECTS
Return losses: a fresh list[float], one pre-update mean loss per inner list.
Original parameters retain their final updated values. Preserve source IDs,
the nested schedule, caller gradient mode, and random state. Do not recreate
or reset parameters, clear gradients, call backward, or apply extra updates
in this wrapper: batch_train_step handles the complete step.
For batch_starts=[], return [] without calling either helper or changing
parameter values or existing gradient buffers. No final loss is appended.

FORMULAS
    K = len(batch_starts); B_k = len(batch_starts[k])
    X_k[b,t] = token_ids[batch_starts[k][b] + t]
    Y_k[b,t] = token_ids[batch_starts[k][b] + t + 1]
    L_k = mean(-log P_theta_k(Y_k | X_k)) over B_k*T positions
    theta_(k+1) = theta_k - learning_rate*gradient(L_k,theta_k)
    losses[k] = L_k; supervised-position budget = T*sum(B_k)
theta_k contains parameters AFTER k updates and BEFORE update k+1.
Data and parameters both affect loss; different-batch losses can increase
even when both fixed-batch losses decrease on a separate comparison.

COMPLETE EXAMPLE: ordered-two-batch-v1, a hand-picked correctness fixture
    token_ids=[0,1,0,1,1,0,1]; batch_starts=[[0,1],[3,4]]; T=2; lr=0.1
    E=[[1],[2]]; W=[[1,0]]; bias=[0,0]; D=1,V=2
    k=0: X=[[0,1],[1,0]], Y=[[1,0],[0,1]], loss=0.7200948493
         E -> [[0.9634470711],[2.0059601461]]
         W -> [[0.9753673633,0.0246326367]]
         bias -> [-0.0305927828,0.0305927828]
    k=1: X=[[1,1],[1,0]], Y=[[1,0],[0,1]], loss=0.8737345841
         E -> [[0.9467718386],[1.9919141624]]
         W -> [[0.9288334710,0.0711665290]]
         bias -> [-0.0629059142,0.0629059142]
    Return [0.7200948493,0.8737345841]; two updates, eight supervised positions.
    Separate fixed-batch losses at theta_0/theta_1/theta_2:
         batch A: 0.7200948493 / 0.6779073607 / 0.6404176541
         batch B: 0.9235114302 / 0.8737345841 / 0.8092791313
    Both fixed-batch losses decrease here, while the mixed history increases.

SYNTAX AND INDENTATION
[] creates a fresh list on each call. for start_positions in batch_starts:
visits the inner lists themselves, preserving their supplied order.
For [[0,1],[3,4]], start_positions is [0,1], then [3,4]; it is not k=0,1.
Use four spaces for setup, the for header, and final return. The loop body
uses eight spaces. Tuple unpacking obtains input_ids,target_ids together.
step_result[0] selects the loss float from the step's four-item tuple.
losses.append(loss_value) adds one float after that update returns; the float
still describes the BEFORE-update state. Return outside the loop.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


def train_batches(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    batch_starts: list[list[int]],
    context_length: int,
    learning_rate: float,
) -> list[float]:
    """Update original parameters in schedule order and return K old-state losses."""
    # ===== START LEARNER CODE =====
    # STEP 1: Four spaces: create a fresh losses list using []. Parameter
    # tensors already exist; keep the caller's objects and current values.
    #
    # STEP 2: Four spaces: for start_positions in batch_starts:. The colon
    # opens an eight-space body. Example visits [0,1], then [3,4]. An empty
    # outer list skips the body and must still reach the final return.
    #
    # STEP 3: Eight spaces: call make_next_token_batch(token_ids,
    # start_positions,context_length) once. Unpack into input_ids,target_ids.
    # First batch X=[[0,1],[1,0]], Y=[[1,0],[0,1]], both shape [2,2].
    # The builder already shifts targets; do not shift them again.
    #
    # STEP 4: Eight spaces: call batch_train_step(embedding_table,input_ids,
    # output_weights,output_bias,target_ids,learning_rate) once and save
    # step_result. The helper clears old grads and updates the SAME objects.
    # Its first result item is the pre-update loss; do not use an outer no-grad.
    #
    # STEP 5: Eight spaces: select step_result[0], saving loss_value.
    # Example first call returns 0.7200948493; second returns 0.8737345841.
    # The second call starts from the already updated theta_1 parameters.
    #
    # STEP 6: Eight spaces: losses.append(loss_value). Append one float,
    # not the complete tuple or gradients. After two updates, length is two.
    # Do not run another forward, backward, clear, or parameter update here.
    #
    # STEP 7: Back at four spaces AFTER the loop: return losses.
    # Return [] for an empty schedule; otherwise one entry per scheduled batch.
    # Caller-owned parameters retain the final state. No final extra loss item.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement train_batches for this exercise")
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


make_next_token_batch = _load_module("03_19_next_token_batch.py", "ordered_batch_builder").make_next_token_batch
_step_module = _load_module("03_21_batch_train_step.py", "ordered_batch_step")
batch_train_step = _step_module.batch_train_step


def _parameter(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=True)


def _fixture():
    return (_parameter([[1.], [2.]]), torch.tensor([0, 1, 0, 1, 1, 0, 1]),
            _parameter([[1., 0.]]), _parameter([0., 0.]), [[0, 1], [3, 4]], 2)


def _objective(parameters, inputs, targets):
    table, weights, bias = parameters
    scores = table[inputs] @ weights + bias
    return F.cross_entropy(scores.reshape(-1, bias.numel()), targets.reshape(-1))


def _reference(parameters, source, schedule, context_length, learning_rate):
    copies = tuple(p.detach().clone().requires_grad_() for p in parameters)
    history, last_gradients = [], None
    for starts in schedule:
        # Independent reference indexing, rather than the learner's builder.
        positions = torch.tensor(starts, dtype=torch.long, device=source.device)[:, None]
        positions = positions + torch.arange(context_length, device=source.device)[None, :]
        inputs, targets = source[positions], source[positions+1]
        loss = _objective(copies, inputs, targets)
        gradients = torch.autograd.grad(loss, copies)
        history.append(loss.item())
        with torch.no_grad():
            for p, gradient in zip(copies, gradients):
                p -= learning_rate*gradient
        last_gradients = gradients
    return history, copies, last_gradients


def _checked_loop(arguments, learning_rate):
    table, source, weights, bias, schedule, length = arguments
    parameters = (table, weights, bias)
    expected, final, last_gradients = _reference(parameters, source, schedule, length, learning_rate)
    pointers = tuple(p.data_ptr() for p in parameters)
    old_grads = tuple((p.grad, None if p.grad is None else p.grad.clone()) for p in parameters)
    fixed_source, fixed_schedule = source.clone(), copy.deepcopy(schedule)
    rng, mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    losses = train_batches(table, source, weights, bias, schedule, length, learning_rate)
    assert isinstance(losses, list) and len(losses) == len(schedule)
    assert all(type(value) is float for value in losses)
    tolerance = 3e-6 if table.dtype == torch.float32 else 1e-10
    torch.testing.assert_close(torch.tensor(losses, dtype=torch.float64),
                               torch.tensor(expected, dtype=torch.float64),
                               rtol=tolerance, atol=tolerance)
    for index, (p, value, pointer) in enumerate(zip(parameters, final, pointers)):
        assert p.is_leaf and p.requires_grad and p.data_ptr() == pointer
        assert p.dtype == value.dtype and p.device == value.device
        torch.testing.assert_close(p.detach(), value.detach(), rtol=tolerance, atol=tolerance)
        if schedule:
            torch.testing.assert_close(p.grad, last_gradients[index], rtol=tolerance, atol=tolerance)
        else:
            grad, old_value = old_grads[index]
            assert p.grad is grad
            if grad is not None:
                assert torch.equal(p.grad, old_value)
    assert torch.equal(source, fixed_source) and schedule == fixed_schedule
    assert torch.equal(torch.get_rng_state(), rng) and torch.is_grad_enabled() == mode
    return losses


def check_training():
    fixture = _fixture()
    losses = _checked_loop(fixture, .1)
    assert abs(losses[0]-.7200948492805977) < 1e-10
    assert abs(losses[1]-.8737345841102824) < 1e-10 and losses[1] > losses[0]
    expected_final = ([[.9467718386], [1.9919141624]],
                      [[.9288334710, .0711665290]], [-.0629059142, .0629059142])
    for p, values in zip((fixture[0], fixture[2], fixture[3]), expected_final):
        torch.testing.assert_close(p.detach(), torch.tensor(values, dtype=torch.float64), rtol=0, atol=1e-9)
    print("PASS: two different batches, pre-update history, and persistent final parameters")

    global make_next_token_batch, batch_train_step
    original_builder, original_step = make_next_token_batch, batch_train_step
    def forbidden(*args, **kwargs):
        raise AssertionError("An empty schedule must call neither helper")
    make_next_token_batch = batch_train_step = forbidden
    try:
        fixture = list(_fixture()); fixture[4] = []
        for p in (fixture[0], fixture[2], fixture[3]):
            p.grad = torch.full_like(p, 7.)
        first = _checked_loop(fixture, .1)
        second = _checked_loop(fixture, .1)
        assert first == second == [] and first is not second
    finally:
        make_next_token_batch, batch_train_step = original_builder, original_step
    print("PASS: empty schedule returns a fresh list and preserves values and existing gradient buffers")

    fixture = list(_fixture()); fixture[4] = [[1], [4, 0], [2, 2, 0]]
    _checked_loop(fixture, .07)
    print("PASS: supplied batch order, varying batch sizes, repeated starts, and aligned targets")

    fixture = _fixture()
    parameters = (fixture[0], fixture[2], fixture[3])
    events, built = [], []
    def guarded_builder(source, starts, length):
        index = len(built)
        assert events == [event for _ in range(index) for event in ("build", "step")]
        assert source is fixture[1] and starts is fixture[4][index] and length == fixture[5]
        assert torch.is_grad_enabled()
        result = original_builder(source, starts, length)
        events.append("build"); built.append(result)
        return result
    def guarded_step(table, inputs, weights, bias, targets, rate):
        assert all(actual is expected for actual, expected in zip((table, weights, bias), parameters))
        assert inputs is built[-1][0] and targets is built[-1][1] and rate == .1
        assert torch.is_grad_enabled() and events[-1] == "build"
        events.append("step")
        return original_step(table, inputs, weights, bias, targets, rate)
    make_next_token_batch, batch_train_step = guarded_builder, guarded_step
    try:
        _checked_loop(fixture, .1)
        assert events == ["build", "step", "build", "step"]
    finally:
        make_next_token_batch, batch_train_step = original_builder, original_step
    print("PASS: one builder and one step call per batch, same parameter objects, and normal gradient mode")

    fixture = _fixture()
    first = _checked_loop(fixture, .1); saved = first.copy()
    second = _checked_loop(fixture, .1)
    assert first == saved and first is not second and second[0] < first[0]
    print("PASS: separate calls continue from current parameters and keep independent loss histories")

    fixture = _fixture()
    old = tuple(p.detach().clone() for p in (fixture[0], fixture[2], fixture[3]))
    calls = []
    def zero_rate_step(*args):
        calls.append(args[-1])
        return original_step(*args)
    batch_train_step = zero_rate_step
    try:
        _checked_loop(fixture, 0.)
        assert calls == [0., 0.]
        assert all(torch.equal(p.detach(), value) for p, value in zip((fixture[0], fixture[2], fixture[3]), old))
    finally:
        batch_train_step = original_step
    print("PASS: zero learning rate still visits every scheduled batch and records one loss each")

    V, D = 258, 2
    table = _parameter(torch.arange(V*D).reshape(V, D).div(1000).tolist(), torch.float32)
    weights = _parameter(torch.arange(D*V).reshape(D, V).div(2000).tolist(), torch.float32)
    bias = _parameter(torch.zeros(V).tolist(), torch.float32)
    source = torch.tensor([0, 99, 256, 99, 257, 99, 1, 99, 0, 99, 256, 99])[::2]
    assert not source.is_contiguous()
    _checked_loop((table, source, weights, bias, [[0, 2], [1]], 2), .05)
    print("PASS: strided source, IDs above 255, dtype preservation, and unchanged source/schedule/random state")
    print("Lesson 03.22 checks passed.")


if __name__ == "__main__":
    check_training()
