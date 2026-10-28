# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.12: repeat the completed joint update and record its losses.

Run from the repository root in the existing PyTorch environment:
    python exercises/03_12_next_token_training_loop.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse Lesson 03.11's next_token_train_step to update the same embedding
table, output weights, and bias num_steps times on the supplied token pairs.
Review the scalar/tensor loops from 03.03 and 03.07, then read the bilingual
formula, timeline figure, full trace, and numbered implementation plan.
This teaching loop uses a fixed input window and plain gradient descent.
It does not yet load a corpus or generate new tokens.

EXACT INPUTS
embedding_table: supplied trainable floating leaf torch.Tensor [V,D].
token_ids: fixed torch.long tensor [T], input row selectors.
output_weights: supplied trainable floating leaf torch.Tensor [D,V].
output_bias: supplied trainable floating leaf torch.Tensor [V].
target_ids: fixed torch.long tensor [T], observed next-token IDs.
learning_rate: nonnegative Python float, passed to the completed step.
num_steps: nonnegative Python int, the exact number of parameter updates.
Assume the valid shapes, finite values, matching dtype/device, separate
parameter storage, and normal gradient mode specified in Lesson 03.11.
Checks construct CPU float64 parameters. No learner input construction or
validation is needed. All parameter objects already exist before looping.

RETURN AND SIDE EFFECTS
Return losses: a fresh list of Python floats, length num_steps, in order.
losses[k] measures the parameter state BEFORE update k+1.
The original table, weights, and bias now contain the final updated values.
Do not return replacement parameters: the caller already holds these objects.
For zero updates, return []; parameter values and existing gradients stay
unchanged. Keep input IDs, target IDs, and normal gradient mode unchanged.

FORMULAS
    theta_k = (E_k, W_k, b_k), the parameter state after k updates
    L_k = cross_entropy(E_k[token_ids] @ W_k + b_k, target_ids)
    theta_(k+1) = theta_k - learning_rate * gradient(L_k, theta_k)
    losses[k] = L_k, for k = 0,...,num_steps-1
After N updates, parameters contain theta_N, while history ends at L_(N-1).
A separate evaluation computes L_N; it is not another returned history item.
The first helper call uses theta_0, then the second uses updated theta_1.
Keep the same objects rather than resetting their values each iteration.

HELPER CONTRACT
next_token_train_step(embedding_table, token_ids, output_weights, output_bias,
                     target_ids, learning_rate)
returns (loss_value, gradient_table, gradient_weights, gradient_bias).
It already clears old gradients, calculates a new forward/backward, and
updates the original three parameters inside its own no-grad block.
step_result[0] selects its pre-update Python loss. This wrapper records only
that number. The gradient snapshots are not stored in the history.
Call the helper exactly once per iteration; no extra backward, update, or
outer torch.no_grad() block is required.

COMPLETE WORKED EXAMPLE: THREE UPDATES
    embedding_table = [[1],[2]], output_weights = [[1,0]], output_bias = [0,0]
    token_ids = [0], target_ids = [1], learning_rate = 0.1, num_steps = 3
    Initial loss L_0 = 1.3132616875; losses = [].
    Call 1 returns L_0 and updates all three parameters to theta_1.
        losses = [1.3132616875]
    Call 2 uses theta_1, returns L_1 = 1.0668778121, and produces theta_2.
        losses = [1.3132616875, 1.0668778121]
    Call 3 uses theta_2, returns L_2 = 0.8894180392, and produces theta_3.
        losses = [1.3132616875, 1.0668778121, 0.8894180392]
    Return that three-item list. Final parameters are approximately:
        embedding_table = [[0.8277585808],[2]]
        output_weights = [[0.8147926734,0.1852073266]]
        output_bias = [-0.1976082643,0.1976082643]
    A separate final forward gives L_3 = 0.7580921817.
    It is not appended, because the budget requested three updates.

SYNTAX AND INDENTATION
[] creates a fresh empty list. for step in range(num_steps): repeats the
eight-space body exactly num_steps times. range(3) yields 0,1,2; range(0)
skips the body. step counts repetitions, not generated tokens or candidate IDs.
The helper returns a four-item tuple; [0] selects its first item.
losses.append(loss_value) adds one float to the list. This statement executes
after the helper returns, but that float describes the pre-update state.
Ordinary setup, the for header, and final return use four spaces.
The helper call, result selection, and append use eight spaces.
The final return stays outside the loop so zero and multiple steps work.

HOW TO REVIEW
The learner core is complete. Review or reconstruct it using the five hints.
Keep the supplied setup/check driver and all earlier learner code unchanged.
Expected final success line after implementation:
    Lesson 03.12 checks passed.
"""

from __future__ import annotations

import importlib.util
from math import isclose
from pathlib import Path

import torch
import torch.nn.functional as F


def train_next_token(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    target_ids: torch.Tensor,
    learning_rate: float,
    num_steps: int,
) -> list[float]:
    """Update the supplied parameters N times and return N pre-update losses."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, create a fresh losses list using [].
    # The parameter tensors already exist. Reuse them without reconstruction.
    # Checkpoint: before any update, losses is [].
    #
    # STEP 2: At four spaces, start for step in range(num_steps):.
    # The colon opens the eight-space body. range(3) repeats three times;
    # range(0) skips the body and still reaches the final return.
    #
    # STEP 3: At eight spaces inside the loop, call next_token_train_step
    # with embedding_table, token_ids, output_weights, output_bias,
    # target_ids, learning_rate, in that order. Save its tuple as step_result.
    # It already changes the SAME original parameter tensors in place.
    # Example call 1 returns original loss 1.3132616875 and produces theta_1.
    # Call 2 must use those updated tensors rather than reinitializing them.
    #
    # STEP 4: Still at eight spaces, select step_result[0] as loss_value,
    # then use losses.append(loss_value). The first tuple item is a float.
    # Example after two calls: [1.3132616875,1.0668778121].
    # Record neither the whole tuple nor the gradient snapshots. Do not
    # apply another backward or update; the completed helper handles them.
    #
    # STEP 5: At four spaces, AFTER leaving the loop, return losses.
    # Example three entries [1.3132616875,1.0668778121,0.8894180392].
    # Final parameters remain in the caller's original tensors. Do not add
    # final evaluation loss 0.7580921817 as an extra history entry.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement train_next_token for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_step():
    path = Path(__file__).with_name("03_11_next_token_train_step.py")
    specification = importlib.util.spec_from_file_location("completed_joint_step", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.next_token_train_step


next_token_train_step = _load_step()


def _parameter(values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _ids(values):
    return torch.tensor(values, dtype=torch.long)


def _example():
    return (_parameter([[1.], [2.]]), _ids([0]), _parameter([[1., 0.]]),
            _parameter([0., 0.]), _ids([1]))


def _objective(parameters, token_ids, target_ids):
    table, weights, bias = parameters
    return F.cross_entropy(table[token_ids] @ weights + bias, target_ids)


def _reference_run(parameters, token_ids, target_ids, learning_rate, num_steps):
    references = tuple(value.detach().clone().requires_grad_() for value in parameters)
    losses, gradients = [], None
    for _ in range(num_steps):
        loss = _objective(references, token_ids, target_ids)
        losses.append(loss.item())
        gradients = torch.autograd.grad(loss, references)
        with torch.no_grad():
            for value, gradient in zip(references, gradients):
                value -= learning_rate * gradient
    return losses, tuple(value.detach().clone() for value in references), gradients


def _check_run(embedding_table, token_ids, output_weights, output_bias,
               target_ids, learning_rate, num_steps):
    parameters = (embedding_table, output_weights, output_bias)
    original_gradients = tuple(None if p.grad is None else p.grad.clone() for p in parameters)
    pointers = tuple(p.data_ptr() for p in parameters)
    ids_before, targets_before = token_ids.clone(), target_ids.clone()
    mode_before = torch.is_grad_enabled()
    expected_losses, expected_parameters, expected_gradients = _reference_run(
        parameters, token_ids, target_ids, learning_rate, num_steps
    )
    actual = train_next_token(embedding_table, token_ids, output_weights, output_bias,
                              target_ids, learning_rate, num_steps)
    assert isinstance(actual, list) and len(actual) == num_steps, "Return exactly N loss numbers"
    for loss, expected in zip(actual, expected_losses):
        assert isinstance(loss, float), "Store only the loss float, not the helper tuple"
        assert isclose(loss, expected, rel_tol=1e-9, abs_tol=1e-12), (
            "Record each pre-update loss from the current persistent parameters"
        )
    for index, (parameter, expected, pointer) in enumerate(zip(parameters, expected_parameters, pointers)):
        assert torch.allclose(parameter.detach(), expected, rtol=1e-9, atol=1e-12)
        assert parameter.data_ptr() == pointer and parameter.requires_grad and parameter.is_leaf
        assert parameter.grad_fn is None
        if num_steps:
            assert parameter.grad is not None
            assert torch.allclose(parameter.grad, expected_gradients[index], rtol=1e-9, atol=1e-12)
        elif original_gradients[index] is None:
            assert parameter.grad is None, "Zero updates should leave gradients untouched"
        else:
            assert torch.equal(parameter.grad, original_gradients[index])
    assert torch.equal(token_ids, ids_before) and torch.equal(target_ids, targets_before)
    assert torch.is_grad_enabled() == mode_before
    return actual


def _check_helper_reuse():
    global next_token_train_step
    original_step = next_token_train_step
    arguments = _example()
    parameters = (arguments[0], arguments[2], arguments[3])
    expected_before = []
    references = tuple(p.detach().clone().requires_grad_() for p in parameters)
    for _ in range(3):
        expected_before.append(tuple(p.detach().clone() for p in references))
        loss = _objective(references, arguments[1], arguments[4])
        gradients = torch.autograd.grad(loss, references)
        with torch.no_grad():
            for p, g in zip(references, gradients):
                p -= 0.1 * g
    calls = []

    def observed_step(table, ids, weights, bias, targets, learning_rate):
        assert torch.is_grad_enabled(), "Keep the training loop in normal gradient mode"
        assert len(calls) < 3, "Call the helper exactly N times"
        current = (table, weights, bias)
        for actual, original, expected in zip(current, parameters, expected_before[len(calls)]):
            assert actual is original, "Reuse the same caller-owned parameter objects"
            assert torch.allclose(actual.detach(), expected, rtol=1e-9, atol=1e-12)
        assert ids is arguments[1] and targets is arguments[4] and learning_rate == 0.1
        calls.append(current)
        return original_step(table, ids, weights, bias, targets, learning_rate)

    try:
        next_token_train_step = observed_step
        _check_run(*arguments, 0.1, 3)
        assert len(calls) == 3, "Reuse the completed helper once per requested update"
        _check_run(*arguments, 0.1, 0)
        assert len(calls) == 3, "A zero-step call must not invoke the helper"
    finally:
        next_token_train_step = original_step


def check_training_loop():
    arguments = _example()
    history = _check_run(*arguments, 0.1, 3)
    for actual, expected in zip(history, [1.3132616875182228, 1.066877812077227, 0.8894180391836587]):
        assert isclose(actual, expected, rel_tol=1e-12)
    final_loss = _objective((arguments[0], arguments[2], arguments[3]), arguments[1], arguments[4]).item()
    assert isclose(final_loss, 0.7580921816742527, rel_tol=1e-12)
    assert not isclose(history[-1], final_loss), "Final evaluation is separate from history"
    print("PASS: three updates, ordered pre-update losses, and final parameter values")

    arguments = _example()
    for parameter in (arguments[0], arguments[2], arguments[3]):
        parameter.grad = torch.full_like(parameter, 99.)
    assert _check_run(*arguments, 0.1, 0) == []
    _check_run(*arguments, 0.1, 1)
    print("PASS: zero and one step preserve the specified budget and gradient state")

    _check_helper_reuse()
    print("PASS: supplied step reused exactly once per update with persistent parameters")

    arguments = _example()
    first = _check_run(*arguments, 0.1, 3)
    second = _check_run(*arguments, 0.1, 2)
    uninterrupted = _check_run(*_example(), 0.1, 5)
    assert all(isclose(a, b, rel_tol=1e-12) for a, b in zip(first + second, uninterrupted))
    first.append(99.)
    assert len(second) == 2 and len(uninterrupted) == 5
    print("PASS: fresh histories and continuation across separate calls")

    varied = (_parameter([[0.2, -0.1], [0.3, 0.4], [0.5, -0.2]]), _ids([2, 0, 2]),
              _parameter([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]),
              _parameter([0.05, -0.2, 0.1]), _ids([1, 2, 0]))
    _check_run(*varied, 0.05, 4)
    assert torch.equal(varied[0][1].detach(), torch.tensor([0.3, 0.4], dtype=torch.float64))
    print("PASS: varied positions, vocabulary dimensions, repeated IDs, and learning rates")

    frozen = _check_run(*_example(), 0.0, 3)
    assert all(isclose(value, frozen[0], rel_tol=1e-12) for value in frozen)
    zero = (_parameter([[2., -1.]]), _ids([0, 0]), _parameter([[0.3], [-0.2]]),
            _parameter([0.1]), _ids([0, 0]))
    assert _check_run(*zero, 0.4, 3) == [0., 0., 0.]
    print("PASS: zero learning rate and single-candidate zero loss still record every step")

    large = _check_run(*_example(), 10.0, 2)
    assert large[1] > large[0], "Preserve observed loss increases; do not filter or sort history"
    print("PASS: large-step loss increases remain visible without extra updates")
    print("Lesson 03.12 checks passed.")


if __name__ == "__main__":
    check_training_loop()
