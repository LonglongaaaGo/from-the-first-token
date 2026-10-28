# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.11: one joint next-token training step.

Run from the repository root in the existing PyTorch environment:
    python exercises/03_11_next_token_train_step.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

IMPLEMENTATION NOTES
The reset-loop name paramemter is used consistently and is valid Python.
loss_values holds the scalar tensor; loss_value is its extracted float.
The learner uses three explicit in-place updates inside no-grad, with no
nested update loop. Those update statements correctly use eight spaces.
Twelve spaces apply to the alternative nested-loop form in the hints.

PURPOSE AND PREREQUISITES
Lesson 03.08 trained an embedding table against teaching target vectors.
Lessons 03.09 and 03.10 supply compute_logits and cross_entropy_loss.
Now one observed next-token loss trains the embedding table, output weights,
and bias together. This lookup-plus-head model uses the current input token;
contextual attention and the repeated-step loop are later lessons.
Read the bilingual formulas, figures, and complete example before coding.

EXACT INPUTS
embedding_table: trainable floating leaf torch.Tensor [V,D].
token_ids: fixed torch.long tensor [T], selecting input rows.
output_weights: trainable floating leaf torch.Tensor [D,V].
output_bias: trainable floating leaf torch.Tensor [V].
target_ids: fixed torch.long tensor [T], observed next-token IDs.
learning_rate: nonnegative float; update multiplier for all three parameters.
Assume T,D,V>0, valid IDs, finite values, matching dtype/device, normal
gradient mode, and separately stored, untied parameter tensors. Setup is
supplied. Checks use CPU float64; no learner input validation is required.

RETURN AND SIDE EFFECTS
Return (loss_value, gradient_table, gradient_weights, gradient_bias).
loss_value: Python float describing the loss BEFORE this update.
The three gradients: independent tensor copies, shapes [V,D], [D,V], [V],
all describing the same pre-update parameter state.
Update the original three parameters in place. Keep them trainable leaves
with the same storage. Preserve token_ids and target_ids. Leave the current
gradients stored; reset all three at the start of the next call.

FORMULAS
    H[t,d] = E[token_ids[t],d]; H has shape [T,D]
    Z = H @ W + b; Z has shape [T,V]
    L = mean(-ln P[t,target_ids[t]]) over T positions
    G_E = dL/dE; G_W = dL/dW; G_b = dL/db
    E_new = E - learning_rate * G_E
    W_new = W - learning_rate * G_W
    b_new = b - learning_rate * G_b
Call loss.backward() once. Autograd sends this one loss through the full
connected graph and fills all three .grad attributes before any update.

COMPLETE SMALL EXAMPLE: T=1, D=1, V=2
    embedding_table = [[1],[2]]; token_ids = [0]; target_ids = [1]
    output_weights = [[1,0]]; output_bias = [0,0]; learning_rate = 0.1
    selected_vectors = [[1]]; logits = [[1,0]]
    probabilities = [[0.7310585786,0.2689414214]]
    loss = -ln(0.2689414214), approximately 1.3132616875
    gradient_table = [[0.7310585786],[0]]
    gradient_weights = [[0.7310585786,-0.7310585786]]
    gradient_bias = [0.7310585786,-0.7310585786]
    updated table = [[0.9268941421],[2]]
    updated weights = [[0.9268941421,0.0731058579]]
    updated bias = [-0.0731058579,0.0731058579]
    Return the ORIGINAL loss and the three gradient snapshots.
    Recomputing after the update gives logits about [[0.7860268929,0.1408672493]],
    true-target probability about 0.3440811280, loss about 1.0668778121.
    This after-update loss is a separate check, not the function's return.

NEW SYNTAX AND INDENTATION
parameters = (embedding_table, output_weights, output_bias) groups references
to the existing tensors; it does not copy or stack their values.
for parameter in parameters: visits the three tensor objects in order.
parameter.grad = None clears stored gradients without changing parameters.
At function scope use four spaces; a for-loop body uses eight.
with torch.no_grad(): also starts at four spaces for the update block.
The parameter loop inside this context starts at eight spaces; its update
body uses twelve. parameter -= learning_rate * parameter.grad changes the
original tensor in place. Rebinding parameter = parameter - ... does not.
The return is at four spaces, outside both the loop and context.
Forward/backward must stay outside no-grad; clone gradients for copies.

HOW TO REVIEW
The learner core is complete. Review or reconstruct it using the eight hints.
The supplied loader exposes compute_logits and cross_entropy_loss from the
completed sibling exercises. Reuse them; do not rewrite softmax or the loss,
call the squared-error update helper, or manually implement derivatives.
Expected final success line after implementation:
    Lesson 03.11 checks passed.
"""

from __future__ import annotations

import importlib.util
from math import isclose
from pathlib import Path

import torch
import torch.nn.functional as F


def next_token_train_step(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    target_ids: torch.Tensor,
    learning_rate: float,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the original three parameters; return pre-update loss/gradients."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, group the three trainable tensors into
    # parameters = (embedding_table, output_weights, output_bias).
    # A tuple stores references. At four spaces, write a for-loop over
    # parameters; at eight spaces, set each parameter.grad to None.
    # IDs are selectors, not trainable parameters. All values stay unchanged.
    #
    # STEP 2: Back at four spaces, obtain selected_vectors using
    # embedding_table[token_ids]. Shape [T,D]; example [[1]].
    # Use the supplied table directly so lookup stays connected to it.
    #
    # STEP 3: At four spaces, obtain logits by calling the supplied
    # compute_logits(selected_vectors, output_weights, output_bias).
    # Shape [T,V]; example [[1,0]]. No softmax call is needed here.
    #
    # STEP 4: At four spaces, obtain loss by calling the supplied
    # cross_entropy_loss(logits, target_ids). Example scalar 1.3132616875.
    # Keep this tensor connected; do not overwrite it with .item().
    #
    # STEP 5: At four spaces, call loss.backward() once.
    # All three gradients now describe the SAME old parameters.
    # Example table grad [[0.7310585786],[0]], weights grad
    # [[0.7310585786,-0.7310585786]], bias grad [0.7310585786,-0.7310585786].
    # Do not update any parameter before this full backward completes.
    #
    # STEP 6: At four spaces, save loss_value using loss.item(), and
    # gradient_table, gradient_weights, gradient_bias by calling .clone()
    # on the corresponding parameter.grad. These are independent snapshots.
    # Ordinary backward gradients are untracked; no extra detach is needed.
    #
    # STEP 7: At four spaces, enter with torch.no_grad():.
    # At eight spaces, loop for parameter in parameters:.
    # At twelve spaces, use -= to subtract learning_rate * parameter.grad.
    # Example E[0,0] becomes 0.9268941421; E[1,0] stays 2.
    # Keep forward/backward outside this block and avoid .data access.
    #
    # STEP 8: At four spaces, return loss_value, gradient_table,
    # gradient_weights, gradient_bias, in that order.
    # Return pre-update observations. No second forward or outer step loop.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement next_token_train_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_helper(filename: str, function_name: str):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(function_name + "_helper", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return getattr(module, function_name)


compute_logits = _load_helper("03_09_logits_and_softmax.py", "compute_logits")
cross_entropy_loss = _load_helper("03_10_cross_entropy_loss.py", "cross_entropy_loss")


def _parameter(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _example():
    return (_parameter([[1.0], [2.0]]), _ids([0]), _parameter([[1.0, 0.0]]),
            _parameter([0.0, 0.0]), _ids([1]))


def _objective(parameters, token_ids, target_ids) -> torch.Tensor:
    table, weights, bias = parameters
    return F.cross_entropy(table[token_ids] @ weights + bias, target_ids)


def _check_step(embedding_table, token_ids, output_weights, output_bias,
                target_ids, learning_rate):
    parameters = (embedding_table, output_weights, output_bias)
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    pointers = tuple(parameter.data_ptr() for parameter in parameters)
    ids_before, targets_before = token_ids.clone(), target_ids.clone()
    mode_before = torch.is_grad_enabled()
    references = tuple(value.clone().requires_grad_() for value in originals)
    reference_loss = _objective(references, token_ids, target_ids)
    expected_gradients = torch.autograd.grad(reference_loss, references)
    expected_values = tuple(value - learning_rate * gradient
                            for value, gradient in zip(originals, expected_gradients))

    actual = next_token_train_step(embedding_table, token_ids, output_weights,
                                  output_bias, target_ids, learning_rate)
    assert isinstance(actual, tuple) and len(actual) == 4, "Return loss plus three gradients"
    loss_value, *snapshots = actual
    assert isinstance(loss_value, float), "Return the pre-update scalar as a Python float"
    assert isclose(loss_value, reference_loss.item(), rel_tol=1e-9, abs_tol=1e-12), (
        "Return the original loss, not a second forward pass after updating"
    )
    for parameter, snapshot, gradient, expected, pointer in zip(
            parameters, snapshots, expected_gradients, expected_values, pointers):
        assert isinstance(snapshot, torch.Tensor) and snapshot.shape == parameter.shape
        assert snapshot.dtype == parameter.dtype and snapshot.device == parameter.device
        assert not snapshot.requires_grad and snapshot.grad_fn is None
        assert torch.allclose(snapshot, gradient, rtol=1e-9, atol=1e-12)
        assert torch.allclose(parameter.detach(), expected, rtol=1e-9, atol=1e-12), (
            "Update all three original parameters with their own current gradients"
        )
        assert parameter.data_ptr() == pointer, "Keep the original parameter storage"
        assert parameter.requires_grad and parameter.is_leaf and parameter.grad_fn is None
        assert parameter.grad is not None
        assert torch.allclose(parameter.grad, gradient, rtol=1e-9, atol=1e-12)
        assert snapshot.data_ptr() != parameter.grad.data_ptr(), "Clone gradient snapshots"
    assert torch.equal(token_ids, ids_before) and torch.equal(target_ids, targets_before)
    assert not token_ids.requires_grad and not target_ids.requires_grad
    assert torch.is_grad_enabled() == mode_before
    return actual


def check_next_token_step() -> None:
    arguments = _example()
    loss, table_gradient, weights_gradient, bias_gradient = _check_step(*arguments, 0.1)
    g = 0.7310585786300049
    assert isclose(loss, 1.3132616875182228, rel_tol=1e-12)
    assert torch.allclose(table_gradient, torch.tensor([[g], [0.]], dtype=torch.float64))
    assert torch.allclose(weights_gradient, torch.tensor([[g, -g]], dtype=torch.float64))
    assert torch.allclose(bias_gradient, torch.tensor([g, -g], dtype=torch.float64))
    assert torch.equal(arguments[0][1].detach(), torch.tensor([2.], dtype=torch.float64))
    print("PASS: worked example, three parameter gradients, and joint in-place update")

    table = _parameter([[1., 2.], [0., 0.], [2., 1.]])
    weights = _parameter([[1., 0., 1.], [0., 1., 1.]])
    bias = _parameter([0., 0., 0.])
    base_loss, *_ = _check_step(table, _ids([0, 2]), weights, bias, _ids([2, 0]), 0.1)
    assert isclose(base_loss, 0.9076059644443804, rel_tol=1e-12)
    assert torch.equal(table.grad[1], torch.zeros(2, dtype=torch.float64))
    repeated = _parameter([[0.2, -0.1], [0.3, 0.4], [0.5, -0.2]])
    _check_step(repeated, _ids([2, 0, 2]), _parameter([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]),
                _parameter([0.05, -0.2, 0.1]), _ids([1, 2, 0]), 0.05)
    assert torch.equal(repeated.grad[1], torch.zeros(2, dtype=torch.float64))
    print("PASS: multiple positions, varied dimensions, repeated IDs, and untouched input rows")

    _check_step(*_example(), 0.0)
    zero = (_parameter([[2., -1.]]), _ids([0, 0]), _parameter([[0.3], [-0.2]]),
            _parameter([0.1]), _ids([0, 0]))
    zero_loss, *zero_gradients = _check_step(*zero, 0.4)
    assert zero_loss == 0.0 and all(torch.count_nonzero(g) == 0 for g in zero_gradients)
    print("PASS: zero learning rate and single-candidate zero gradients")

    arguments = _example()
    parameters = (arguments[0], arguments[2], arguments[3])
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 99.)
    _check_step(*arguments, 0.1)
    _check_step(*arguments, 0.1)
    print("PASS: all old gradients cleared and repeated calls use updated trainable parameters")

    arguments = _example()
    result = _check_step(*arguments, 0.1)
    snapshots = tuple(snapshot.clone() for snapshot in result[1:])
    for parameter in (arguments[0], arguments[2], arguments[3]):
        parameter.grad.fill_(123.)
    assert all(torch.equal(actual, expected) for actual, expected in zip(result[1:], snapshots))
    print("PASS: independent table, weight, and bias gradient snapshots")

    arguments = (_parameter([[0.2, -0.1], [0.3, 0.4], [0.5, -0.2]]), _ids([2, 0, 2]),
                 _parameter([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]),
                 _parameter([0.05, -0.2, 0.1]), _ids([1, 2, 0]))
    parameters = (arguments[0], arguments[2], arguments[3])
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    _, *gradients = _check_step(*arguments, 0.0)
    h = 1e-6
    for parameter_index, values in enumerate(originals):
        for component in range(values.numel()):
            plus = [value.clone() for value in originals]
            minus = [value.clone() for value in originals]
            plus[parameter_index].view(-1)[component] += h
            minus[parameter_index].view(-1)[component] -= h
            numerical = (_objective(plus, arguments[1], arguments[4]).item()
                         - _objective(minus, arguments[1], arguments[4]).item()) / (2 * h)
            assert isclose(gradients[parameter_index].view(-1)[component].item(), numerical,
                           rel_tol=1e-6, abs_tol=1e-8)
    print("PASS: all parameter gradients agree with trusted autograd and finite differences")

    arguments = _example()
    before, *_ = _check_step(*arguments, 0.1)
    after = _objective((arguments[0], arguments[2], arguments[3]), arguments[1], arguments[4]).item()
    assert isclose(after, 1.066877812077227, rel_tol=1e-12) and after < before
    arguments = _example()
    before, *_ = _check_step(*arguments, 10.0)
    after = _objective((arguments[0], arguments[2], arguments[3]), arguments[1], arguments[4]).item()
    assert after > before, "A large step can increase loss; do not clamp or hide it"
    print("PASS: pre-update reporting, small-step improvement, and possible large-step loss increase")
    print("Lesson 03.11 checks passed.")


if __name__ == "__main__":
    check_next_token_step()
