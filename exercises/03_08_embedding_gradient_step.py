# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.08: train an embedding table through indexed lookup.

Run in the existing PyTorch environment from the repository root:
    python exercises/03_08_embedding_gradient_step.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

IMPLEMENTATION NOTES
The learner uses errors for the elementwise deviations. loss_value is first
the scalar tensor, then becomes a Python float through .item() after backward.
The returned gradient_table.detach() is valid; its .clone() already provides
independent storage, and the gradient is already untracked after ordinary
backward. Keep this distinction between copying and stopping gradient tracking.

PURPOSE AND PREREQUISITES
Connect Lesson 03.01's row lookup to Lesson 03.06's autograd/update logic.
The table now contains many trainable numbers. Read the paired lesson's
formula, complete small example, and repeated-ID explanation before coding.
Target vectors are supplied for this teaching objective; a language model
later receives embedding gradients through its next-token prediction loss.

FORMULAS
    E: embedding_table, shape [V, D]
    ids: token_ids, shape [T]
    Z[t, d] = E[ids[t], d]
    Y: target_vectors, shape [T, D]
    loss = sum((Z[t, d] - Y[t, d]) ** 2 for all t, d) / (T * D)
    G[r, d] = 2 / (T * D) * sum(E[r, d] - Y[t, d] for ids[t] == r)
    E_new = E - learning_rate * G
Repeated IDs send multiple contributions to the same table row. In this
lookup-only objective and plain gradient update, unselected rows have zero
gradient. Token IDs select rows; the floating-point table values are trained.

INPUTS
embedding_table: floating-point torch.Tensor, shape [V, D], supplied as a
    trainable leaf with requires_grad=True. Update this tensor in place.
token_ids: torch.Tensor, shape [T], dtype=torch.long. Valid row indices.
target_vectors: fixed floating-point torch.Tensor, shape [T, D], same dtype
    and device as the table, with requires_grad=False.
learning_rate: nonnegative float, the update multiplier.
Assume finite values, V>0, D>0, T>0, valid IDs, and normal gradient mode.
An empty lookup is possible, but this mean-loss assignment needs T>0.
No input validation or tensor construction is required in the learner code.

OUTPUT AND SIDE EFFECT
Return (loss_value, gradient_table). loss_value is a Python float from
before updating. gradient_table is an independent tensor copy, shape [V, D],
of the current parameter gradient. Use .clone() on the table's grad.
The supplied embedding_table is updated in place and remains trainable.
Leave the current gradient stored; the next call resets it at the start.
Keep token_ids and target_vectors unchanged.

COMPLETE WORKED EXAMPLE
    embedding_table = [[1, 2], [3, 4], [5, 6]]; token_ids = [0, 2]
    target_vectors = [[0, 0], [4, 4]]; learning_rate = 0.1
    selected_vectors = [[1, 2], [5, 6]], shape [2, 2]
    error = [[1, 2], [1, 2]]
    squared errors = [[1, 4], [1, 4]]
    loss = (1 + 4 + 1 + 4) / 4 = 2.5
    gradient_table = [[0.5, 1], [0, 0], [0.5, 1]], shape [3, 2]
    updated table = [[0.95, 1.9], [3, 4], [4.95, 5.9]]
    Return loss 2.5 and the gradient tensor; row 1 is unchanged.
    With IDs [0, 0] and zero targets, two contributions [0.5, 1]
    instead add to gradient [1, 2] on row 0; the mean denominator is still 4.

NEW SYNTAX
embedding_table[token_ids] selects rows in order, including repeated IDs.
    Keep the selected_vectors tensor connected to the table for autograd.
Tensor subtraction and ** 2 act element by element for matching shapes.
tensor.mean() averages ALL its elements when called without a dimension.
    Squared error shape [T, D] becomes a scalar loss; divide by T*D.
tensor.clone() copies tensor values into separate storage. Here copy grad,
    which is already untracked after ordinary backward; clone alone is not
    a general way to disable gradient tracking.
.item() extracts the scalar loss. A multi-element gradient needs a tensor
    return, not .item(). Ordinary statements use four spaces of indentation;
    only the in-place update inside with torch.no_grad(): uses eight.

HOW TO REVIEW
The learner core is complete. Use the seven numbered hints to review or
independently reconstruct it. The setup and check driver are supplied.
No learner loop is needed: tensor operations apply across the selected rows
and components together. Expected final success line:
    Lesson 03.08 checks passed.
"""

from __future__ import annotations

from math import isclose

import torch


def embedding_gradient_step(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    target_vectors: torch.Tensor,
    learning_rate: float,
) -> tuple[float, torch.Tensor]:
    """Return pre-update loss/gradient snapshot and update the supplied table."""
    # ===== START LEARNER CODE =====
    # STEP 1: Clear the previous gradient of embedding_table.
    # Syntax: object.attribute = None. Use the table's grad attribute.
    # Four spaces. This leaves all table values unchanged.
    #
    # STEP 2: Look up selected_vectors using the supplied token_ids tensor.
    # Syntax: table[indices]. This result has shape [T, D].
    # Preserve the tensor graph; do not convert the selected values to lists.
    # Checkpoint: example selected_vectors is [[1, 2], [5, 6]].
    #
    # STEP 3: Calculate error from selected_vectors and target_vectors.
    # Subtract element by element, then square with ** 2 and average with
    # .mean() to obtain loss. No dimension argument means all T*D elements.
    # Checkpoint: error is [[1, 2], [1, 2]]; scalar loss is 2.5.
    #
    # STEP 4: Run backward on the scalar loss once, outside no-grad mode.
    # Syntax: object.method(). The table's grad now has shape [V, D].
    # Checkpoint: [[0.5, 1], [0, 0], [0.5, 1]] in the example.
    # Autograd adds contributions when several positions select the same row.
    #
    # STEP 5: Save the pre-update loss_value and gradient_table.
    # Extract loss with .item(); copy embedding_table.grad using .clone().
    # Keep gradient_table as a tensor so all rows and columns remain visible.
    # A copied snapshot stays independent of later changes to stored grad.
    #
    # STEP 6: At four spaces, enter with torch.no_grad():.
    # At eight spaces inside it, update embedding_table directly with -=.
    # Subtract learning_rate multiplied by embedding_table.grad.
    # Checkpoint: row 0 becomes [0.95, 1.9]; unselected row 1 stays [3, 4].
    # Do not modify selected_vectors instead or access the parameter via .data.
    #
    # STEP 7: At four spaces, outside the context, return loss_value first,
    # then gradient_table. Syntax: return first, second.
    # Both describe the original table; the passed table is now updated.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement embedding_gradient_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _table(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _targets(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _check_step(
    embedding_table: torch.Tensor,
    token_ids: torch.Tensor,
    target_vectors: torch.Tensor,
    learning_rate: float,
    expected_loss: float,
    expected_gradient: torch.Tensor,
    expected_table: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    pointer_before = embedding_table.data_ptr()
    ids_before = token_ids.clone()
    targets_before = target_vectors.clone()
    mode_before = torch.is_grad_enabled()
    actual = embedding_gradient_step(
        embedding_table, token_ids, target_vectors, learning_rate
    )
    assert isinstance(actual, tuple) and len(actual) == 2
    loss_value, gradient_table = actual
    assert isinstance(loss_value, float), "Extract the scalar loss using .item()"
    assert isclose(loss_value, expected_loss, rel_tol=1e-9, abs_tol=1e-12), (
        f"Loss: expected {expected_loss}, got {loss_value}"
    )
    assert isinstance(gradient_table, torch.Tensor), "Return the full gradient tensor"
    assert gradient_table.shape == embedding_table.shape == expected_gradient.shape
    assert gradient_table.dtype == torch.float64 and not gradient_table.requires_grad
    assert torch.allclose(gradient_table, expected_gradient, rtol=1e-9, atol=1e-12), (
        f"Gradient: expected {expected_gradient}, got {gradient_table}"
    )
    assert torch.allclose(embedding_table, expected_table, rtol=1e-9, atol=1e-12), (
        "Update the passed embedding table using its full gradient"
    )
    assert embedding_table.data_ptr() == pointer_before
    assert embedding_table.is_leaf and embedding_table.requires_grad
    assert embedding_table.grad_fn is None
    assert embedding_table.grad is not None
    assert torch.allclose(embedding_table.grad, expected_gradient, rtol=1e-9, atol=1e-12)
    assert torch.is_grad_enabled() == mode_before
    assert torch.equal(token_ids, ids_before) and torch.equal(target_vectors, targets_before)
    assert not token_ids.requires_grad and not target_vectors.requires_grad
    return actual


def check_embedding_step() -> None:
    values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    token_ids = _ids([0, 2])
    target_vectors = _targets([[0.0, 0.0], [4.0, 4.0]])
    gradient = _targets([[0.5, 1.0], [0.0, 0.0], [0.5, 1.0]])
    updated = _targets([[0.95, 1.9], [3.0, 4.0], [4.95, 5.9]])
    _check_step(_table(values), token_ids, target_vectors, 0.1, 2.5, gradient, updated)
    print("PASS: indexed lookup, mean loss, table gradients, and untouched rows")

    _check_step(
        _table(values), _ids([0, 0]), _targets([[0.0, 0.0], [0.0, 0.0]]), 0.1, 2.5,
        _targets([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
        _targets([[0.9, 1.8], [3.0, 4.0], [5.0, 6.0]]),
    )
    print("PASS: repeated token positions add gradient contributions to one row")

    _check_step(
        _table([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]), _ids([1]),
        _targets([[5.0, 3.0, 7.0]]), 0.3, 2.0,
        _targets([[0.0, 0.0, 0.0], [-2 / 3, 4 / 3, -2 / 3]]),
        _targets([[1.0, -2.0, 3.0], [4.2, 4.6, 6.2]]),
    )
    large_values = torch.zeros((258, 2), dtype=torch.float64)
    large_values[5] = _targets([[7.0, 8.0]])[0]
    large_values[97] = _targets([[1.0, 2.0]])[0]
    large_values[256] = _targets([[3.0, 4.0]])[0]
    large_table = large_values.clone().requires_grad_()
    large_gradient = torch.zeros_like(large_values)
    large_gradient[97] = _targets([[1 / 3, 2 / 3]])[0]
    large_gradient[256] = _targets([[2.0, 8 / 3]])[0]
    large_updated = large_values.clone()
    large_updated[97] = _targets([[0.95, 1.9]])[0]
    large_updated[256] = _targets([[2.7, 3.6]])[0]
    _check_step(
        large_table, _ids([256, 97, 256]), torch.zeros((3, 2), dtype=torch.float64),
        0.15, 55 / 6, large_gradient, large_updated,
    )
    print("PASS: different vector dimensions and vocabulary IDs above 255")

    _check_step(
        _table(values), _ids([0]), _targets([[1.0, 2.0]]), 0.1, 0.0,
        torch.zeros((3, 2), dtype=torch.float64), _targets(values),
    )
    _check_step(_table(values), token_ids, target_vectors, 0.0, 2.5, gradient, _targets(values))
    print("PASS: zero loss and zero learning rate")

    embedding_table = _table(values)
    embedding_table.grad = torch.full_like(embedding_table, 99.0)
    _check_step(embedding_table, token_ids, target_vectors, 0.1, 2.5, gradient, updated)
    _check_step(
        embedding_table, token_ids, target_vectors, 0.1, 2.25625,
        _targets([[0.475, 0.95], [0.0, 0.0], [0.475, 0.95]]),
        _targets([[0.9025, 1.805], [3.0, 4.0], [4.9025, 5.805]]),
    )
    print("PASS: old gradients are cleared and repeated steps use the updated table")

    embedding_table = _table(values)
    _, snapshot = _check_step(
        embedding_table, token_ids, target_vectors, 0.1, 2.5, gradient, updated
    )
    embedding_table.grad.fill_(123.0)
    assert torch.allclose(snapshot, gradient, rtol=1e-9, atol=1e-12), (
        "Return an independent gradient copy using .clone()"
    )
    print("PASS: returned gradient snapshots have independent storage")

    embedding_table = _table([[0.2, -0.1], [0.3, 0.4], [0.5, -0.2]])
    fd_ids = _ids([2, 0, 2])
    fd_targets = _targets([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    original = embedding_table.detach().clone()
    expected_gradient = _targets([[1 / 15, -11 / 30], [0.0, 0.0], [1 / 3, -7 / 15]])
    _, actual_gradient = _check_step(
        embedding_table, fd_ids, fd_targets, 0.0, 5.23 / 6,
        expected_gradient, original,
    )
    h = 1e-6
    for row in range(original.shape[0]):
        for column in range(original.shape[1]):
            plus = original.clone()
            minus = original.clone()
            plus[row, column] += h
            minus[row, column] -= h
            loss_plus = ((plus[fd_ids] - fd_targets) ** 2).mean().item()
            loss_minus = ((minus[fd_ids] - fd_targets) ** 2).mean().item()
            numerical = (loss_plus - loss_minus) / (2 * h)
            assert isclose(actual_gradient[row, column].item(), numerical,
                           rel_tol=1e-6, abs_tol=1e-8)
    print("PASS: full table gradients agree with finite differences")
    print("Lesson 03.08 checks passed.")


if __name__ == "__main__":
    check_embedding_step()
