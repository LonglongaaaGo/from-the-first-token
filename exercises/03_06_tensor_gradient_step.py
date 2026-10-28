# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.06: update one persistent tensor parameter using autograd.

Run from the repository root in an existing PyTorch environment:
    python exercises/03_06_tensor_gradient_step.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect Lesson 03.05's backward calculation to gradient descent. Review
the paired lesson's formulas, complete trace, gradient reset, no_grad,
and indentation before implementing. Earlier learner code is preserved.

FORMULAS
    prediction = w * x
    error = prediction - target
    loss = error ** 2
    gradient = d(loss)/d(w) = 2 * error * x
    w_new = w - learning_rate * gradient
Use the gradient formula for reasoning; get the implemented gradient
from loss.backward() and w_tensor.grad.

INPUTS
w_tensor: a supplied scalar, floating-point torch.Tensor. It is a leaf
    with requires_grad=True. Update this same tensor in place. Do not
    construct a replacement parameter. Its grad may be None or old data.
x: float, a fixed input multiplier.
target: float, a fixed desired prediction.
learning_rate: float, a nonnegative update multiplier.
Assume finite inputs and the stated tensor prerequisites; no validation
is required. Supplied checks create float64 CPU parameter tensors.

OUTPUT AND SIDE EFFECT
Return (loss_value, grad_value), two Python floats describing the
parameter BEFORE updating. The passed w_tensor now contains w_new and
remains a trainable leaf. Leave this step's gradient in w_tensor.grad;
the next call resets it at the start.

COMPLETE WORKED EXAMPLE
    w_tensor starts at 0.2; x=2.0; target=1.0; learning_rate=0.1.
    Reset any old gradient to None.
    Forward: predicted_tensor=0.4, error=-0.6, loss=0.36.
    Backward: w_tensor.grad=-2.4; w_tensor still holds 0.2.
    Save loss_value=0.36 and grad_value=-2.4 with .item().
    Update: w_new=0.2 - 0.1 * (-2.4)=0.44.
    Return approximately (0.36, -2.4); the caller's tensor holds 0.44.
    A second call uses 0.44, giving loss=0.0144, gradient=-0.48,
    and the next parameter value 0.488.

NEW SYNTAX AND PRINCIPLES
w_tensor.grad = None discards the previous stored gradient; it does not
    change the parameter value. Backward can then store a fresh gradient.
with torch.no_grad(): temporarily stops reverse-mode graph recording
    for its indented block. Keep forward and backward outside this block.
    The colon starts the block; leaving it restores the previous mode.
-= subtracts a value in place for this supplied tensor, preserving the
    parameter the caller holds. Example syntax elsewhere: counter -= 2.
Inside this function, ordinary lines use four spaces; the with block's
update uses eight. Return at four spaces, outside the with block.

HOW TO WORK
Review the completed learner region using its numbered hints. Preserve
the learner implementation. The imports, setup, and checks are supplied.
Verified final success line:
    Lesson 03.06 checks passed.
"""

from __future__ import annotations

from math import isclose

import torch


def tensor_gradient_step(
    w_tensor: torch.Tensor,
    x: float,
    target: float,
    learning_rate: float,
) -> tuple[float, float]:
    """Return pre-update loss/gradient and update the supplied tensor in place."""
    # ===== START LEARNER CODE =====
    # STEP 1: Reset the previous gradient on w_tensor to None.
    # Syntax: object.attribute = value. Use the grad attribute and None.
    # Do this before backward; grad may be None on the first call already.
    # Checkpoint: the example parameter still holds 0.2.
    #
    # STEP 2: Calculate predicted_tensor using w_tensor and x.
    # Hint: ordinary assignment and multiplication with *.
    # Checkpoint: the example predicted_tensor holds 0.4.
    #
    # STEP 3: Calculate error, then the scalar loss.
    # Hint: subtract target from predicted_tensor; square with ** 2.
    # Checkpoint: error=-0.6 and loss=0.36, both still tensors.
    #
    # STEP 4: Run backward on loss once.
    # Syntax: object.method(). This is still in normal gradient mode.
    # Checkpoint: w_tensor.grad holds -2.4; w_tensor still holds 0.2.
    #
    # STEP 5: Save loss_value and grad_value as Python floats.
    # Hint: .item() on loss and on w_tensor.grad, respectively.
    # Both values describe the parameter before the update.
    #
    # STEP 6: Enter with torch.no_grad(): at four-space indentation.
    # Inside its block, update w_tensor using -= at eight spaces.
    # Subtract the product of learning_rate and w_tensor.grad.
    # Checkpoint: this SAME tensor now holds 0.44 and remains trainable.
    # Keep forward/backward outside this block; avoid replacing the tensor.
    #
    # STEP 7: Return loss_value, then grad_value at four-space indentation.
    # Syntax: return first, second. This line is outside the with block.
    # Checkpoint: return approximately (0.36, -2.4), not the new loss.
    # Leave the current gradient stored; the next call clears it at STEP 1.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement tensor_gradient_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _make_parameter(value: float) -> torch.Tensor:
    """Create the scalar trainable leaf used by the supplied checks."""
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def _check_step(
    w_tensor: torch.Tensor,
    x: float,
    target: float,
    learning_rate: float,
    expected: tuple[float, float, float],
) -> tuple[float, float]:
    pointer_before = w_tensor.data_ptr()
    mode_before = torch.is_grad_enabled()
    actual = tensor_gradient_step(w_tensor, x, target, learning_rate)
    assert isinstance(actual, tuple) and len(actual) == 2, (
        "Return only (loss_value, grad_value); update the supplied tensor"
    )
    for name, value, wanted in zip(("loss", "gradient"), actual, expected[:2]):
        assert isinstance(value, float), f"{name}: use .item() to return a float"
        assert isclose(value, wanted, rel_tol=1e-9, abs_tol=1e-12), (
            f"{name}: expected {wanted}, received {value}"
        )
    assert isclose(w_tensor.item(), expected[2], rel_tol=1e-9, abs_tol=1e-12), (
        f"Update the caller's tensor: expected {expected[2]}, got {w_tensor.item()}"
    )
    assert w_tensor.data_ptr() == pointer_before, "Preserve parameter storage"
    assert w_tensor.ndim == 0 and w_tensor.dtype == torch.float64
    assert w_tensor.is_leaf and w_tensor.requires_grad
    assert w_tensor.grad_fn is None, "Keep the update outside the backward graph"
    assert torch.is_grad_enabled() == mode_before, "Restore the prior grad mode"
    assert w_tensor.grad is not None, "Keep the current gradient after the step"
    assert isclose(
        w_tensor.grad.item(), expected[1], rel_tol=1e-9, abs_tol=1e-12
    ), "Stored gradient should describe this step, not accumulated old steps"
    return actual


def check_tensor_step() -> None:
    """Check updates, persistent state, resets, and boundary cases."""
    _check_step(_make_parameter(0.2), 2.0, 1.0, 0.1, (0.36, -2.4, 0.44))
    _check_step(_make_parameter(0.6), 2.0, 1.0, 0.1, (0.04, 0.8, 0.52))
    print("PASS: both gradient directions and in-place trainable parameters")

    w_tensor = _make_parameter(0.2)
    _check_step(w_tensor, 2.0, 1.0, 0.1, (0.36, -2.4, 0.44))
    _check_step(w_tensor, 2.0, 1.0, 0.1, (0.0144, -0.48, 0.488))
    print("PASS: repeated steps use current parameters and fresh gradients")

    w_tensor = _make_parameter(0.2)
    w_tensor.grad = torch.tensor(99.0, dtype=torch.float64)
    _check_step(w_tensor, 2.0, 1.0, 0.1, (0.36, -2.4, 0.44))
    print("PASS: previously stored gradients are cleared")

    _check_step(_make_parameter(0.5), 2.0, 1.0, 0.1, (0.0, 0.0, 0.5))
    _check_step(_make_parameter(0.7), 0.0, 1.0, 0.1, (1.0, 0.0, 0.7))
    _check_step(_make_parameter(0.2), 2.0, 1.0, 0.0, (0.36, -2.4, 0.2))
    print("PASS: zero gradients and zero learning rate")

    _check_step(_make_parameter(0.6), -2.0, 1.0, 0.1, (4.84, 8.8, -0.28))
    _check_step(_make_parameter(0.6), 3.0, -0.4, 0.05, (4.84, 13.2, -0.06))
    print("PASS: changed inputs, targets, and learning rates")

    w_tensor = _make_parameter(0.2)
    original_loss, _ = _check_step(w_tensor, 2.0, 1.0, 1.0, (0.36, -2.4, 2.6))
    updated_loss = (w_tensor.item() * 2.0 - 1.0) ** 2
    assert isclose(updated_loss, 17.64) and updated_loss > original_loss
    print("PASS: a large learning rate can increase loss")
    print("Lesson 03.06 checks passed.")


if __name__ == "__main__":
    check_tensor_step()
