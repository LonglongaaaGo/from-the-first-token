# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.05: calculate a scalar loss and gradient with PyTorch autograd.

Run from the repository root in an existing PyTorch environment:
    python exercises/03_05_scalar_autograd.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect the chain-rule formula to automatic differentiation. Review the
paired lesson's formula, reverse rule, and complete trace before editing.
This first tensor exercise calculates a loss and gradient; parameter
updates and repeated training follow in a later lesson.

FORMULAS
    prediction = w * x
    error = prediction - target
    loss = error ** 2
    d(loss)/d(w) = 2 * error * x
Use the derivative formula to understand and check the result. In the
learner code, obtain the gradient through backward() and w_tensor.grad.
The formula's w and prediction are named w_tensor and prediction_tensor
in the learner implementation; these names represent the same quantities.

INPUTS
w_value: float, the initial parameter number, for example 0.2.
x: float, the fixed input multiplier, for example 2.0.
target: float, the fixed desired prediction, for example 1.0.
Assume finite inputs. A learning rate is not an input to this function.

OUTPUT
A tuple (loss_value, grad_value) containing two Python floats.
Both describe the original parameter value. No parameter update is needed.

COMPLETE WORKED EXAMPLE
    w_value=0.2, x=2.0, target=1.0
    Forward: prediction=0.4, error=-0.6, loss=0.36.
    Reverse seed: d(loss)/d(loss)=1.
    Through square: d(loss)/d(error)=1 * 2 * (-0.6)=-1.2.
    Through subtraction: d(loss)/d(prediction)=-1.2 * 1=-1.2.
    Through multiplication: d(loss)/d(w)=-1.2 * 2.0=-2.4.
    Before backward: w_tensor.grad is None.
    After backward: w_tensor.grad contains approximately -2.4;
                    w_tensor remains 0.2.
    Return approximately (0.36, -2.4).

NEW SYNTAX
torch.tensor(value, dtype=..., requires_grad=...) creates a tensor.
torch.float64 selects floating-point precision; True enables recording.
object.method() calls a method. object.attribute reads a stored attribute.
loss.backward() runs reverse differentiation on the scalar loss.
w_tensor.grad stores the resulting parameter gradient as a tensor.
one_element_tensor.item() extracts its ordinary Python number.
return first, second returns a two-item tuple.

HOW TO WORK
Review the completed learner region using its numbered hints. Preserve
the learner's implementation. All lines use four spaces of indentation;
there is no loop or conditional in this function. The supplied imports and
check driver are already complete. Expected final success line:
    Lesson 03.05 checks passed.
"""

from __future__ import annotations

from math import isclose

import torch


def autograd_loss_and_gradient(
    w_value: float, x: float, target: float
) -> tuple[float, float]:
    """Return the original loss and its parameter gradient as Python floats."""
    # ===== START LEARNER CODE =====
    # STEP 1: Create a new scalar tensor w_tensor from the argument w_value.
    # Syntax: torch.tensor(value, dtype=..., requires_grad=...).
    # Use dtype=torch.float64 and requires_grad=True.
    # True is Python's Boolean true value; the capital T matters.
    # Checkpoint: w_tensor holds 0.2; w_tensor.grad is initially None.
    # A fresh tensor per call gives each calculation its own graph/gradient.
    #
    # STEP 2: Calculate prediction_tensor from w_tensor and fixed float x.
    # Hint: multiply with * and store the result in prediction_tensor.
    # Keep this result as a tensor; do not extract a Python number yet.
    # Checkpoint: the example prediction holds 0.4.
    #
    # STEP 3: Calculate error from prediction_tensor and target.
    # Hint: subtract the fixed target and store the result in error.
    # Checkpoint: the example error holds -0.6.
    #
    # STEP 4: Calculate loss by squaring error.
    # Syntax: value ** 2. Keep loss as a tensor connected to w_tensor.
    # Checkpoint: the example scalar loss holds 0.36.
    #
    # STEP 5: Ask the scalar loss to run backward differentiation.
    # Syntax: call the backward method on loss with empty parentheses.
    # Call it once, after the forward calculations and before reading grad.
    # Checkpoint: w_tensor.grad now holds -2.4; w_tensor still holds 0.2.
    # This call populates gradients; it does not return the gradient number.
    #
    # STEP 6: Read the parameter gradient and extract its Python number.
    # Syntax: w_tensor.grad reads the tensor; .item() extracts its value.
    # Store that number in grad_value. It should be -2.4 in the example.
    # Do not implement the hand-derived gradient formula in this exercise.
    #
    # STEP 7: Return the loss number and grad_value, in that order.
    # Hint: call .item() on loss for the first returned value.
    # Syntax: return first, second creates a two-item tuple.
    # Checkpoint: the example returns approximately (0.36, -2.4).
    # Keep return at four-space indentation, like the calculations above.
    #
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement autograd_loss_and_gradient for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def _check_result(
    w_value: float,
    x: float,
    target: float,
    expected: tuple[float, float],
) -> tuple[float, float]:
    actual = autograd_loss_and_gradient(w_value, x, target)
    assert isinstance(actual, tuple) and len(actual) == 2, (
        "Return (loss_value, gradient_value) in that order"
    )
    for name, value, wanted in zip(("loss", "gradient"), actual, expected):
        assert isinstance(value, float), (
            f"{name}: return a Python float using .item(), not a tensor"
        )
        assert isclose(value, wanted, rel_tol=1e-9, abs_tol=1e-12), (
            f"{name}: expected {wanted}, received {value}"
        )
    return actual


def check_autograd() -> None:
    """Check known gradients, changed inputs, independence, and a numerical check."""
    _check_result(0.2, 2.0, 1.0, (0.36, -2.4))
    _check_result(0.6, 2.0, 1.0, (0.04, 0.8))
    print("PASS: negative and positive parameter gradients")

    _check_result(0.2, 1.0, 1.0, (0.64, -1.6))
    print("PASS: unit input matches the earlier scalar loss")

    _check_result(0.6, -2.0, 1.0, (4.84, 8.8))
    _check_result(0.6, 3.0, -0.4, (4.84, 13.2))
    print("PASS: changed inputs and targets")

    _check_result(0.5, 2.0, 1.0, (0.0, 0.0))
    _check_result(0.7, 0.0, 1.0, (1.0, 0.0))
    print("PASS: zero gradients with zero and nonzero losses")

    _check_result(0.6, 2.0, 1.0, (0.04, 0.8))
    _check_result(0.2, 2.0, 1.0, (0.36, -2.4))
    _check_result(0.6, 2.0, 1.0, (0.04, 0.8))
    print("PASS: separate calls produce independent results")

    h = 1e-6
    for w_value, x, target in (
        (-0.7, 1.5, 0.2), (0.3, -0.5, 0.4),
        (0.5, 2.0, 1.0), (0.9, 0.0, -0.8),
    ):
        _, gradient = autograd_loss_and_gradient(w_value, x, target)
        loss_plus = ((w_value + h) * x - target) ** 2
        loss_minus = ((w_value - h) * x - target) ** 2
        numerical_gradient = (loss_plus - loss_minus) / (2 * h)
        assert isclose(gradient, numerical_gradient, rel_tol=1e-6, abs_tol=1e-8), (
            "Autograd gradient should agree with a small central difference"
        )
    print("PASS: finite-difference gradient agreement")
    print("Lesson 03.05 checks passed.")


if __name__ == "__main__":
    check_autograd()
