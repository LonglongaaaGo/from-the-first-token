# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.02: implement one scalar gradient-descent update.

Run from the repository root:
    python3 exercises/03_02_scalar_gradient_step.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Turn the lesson's squared-error example into one update of one parameter.
Review loss, gradient sign, learning rate, and the worked update first.
This exercise uses ordinary Python floats. It needs no additional package.
Later lessons introduce automatic differentiation and trainable tensors.

TASK
The prediction is w itself. The loss is (w - target) squared, its gradient
is twice the prediction error, and gradient descent subtracts the learning
rate times that gradient from the current parameter. Perform one update.

INPUTS
w: float, the current adjustable parameter and prediction, for example 0.2.
target: float, the fixed desired prediction, for example 1.0.
learning_rate: float, a positive step-size multiplier, for example 0.1.
Assume finite inputs and a positive learning_rate; validation is supplied
by the exercise contract and is not part of the learner task.

OUTPUT
A tuple of three floats, in this order: (loss, gradient, w_new).
loss and gradient describe the original w; w_new is the updated parameter.
Returning w_new makes it available to the caller for the next update.

COMPLETE WORKED EXAMPLE
    w = 0.2, target = 1.0, learning_rate = 0.1
    error: 0.2 - 1.0 = -0.8
    loss: (-0.8) squared = 0.64
    gradient: 2 times (-0.8) = -1.6
    w_new: 0.2 - 0.1 times (-1.6) = 0.36
    returned tuple: (0.64, -1.6, 0.36), allowing float rounding.
    The supplied driver checks the new loss separately: 0.4096.

HOW TO WORK
Review the completed learner region in gradient_step using the retained
numbered hints. Calculation and return lines use four spaces of indentation.
No loop is needed for this one-update function. The supplied check driver
below remains unchanged. Its final success line is:
    Lesson 03.02 checks passed.
"""

from math import isclose


def gradient_step(
    w: float, target: float, learning_rate: float
) -> tuple[float, float, float]:
    """Return original loss, original gradient, and the updated parameter."""
    # ===== START LEARNER CODE =====
    # STEP 1: Calculate the prediction error and call it error.
    # Hint: subtract the fixed target from the current w.
    # Checkpoint: w=0.2 and target=1.0 give error=-0.8.
    #
    # STEP 2: Square error and store the result in loss.
    # Syntax: value ** 2 squares a number. ^ is not Python's power operator.
    # Checkpoint: the example loss is 0.64, a nonnegative value.
    #
    # STEP 3: Calculate gradient using the derivative taught in the lesson.
    # Hint: multiply error by 2, keeping its sign. Use * for multiplication.
    # Checkpoint: the example gradient is -1.6, calculated at the old w.
    #
    # STEP 4: Calculate w_new using gradient descent.
    # Hint: scale gradient by learning_rate, then subtract that quantity
    # from w. Use the arguments rather than hard-coded example numbers.
    # Checkpoint: the example w_new is 0.36.
    #
    # STEP 5: Return loss, gradient, and w_new in that exact order.
    # Syntax: return first, second, third returns a three-item tuple.
    # Keep return at the same indentation as the calculations above.
    #
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement gradient_step for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def _check_result(
    w: float,
    target: float,
    learning_rate: float,
    expected: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Compare known results with tolerances for floating-point rounding."""
    actual = gradient_step(w, target, learning_rate)
    assert isinstance(actual, tuple) and len(actual) == 3, (
        "Return a tuple in the order (loss, gradient, w_new)"
    )
    for name, value, wanted in zip(("loss", "gradient", "w_new"), actual, expected):
        assert isclose(value, wanted, rel_tol=1e-9, abs_tol=1e-12), (
            f"{name}: expected {wanted}, received {value}"
        )
    return actual


def check_step() -> None:
    """Check update direction, target dependence, and learning-rate effects."""
    loss, gradient, w_new = _check_result(0.2, 1.0, 0.1, (0.64, -1.6, 0.36))
    assert gradient < 0 and w_new > 0.2
    assert (w_new - 1.0) ** 2 < loss
    print("PASS: negative gradient increases w and reduces loss")

    loss, gradient, w_new = _check_result(1.2, 1.0, 0.1, (0.04, 0.4, 1.16))
    assert gradient > 0 and w_new < 1.2
    assert (w_new - 1.0) ** 2 < loss
    print("PASS: positive gradient decreases w and reduces loss")

    _check_result(1.0, 1.0, 0.1, (0.0, 0.0, 1.0))
    print("PASS: zero gradient leaves the parameter unchanged")

    _check_result(0.6, -0.4, 0.25, (1.0, 2.0, 0.1))
    _check_result(0.2, 1.0, 0.05, (0.64, -1.6, 0.28))
    print("PASS: supplied target and learning rate control the update")

    loss, gradient, w_new = _check_result(0.2, 1.0, 1.2, (0.64, -1.6, 2.12))
    assert (w_new - 1.0) ** 2 > loss
    print("PASS: a large learning rate can increase loss")
    print("Lesson 03.02 checks passed.")


if __name__ == "__main__":
    check_step()
