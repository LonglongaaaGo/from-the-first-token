# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.03: connect scalar updates into a fixed-length training loop.

Run from the repository root to review the completed train_scalar:
    python3 exercises/03_03_scalar_training_loop.py
Expected final success line:
    Lesson 03.03 checks passed.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Lesson 03.02 performs one update. This lesson repeats that operation while
carrying the latest parameter forward and recording the loss used each time.
Reuse the completed gradient_step; no additional package is required.

INPUTS
initial_w: float, the parameter before training begins, for example 0.2.
target: float, the fixed desired prediction, for example 1.0.
learning_rate: float, a positive step-size multiplier, for example 0.1.
num_steps: int, the number of updates to perform, for example 3.
Assume finite floats, a positive learning_rate, and num_steps >= 0.

OUTPUT
A tuple (w, losses): the final parameter as a float and a list of floats.
losses records the loss BEFORE each update, in order. Its length is num_steps.
For zero updates, return the initial parameter and an empty list.

COMPLETE WORKED EXAMPLE
    initial_w=0.2, target=1.0, learning_rate=0.1, num_steps=3
    Before looping: w=0.2, losses=[]
    Iteration 1: gradient_step returns approximately (0.64, -1.6, 0.36).
        Record loss 0.64 and keep w=0.36; losses=[0.64].
    Iteration 2: gradient_step returns approximately (0.4096, -1.28, 0.488).
        Record loss 0.4096 and keep w=0.488; losses=[0.64, 0.4096].
    Iteration 3: gradient_step returns approximately (0.262144, -1.024, 0.5904).
        Record loss 0.262144 and keep w=0.5904.
    Return approximately (0.5904, [0.64, 0.4096, 0.262144]).
    The loss at the final parameter is 0.16777216; it is not another history
    item, because no fourth update was requested.

SYNTAX BEFORE IMPLEMENTATION
range(3) produces iteration counters 0, 1, 2: exactly three iterations.
for step in range(num_steps): repeats its indented body num_steps times.
range(0) is empty, so its loop body does not run.
Tuple unpacking assigns values by position: a, b, c = (10, 20, 30).
The supplied gradient_step returns (loss, gradient, w_new) in that order.
list.append(value) adds one item. Here append one scalar loss per iteration.

HOW TO WORK
Review the completed learner region in train_scalar with the numbered hints.
Setup, loop header, and final return use four spaces of indentation; the loop
body uses eight spaces. The learner combines helper-result unpacking and
parameter replacement by assigning the third result directly to w.
The supplied setup and check driver remain unchanged.
"""

from math import isclose
from pathlib import Path
import runpy


# PROVIDED SETUP: load the completed helper without running its check driver.
_exercise_dir = Path(__file__).resolve().parent
gradient_step = runpy.run_path(
    str(_exercise_dir / "03_02_scalar_gradient_step.py")
)["gradient_step"]


def train_scalar(
    initial_w: float, target: float, learning_rate: float, num_steps: int
) -> tuple[float, list[float]]:
    """Return the final parameter and the loss before each requested update."""
    # ===== START LEARNER CODE =====
    # STEP 1: Initialize the working parameter w from initial_w.
    # Hint: name = value creates or updates a variable binding.
    # Start here once per call, outside the loop, at four-space indentation.
    #
    # STEP 2: Create a fresh empty list called losses, also outside the loop.
    # Syntax: [] is an empty list. For zero updates, it should stay empty.
    #
    # STEP 3: Repeat exactly num_steps times with a for loop.
    # Syntax: for step in range(num_steps):
    # The header uses four spaces; steps 4 through 6 use eight spaces.
    # step counts repetitions; it is not the value of w or a step size.
    #
    # STEP 4: Call gradient_step with the current w, target, and learning_rate.
    # Unpack its returned tuple into loss, gradient, and w_new in that order.
    # Syntax example: first, second, third = some_function(arguments).
    # Reusing this helper recalculates loss and gradient at the current w.
    # Checkpoint: iteration 2 should call it with w=0.36, not initial_w=0.2.
    #
    # STEP 5: Append the returned loss to losses, inside the loop.
    # Syntax: collection.append(item). Store a float, not the whole tuple.
    # Checkpoint: after two iterations, losses is approximately [0.64, 0.4096].
    #
    # STEP 6: Keep w_new as the working w for the next iteration.
    # Hint: update the variable that the next gradient_step call reads.
    # The completed learner code combines steps 4 and 6 by assigning the
    # third returned value directly to w, after the helper uses the old w.
    # Checkpoint: after the second iteration, w should be 0.488.
    #
    # STEP 7: After the loop, return w and losses in that order.
    # Syntax: return first, second returns a two-item tuple.
    # Use four-space indentation; returning inside the loop stops too early
    # and does not handle num_steps=0.
    #
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement train_scalar for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def _check_result(
    initial_w: float,
    target: float,
    learning_rate: float,
    num_steps: int,
    expected_w: float,
    expected_losses: list[float],
) -> tuple[float, list[float]]:
    actual = train_scalar(initial_w, target, learning_rate, num_steps)
    assert isinstance(actual, tuple) and len(actual) == 2, (
        "Return (w, losses) in that order"
    )
    w, losses = actual
    assert isclose(w, expected_w, rel_tol=1e-9, abs_tol=1e-12), (
        f"Final w: expected {expected_w}, received {w}"
    )
    assert isinstance(losses, list) and len(losses) == num_steps, (
        "Record exactly one pre-update loss per requested step"
    )
    for index, (loss, expected) in enumerate(zip(losses, expected_losses)):
        assert isclose(loss, expected, rel_tol=1e-9, abs_tol=1e-12), (
            f"Loss at iteration {index + 1}: expected {expected}, received {loss}"
        )
    return w, losses


def check_training() -> None:
    _check_result(0.2, 1.0, 0.1, 3, 0.5904, [0.64, 0.4096, 0.262144])
    print("PASS: current parameter, recalculated gradients, and loss history")

    _check_result(-2.0, 1.0, 0.1, 0, -2.0, [])
    _check_result(0.2, 1.0, 0.1, 1, 0.36, [0.64])
    print("PASS: zero and one update")

    _check_result(
        0.2, 1.0, 0.1, 5, 0.737856,
        [0.64, 0.4096, 0.262144, 0.16777216, 0.1073741824],
    )
    _, first_losses = _check_result(0.2, 1.0, 0.1, 1, 0.36, [0.64])
    _, second_losses = _check_result(0.2, 1.0, 0.1, 1, 0.36, [0.64])
    first_losses.append(99.0)
    assert len(second_losses) == 1 and isclose(
        second_losses[0], 0.64, rel_tol=1e-9, abs_tol=1e-12
    ), "Each call needs its own loss history"
    print("PASS: requested step count and separate calls")

    _check_result(0.6, -0.4, 0.25, 3, -0.275, [1.0, 0.25, 0.0625])
    print("PASS: different starting values, targets, and learning rates")

    _check_result(1.0, 1.0, 0.25, 4, 1.0, [0.0, 0.0, 0.0, 0.0])
    print("PASS: zero gradient still records every requested step")

    _check_result(0.2, 1.0, 1.2, 2, -0.568, [0.64, 1.2544])
    print("PASS: large steps remain visible in loss history")
    print("Lesson 03.03 checks passed.")


if __name__ == "__main__":
    check_training()
