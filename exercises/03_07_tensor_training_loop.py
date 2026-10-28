# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.07: repeat tensor updates and record pre-update losses.

Run from the repository root in an existing PyTorch environment:
    python exercises/03_07_tensor_training_loop.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect the completed Lesson 03.06 tensor_gradient_step to a fixed-length
training loop. Review the formula, complete trace, and numbered hints in
the paired lesson before editing. Earlier learner implementations remain
unchanged. Imports and helper loading are supplied below.

FORMULAS
    w_0 = initial_w
    prediction_t = w_t * x
    loss_t = (prediction_t - target) ** 2
    gradient_t = 2 * (w_t * x - target) * x
    w_(t+1) = w_t - learning_rate * gradient_t
Each step uses the latest parameter. The helper obtains the gradient
with autograd; the handwritten formula is for reasoning and checking.

INPUTS
initial_w: float, the initial parameter value, for example 0.2.
x: float, the fixed input multiplier, for example 2.0.
target: float, the fixed desired prediction, for example 1.0.
learning_rate: float, a nonnegative update multiplier, for example 0.1.
num_steps: int, the number of updates to perform, for example 3.
Assume finite floats and num_steps >= 0. Input validation is not required.

OUTPUT
A tuple (final_w, losses): a Python float and a fresh list of Python floats.
losses records loss BEFORE each requested update, in execution order.
Its length is num_steps. For zero updates, return (initial_w, []).

HELPER CONTRACT
tensor_gradient_step(w_tensor, x, target, learning_rate) returns the
pre-update (loss_value, grad_value) and changes w_tensor in place.
It already clears old gradients, computes a new forward graph, runs
backward, and applies the update inside its own torch.no_grad() block.
Call it once per loop iteration, passing the same parameter tensor.
No extra backward call, parameter update, or outer no-grad block is needed.

COMPLETE WORKED EXAMPLE
    initial_w=0.2, x=2.0, target=1.0, learning_rate=0.1, num_steps=3
    Before looping: one w_tensor holds 0.2; losses=[].
    Iteration 1: helper returns approximately (0.36, -2.4).
        That same tensor now holds 0.44; losses becomes [0.36].
    Iteration 2: helper returns approximately (0.0144, -0.48).
        That same tensor now holds 0.488; losses becomes [0.36, 0.0144].
    Iteration 3: helper returns approximately (0.000576, -0.096).
        That same tensor now holds 0.4976.
        losses becomes [0.36, 0.0144, 0.000576].
    Return approximately (0.4976, [0.36, 0.0144, 0.000576]).
    Loss at the final parameter is 0.00002304. It is not another history
    item because a fourth update was not requested.

SYNTAX AND INDENTATION
torch.tensor(value, dtype=torch.float64, requires_grad=True) creates the
    trainable scalar. Create it once per function call, outside the loop.
[] creates an empty list. Make a fresh one for each function call.
for i in range(num_steps): repeats its indented body num_steps times.
    range(3) gives counters 0, 1, 2; range(0) runs no iterations.
    i counts repetitions; it is not a learning rate or parameter value.
Tuple unpacking assigns results by position: first, second = (10, 20).
    The helper returns loss_value first and grad_value second.
losses.append(value) adds one loss number, not the whole returned tuple.
w_tensor.item() extracts the final Python number after the loop.
return first, second returns a two-item tuple.
Setup, loop header, and final return use four spaces of indentation.
The helper call and append inside the loop use eight spaces.

HOW TO WORK
Review the completed learner region using its numbered hints. Preserve
the learner implementation and the supplied helper loader/check driver.
The loop uses normal gradient mode; the helper controls its update block.
Verified final success line:
    Lesson 03.07 checks passed.
"""

from __future__ import annotations

from math import isclose
from pathlib import Path
import runpy

import torch


# PROVIDED SETUP: load the completed helper without running its check driver.
_exercise_dir = Path(__file__).resolve().parent
tensor_gradient_step = runpy.run_path(
    str(_exercise_dir / "03_06_tensor_gradient_step.py")
)["tensor_gradient_step"]


def train_tensor(
    initial_w: float,
    x: float,
    target: float,
    learning_rate: float,
    num_steps: int,
) -> tuple[float, list[float]]:
    """Return the final parameter number and each pre-update loss."""
    # ===== START LEARNER CODE =====
    # STEP 1: Create w_tensor from initial_w once, outside the loop.
    # Syntax: torch.tensor(value, dtype=..., requires_grad=...).
    # Use dtype=torch.float64 and requires_grad=True, as in Lesson 03.05.
    # Four-space indentation. Checkpoint: the example tensor holds 0.2.
    # Do not recreate it inside the loop: updates must carry forward.
    #
    # STEP 2: Create a fresh empty list called losses, outside the loop.
    # Syntax: []. Four-space indentation. It is [] before training.
    #
    # STEP 3: Repeat exactly num_steps times with a for loop.
    # Syntax: for i in range(num_steps): at four spaces.
    # The colon begins its body; steps 4 and 5 use eight spaces.
    # range(0) skips the body; the final return still executes.
    #
    # STEP 4: Call tensor_gradient_step with w_tensor, x, target,
    # and learning_rate, in that order. Do this inside the loop.
    # Unpack the two results into loss_value and grad_value.
    # Syntax: first, second = helper(arguments).
    # The helper changes the same w_tensor in place; no w_new is returned.
    # Checkpoint: call 1 returns (0.36, -2.4) and leaves w_tensor at 0.44.
    # Call 2 must use that tensor at 0.44, not initial_w=0.2 again.
    # grad_value is unpacked for clarity; this task does not record it.
    #
    # STEP 5: Append only loss_value to losses inside the loop.
    # Syntax: collection.append(item). Add one Python float per update.
    # Checkpoint: after two iterations, losses is [0.36, 0.0144].
    # Do not add a second parameter update; the helper already did it.
    #
    # STEP 6: After the loop, extract w_tensor's final number with .item().
    # Return that number first, then losses. Four-space indentation.
    # Syntax: return first, second. This executes even for num_steps=0.
    # Checkpoint: three steps return (0.4976, [0.36, 0.0144, 0.000576]).
    # Do not put return inside the loop or append a final extra loss.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement train_tensor for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED CHECK DRIVER: DO NOT EDIT =====
def _check_result(
    initial_w: float,
    x: float,
    target: float,
    learning_rate: float,
    num_steps: int,
    expected_w: float,
    expected_losses: list[float],
) -> tuple[float, list[float]]:
    actual = train_tensor(initial_w, x, target, learning_rate, num_steps)
    assert isinstance(actual, tuple) and len(actual) == 2, (
        "Return (final_w, losses) in that order"
    )
    final_w, losses = actual
    assert isinstance(final_w, float), "Extract the final number with .item()"
    assert isclose(final_w, expected_w, rel_tol=1e-9, abs_tol=1e-12), (
        f"Final parameter: expected {expected_w}, received {final_w}"
    )
    assert isinstance(losses, list) and len(losses) == num_steps, (
        "Record exactly one pre-update loss per requested update"
    )
    for index, (loss_value, wanted) in enumerate(zip(losses, expected_losses)):
        assert isinstance(loss_value, float), "Record loss numbers, not tensors"
        assert isclose(loss_value, wanted, rel_tol=1e-9, abs_tol=1e-12), (
            f"Loss at iteration {index + 1}: expected {wanted}, got {loss_value}"
        )
    return actual


def _check_helper_reuse() -> None:
    """Observe integration with the supplied step and persistent parameter."""
    global tensor_gradient_step
    original_step = tensor_gradient_step
    calls: list[tuple[torch.Tensor, float]] = []

    def observed_step(
        w_tensor: torch.Tensor, x: float, target: float, learning_rate: float
    ) -> tuple[float, float]:
        assert torch.is_grad_enabled(), "Keep the outer loop in normal grad mode"
        assert w_tensor.ndim == 0 and w_tensor.dtype == torch.float64
        assert w_tensor.is_leaf and w_tensor.requires_grad
        calls.append((w_tensor, w_tensor.item()))
        return original_step(w_tensor, x, target, learning_rate)

    try:
        tensor_gradient_step = observed_step
        _check_result(0.2, 2.0, 1.0, 0.1, 3, 0.4976, [0.36, 0.0144, 0.000576])
    finally:
        tensor_gradient_step = original_step

    assert len(calls) == 3, "Call the supplied helper exactly once per update"
    first_tensor = calls[0][0]
    for (w_tensor, before), wanted in zip(calls, (0.2, 0.44, 0.488)):
        assert w_tensor is first_tensor, "Reuse one parameter tensor across steps"
        assert isclose(before, wanted, rel_tol=1e-9, abs_tol=1e-12), (
            "Each helper call must see the current parameter value"
        )


def check_training() -> None:
    """Check numeric traces, zero steps, independent calls, and helper reuse."""
    _check_result(0.2, 2.0, 1.0, 0.1, 3, 0.4976, [0.36, 0.0144, 0.000576])
    print("PASS: repeated tensor updates and pre-update loss history")

    _check_result(-2.0, 2.0, 1.0, 0.1, 0, -2.0, [])
    _check_result(0.2, 2.0, 1.0, 0.1, 1, 0.44, [0.36])
    print("PASS: zero and one update")

    _check_result(
        0.2, 2.0, 1.0, 0.1, 5, 0.499904,
        [0.36, 0.0144, 0.000576, 0.00002304, 0.0000009216],
    )
    _, first_losses = _check_result(0.2, 2.0, 1.0, 0.1, 1, 0.44, [0.36])
    _, second_losses = _check_result(0.2, 2.0, 1.0, 0.1, 1, 0.44, [0.36])
    first_losses.append(99.0)
    assert len(second_losses) == 1 and isclose(
        second_losses[0], 0.36, rel_tol=1e-9, abs_tol=1e-12
    ), "Each call needs a fresh loss-history list"
    print("PASS: requested step count and separate calls")

    _check_helper_reuse()
    print("PASS: supplied helper reuse and one persistent parameter tensor")

    _check_result(0.6, -2.0, 1.0, 0.1, 3, -0.4912, [4.84, 0.1936, 0.007744])
    _check_result(0.6, 3.0, -0.4, 0.05, 3, -0.1326, [4.84, 0.0484, 0.000484])
    print("PASS: changed inputs, targets, and learning rates")

    _check_result(0.5, 2.0, 1.0, 0.1, 4, 0.5, [0.0, 0.0, 0.0, 0.0])
    _check_result(0.7, 0.0, 1.0, 0.1, 3, 0.7, [1.0, 1.0, 1.0])
    _check_result(0.2, 2.0, 1.0, 0.0, 3, 0.2, [0.36, 0.36, 0.36])
    print("PASS: zero gradients and zero learning rate still record every step")

    _check_result(0.2, 2.0, 1.0, 1.0, 2, -14.2, [0.36, 17.64])
    print("PASS: large-step loss increases remain visible")
    print("Lesson 03.07 checks passed.")


if __name__ == "__main__":
    check_training()
