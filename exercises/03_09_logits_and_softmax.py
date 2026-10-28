# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.09: vocabulary logits and stable softmax probabilities.

Run from the repository root in the existing PyTorch environment:
    python exercises/03_09_logits_and_softmax.py --check logits
    python exercises/03_09_logits_and_softmax.py --check softmax
    python exercises/03_09_logits_and_softmax.py --check all

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Connect floating-point vectors to one score per vocabulary token, then turn
each position's scores into a probability distribution. Review tensor shapes
and keeping the autograd connection from Lessons 03.01 and 03.05-03.08.
Read the paired chapter's rendered formulas and complete trace before coding.
These functions perform a forward calculation, without parameter updates.

FORMULAS AND EXACT INPUTS
    H = hidden_vectors, floating tensor [T, D]
    W = output_weights, floating tensor [D, V]
    b = output_bias, floating tensor [V]
    z[t, v] = sum(H[t, d] * W[d, v] for d in range(D)) + b[v]
    logits = H @ W + b, floating tensor [T, V]
    m[t] = maximum logit at position t
    probabilities[t, v] = exp(z[t, v] - m[t]) /
        sum(exp(z[t, j] - m[t]) for j in range(V))
Each probability row sums to one. Normalize over vocabulary columns,
independently for each input position. Scores can be negative and need not
sum to one. Subtracting one constant from a whole row preserves softmax.

CONTRACT
Assume finite values, T>0, D>0, V>0, normal gradient mode, matching input
shapes, and the same floating dtype/device. Supplied checks use CPU float64.
Inputs may require gradients. Preserve their values and the autograd graph.
Return tensors of the same dtype/device, not lists or extracted floats.
No input validation, construction, gradient reset, backward, or update is
required inside either learner function. Backward is used only by the driver
to verify that later training can still reach the input parameters.

COMPLETE SMALL EXAMPLE
    hidden_vectors = [[1, 2], [2, 1]], shape [2, 2]
    output_weights = [[1, 0, 1], [0, 1, 1]], shape [2, 3]
    output_bias = [0, 0, 0], shape [3]
    First row: [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
    Second row: [2*1+1*0, 2*0+1*1, 2*1+1*1] = [2, 1, 3]
    logits = [[1, 2, 3], [2, 1, 3]], shape [2, 3]
    row_max = [[3], [3]], shape [2, 1]
    shifted_logits = [[-2, -1, 0], [-1, -2, 0]]
    exp_scores first row = [0.1353352832, 0.3678794412, 1.0]
    row_sums = [[1.5032147244], [1.5032147244]], shape [2, 1]
    probabilities first row = [0.0900305732, 0.2447284711, 0.6652409558]
    probabilities second row = [0.2447284711, 0.0900305732, 0.6652409558]
    Changing the bias to [0.5, -1, 2] changes the logits to
    [[1.5, 1, 5], [2.5, 0, 5]]. Add that same bias vector to each row.

NEW SYNTAX
For two 2-D tensors, @ is matrix multiplication: [T,D] @ [D,V] -> [T,V].
It takes a row/column weighted sum; * instead multiplies elementwise.
Adding [V] bias to [T,V] broadcasts that vector across every row.
logits.max(dim=-1, keepdim=True) returns maximum values and their indices.
Use .values to select the values tensor. dim=-1 is the last dimension,
the vocabulary axis here. keepdim=True keeps a [T,1] column so row-wise
subtraction broadcasts correctly, rather than producing a [T] vector.
torch.exp(tensor) applies the exponential elementwise. It does not detach.
tensor.sum(dim=-1, keepdim=True) adds columns per row, keeping shape [T,1].
Dividing [T,V] by [T,1] divides each row by its own denominator.
All learner statements use four spaces; no loop or context block is needed.

HOW TO REVIEW
Both learner stages are complete. Review Stage A with --check logits and
Stage B with --check softmax. The learner implements Stage B from
max/subtraction/exp/sum/division; torch.softmax is a supplied reference.
Preserve the learner implementation and supplied checks. Run --check all
for the connected pipeline. Expected final line:
    Lesson 03.09 checks passed.
"""

from __future__ import annotations

import argparse

import torch


def compute_logits(
    hidden_vectors: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
) -> torch.Tensor:
    """Return connected [T,V] vocabulary scores from supplied vectors/weights."""
    # ===== START LEARNER CODE =====
    # STEP 1: Form weighted_scores from hidden_vectors and output_weights.
    # Four spaces. Use the matrix-product operator @, not elementwise *.
    # Shapes: [T,D] @ [D,V] -> [T,V]. Example: [[1,2,3],[2,1,3]].
    #
    # STEP 2: Add output_bias to weighted_scores to obtain logits.
    # Tensor addition broadcasts [V] over all T rows. Example zero bias
    # leaves the scores unchanged; bias [0.5,-1,2] gives [[1.5,1,5],[2.5,0,5]].
    #
    # STEP 3: Return logits as a tensor, preserving its graph and the inputs.
    # Four spaces. Do not call .item(), .detach(), or torch.no_grad().
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement compute_logits for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE A =====


def softmax_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return stable, connected [T,V] probabilities normalized per row."""
    # ===== START LEARNER CODE =====
    # STEP 1: Find row_max using logits.max(dim=-1, keepdim=True).values.
    # Four spaces. .max(...) returns values and indices; .values chooses
    # the maxima themselves. Example row_max is [[3],[3]], shape [T,1].
    #
    # STEP 2: Subtract row_max from logits to obtain shifted_logits.
    # Each row uses its own maximum. Example: [[-2,-1,0],[-1,-2,0]].
    # Use a new subtraction result; do not modify logits in place.
    #
    # STEP 3: Apply torch.exp to shifted_logits, saving exp_scores.
    # First row: [0.1353352832,0.3678794412,1.0], shape [T,V].
    # Subtracting the maximum avoids exponentiating a large positive score.
    #
    # STEP 4: Obtain row_sums from exp_scores.sum(dim=-1, keepdim=True).
    # Sum across the vocabulary columns, not positions or the whole tensor.
    # Example: [[1.5032147244],[1.5032147244]], shape [T,1].
    #
    # STEP 5: Divide exp_scores by row_sums to obtain probabilities.
    # This is elementwise / with broadcasting, shape [T,V]. Each row sums
    # to one. First row: [0.0900305732,0.2447284711,0.6652409558].
    #
    # STEP 6: Return probabilities as a tensor. Keep autograd connected.
    # All statements use four spaces. No loop, backward, or update here.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement softmax_probabilities for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE B =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _tensor(
    values: list[float] | list[list[float]], *, requires_grad: bool = False
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _check_logits(
    hidden_vectors: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    inputs = (hidden_vectors, output_weights, output_bias)
    originals = tuple(value.detach().clone() for value in inputs)
    mode_before = torch.is_grad_enabled()
    actual = compute_logits(*inputs)
    assert isinstance(actual, torch.Tensor), "Return logits as a tensor"
    assert actual.shape == expected.shape
    assert actual.dtype == hidden_vectors.dtype and actual.device == hidden_vectors.device
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12), (
        f"Logits: expected {expected}, got {actual}"
    )
    assert all(torch.equal(value.detach(), saved) for value, saved in zip(inputs, originals))
    assert torch.is_grad_enabled() == mode_before
    return actual


def check_logits() -> None:
    hidden_vectors = _tensor([[1.0, 2.0], [2.0, 1.0]])
    output_weights = _tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    _check_logits(hidden_vectors, output_weights, _tensor([0.0, 0.0, 0.0]),
                  _tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]))
    _check_logits(hidden_vectors, output_weights, _tensor([0.5, -1.0, 2.0]),
                  _tensor([[1.5, 1.0, 5.0], [2.5, 0.0, 5.0]]))
    print("PASS: row-column score projection and shared vocabulary bias")

    _check_logits(_tensor([[1.0, -2.0, 3.0]]),
                  _tensor([[1.0, 2.0], [0.0, -1.0], [2.0, 1.0]]),
                  _tensor([-1.0, 0.5]), _tensor([[6.0, 7.5]]))
    _check_logits(_tensor([[2.0], [-1.0], [0.0]]),
                  _tensor([[1.0, -2.0, 0.0, 3.0]]),
                  _tensor([0.0, 1.0, -1.0, 2.0]),
                  _tensor([[2.0, -3.0, -1.0, 8.0],
                           [-1.0, 3.0, -1.0, -1.0], [0.0, 1.0, -1.0, 2.0]]))
    print("PASS: different vector/vocabulary dimensions and preserved inputs")

    hidden_vectors = _tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    output_weights = _tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], requires_grad=True)
    output_bias = _tensor([0.5, -1.0, 2.0], requires_grad=True)
    logits = _check_logits(hidden_vectors, output_weights, output_bias,
                           _tensor([[1.5, 1.0, 5.0], [2.5, 0.0, 5.0]]))
    assert logits.requires_grad, "Preserve the projection's autograd connection"
    logits.sum().backward()
    assert torch.allclose(hidden_vectors.grad, _tensor([[2.0, 2.0], [2.0, 2.0]]))
    assert torch.allclose(output_weights.grad, torch.full_like(output_weights, 3.0))
    assert torch.allclose(output_bias.grad, _tensor([2.0, 2.0, 2.0]))
    print("PASS: projection gradients reach vectors, weights, and bias")


def _check_softmax(logits: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    original = logits.detach().clone()
    mode_before = torch.is_grad_enabled()
    actual = softmax_probabilities(logits)
    assert isinstance(actual, torch.Tensor), "Return probabilities as a tensor"
    assert actual.shape == logits.shape == expected.shape
    assert actual.dtype == logits.dtype and actual.device == logits.device
    assert torch.isfinite(actual).all(), "Stable probabilities must be finite"
    assert ((actual >= 0) & (actual <= 1)).all()
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12), (
        f"Probabilities: expected {expected}, got {actual}"
    )
    assert torch.allclose(actual.sum(dim=-1), torch.ones_like(actual[:, 0]),
                          rtol=1e-10, atol=1e-12), "Normalize each vocabulary row"
    assert torch.equal(logits.detach(), original), "Preserve the input logits"
    assert torch.is_grad_enabled() == mode_before
    return actual


def check_softmax() -> None:
    known = _tensor([[0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
                     [0.24472847105479764, 0.09003057317038046, 0.6652409557748218]])
    base = _tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    _check_softmax(base, known)
    print("PASS: worked-example probabilities and vocabulary-axis normalization")

    _check_softmax(_tensor([[0.0, 0.0, 0.0], [-7.0, -7.0, -7.0]]),
                   torch.full((2, 3), 1 / 3, dtype=torch.float64))
    _check_softmax(_tensor([[5.0], [-1000.0]]), _tensor([[1.0], [1.0]]))
    print("PASS: equal logits, negative logits, and a one-token vocabulary")

    _check_softmax(base + _tensor([[10000.0], [-10000.0]]), known)
    _check_softmax(_tensor([[1001.0, 1002.0, 1003.0], [-998.0, -999.0, -997.0]]), known)
    _check_softmax(_tensor([[10000.0, 10000.0, 10000.0], [-10000.0, -10000.0, -10000.0]]),
                   torch.full((2, 3), 1 / 3, dtype=torch.float64))
    print("PASS: row-wise shift invariance and large-score numerical stability")

    logits = _tensor([[-5.0, 0.0, 2.0, -1.0], [4.0, -2.0, 1.0, 3.0],
                      [-0.5, 0.2, 0.8, -0.1]])
    _check_softmax(logits, torch.softmax(logits, dim=-1))
    print("PASS: varied shapes and unchanged input scores")

    logits = _tensor([[0.2, -0.1, 0.8], [-0.4, 0.7, 0.3]], requires_grad=True)
    coefficients = _tensor([[1.0, -2.0, 0.5], [-0.3, 0.4, 1.2]])
    probabilities = _check_softmax(logits, torch.softmax(logits.detach(), dim=-1))
    assert probabilities.requires_grad, "Keep probabilities connected for later training"
    (probabilities * coefficients).sum().backward()
    reference = logits.detach().clone().requires_grad_()
    (torch.softmax(reference, dim=-1) * coefficients).sum().backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, reference.grad, rtol=1e-9, atol=1e-12)
    print("PASS: softmax gradients agree with the PyTorch reference")


def check_connected_pipeline() -> None:
    hidden_vectors = _tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    output_weights = _tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], requires_grad=True)
    output_bias = _tensor([0.2, -0.1, 0.3], requires_grad=True)
    inputs = (hidden_vectors, output_weights, output_bias)
    originals = tuple(value.detach().clone() for value in inputs)
    coefficients = _tensor([[0.2, -0.3, 1.0], [1.2, 0.1, -0.5]])
    logits = compute_logits(*inputs)
    probabilities = softmax_probabilities(logits)
    (probabilities * coefficients).sum().backward()
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    reference_logits = reference_inputs[0] @ reference_inputs[1] + reference_inputs[2]
    reference_probabilities = torch.softmax(reference_logits, dim=-1)
    (reference_probabilities * coefficients).sum().backward()
    assert torch.allclose(probabilities, reference_probabilities, rtol=1e-10, atol=1e-12)
    for value, reference, original in zip(inputs, reference_inputs, originals):
        assert value.grad is not None
        assert torch.allclose(value.grad, reference.grad, rtol=1e-9, atol=1e-12)
        assert torch.equal(value.detach(), original)
    print("PASS: connected vectors-to-probabilities forward and backward pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("logits", "softmax", "all"), default="all")
    arguments = parser.parse_args()
    if arguments.check in ("logits", "all"):
        check_logits()
        print("PASS logits: all three check categories")
    if arguments.check in ("softmax", "all"):
        check_softmax()
        print("PASS softmax: all five check categories")
    if arguments.check == "all":
        check_connected_pipeline()
        print("Lesson 03.09 checks passed.")
