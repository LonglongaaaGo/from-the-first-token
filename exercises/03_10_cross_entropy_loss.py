# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.10: stable mean cross-entropy for next-token targets.

Run from the repository root in the existing PyTorch environment:
    python exercises/03_10_cross_entropy_loss.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Use the true next-token ID at each position to score a prediction. Review
Lesson 03.09's logits, per-row normalization, and row-maximum stabilization.
Read the paired chapter's formulas and complete example before implementing.
The function returns a connected scalar loss, without backward or updates.

EXACT INPUTS AND RETURN
logits: floating torch.Tensor [T,V], raw vocabulary scores at each position.
target_ids: torch.Tensor [T], dtype=torch.long, fixed true next-token IDs.
Assume T>0, V>0, finite logits, valid IDs, matching devices, and normal
gradient mode. Supplied checks use CPU float64 scores. Inputs are supplied;
no tensor construction or validation is needed except the row-index tensor.
Return loss: a scalar tensor, the mean of T negative target log-probabilities.
Preserve both inputs and the logits' autograd connection. Do not extract a
float, detach, call backward, clear gradients, or update any parameter here.
This exercise uses no ignored positions, class weights, or label smoothing.

FORMULAS
    y[t] = target_ids[t]
    p[t,v] = exp(logits[t,v]) / sum(exp(logits[t,j]) over candidates j)
    loss = mean(-log(p[t,y[t]])) over positions t
    m[t] = max(logits[t,:])
    shifted_logits[t,v] = logits[t,v] - m[t]
    log_probabilities[t,v] = shifted_logits[t,v] -
        log(sum(exp(shifted_logits[t,j]) over candidates j))
Compute log-probabilities directly: tiny probabilities can underflow before
a subsequent log, while the equivalent shifted-score expression stays
useful. The mean divides by T, not by V or T*V.

COMPLETE WORKED EXAMPLE
    logits = [[1,2,3], [2,1,3]], shape [2,3]
    target_ids = [2,0], shape [2]
    row_max = [[3],[3]]
    shifted_logits = [[-2,-1,0], [-1,-2,0]]
    row_sums = [[1.5032147244],[1.5032147244]]
    log_row_sums = [[0.4076059644],[0.4076059644]]
    log_probabilities = [[-2.4076059644,-1.4076059644,-0.4076059644],
                         [-1.4076059644,-2.4076059644,-0.4076059644]]
    row_indices = [0,1]
    Select coordinates (0,2) and (1,0), not every combination of rows/IDs.
    target_log_probabilities = [-0.4076059644,-1.4076059644]
    loss = (0.4076059644 + 1.4076059644) / 2 = 0.9076059644

NEW SYNTAX
torch.log(tensor) applies the natural logarithm elementwise.
logits.shape[0] is T. torch.arange(T, device=logits.device) constructs
integer indices [0,...,T-1] on the same device as the scores.
matrix[row_indices, target_ids] selects one coordinate per paired index:
row_indices[t] chooses the row and target_ids[t] chooses the column.
The result has shape [T]. Negate those selected log-probabilities and use
.mean() to obtain a scalar. Preserve the tensor for later backward.

HOW TO REVIEW
The learner core is complete. Use the eight numbered hints to review or
independently reconstruct the max/subtraction/exp/sum/log/indexing/mean path.
Built-in cross_entropy and logsumexp remain supplied references.
The setup loads the completed Lesson 03.09 projection by its portable
sibling filename to verify gradients reaching the embedding table and head.
Expected final success line after implementation:
    Lesson 03.10 checks passed.
"""

from __future__ import annotations

import importlib.util
from math import isclose, log
from pathlib import Path

import torch
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return connected scalar mean target negative log-likelihood."""
    # ===== START LEARNER CODE =====
    # STEP 1: Find row_max with logits.max(dim=-1, keepdim=True).values.
    # Four spaces. Example [[3],[3]], shape [T,1]. Reuse the 03.09 syntax.
    #
    # STEP 2: Subtract row_max to obtain shifted_logits, preserving logits.
    # Example [[-2,-1,0],[-1,-2,0]], shape [T,V].
    #
    # STEP 3: Obtain exp_scores with torch.exp, then row_sums with
    # .sum(dim=-1, keepdim=True). Example row_sums [[1.5032147244],[1.5032147244]].
    #
    # STEP 4: Apply torch.log to row_sums, saving log_row_sums, shape [T,1].
    # Example [[0.4076059644],[0.4076059644]]. It is log of the sum of
    # exponentials, not the sum of logs or the log of raw logits.
    #
    # STEP 5: Subtract log_row_sums from shifted_logits to get
    # log_probabilities. [T,1] broadcasts per row. Example row 0 is
    # [-2.4076059644,-1.4076059644,-0.4076059644].
    #
    # STEP 6: Build row_indices with torch.arange(logits.shape[0],
    # device=logits.device). Integral bounds produce integer indices.
    # Example [0,1]. Every statement still uses four spaces.
    #
    # STEP 7: Obtain target_log_probabilities using the paired indexing
    # log_probabilities[row_indices, target_ids], shape [T].
    # Coordinates (0,2) and (1,0) yield [-0.4076059644,-1.4076059644].
    #
    # STEP 8: Negate the selected vector, average with .mean(), and return
    # the scalar loss tensor. Parentheses group negation before the mean.
    # Example 0.9076059644. The mean is over T positions; retain the graph.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement cross_entropy_loss for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _scores(values: list[list[float]], *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def _check_loss(logits: torch.Tensor, target_ids: torch.Tensor, expected: float) -> torch.Tensor:
    scores_before = logits.detach().clone()
    targets_before = target_ids.clone()
    mode_before = torch.is_grad_enabled()
    actual = cross_entropy_loss(logits, target_ids)
    assert isinstance(actual, torch.Tensor), "Return a scalar tensor, not .item()"
    assert actual.shape == torch.Size([]), "Average to one scalar loss"
    assert actual.dtype == logits.dtype and actual.device == logits.device
    assert torch.isfinite(actual), "Stable loss should be finite in these cases"
    assert isclose(actual.item(), expected, rel_tol=1e-10, abs_tol=1e-12), (
        f"Loss: expected {expected}, got {actual.item()}"
    )
    assert torch.equal(logits.detach(), scores_before), "Keep logits unchanged"
    assert torch.equal(target_ids, targets_before), "Keep targets unchanged"
    assert torch.is_grad_enabled() == mode_before
    if logits.requires_grad:
        assert actual.requires_grad, "Keep the loss connected for backward"
    return actual


def _projection_helper():
    path = Path(__file__).with_name("03_09_logits_and_softmax.py")
    specification = importlib.util.spec_from_file_location("lesson_03_09_projection", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.compute_logits


def check_cross_entropy() -> None:
    base = _scores([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]])
    targets = _ids([2, 0])
    base_loss = 0.9076059644443804
    _check_loss(base, targets, base_loss)
    print("PASS: true-target selection and mean loss in the worked example")

    _check_loss(base, _ids([0, 2]), 1.4076059644443804)
    _check_loss(base, _ids([1, 1]), 1.9076059644443804)
    _check_loss(torch.zeros((2, 3), dtype=torch.float64), _ids([0, 2]), log(3))
    _check_loss(_scores([[7.0]]), _ids([0]), 0.0)
    _check_loss(torch.zeros((2, 258), dtype=torch.float64), _ids([256, 97]), log(258))
    varied = _scores([[0.3, -0.5, 1.2, 0.0], [-0.7, 0.8, 0.2, 1.1],
                      [0.6, 0.1, -0.4, 0.9]])
    varied_targets = _ids([3, 1, 0])
    _check_loss(varied, varied_targets, F.cross_entropy(varied, varied_targets).item())
    print("PASS: varied targets, vocabulary sizes, single-candidate case, and preserved inputs")

    _check_loss(base + _scores([[10000.0], [-10000.0]]), targets, base_loss)
    _check_loss(_scores([[1000.0, 0.0, -1000.0]]), _ids([2]), 2000.0)
    _check_loss(_scores([[2000.0, -2000.0, 0.0]]), _ids([1]), 4000.0)
    print("PASS: row-shift invariance and finite loss for very unlikely targets")

    logits = base.clone().requires_grad_()
    loss = _check_loss(logits, targets, base_loss)
    loss.backward()
    expected_gradient = torch.softmax(base, dim=-1)
    expected_gradient[torch.arange(2), targets] -= 1
    expected_gradient /= 2
    assert logits.grad is not None
    assert torch.allclose(logits.grad, expected_gradient, rtol=1e-9, atol=1e-12)
    assert torch.allclose(logits.grad.sum(dim=-1), torch.zeros(2, dtype=torch.float64), atol=1e-12)
    print("PASS: logit gradients equal mean-normalized probability minus target indicator")

    logits = _scores([[0.2, -0.1, 0.8], [-0.4, 0.7, 0.3]], requires_grad=True)
    fd_targets = _ids([2, 1])
    reference_loss = F.cross_entropy(logits, fd_targets)
    loss = _check_loss(logits, fd_targets, reference_loss.item())
    loss.backward()
    original = logits.detach().clone()
    h = 1e-6
    for row in range(original.shape[0]):
        for column in range(original.shape[1]):
            plus, minus = original.clone(), original.clone()
            plus[row, column] += h
            minus[row, column] -= h
            numerical = (F.cross_entropy(plus, fd_targets).item()
                         - F.cross_entropy(minus, fd_targets).item()) / (2 * h)
            assert isclose(logits.grad[row, column].item(), numerical,
                           rel_tol=1e-6, abs_tol=1e-8)
    print("PASS: PyTorch loss agreement and finite-difference gradient agreement")

    compute_logits = _projection_helper()
    embedding_table = _scores([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    input_ids = _ids([0, 2, 0])
    weights = _scores([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]], requires_grad=True)
    bias = torch.tensor([0.05, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    pipeline_targets = _ids([1, 2, 0])
    parameters = (embedding_table, weights, bias)
    original_parameters = tuple(value.detach().clone() for value in parameters)
    logits = compute_logits(embedding_table[input_ids], weights, bias)
    expected_loss = F.cross_entropy(logits.detach(), pipeline_targets).item()
    _check_loss(logits, pipeline_targets, expected_loss).backward()
    references = tuple(value.detach().clone().requires_grad_() for value in parameters)
    reference_logits = references[0][input_ids] @ references[1] + references[2]
    F.cross_entropy(reference_logits, pipeline_targets).backward()
    for value, reference, original in zip(parameters, references, original_parameters):
        assert value.grad is not None
        assert torch.allclose(value.grad, reference.grad, rtol=1e-9, atol=1e-12)
        assert torch.equal(value.detach(), original)
    assert torch.equal(embedding_table.grad[1], torch.zeros(2, dtype=torch.float64))
    print("PASS: loss gradients reach the embedding table and completed output projection")
    print("Lesson 03.10 checks passed.")


if __name__ == "__main__":
    check_cross_entropy()
