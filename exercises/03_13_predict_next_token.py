# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.13: one greedy next-token prediction with fixed parameters.

Run from the repository root in the existing PyTorch environment:
    python exercises/03_13_predict_next_token.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse Lesson 03.09's compute_logits and softmax_probabilities to predict
from a supplied prompt. Review its [T,V] distributions, the training step
from 03.11, and the persistent loop from 03.12. Read the bilingual formulas,
figures, worked example, and implementation plan before coding.
Inference uses fixed parameters and no target labels, backward, or updates.
This lookup-plus-head model uses only the final input token for the next
distribution. Contextual attention and the generation loop follow later.

EXACT INPUTS
embedding_table: floating torch.Tensor [V,D], input vector table.
context_ids: fixed torch.long tensor [T], nonempty prompt token IDs.
output_weights: floating torch.Tensor [D,V], vocabulary projection.
output_bias: floating torch.Tensor [V], shared candidate biases.
Assume T,D,V>0, valid IDs, matching dtype/device, and finite forward scores.
The three floating tensors may require gradients; preserve their flags,
values, storage, and any stored gradients. Checks use CPU float64 leaves.
All input construction is supplied; no learner validation is required.

RETURN
Return (next_token_id, next_probabilities).
next_token_id: a Python int in [0,V), the largest-probability candidate.
Equal largest probabilities select the first index, the lowest tied ID.
next_probabilities: untracked floating torch.Tensor [V], the full last-row
distribution, same floating dtype/device as the supplied parameters.
Preserve context_ids and the caller's gradient mode. Do not append to the
prompt, reset gradients, or change parameter values in this helper.

FORMULAS
    H = E[context_ids], shape [T,D]
    Z = H @ W + b, shape [T,V]
    P = softmax(Z) over candidates in each row, shape [T,V]
    next_probabilities = P[T-1,:], shape [V]
    next_token_id = argmax(next_probabilities) over candidate IDs
The final input position predicts the token following the whole prompt.
Earlier rows predict at earlier positions; they are not new future tokens
generated together. This model's rows are independent token lookups.

COMPLETE WORKED EXAMPLE: T=2, D=2, V=2
    embedding_table = [[1,2],[3,1]]
    context_ids = [0,1]
    output_weights = [[1,0],[0,1]]; output_bias = [0,0]
    selected_vectors = [[1,2],[3,1]]
    logits = [[1,2],[3,1]]
    probabilities = [[0.2689414214,0.7310585786],
                     [0.8807970780,0.1192029220]]
    Last input row: next_probabilities = [0.8807970780,0.1192029220].
    Its largest entry is at index 0. Return Python int 0 and that vector.
    The first row instead favors ID 1, so selecting the first row is wrong.
These are hand-picked arithmetic parameters, not a reported training run.
The supplied checks also connect this helper to the completed training loop
on a synthetic UTF-8 byte example with IDs 97 ('a') and 98 ('b').

NEW SYNTAX AND SCOPE
with torch.no_grad(): starts at four spaces; its forward body uses eight.
It disables tracking of these computations while leaving the original
parameters' requires_grad flags and stored gradients unchanged.
probabilities[-1] selects the last row; -1 means last position, not token ID.
On a one-dimensional probability vector, .argmax() returns the index of
its largest value as a scalar integer tensor. .item() extracts a Python int.
The maximum probability itself is not the selected ID.
Return at four spaces, outside the context, after gradient mode is restored.

HOW TO WORK
Replace only the learner placeholder using the seven numbered hints.
The supplied loader exposes compute_logits and softmax_probabilities.
Reuse them; do not call the loss or training helper for prediction.
Expected final success line after implementation:
    Lesson 03.13 checks passed.
"""

from __future__ import annotations

import importlib.util
from math import isclose, log
from pathlib import Path

import torch
import torch.nn.functional as F


def predict_next_token(
    embedding_table: torch.Tensor,
    context_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    """Return a greedy ID and the last-row probability vector without training."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, enter with torch.no_grad():.
    # Steps 2-6 go at eight spaces inside it. The whole prediction is
    # forward-only; preserve the original parameter flags and stored grads.
    #
    # STEP 2: At eight spaces, obtain selected_vectors with
    # embedding_table[context_ids]. Shape [T,D]; example [[1,2],[3,1]].
    #
    # STEP 3: At eight spaces, call compute_logits(selected_vectors,
    # output_weights, output_bias), saving logits. Shape [T,V].
    # Example [[1,2],[3,1]]. Use the completed helper, including its bias.
    #
    # STEP 4: At eight spaces, call softmax_probabilities(logits), saving
    # probabilities. Example last row [0.8807970780,0.1192029220].
    # Rows normalize over vocabulary columns, independently by position.
    #
    # STEP 5: At eight spaces, select probabilities[-1] as
    # next_probabilities. This removes the position axis: [T,V] -> [V].
    # Example [0.8807970780,0.1192029220], not the first row's distribution.
    #
    # STEP 6: At eight spaces, apply .argmax() to next_probabilities,
    # then .item(), saving next_token_id. Example Python int 0.
    # Select an index, not the maximum probability. Ties use the first ID.
    #
    # STEP 7: At four spaces, outside no-grad, return next_token_id,
    # next_probabilities in that order. Input prompt and parameters remain
    # unchanged; the next chapter's caller will append generated IDs.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement predict_next_token for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_module(filename, module_name):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_projection_module = _load_module("03_09_logits_and_softmax.py", "completed_prediction_head")
compute_logits = _projection_module.compute_logits
softmax_probabilities = _projection_module.softmax_probabilities


def _parameter(values, *, requires_grad=True):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _ids(values):
    return torch.tensor(values, dtype=torch.long)


def _example():
    return (_parameter([[1., 2.], [3., 1.]]), _ids([0, 1]),
            _parameter([[1., 0.], [0., 1.]]), _parameter([0., 0.]))


def _check_prediction(embedding_table, context_ids, output_weights, output_bias):
    parameters = (embedding_table, output_weights, output_bias)
    originals = tuple(p.detach().clone() for p in parameters)
    pointers = tuple(p.data_ptr() for p in parameters)
    flags = tuple(p.requires_grad for p in parameters)
    grad_objects = tuple(p.grad for p in parameters)
    gradients = tuple(None if p.grad is None else p.grad.clone() for p in parameters)
    context_before = context_ids.clone()
    mode_before = torch.is_grad_enabled()
    reference_logits = originals[0][context_ids] @ originals[1] + originals[2]
    expected = torch.softmax(reference_logits, dim=-1)[-1]
    actual = predict_next_token(embedding_table, context_ids, output_weights, output_bias)
    assert isinstance(actual, tuple) and len(actual) == 2
    token_id, probabilities = actual
    assert type(token_id) is int, "Extract the integer index with .item()"
    assert token_id == expected.argmax().item(), "Choose the last row's largest-probability ID"
    assert isinstance(probabilities, torch.Tensor) and probabilities.shape == expected.shape
    assert probabilities.dtype == embedding_table.dtype and probabilities.device == embedding_table.device
    assert not probabilities.requires_grad and probabilities.grad_fn is None, "Inference must not retain a graph"
    assert torch.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert isclose(probabilities.sum().item(), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert torch.allclose(probabilities, expected, rtol=1e-10, atol=1e-12)
    for p, original, pointer, flag, grad_object, gradient in zip(
            parameters, originals, pointers, flags, grad_objects, gradients):
        assert torch.equal(p.detach(), original) and p.data_ptr() == pointer
        assert p.requires_grad == flag and p.is_leaf and p.grad_fn is None
        assert p.grad is grad_object, "Preserve existing gradients without clearing them"
        if gradient is not None:
            assert torch.equal(p.grad, gradient)
    assert torch.equal(context_ids, context_before)
    assert torch.is_grad_enabled() == mode_before
    return actual


def _check_forward_reuse():
    global compute_logits, softmax_probabilities
    original_logits, original_softmax = compute_logits, softmax_probabilities
    arguments = _example()
    calls = []

    def observed_logits(hidden, weights, bias):
        assert not torch.is_grad_enabled(), "Use no-grad around the entire forward"
        assert not hidden.requires_grad and hidden.grad_fn is None
        assert weights is arguments[2] and bias is arguments[3]
        assert torch.equal(hidden, arguments[0].detach()[arguments[1]])
        calls.append("logits")
        return original_logits(hidden, weights, bias)

    def observed_softmax(logits):
        assert not torch.is_grad_enabled() and not logits.requires_grad
        calls.append("softmax")
        return original_softmax(logits)

    try:
        compute_logits, softmax_probabilities = observed_logits, observed_softmax
        _check_prediction(*arguments)
    finally:
        compute_logits, softmax_probabilities = original_logits, original_softmax
    assert calls == ["logits", "softmax"], "Reuse both completed forward helpers"


def check_prediction():
    token_id, probabilities = _check_prediction(*_example())
    assert token_id == 0
    assert torch.allclose(probabilities, torch.tensor([0.8807970779778823, 0.11920292202211755], dtype=torch.float64))
    print("PASS: worked example, last-position distribution, and Python integer selection")

    arguments = _example()
    _check_prediction(arguments[0], _ids([0]), arguments[2], arguments[3])
    varied = (_parameter([[1., -2.], [0.3, 0.4], [-0.5, 0.9]]), _ids([2, 0, 2]),
              _parameter([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]), _parameter([0.05, -0.2, 0.1]))
    _check_prediction(*varied)
    _check_prediction(_parameter([[2.]]), _ids([0]), _parameter([[0.3]]), _parameter([0.1]))
    large_table = torch.zeros((258, 2), dtype=torch.float64)
    large_table[256] = torch.tensor([1., 2.], dtype=torch.float64)
    large_bias = [0.] * 258
    large_bias[256] = 2.
    merged_prediction = _check_prediction(large_table.requires_grad_(), _ids([97, 256]),
                                          _parameter(torch.zeros((2, 258)).tolist()), _parameter(large_bias))
    assert merged_prediction[0] == 256, "Output candidate IDs can exceed the raw-byte range"
    print("PASS: single/multiple positions, repeated IDs, varied shapes, and merged vocabulary IDs")

    _check_prediction(_parameter([[1000., 0., -1000.]]), _ids([0]),
                      _parameter(torch.eye(3).tolist()), _parameter([0., 0., 0.]))
    tied = _check_prediction(_parameter([[1.]]), _ids([0]),
                             _parameter([[0., 0., 0.]]), _parameter([1., 2., 2.]))
    assert tied[0] == 1, "Equal maxima choose the lowest tied candidate ID"
    print("PASS: stable extreme logits and deterministic first-index tie handling")

    arguments = _example()
    for p in (arguments[0], arguments[2], arguments[3]):
        p.grad = torch.full_like(p, 99.)
    _check_prediction(*arguments)
    with torch.no_grad():
        _check_prediction(*arguments)
    frozen = tuple(value.detach().clone() if isinstance(value, torch.Tensor) else value for value in arguments)
    _check_prediction(*frozen)
    print("PASS: unchanged parameters, prompts, gradient buffers, trainability, and caller mode")

    _check_forward_reuse()
    print("PASS: completed helpers reused under no-grad without retaining a forward graph")

    arguments = _example()
    first = _check_prediction(*arguments)
    second = _check_prediction(arguments[0], _ids([1]), arguments[2], arguments[3])
    third = _check_prediction(arguments[0], _ids([1, 0, 1]), arguments[2], arguments[3])
    assert torch.equal(first[1], second[1]) and torch.equal(second[1], third[1])
    first[1].fill_(0.)
    assert isclose(second[1].sum().item(), 1., rel_tol=1e-12)
    print("PASS: repeatable predictions, independent outputs, and current-token model scope")

    loop_module = _load_module("03_12_next_token_training_loop.py", "completed_prediction_training")
    table = torch.zeros((256, 2), dtype=torch.float64)
    table[97], table[98] = torch.tensor([1., 2.]), torch.tensor([3., 1.])
    table.requires_grad_()
    weights = _parameter(torch.zeros((2, 256)).tolist())
    bias = _parameter([0.] * 256)
    input_ids, target_ids = _ids([97, 98]), _ids([98, 97])
    before = F.cross_entropy(table[input_ids] @ weights + bias, target_ids).item()
    history = loop_module.train_next_token(table, input_ids, weights, bias, target_ids, 0.1, 100)
    after = F.cross_entropy(table[input_ids] @ weights + bias, target_ids).item()
    assert len(history) == 100 and isclose(before, log(256), rel_tol=1e-12)
    assert isclose(after, 0.018814872487891755, rel_tol=1e-9) and after < before
    after_a = _check_prediction(table, _ids([97]), weights, bias)
    after_ab = _check_prediction(table, _ids([97, 98]), weights, bias)
    assert after_a[0] == 98 and after_ab[0] == 97
    assert after_a[1][98].item() > 0.97 and after_ab[1][97].item() > 0.98
    print("PASS: completed training loop connects to byte-level next-token inference")
    print("Lesson 03.13 checks passed.")


if __name__ == "__main__":
    check_prediction()
