# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.14: grow a prompt by repeated greedy next-token prediction.

Run in the existing PyTorch environment from the repository root:
    python exercises/03_14_autoregressive_generation.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse the completed 03.13 predict_next_token once for each new token.
Autoregressive generation feeds each selected ID into the next prompt.
The sequence grows; embedding_table, output_weights, and output_bias stay
fixed. The completed helper already puts its forward under no-grad.
No target IDs, losses, backward calls, or training updates are needed.

INPUTS
embedding_table: floating torch.Tensor [V,D], fixed token vectors.
context_ids: torch.long tensor [T], the original nonempty prompt.
output_weights: floating torch.Tensor [D,V], fixed vocabulary projection.
output_bias: floating torch.Tensor [V], fixed vocabulary biases.
num_new_tokens: Python int >= 0, the EXACT number of new IDs to append.
Assume valid IDs, compatible shapes/dtypes/devices, and finite scores.
Input construction and validation cases are supplied, not learner tasks.

RETURN
Return generated_ids: a fresh torch.long tensor [T + num_new_tokens],
containing the original prompt followed by the selected new IDs.
Keep the prompt unchanged. Preserve parameter values, storage, trainability,
stored gradients, and caller gradient mode. Return a fresh prompt copy when
num_new_tokens is zero; do not query the model in that case.
No EOS rule or context truncation is used: the budget determines stopping.
The caller decodes accumulated IDs using the matching tokenizer vocabulary.

FORMULAS
    s^(0) = context_ids
    v_k = argmax predict_distribution(s^(k); fixed parameters)
    s^(k+1) = concat(s^(k), [v_k])
For k=0,...,N-1, lengths are T+k before prediction and T+k+1 afterwards.
N=num_new_tokens gives N model calls and T+N final IDs, not N total IDs.

COMPLETE WORKED EXAMPLE
    embedding_table = [[1,2],[3,1]]
    output_weights = [[1,0],[0,1]]; output_bias = [0,0]
    context_ids = [0,1]; num_new_tokens = 3
    call 1: [0,1]     -> probabilities [0.8808,0.1192] -> ID 0
    call 2: [0,1,0]   -> probabilities [0.2689,0.7311] -> ID 1
    call 3: [0,1,0,1] -> probabilities [0.8808,0.1192] -> ID 0
    return tensor([0,1,0,1,0]); original context_ids stays [0,1].
These are hand-picked arithmetic parameters. This lookup-plus-head model
depends only on its last ID; attention will add contextual processing later.

NEW SYNTAX
context_ids.clone() makes an independent tensor copy, including at zero
budget. Plain assignment creates another name for the same tensor.
torch.tensor([next_token_id], dtype=generated_ids.dtype,
             device=generated_ids.device) creates a ONE-DIMENSIONAL [1]
integer tensor. The brackets matter: a scalar tensor has shape [], not [1].
dtype is the tensor's numeric type; device is where it is stored.
torch.cat((generated_ids, next_id_tensor), dim=0) joins along the existing
one-dimensional sequence axis and returns a new tensor. It does not change
the old tensor in place, so save the returned result as generated_ids.
The tuple in cat supplies TWO tensors; addition would add values instead.

HOW TO WORK
Follow the seven hints below; preserve supplied setup and checks.
Expected final success line:
    Lesson 03.14 checks passed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def generate_token_ids(
    embedding_table: torch.Tensor,
    context_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    num_new_tokens: int,
) -> torch.Tensor:
    """Return a fresh full prompt-plus-continuation tensor using fixed parameters."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, make generated_ids from context_ids.clone().
    # Example [0,1], shape [2]. This also defines the zero-budget return.
    #
    # STEP 2: At four spaces, loop over range(num_new_tokens).
    # A for ...: block has its body at eight spaces. The counter counts
    # new tokens, not positions in the original prompt. Example 0,1,2.
    #
    # STEP 3: At eight spaces, call predict_next_token with embedding_table,
    # generated_ids, output_weights, output_bias in that order. Save
    # prediction_result. Use the CURRENT growing sequence, not context_ids.
    # The helper already controls no-grad for its complete forward.
    #
    # STEP 4: At eight spaces, save prediction_result[0] as next_token_id.
    # This is a Python int, not the returned probability vector at index 1.
    # Example ID 0, then 1, then 0 on successive calls.
    #
    # STEP 5: At eight spaces, create next_id_tensor with torch.tensor from
    # [next_token_id], specifying dtype=generated_ids.dtype and
    # device=generated_ids.device. Shape [1], example tensor([0]).
    #
    # STEP 6: At eight spaces, join generated_ids and next_id_tensor with
    # torch.cat((generated_ids, next_id_tensor), dim=0). Assign the returned
    # tensor back to generated_ids. First result [0,1,0], shape [3].
    #
    # STEP 7: At four spaces AFTER the loop, return generated_ids.
    # Three iterations give [0,1,0,1,0], shape [5]. Zero iterations return
    # an independent copy of the prompt. Do not update parameters or decode
    # individual tokens in this helper; decoding is supplied separately.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement generate_token_ids for this exercise")
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


_prediction = _load_module("03_13_predict_next_token.py", "completed_generation_prediction")
predict_next_token = _prediction.predict_next_token


def _parameter(values, dtype=torch.float64, requires_grad=True):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def _example():
    return (_parameter([[1., 2.], [3., 1.]]), torch.tensor([0, 1]),
            _parameter([[1., 0.], [0., 1.]]), _parameter([0., 0.]))


def _check_case(parameters, prompt, budget, expected=None):
    table, weights, bias = parameters
    snapshots = [p.detach().clone() for p in parameters]
    pointers = [p.data_ptr() for p in parameters]
    flags = [p.requires_grad for p in parameters]
    grad_objects = [p.grad for p in parameters]
    gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    prompt_before = prompt.clone()
    caller_mode = torch.is_grad_enabled()
    actual = generate_token_ids(table, prompt, weights, bias, budget)
    assert isinstance(actual, torch.Tensor), "Return the complete ID tensor."
    assert actual.shape == (prompt.numel() + budget,)
    assert actual.dtype == prompt.dtype and actual.device == prompt.device
    assert not actual.requires_grad and actual.grad_fn is None
    assert actual.data_ptr() != prompt.data_ptr(), "Even zero budget returns a fresh copy."
    assert torch.equal(actual[:prompt.numel()], prompt_before)
    assert torch.equal(prompt, prompt_before), "Preserve the original prompt."
    assert torch.is_grad_enabled() == caller_mode
    for index, parameter in enumerate(parameters):
        assert torch.equal(parameter.detach(), snapshots[index]), "Generation must not train."
        assert parameter.data_ptr() == pointers[index]
        assert parameter.requires_grad == flags[index]
        assert parameter.grad is grad_objects[index]
        if gradients[index] is not None:
            assert torch.equal(parameter.grad, gradients[index])
    if expected is None:
        reference = prompt.clone()
        with torch.no_grad():
            for _ in range(budget):
                scores = table[reference[-1]] @ weights + bias
                next_id = int(scores.argmax().item())
                reference = torch.cat((reference, reference.new_tensor([next_id])))
        expected = reference.tolist()
    assert actual.tolist() == expected, (actual.tolist(), expected)
    return actual


def run_checks():
    global predict_next_token
    table, prompt, weights, bias = _example()
    _check_case((table, weights, bias), prompt, 3, [0, 1, 0, 1, 0])
    print("PASS: worked trace, full prompt-plus-continuation tensor, and original prompt preservation")

    for budget in (0, 1, 2, 6):
        _check_case((table, weights, bias), prompt, budget)
    single = (_parameter([[2.]]), _parameter([[3.]]), _parameter([1.]))
    _check_case(single, torch.tensor([0]), 4, [0, 0, 0, 0, 0])
    varied = (_parameter([[1., -2., 0.], [0., 3., 1.], [2., 0., -1.]]),
              _parameter([[0., 1., -1.], [1., 0., 2.], [-1., 2., 0.]]),
              _parameter([0.2, -0.1, 0.]))
    noncontiguous_prompt = torch.tensor([2, 9, 0, 9, 2])[::2]
    _check_case(varied, noncontiguous_prompt, 5)
    print("PASS: zero/one/multiple-token budgets, varied shapes, repeated IDs, and noncontiguous prompts")

    original_helper = predict_next_token
    calls = []
    def spy(table_arg, prompt_arg, weights_arg, bias_arg):
        assert table_arg is table and weights_arg is weights and bias_arg is bias
        calls.append(prompt_arg.clone())
        next_id = (int(prompt_arg.sum().item()) + prompt_arg.numel()) % 2
        return next_id, torch.tensor([0.5, 0.5], dtype=table_arg.dtype)
    predict_next_token = spy
    try:
        _check_case((table, weights, bias), prompt, 0, [0, 1])
        assert calls == [], "Zero budget must not query the model."
        _check_case((table, weights, bias), prompt, 3, [0, 1, 1, 1, 1])
        assert [item.tolist() for item in calls] == [[0, 1], [0, 1, 1], [0, 1, 1, 1]]
    finally:
        predict_next_token = original_helper
    print("PASS: exactly one completed-helper call per new token with the full growing prefix")

    for requires_grad in (False, True):
        state = tuple(_parameter(values, torch.float32, requires_grad) for values in
                      ([[1., 2.], [3., 1.]], [[1., 0.], [0., 1.]], [0., 0.]))
        for parameter in state:
            parameter.grad = torch.full_like(parameter, 99.)
        _check_case(state, prompt, 3, [0, 1, 0, 1, 0])
        with torch.no_grad():
            _check_case(state, prompt, 2, [0, 1, 0, 1])
    print("PASS: fixed parameters, stored gradient objects/values, trainability, and caller mode")

    wide_table = torch.zeros((258, 1), dtype=torch.float64, requires_grad=True)
    wide_weights = torch.zeros((1, 258), dtype=torch.float64, requires_grad=True)
    bias_values = [0.] * 258
    bias_values[256] = 5.
    wide_bias = _parameter(bias_values)
    merged_ids = _check_case((wide_table, wide_weights, wide_bias), torch.tensor([97]), 2,
                            [97, 256, 256])
    decoder = _load_module("02_06_bpe_decode.py", "completed_generation_decoder").decode_bpe
    vocabulary = {index: bytes([index]) for index in range(256)}
    vocabulary[256] = b"ab"
    vocabulary[257] = b"unused"
    assert decoder(merged_ids.tolist(), vocabulary) == "aabab"
    print("PASS: vocabulary IDs above 255 remain IDs and decode through the matching byte vocabulary")

    first = _check_case((table, weights, bias), prompt, 3)
    second = _check_case((table, weights, bias), prompt, 3)
    continued = _check_case((table, weights, bias), first, 2)
    whole = _check_case((table, weights, bias), prompt, 5)
    assert torch.equal(first, second) and torch.equal(continued, whole)
    first[0] = 1
    assert prompt.tolist() == [0, 1] and second.tolist() == [0, 1, 0, 1, 0]
    raw_vocabulary = {index: bytes([index]) for index in range(256)}
    assert decoder([228, 184, 173], raw_vocabulary) == "中"
    try:
        decoder([228, 184], raw_vocabulary)
    except UnicodeDecodeError:
        pass
    else:
        raise AssertionError("Strict decoding must expose an incomplete UTF-8 suffix.")
    print("PASS: repeatable independent outputs, continued calls, and accumulated UTF-8 decoding")

    training = _load_module("03_12_next_token_training_loop.py", "completed_generation_training")
    initial_table = torch.zeros((256, 2), dtype=torch.float64)
    initial_table[97], initial_table[98] = torch.tensor([1., 2.]), torch.tensor([3., 1.])
    trained_table = initial_table.requires_grad_()
    trained_weights = torch.zeros((2, 256), dtype=torch.float64, requires_grad=True)
    trained_bias = torch.zeros(256, dtype=torch.float64, requires_grad=True)
    training.train_next_token(trained_table, torch.tensor([97, 98]), trained_weights,
                             trained_bias, torch.tensor([98, 97]), 0.1, 100)
    trained_ids = _check_case((trained_table, trained_weights, trained_bias), torch.tensor([97]),
                             6, [97, 98, 97, 98, 97, 98, 97])
    text = decoder(trained_ids.tolist(), raw_vocabulary)
    assert text == "abababa"
    print("PASS: completed training -> fixed-parameter generation -> matching decoder: abababa")
    print("Lesson 03.14 checks passed.")


if __name__ == "__main__":
    run_checks()
