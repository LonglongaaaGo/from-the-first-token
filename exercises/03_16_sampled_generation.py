# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.16: connect temperature and sampling to autoregressive generation.

Run AFTER completing both Lesson 03.15 stages in the environment described in README.md:
    python exercises/03_16_sampled_generation.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse compute_logits from 03.09 and temperature_probabilities and
sample_token_id from 03.15. Review 03.14's fresh prompt copy, exact new-token
budget, one-ID tensor, concatenation reassignment, and return after the loop.
This function samples from the final position instead of using greedy argmax.
The original greedy functions and learner implementations stay intact.

EXACT INPUTS
embedding_table: floating torch.Tensor [V,D], fixed token vectors.
context_ids: torch.long tensor [T], original nonempty prompt.
output_weights: floating torch.Tensor [D,V], fixed vocabulary projection.
output_bias: floating torch.Tensor [V], fixed candidate biases.
num_new_tokens: Python int >= 0, EXACT number of new IDs to append.
temperature: positive finite Python float, applied to logits before softmax.
generator: caller-owned torch.Generator, already seeded by supplied setup.
Assume valid IDs, compatible shapes/dtypes/devices and supported generator,
finite scores and scaled scores. Input validation is not a learner task.

RETURN AND STATE
Return generated_ids, a fresh torch.long tensor [T + num_new_tokens],
containing the prompt plus sampled continuation on the prompt's device.
Preserve the original prompt, parameter values/storage/trainability/stored
gradients, caller gradient mode, and global random stream. Consume the
supplied generator continuously, exactly one completed sampler call per
new ID. Zero budget returns an independent prompt copy without helper calls
or generator-state changes. No training, EOS stopping, or truncation here.

FORMULAS
    s^(0) = context_ids
    H^(k) = E[s^(k)]; Z^(k) = H^(k) @ W + b
    P^(k) = softmax(Z^(k) / temperature), independently by row
    p^(k) = P^(k)[-1]; v_k ~ Categorical(p^(k))
    s^(k+1) = concat(s^(k), [v_k])
N=num_new_tokens gives N sampler calls and T+N total IDs. Parameters stay
fixed. This lookup-plus-head model's next distribution uses only its last ID.

COMPLETE WORKED EXAMPLE: HYPOTHETICAL SAMPLED CHOICES
    E = [[1,2],[3,1]], W = [[1,0],[0,1]], b = [0,0]
    context_ids = [0,1]; temperature = 2.; num_new_tokens = 3
    current [0,1]   -> last scores [3,1] -> p=[0.7311,0.2689]
        illustrative u=0.2 selects ID 0; new sequence [0,1,0]
    current [0,1,0] -> last scores [1,2] -> p=[0.3775,0.6225]
        illustrative u=0.3 selects ID 0; new sequence [0,1,0,0]
    current [0,1,0,0] -> same last scores and p
        illustrative u=0.8 selects ID 1; new sequence [0,1,0,0,1]
Return [0,1,0,0,1] for those hypothetical choices. The second choice is a
valid smaller-probability sample. This explains recurrence, NOT a promised
PyTorch seeded trace or its internal algorithm. The real helper consumes
the supplied generator; checks compare it to a same-runtime reference.

SCOPE AND INDENTATION
Copy the integer prompt at four spaces. Enter with torch.no_grad(): at four
spaces. The for loop is at eight spaces inside that context; its forward,
selection, and append body is at twelve spaces. The whole lookup/projection/
temperature forward is untracked. Return at four spaces after both blocks,
restoring caller mode. The mathematical helper from 03.15 remains unchanged.

HOW TO WORK
Follow the ten hints. Helpers load from their exact exercise files; complete
03.15 first. Do not seed inside the loop or call the greedy predictor here.
Expected final line after learner implementation and prerequisite completion:
    Lesson 03.16 checks passed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def generate_sampled_ids(
    embedding_table: torch.Tensor,
    context_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    num_new_tokens: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a fresh prompt-plus-continuation tensor using temperature sampling."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, initialize generated_ids with context_ids.clone().
    # Example [0,1]. Zero budget must also return independent storage.
    #
    # STEP 2: At four spaces, enter with torch.no_grad():. Steps 3-9 are
    # inside it. Preserve parameter flags and stored gradient buffers.
    #
    # STEP 3: At eight spaces, write a for loop over range(num_new_tokens).
    # Steps 4-9 use TWELVE spaces, inside both the context and loop.
    #
    # STEP 4: At twelve spaces, obtain selected_vectors by indexing
    # embedding_table[generated_ids]. First shape [2,2], then [3,2], etc.
    # Use the current growing sequence, not the unchanged original prompt.
    #
    # STEP 5: At twelve spaces, call compute_logits(selected_vectors,
    # output_weights, output_bias), saving logits. Shape [T+k,V].
    #
    # STEP 6: At twelve spaces, call temperature_probabilities(logits,
    # temperature), saving probabilities. Shape still [T+k,V].
    #
    # STEP 7: At twelve spaces, select probabilities[-1] as
    # next_probabilities, shape [V]. First example [0.7311,0.2689].
    #
    # STEP 8: At twelve spaces, call sample_token_id(next_probabilities,
    # generator), saving next_token_id. The same supplied stream is reused.
    # Do not reset it or replace this draw with argmax.
    #
    # STEP 9: At twelve spaces, create next_id_tensor from [next_token_id]
    # with torch.tensor, dtype=generated_ids.dtype, device=generated_ids.device.
    # Join it with generated_ids using torch.cat((generated_ids,
    # next_id_tensor), dim=0), assigning the result back to generated_ids.
    # First example sequence after append is [0,1,0], shape [3].
    #
    # STEP 10: Back at four spaces AFTER the context and loop, return
    # generated_ids. Shape [T+N]. Neither return inside the loop nor extra
    # prediction after the budget is part of this function.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement generate_sampled_ids for this exercise")
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


_sampling = _load_module("03_15_temperature_and_sampling.py", "completed_sampled_generation_selection")
temperature_probabilities = _sampling.temperature_probabilities
sample_token_id = _sampling.sample_token_id
_projection = _load_module("03_09_logits_and_softmax.py", "completed_sampled_generation_projection")
compute_logits = _projection.compute_logits


def _tensor(values, *, requires_grad=True, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def _example():
    return (_tensor([[1., 2.], [3., 1.]]), torch.tensor([0, 1]),
            _tensor([[1., 0.], [0., 1.]]), _tensor([0., 0.]))


def _check_case(parameters, prompt, budget, temperature, generator, expected=None):
    table, weights, bias = parameters
    values = [p.detach().clone() for p in parameters]
    pointers, flags = [p.data_ptr() for p in parameters], [p.requires_grad for p in parameters]
    grads = [p.grad for p in parameters]
    grad_values = [None if p.grad is None else p.grad.clone() for p in parameters]
    prompt_before, generator_before = prompt.clone(), generator.get_state().clone()
    global_before, mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    result = generate_sampled_ids(table, prompt, weights, bias, budget, temperature, generator)
    assert isinstance(result, torch.Tensor) and result.shape == (prompt.numel() + budget,)
    assert result.dtype == prompt.dtype and result.device == prompt.device
    assert result.data_ptr() != prompt.data_ptr() and result.grad_fn is None
    assert torch.equal(result[:prompt.numel()], prompt_before) and torch.equal(prompt, prompt_before)
    assert torch.equal(torch.get_rng_state(), global_before) and torch.is_grad_enabled() == mode
    for index, parameter in enumerate(parameters):
        assert torch.equal(parameter.detach(), values[index])
        assert parameter.data_ptr() == pointers[index] and parameter.requires_grad == flags[index]
        assert parameter.grad is grads[index]
        if grad_values[index] is not None:
            assert torch.equal(parameter.grad, grad_values[index])
    if budget == 0:
        assert torch.equal(generator.get_state(), generator_before)
    if expected is None:
        reference_generator = torch.Generator().set_state(generator_before)
        reference = prompt.clone()
        with torch.no_grad():
            for _ in range(budget):
                scores = table[reference[-1]] @ weights + bias
                probabilities = torch.softmax(scores / temperature, dim=-1)
                next_id = int(torch.multinomial(probabilities, 1, generator=reference_generator).item())
                reference = torch.cat((reference, reference.new_tensor([next_id])))
        expected = reference.tolist()
        assert torch.equal(generator.get_state(), reference_generator.get_state())
    assert result.tolist() == expected, (result.tolist(), expected)
    return result


def run_checks():
    global compute_logits, temperature_probabilities, sample_token_id
    table, prompt, weights, bias = _example()
    parameters = (table, weights, bias)
    real_sampler = sample_token_id
    scripted_choices = iter([0, 0, 1])
    distributions = []
    scripted_generator = torch.Generator().manual_seed(5)
    def scripted_sampler(probabilities, generator):
        assert generator is scripted_generator
        distributions.append(probabilities.clone())
        return next(scripted_choices)
    sample_token_id = scripted_sampler
    try:
        _check_case(parameters, prompt, 3, 2., scripted_generator, [0, 1, 0, 0, 1])
    finally:
        sample_token_id = real_sampler
    expected = [[0.7310585786300049, 0.26894142136999516],
                [0.37754066879814546, 0.6224593312018546],
                [0.37754066879814546, 0.6224593312018546]]
    assert torch.allclose(torch.stack(distributions), _tensor(expected), rtol=1e-12, atol=1e-12)
    print("PASS: worked recurrence with scripted valid choices, including a non-greedy selection")

    for temperature in (0.5, 1., 2.):
        for budget in (0, 1, 3, 6):
            _check_case(parameters, prompt, budget, temperature, torch.Generator().manual_seed(137))
    varied = (_tensor([[1., -2., 0.], [0., 3., 1.], [2., 0., -1.]]),
              _tensor([[0., 1., -1.], [1., 0., 2.], [-1., 2., 0.]]), _tensor([0.2, -0.1, 0.]))
    for seed in (7, 29):
        _check_case(varied, torch.tensor([2, 9, 0, 9, 2])[::2], 5, 1.7,
                    torch.Generator().manual_seed(seed))
    print("PASS: positive temperatures, exact budgets, varied shapes/prompts, and reference random streams")

    old_projection, old_temperature = compute_logits, temperature_probabilities
    events, prefixes = [], []
    generator = torch.Generator().manual_seed(17)
    def projection_spy(vectors, supplied_weights, supplied_bias):
        events.append("logits")
        assert not torch.is_grad_enabled() and not vectors.requires_grad
        assert supplied_weights is weights and supplied_bias is bias
        prefixes.append(vectors.clone())
        return old_projection(vectors, supplied_weights, supplied_bias)
    def temperature_spy(scores, temperature):
        events.append("temperature")
        assert not torch.is_grad_enabled() and not scores.requires_grad and temperature == 2.
        return old_temperature(scores, temperature)
    def sampling_spy(probabilities, supplied_generator):
        events.append("sample")
        assert probabilities.shape == (2,) and not probabilities.requires_grad
        assert supplied_generator is generator
        return real_sampler(probabilities, supplied_generator)
    compute_logits, temperature_probabilities, sample_token_id = projection_spy, temperature_spy, sampling_spy
    try:
        result = _check_case(parameters, prompt, 3, 2., generator)
    finally:
        compute_logits, temperature_probabilities, sample_token_id = old_projection, old_temperature, real_sampler
    assert events == ["logits", "temperature", "sample"] * 3
    assert [vectors.shape[0] for vectors in prefixes] == [2, 3, 4]
    for index, vectors in enumerate(prefixes):
        assert torch.equal(vectors, table[result[:2+index]].detach())
    print("PASS: completed helpers reused once per token in order under full no-grad with growing prefixes")

    for trainable in (False, True):
        state = tuple(_tensor(values, requires_grad=trainable, dtype=torch.float32) for values in
                      ([[1., 2.], [3., 1.]], [[1., 0.], [0., 1.]], [0., 0.]))
        for parameter in state:
            parameter.grad = torch.full_like(parameter, 99.)
        _check_case(state, prompt, 4, 1.3, torch.Generator().manual_seed(23))
        with torch.no_grad():
            _check_case(state, prompt, 2, 1.3, torch.Generator().manual_seed(23))
    print("PASS: fixed parameters, stored gradients, original prompts, global RNG, and caller-mode preservation")

    generator = torch.Generator().manual_seed(41)
    calls = []
    def forbidden_helper(*arguments):
        calls.append(arguments)
        raise AssertionError("No forward or draw is allowed with zero new tokens.")
    compute_logits = temperature_probabilities = sample_token_id = forbidden_helper
    try:
        zero = _check_case(parameters, prompt, 0, 2., generator, [0, 1])
    finally:
        compute_logits, temperature_probabilities, sample_token_id = old_projection, old_temperature, real_sampler
    assert calls == []
    zero[0] = 1
    assert prompt.tolist() == [0, 1]
    print("PASS: zero budget returns an independent prompt without forward calls or random-state consumption")

    first_generator, whole_generator = torch.Generator().manual_seed(53), torch.Generator().manual_seed(53)
    first = _check_case(parameters, prompt, 2, 2., first_generator)
    continued = _check_case(parameters, first, 3, 2., first_generator)
    whole = _check_case(parameters, prompt, 5, 2., whole_generator)
    assert torch.equal(continued, whole) and torch.equal(first_generator.get_state(), whole_generator.get_state())
    replay = _check_case(parameters, prompt, 5, 2., torch.Generator().manual_seed(53))
    assert torch.equal(replay, whole)
    replay[0] = 1
    assert prompt.tolist() == [0, 1] and whole[0].item() == 0
    print("PASS: same-state replay and continued calls agree with one longer call while outputs stay independent")

    single = (_tensor([[2.]]), _tensor([[3.]]), _tensor([1.]))
    _check_case(single, torch.tensor([0]), 4, 0.5, torch.Generator().manual_seed(3), [0, 0, 0, 0, 0])
    wide_table, wide_weights = _tensor([[0.]]*258), _tensor([[0.]*258])
    wide_bias = _tensor([0.]*256+[1000., 0.])
    merged = _check_case((wide_table, wide_weights, wide_bias), torch.tensor([97]), 2, 0.5,
                         torch.Generator().manual_seed(7), [97, 256, 256])
    decoder = _load_module("02_06_bpe_decode.py", "completed_sampled_generation_decoder").decode_bpe
    vocabulary = {index: bytes([index]) for index in range(256)}
    vocabulary[256], vocabulary[257] = b"ab", b"unused"
    assert decoder(merged.tolist(), vocabulary) == "aabab"
    print("PASS: single-candidate generation and merged IDs decoded through the matching vocabulary")
    print("Lesson 03.16 checks passed.")


if __name__ == "__main__":
    run_checks()
