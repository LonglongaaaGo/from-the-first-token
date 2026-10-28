# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.17: stop generation at a newly generated EOS or a length limit.

Run AFTER completing Lesson 03.16 in the environment described in README.md:
    python exercises/03_17_generation_stopping.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse 03.16's generate_sampled_ids for ONE new ID per iteration, keeping
parameters fixed and the caller's random stream continuous. Add a maximum
budget and a newly generated end-of-sequence (EOS) check. Append the sampled
ID first, then inspect it; retain EOS in the returned full ID tensor.

EXACT INPUTS
embedding_table: floating torch.Tensor [V,D], fixed token vectors.
context_ids: torch.long tensor [T], original nonempty prompt.
output_weights: floating torch.Tensor [D,V], fixed vocabulary projection.
output_bias: floating torch.Tensor [V], fixed candidate biases.
max_new_tokens: Python int >= 0, the UPPER LIMIT on new IDs, written M.
temperature: positive finite Python float used by the completed wrapper.
generator: caller-owned torch.Generator, already seeded by supplied setup.
eos_token_id: Python int in [0,V), or None to disable EOS stopping.
Assume valid IDs, compatible shapes/dtypes/devices/generator, finite scores
and scaled scores. Input construction is supplied; validation is not assigned.

RETURN AND STOPPING POLICY
Return (generated_ids, stop_reason).
generated_ids: fresh torch.long tensor [T+L], including the prompt and any
newly sampled EOS; 0 <= L <= M. Preserve original prompt and parameter state.
stop_reason: Python str, "eos" if a newly generated EOS ends the loop,
otherwise "length" when the maximum budget is exhausted.
EOS on the final allowed draw has reason "eos". Zero budget returns an
independent prompt copy and "length", with no helper calls or RNG consumption.
Only newly generated IDs are checked. An EOS in the original prompt, even
at its final position, does not itself stop this new generation request.
Pass the same generator to every helper call; do not reseed or draw again
after EOS. Preserve parameter values/storage/trainability/stored gradients,
caller gradient mode, and global random stream. The 03.16 helper manages
full no-grad inference; this wrapper only handles IDs and stopping control.

FORMULA
If K is the index (starting at 1) of the first newly sampled EOS, or infinity
when there is no such EOS or EOS stopping is disabled, then L=min(M,K).
New-token budget differs from full output length: len(output)=T+L.

COMPLETE WORKED EXAMPLE: AN ILLUSTRATIVE THREE-ID VOCABULARY
Vocabulary: 0 -> 'a', 1 -> 'b', 2 -> EOS control marker (not byte value 2).
    E = [[1,0],[0,1],[0,0]]
    W = [[0,1000,0],[0,0,1000]]; b = [0,0,0]
    context_ids=[0]; max_new_tokens=5; temperature=1.; eos_token_id=2
    After ID 0, scores [0,1000,0] yield computed float64 p=[0,1,0].
    helper call 1 appends ID 1 -> [0,1]; 1 is not EOS, so continue.
    After ID 1, scores [0,0,1000] yield computed p=[0,0,1].
    helper call 2 appends ID 2 -> [0,1,2]; 2 is EOS, so stop.
    Return (tensor([0,1,2]), "eos"); exactly two new IDs, not five.
These hand-picked scores force a small test path; they are not a trained
language model or a universal EOS-ID convention. Display ordinary IDs [0,1]
with the matching byte mapping, excluding the control marker in the caller.

NEW SYNTAX
None means EOS stopping is disabled here. Use "eos_token_id is not None"
to test configuration; truthiness would mistakenly disable valid EOS ID 0.
"==" compares IDs. "and" requires both conditions and short-circuits the
right side when the left is false. "if condition:" runs its indented body
only when true. "break" immediately exits the nearest loop; execution then
reaches the return after that loop. It does not return from the function.

HOW TO WORK
Follow the eight numbered hints. The caller supplies inputs and generators.
Do not rewrite the completed sampled-generation implementation.
Expected final line after implementation and prerequisite completion:
    Lesson 03.17 checks passed.
"""

# import sre_compile
# from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch


def generate_until_eos(
    embedding_table: torch.Tensor,
    context_ids: torch.Tensor,
    output_weights: torch.Tensor,
    output_bias: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator,
    eos_token_id: Optional[int],
) -> tuple[torch.Tensor, str]:
    """Return full sampled IDs and the reason for stopping this continuation."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, initialize generated_ids with context_ids.clone().
    # Example [0], shape [1]; zero budget also returns independent storage.
    #
    # STEP 2: At four spaces, initialize stop_reason to the string "length".
    # This is the default if no newly generated EOS ends the loop.
    #
    # STEP 3: At four spaces, loop over range(max_new_tokens).
    # Steps 4-6 use eight spaces. The loop is an upper bound, not a promise
    # that all M iterations run; break may stop it early.
    #
    # STEP 4: At eight spaces, call generate_sampled_ids with embedding_table,
    # generated_ids, output_weights, output_bias, 1, temperature, generator.
    # Save its full returned tensor back into generated_ids. The literal 1
    # requests exactly ONE new ID. First example result [0,1], shape [2].
    #
    # STEP 5: At eight spaces, extract generated_ids[-1].item() as
    # next_token_id, a Python int. It is the newly appended token.
    #
    # STEP 6: At eight spaces, write an if condition combining
    # eos_token_id is not None AND next_token_id == eos_token_id.
    # Use Python's "and" keyword. Its colon starts the twelve-space body.
    #
    # STEP 7: At twelve spaces inside that if, set stop_reason to "eos",
    # then write break at the same twelve-space indentation. Example second
    # draw is ID 2; it is already included in [0,1,2] when the loop stops.
    #
    # STEP 8: Back at four spaces AFTER the loop, return generated_ids,
    # stop_reason in that order. Example (tensor([0,1,2]), "eos").
    # Zero budget returns (fresh prompt, "length") without querying a helper.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement generate_until_eos for this exercise")
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


_generation = _load_module("03_16_sampled_generation.py", "completed_stopping_generation")
generate_sampled_ids = _generation.generate_sampled_ids


def _tensor(values, *, requires_grad=True, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def _example():
    return (_tensor([[1., 0.], [0., 1.], [0., 0.]]), torch.tensor([0]),
            _tensor([[0., 1000., 0.], [0., 0., 1000.]]), _tensor([0., 0., 0.]))


def _check_case(parameters, prompt, budget, temperature, generator, eos, expected=None):
    table, weights, bias = parameters
    originals = [parameter.detach().clone() for parameter in parameters]
    pointers = [parameter.data_ptr() for parameter in parameters]
    flags, grads = [parameter.requires_grad for parameter in parameters], [parameter.grad for parameter in parameters]
    gradient_values = [None if grad is None else grad.clone() for grad in grads]
    prompt_before, generator_before = prompt.clone(), generator.get_state().clone()
    mode, global_before = torch.is_grad_enabled(), torch.get_rng_state().clone()
    result = generate_until_eos(table, prompt, weights, bias, budget, temperature, generator, eos)
    assert isinstance(result, tuple) and len(result) == 2
    ids, reason = result
    assert isinstance(ids, torch.Tensor) and ids.ndim == 1
    assert ids.dtype == prompt.dtype and ids.device == prompt.device and ids.grad_fn is None
    assert prompt.numel() <= ids.numel() <= prompt.numel() + budget
    assert ids.data_ptr() != prompt.data_ptr() and torch.equal(ids[:prompt.numel()], prompt_before)
    assert torch.equal(prompt, prompt_before) and type(reason) is str and reason in ("eos", "length")
    assert mode == torch.is_grad_enabled() and torch.equal(torch.get_rng_state(), global_before)
    for index, parameter in enumerate(parameters):
        assert torch.equal(parameter.detach(), originals[index]) and parameter.data_ptr() == pointers[index]
        assert parameter.requires_grad == flags[index] and parameter.grad is grads[index]
        if gradient_values[index] is not None:
            assert torch.equal(parameter.grad, gradient_values[index])
    reference_generator = torch.Generator().set_state(generator_before)
    reference, reference_reason = prompt.clone(), "length"
    with torch.no_grad():
        for _ in range(budget):
            scores = table[reference[-1]] @ weights + bias
            probabilities = torch.softmax(scores / temperature, dim=-1)
            token = int(torch.multinomial(probabilities, 1, generator=reference_generator).item())
            reference = torch.cat((reference, reference.new_tensor([token])))
            if eos is not None and token == eos:
                reference_reason = "eos"
                break
    assert torch.equal(generator.get_state(), reference_generator.get_state()), "Stop without an extra random draw."
    assert torch.equal(ids, reference) and reason == reference_reason
    if expected is not None:
        assert ids.tolist() == expected[0] and reason == expected[1]
    return ids, reason


def run_checks():
    global generate_sampled_ids
    table, prompt, weights, bias = _example()
    parameters = (table, weights, bias)
    ids, reason = _check_case(parameters, prompt, 5, 1., torch.Generator().manual_seed(5), 2,
                              ([0, 1, 2], "eos"))
    print("PASS: worked EOS path, appended stop marker, two new IDs, and eos reason")

    _check_case(parameters, prompt, 0, 1., torch.Generator().manual_seed(5), 2, ([0], "length"))
    _check_case(parameters, prompt, 1, 1., torch.Generator().manual_seed(5), 2, ([0, 1], "length"))
    _check_case(parameters, prompt, 2, 1., torch.Generator().manual_seed(5), 2, ([0, 1, 2], "eos"))
    _check_case(parameters, prompt, 4, 1., torch.Generator().manual_seed(5), None)
    always_one = (_tensor([[0.], [0.], [0.]]), _tensor([[0., 0., 0.]]), _tensor([0., 1000., 0.]))
    _check_case(always_one, prompt, 3, 1., torch.Generator().manual_seed(7), 2, ([0, 1, 1, 1], "length"))
    print("PASS: zero/short/full budgets, disabled or absent EOS, and eos priority on the final allowed draw")

    original_helper = generate_sampled_ids
    calls = []
    generator = torch.Generator().manual_seed(11)
    def spy(table_arg, ids_arg, weights_arg, bias_arg, count, temperature, generator_arg):
        assert table_arg is table and weights_arg is weights and bias_arg is bias
        assert count == 1 and temperature == 1. and generator_arg is generator
        calls.append(ids_arg.clone())
        return original_helper(table_arg, ids_arg, weights_arg, bias_arg, count, temperature, generator_arg)
    generate_sampled_ids = spy
    try:
        _check_case(parameters, prompt, 5, 1., generator, 2, ([0, 1, 2], "eos"))
    finally:
        generate_sampled_ids = original_helper
    assert [item.tolist() for item in calls] == [[0], [0, 1]]
    print("PASS: completed generation helper reused for one new ID per growing-prefix call with no call after EOS")

    varied = (_tensor([[1., 0.], [0., 1.], [0.3, -0.2]]),
              _tensor([[0.2, 0.8, -0.4], [0.1, -0.3, 0.7]]), _tensor([0., 0.1, -0.2]))
    for seed in (7, 29):
        for temperature in (0.5, 1., 2.):
            for eos in (None, 0, 2):
                _check_case(varied, torch.tensor([1, 0]), 4, temperature,
                            torch.Generator().manual_seed(seed), eos)
    print("PASS: stochastic EOS-or-budget stopping matches a same-runtime reference and generator state")

    always_zero = (_tensor([[0.], [0.]]), _tensor([[0., 0.]]), _tensor([1000., 0.]))
    _check_case(always_zero, torch.tensor([1]), 3, 1., torch.Generator().manual_seed(9), 0,
                ([1, 0], "eos"))
    _check_case(always_one, torch.tensor([2]), 2, 1., torch.Generator().manual_seed(13), 2,
                ([2, 1, 1], "length"))
    print("PASS: EOS ID zero is valid and only newly generated IDs trigger stopping")

    for trainable in (False, True):
        state = tuple(_tensor(values, requires_grad=trainable, dtype=torch.float32) for values in
                      ([[1., 0.], [0., 1.], [0., 0.]], [[0., 1000., 0.], [0., 0., 1000.]], [0., 0., 0.]))
        for parameter in state:
            parameter.grad = torch.full_like(parameter, 99.)
        _check_case(state, prompt, 5, 1., torch.Generator().manual_seed(17), 2, ([0, 1, 2], "eos"))
        with torch.no_grad():
            _check_case(state, prompt, 0, 1., torch.Generator().manual_seed(17), 2, ([0], "length"))
    def forbidden(*arguments):
        raise AssertionError("Zero budget must not call the generation helper.")
    generate_sampled_ids = forbidden
    try:
        zero, _ = _check_case(parameters, prompt, 0, 1., torch.Generator().manual_seed(19), 2, ([0], "length"))
    finally:
        generate_sampled_ids = original_helper
    zero[0] = 1
    assert prompt.tolist() == [0]
    print("PASS: fresh returns, fixed parameters/gradients, caller/global RNG preservation, and zero-budget independence")

    decoder = _load_module("02_06_bpe_decode.py", "completed_stopping_decoder").decode_bpe
    ordinary_ids = ids[:-1].tolist()
    assert reason == "eos" and ids[-1].item() == 2
    assert decoder(ordinary_ids, {0: b"a", 1: b"b"}) == "ab"
    print("PASS: caller separates the illustrative EOS control marker before matching-vocabulary byte decoding")
    print("Lesson 03.17 checks passed.")


if __name__ == "__main__":
    run_checks()
