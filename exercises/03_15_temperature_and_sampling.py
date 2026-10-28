# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.15: temperature probabilities and one categorical token sample.

Run in the existing PyTorch environment from the repository root:
    python exercises/03_15_temperature_and_sampling.py --check temperature
    python exercises/03_15_temperature_and_sampling.py --check sample
    python exercises/03_15_temperature_and_sampling.py --check all

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
Reuse 03.09's stable softmax. Temperature changes the probability
distribution; sampling selects an ID according to those probabilities.
Positive temperature preserves score ordering in exact arithmetic, so
argmax alone keeps the same winner. Sampling may select a smaller entry.
The completed greedy predictor and generation loop remain unchanged.

STAGE A: EXACT INPUTS AND RETURN
temperature_probabilities(logits, temperature)
logits: floating torch.Tensor [T,V], one candidate-score row per position.
temperature: positive finite Python float, written tau in the formulas.
T counts positions; it is not the temperature. Assume T,V>0, finite scores
and finite scaled scores logits/temperature. Input validation is not assigned.
Return probabilities: floating tensor [T,V], same dtype/device, each row
normalized. Preserve logits values and its autograd connection, like 03.09.
No detach, no_grad, backward, in-place scaling, or parameter update here.
The inference caller can place the complete forward under no-grad later.

FORMULA
    p[t,v] = exp(z[t,v]/tau) / sum_j exp(z[t,j]/tau)
Reuse the completed stable softmax on scaled_logits = logits / temperature.
Divide LOGITS before softmax, not probabilities afterwards.

COMPLETE TEMPERATURE EXAMPLE: T=2, V=2
    logits = [[3,1],[1,3]]
    temperature = 0.5
    scaled_logits = [[6,2],[2,6]]
    shifted logits = [[0,-4],[-4,0]]
    exp scores = [[1,0.0183156389],[0.0183156389,1]]
    probabilities = [[0.9820137900,0.0179862100],
                     [0.0179862100,0.9820137900]]
For row [3,1], temperature 1 gives [0.8807970780,0.1192029220];
temperature 2 gives [0.7310585786,0.2689414214]. The greedy ID stays 0.
tau<1 sharpens, tau>1 flattens. tau=0 is outside the division formula.

STAGE B: EXACT INPUTS AND RETURN
sample_token_id(probabilities, generator)
probabilities: floating torch.Tensor [V], one normalized candidate row.
Assume V>0, finite nonnegative values, positive sum, matching generator
device on a supported backend. Zero-probability entries must not be selected.
generator: supplied torch.Generator carrying the pseudorandom stream state.
Return a Python int in [0,V), not the probability value or a tensor.
Preserve probability values/storage/trainability/stored gradients and caller
mode. Advance the supplied generator by ONE categorical draw; do not reseed
it, replace it, or use/alter the global random stream.

COMPLETE SAMPLING EXAMPLE
    probabilities = [0.6,0.3,0.1]
Interpret unit-interval regions [0,0.6), [0.6,0.9), [0.9,1) as IDs 0,1,2.
A hypothetical uniform draw u=0.75 falls in the ID-1 region, although ID 0
has the largest probability. This is a probability illustration, not a claim
about torch.multinomial's internal algorithm or exact seeded draws.
Many samples tend toward the supplied frequencies; one sample is not a
frequency estimate and adjacent samples may repeat.

NEW SYNTAX AND RANDOM STATE
torch.multinomial(probabilities, num_samples=1, generator=generator)
returns a one-element integer tensor [1] when probabilities has shape [V].
.item() extracts its Python int. num_samples counts selected IDs, not
vocabulary size or future generated tokens. Each invocation starts a new
one-draw call; default replacement=False does not prohibit repeats across
separate calls. The caller seeds a generator once and keeps using its state.
Same initial state, inputs, and call order reproduce the stream within a
compatible runtime; exact sequences across versions/backends are not promised.

HOW TO REVIEW
Both learner stages are complete. Review Stage A with --check temperature,
Stage B with --check sample, and the connected pipeline with --check all.
All learner statements use four spaces; neither function needs a loop.
Expected final line with both stages:
    Lesson 03.15 checks passed.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def temperature_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return connected row-wise probabilities after positive temperature scaling."""
    # ===== START LEARNER CODE =====
    # STEP A1: At four spaces, divide logits by temperature and save the
    # new tensor as scaled_logits. Keep shape [T,V]. Example [[6,2],[2,6]].
    # Do not multiply, scale in place, or divide softmax output instead.
    #
    # STEP A2: At four spaces, call the completed
    # softmax_probabilities(scaled_logits), saving probabilities.
    # The helper handles row-wise normalization and stable subtraction.
    # First row at tau=0.5 is [0.9820137900,0.0179862100].
    #
    # STEP A3: At four spaces, return probabilities as the [T,V] tensor.
    # Preserve the input and graph; inference scope belongs to the caller.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement temperature_probabilities for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE A =====


def sample_token_id(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    """Return one Python integer token ID using the caller's random stream."""
    # ===== START LEARNER CODE =====
    # STEP B1: At four spaces, call torch.multinomial with probabilities,
    # num_samples=1, and generator=generator. Save sampled_ids, shape [1].
    # Use the supplied distribution and stream; no argmax or reseeding.
    #
    # STEP B2: At four spaces, use sampled_ids.item() to obtain
    # next_token_id, a Python int. It is an index, not a probability value.
    #
    # STEP B3: At four spaces, return next_token_id. Preserve probabilities.
    # The supplied generator's stream advances; global RNG stays unchanged.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement sample_token_id for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE B =====


# ===== SUPPLIED SETUP AND CHECK DRIVER: DO NOT EDIT =====
def _load_module(filename, module_name):
    path = Path(__file__).with_name(filename)
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_projection = _load_module("03_09_logits_and_softmax.py", "completed_temperature_projection")
softmax_probabilities = _projection.softmax_probabilities
compute_logits = _projection.compute_logits


def _tensor(values, requires_grad=False, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, requires_grad=requires_grad)


def check_temperature():
    global softmax_probabilities
    logits = _tensor([[3., 1.], [1., 3.]])
    expected_rows = {0.5: [0.9820137900379085, 0.017986209962091555],
                     1.: [0.8807970779778823, 0.11920292202211755],
                     2.: [0.7310585786300049, 0.26894142136999516]}
    for temperature, row in expected_rows.items():
        actual = temperature_probabilities(logits, temperature)
        assert isinstance(actual, torch.Tensor) and actual.shape == logits.shape
        assert torch.allclose(actual, _tensor([row, row[::-1]]), rtol=1e-12, atol=1e-12)
        assert actual.argmax(dim=-1).tolist() == [0, 1]
    assert torch.equal(logits, _tensor([[3., 1.], [1., 3.]]))
    print("PASS: temperature worked example, row-wise normalization, and unchanged greedy ordering")

    cases = [([[1000., 0., -1000.]], 0.5), ([[2.], [-1.]], 2.),
             ([[0., 0., 0.], [2., -3., 0.4], [-1., 1., 2.]], 1.7)]
    for values, temperature in cases:
        scores = _tensor(values)
        for dtype in (torch.float64, torch.float32):
            original = scores.to(dtype)
            before = original.clone()
            actual = temperature_probabilities(original, temperature)
            assert actual.dtype == dtype and actual.device == original.device
            assert torch.isfinite(actual).all() and (actual >= 0).all()
            assert torch.allclose(actual.sum(dim=-1), torch.ones(original.shape[0], dtype=dtype))
            assert torch.allclose(actual, torch.softmax(original / temperature, dim=-1))
            assert torch.equal(original, before)
        shifted = temperature_probabilities(scores + 50., temperature)
        assert torch.allclose(shifted, temperature_probabilities(scores, temperature))
    print("PASS: positive temperatures, extreme/tied scores, varied shapes/dtypes, and shift invariance")

    connected = _tensor([[1., -0.5, 2.], [0.2, 0., -1.]], True)
    connected.grad = torch.full_like(connected, 99.)
    stored_grad = connected.grad
    old_helper = softmax_probabilities
    calls = []
    def spy(scaled):
        calls.append(scaled)
        assert torch.allclose(scaled, connected / 1.7)
        assert torch.is_grad_enabled() and scaled.requires_grad
        return old_helper(scaled)
    softmax_probabilities = spy
    try:
        actual = temperature_probabilities(connected, 1.7)
    finally:
        softmax_probabilities = old_helper
    assert len(calls) == 1 and actual.grad_fn is not None
    assert connected.grad is stored_grad and torch.equal(stored_grad, torch.full_like(connected, 99.))
    coefficients = _tensor([[1., 2., -1.], [0.5, -2., 1.]])
    gradient = torch.autograd.grad((actual * coefficients).sum(), connected)[0]
    expected = actual.detach() * (coefficients - (coefficients * actual.detach()).sum(dim=-1, keepdim=True)) / 1.7
    assert torch.allclose(gradient, expected, rtol=1e-12, atol=1e-12)
    print("PASS: completed stable helper reused with connected analytic gradients and preserved inputs/buffers")


def _checked_sample(probabilities, generator):
    before = probabilities.detach().clone()
    pointer, flag, stored_grad = probabilities.data_ptr(), probabilities.requires_grad, probabilities.grad
    gradient_before = None if stored_grad is None else stored_grad.clone()
    mode, global_state = torch.is_grad_enabled(), torch.get_rng_state().clone()
    result = sample_token_id(probabilities, generator)
    assert type(result) is int and 0 <= result < probabilities.numel()
    assert probabilities[result] > 0, "A zero-probability candidate cannot be sampled."
    assert torch.equal(probabilities.detach(), before) and probabilities.data_ptr() == pointer
    assert probabilities.requires_grad == flag and probabilities.grad is stored_grad
    if gradient_before is not None:
        assert torch.equal(stored_grad, gradient_before)
    assert torch.is_grad_enabled() == mode and torch.equal(torch.get_rng_state(), global_state)
    return result


def check_sample():
    forced = _tensor([0., 1., 0.], True)
    forced.grad = torch.full_like(forced, 99.)
    for _ in range(3):
        assert _checked_sample(forced, torch.Generator().manual_seed(5)) == 1
    assert _checked_sample(_tensor([1.]), torch.Generator().manual_seed(9)) == 0
    wide = torch.zeros(258, dtype=torch.float64)
    wide[256] = 1.
    with torch.no_grad():
        assert _checked_sample(wide, torch.Generator().manual_seed(3)) == 256
    print("PASS: Python ID return, zero-probability exclusion, merged IDs, and input/gradient/mode preservation")

    probabilities = _tensor([0.6, 0.3, 0.1])
    for seed in (7, 29):
        actual_generator = torch.Generator().manual_seed(seed)
        reference_generator = torch.Generator().manual_seed(seed)
        for _ in range(24):
            expected = int(torch.multinomial(probabilities, 1, generator=reference_generator).item())
            assert _checked_sample(probabilities, actual_generator) == expected
            assert torch.equal(actual_generator.get_state(), reference_generator.get_state())
    print("PASS: supplied random stream matches one categorical draw per call without global RNG changes")

    def draws(seed, count):
        generator = torch.Generator().manual_seed(seed)
        return [sample_token_id(probabilities, generator) for _ in range(count)]
    assert draws(41, 32) == draws(41, 32)
    samples = torch.tensor(draws(73, 2048))
    frequencies = torch.bincount(samples, minlength=3).to(torch.float64) / samples.numel()
    assert torch.allclose(frequencies, probabilities, atol=0.04, rtol=0.)
    print("PASS: reproducible caller-seeded streams and many-draw frequencies consistent with probabilities")


def check_pipeline():
    table = _tensor([[1., 2.], [3., 1.]], True)
    weights = _tensor([[1., 0.], [0., 1.]], True)
    bias = _tensor([0., 0.], True)
    parameters = (table, weights, bias)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    sequence = torch.tensor([0, 1])
    reference_sequence = sequence.clone()
    generator, reference_generator = torch.Generator().manual_seed(137), torch.Generator().manual_seed(137)
    with torch.no_grad():
        for _ in range(6):
            scores = compute_logits(table[sequence], weights, bias)
            probabilities = temperature_probabilities(scores, 2.)[-1]
            sampled_id = sample_token_id(probabilities, generator)
            reference_probabilities = torch.softmax((table[reference_sequence[-1]] @ weights + bias) / 2., dim=-1)
            reference_id = int(torch.multinomial(reference_probabilities, 1, generator=reference_generator).item())
            assert sampled_id == reference_id and not probabilities.requires_grad
            sequence = torch.cat((sequence, sequence.new_tensor([sampled_id])))
            reference_sequence = torch.cat((reference_sequence, reference_sequence.new_tensor([reference_id])))
    assert torch.equal(sequence, reference_sequence) and sequence.shape == (8,)
    assert all(torch.equal(parameter.detach(), initial) for parameter, initial in zip(parameters, before))
    print("PASS: fixed-parameter logits -> temperature -> last row -> sampling -> growing-prefix pipeline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("temperature", "sample", "all"), default="all")
    arguments = parser.parse_args()
    if arguments.check in ("temperature", "all"):
        check_temperature()
    if arguments.check in ("sample", "all"):
        check_sample()
    if arguments.check == "all":
        check_pipeline()
        print("Lesson 03.15 checks passed.")
    else:
        print(f"Lesson 03.15 {arguments.check} checks passed.")


if __name__ == "__main__":
    main()
