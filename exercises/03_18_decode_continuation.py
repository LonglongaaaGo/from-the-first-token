# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.18: turn a generated continuation into text.

Run AFTER the Lesson 03.17 walkthrough and implementation:
    python exercises/03_18_decode_continuation.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

PURPOSE AND PREREQUISITES
03.17 returns full IDs, including the prompt and any newly generated EOS.
Select only the continuation, remove one configured trailing EOS, then reuse
02.06's decode_bpe. The stopping reason remains separate caller metadata.
There is no forward pass, sampling, or parameter update in this function.

EXACT INPUTS
generated_ids: one-dimensional torch.long tensor [T+L], full returned IDs.
prompt_length: Python int T, with 0 <= T <= generated_ids.numel().
vocab: dict[int, bytes], the matching ORDINARY token-to-bytes vocabulary.
eos_token_id: Python int identifying a separate control token, or None.
When configured, EOS must not be an ordinary key in vocab. The valid 03.17
output protocol allows newly generated EOS only at the continuation's end.
Assume valid tensor shape and prompt length; validation is not assigned.

RETURN AND ERRORS
Return Python str containing only the newly generated ordinary-token text.
Keep generated_ids, vocab, gradient mode, and random streams unchanged.
No new ordinary IDs returns "". Unknown ordinary IDs raise KeyError through
the reused decoder. Invalid/incomplete UTF-8 raises UnicodeDecodeError.
Remove only a trailing configured EOS; do not filter arbitrary unknown IDs.
None disables special removal. EOS ID 0 is valid; check presence explicitly.

FORMULA
    c = generated_ids[prompt_length:]
    q = c[:-1] if c is nonempty, EOS is configured, and c[-1] == EOS
        otherwise c
    raw_bytes = vocab[q[0]] + ... + vocab[q[m-1]]
    text = strict_UTF8_decode(raw_bytes)
q is empty when m=0, so raw_bytes=b"" and text="". m counts ordinary
tokens; their byte count and decoded character count can all differ.

COMPLETE WORKED EXAMPLE: fixture-bpe-utf8-v1 (hand-built, not trained)
All base IDs 0..255 map to their bytes. Additional ordinary entries:
    vocab[256] = b"ab"; vocab[257] = bytes([228,189])
The separate EOS ID is 258; it has no entry in the ordinary byte vocabulary.
    generated_ids = tensor([97,256,257,160,258]); prompt_length = 1
    continuation after slicing: [256,257,160,258]
    after removing trailing EOS: [256,257,160]
    byte chunks: [b"ab", bytes([228,189]), bytes([160])]
    joined bytes: [97,98,228,189,160]
    strict UTF-8 text: "ab你" (3 ordinary tokens, 5 bytes, 3 characters)
The prompt ID 97 is excluded. The two Chinese byte chunks must be joined
before decoding. EOS removal cannot repair incomplete or invalid byte data.

BOUNDARY EXAMPLES
    [97], T=1                         -> ""
    [97,258], T=1, EOS=258            -> ""
    [97,256], T=1, EOS=258            -> "ab" (no trailing EOS)
    [97,228,189,258], T=1, EOS=258    -> UnicodeDecodeError
    [228,189,160], T=1, EOS=None      -> UnicodeDecodeError for new bytes
The last full sequence decodes to 你, but its continuation [189,160] starts
inside a character. Separate full-output rendering is a different policy.

NEW SYNTAX BEFORE IMPLEMENTATION
Tensor .tolist() makes a Python list of IDs. list[T:] selects index T through
the end; list[:-1] excludes the final item. Neither changes the original list.
len(list)>0 guards [-1] on empty input. A and B and C short-circuits from left
to right. is not None checks configuration presence, so valid ID 0 works.
An if body is indented four spaces beyond the surrounding function body.
The final return is outside that if and reuses the already completed decoder.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

import torch


def decode_continuation(
    generated_ids: torch.Tensor,
    prompt_length: int,
    vocab: dict[int, bytes],
    eos_token_id: Optional[int],
) -> str:
    """Select new IDs, remove a trailing EOS, and strictly decode their bytes."""
    # ===== START LEARNER CODE =====
    # STEP 1: At four spaces, call generated_ids.tolist(), storing all_ids.
    # The example list is [97,256,257,160,258]. Do not alter the input tensor.
    #
    # STEP 2: At four spaces, slice all_ids[prompt_length:], storing
    # continuation_ids. Example [256,257,160,258]. T counts prompt tokens,
    # not prompt characters or bytes; do not remove a text substring later.
    #
    # STEP 3: At four spaces, start an if with THREE checks joined by and:
    # len(continuation_ids)>0, eos_token_id is not None, and equality of
    # continuation_ids[-1] to eos_token_id. Check nonemptiness before [-1].
    # A colon starts the body. Never use truthiness of the EOS number.
    #
    # STEP 4: At eight spaces inside the if, replace continuation_ids with
    # its [:-1] slice. Example [256,257,160]. Remove one final marker only.
    # Do not discard prompt markers, every special-looking ID, or errors.
    #
    # STEP 5: Back at four spaces AFTER the if, call decode_bpe with
    # continuation_ids and vocab, and return its str. It already looks up,
    # joins bytes in order, and decodes UTF-8 once. Example "ab你".
    # Do not reimplement that decoder or catch its required exceptions.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement decode_continuation for this exercise")
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


decode_bpe = _load_module("02_06_bpe_decode.py", "completed_continuation_decoder").decode_bpe
generate_until_eos = _load_module("03_17_generation_stopping.py", "completed_continuation_stopping").generate_until_eos


def _fixture():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256], vocab[257] = b"ab", bytes([228, 189])
    return vocab, 258


def _check(ids, prompt_length, vocab, eos, expected):
    original_ids, original_vocab = ids.clone(), vocab.copy()
    global_state, mode = torch.get_rng_state().clone(), torch.is_grad_enabled()
    text = decode_continuation(ids, prompt_length, vocab, eos)
    assert isinstance(text, str) and text == expected, (text, expected)
    assert torch.equal(ids, original_ids) and vocab == original_vocab
    assert torch.equal(torch.get_rng_state(), global_state) and torch.is_grad_enabled() == mode
    return text


def _expect_error(error_type, ids, prompt_length, vocab, eos):
    before_ids, before_vocab = ids.clone(), vocab.copy()
    try:
        decode_continuation(ids, prompt_length, vocab, eos)
    except error_type:
        assert torch.equal(ids, before_ids) and vocab == before_vocab
        return
    raise AssertionError(f"Expected {error_type.__name__} to propagate")


def run_checks():
    global decode_bpe
    vocab, eos = _fixture()
    _check(torch.tensor([97, 256, 257, 160, eos]), 1, vocab, eos, "ab你")
    assert b"".join([vocab[256], vocab[257], vocab[160]]) == "ab你".encode("utf-8")
    print("PASS: prompt exclusion, trailing EOS removal, merged IDs, and cross-token UTF-8")

    for ids, length, expected in (([97], 1, ""), ([97, eos], 1, ""),
                                  ([97, 256], 1, "ab"), ([256], 0, "ab"),
                                  ([97, 97, 98], 1, "ab"), ([], 0, "")):
        _check(torch.tensor(ids, dtype=torch.long), length, vocab, eos, expected)
    _check(torch.tensor([256, 256, 99]), 0, vocab, None, "ababc")
    print("PASS: empty output, immediate EOS, length stopping, zero prompt, and ordered repeats")

    _check(torch.tensor([1, 2, 0]), 1, {1: b"p", 2: b"a"}, 0, "a")
    _check(torch.tensor([1, 0]), 1, {1: b"p", 0: b"\x00"}, None, "\x00")
    _check(torch.tensor([eos, 97]), 1, vocab, eos, "a")
    _expect_error(KeyError, torch.tensor([97, eos]), 1, vocab, None)
    _expect_error(KeyError, torch.tensor([97, eos, 98]), 1, vocab, eos)
    print("PASS: explicit EOS ID zero, disabled removal, prompt control IDs, and unhidden interior markers")

    real_decoder, calls = decode_bpe, []
    def decoder_spy(token_ids, supplied_vocab):
        assert isinstance(token_ids, list) and supplied_vocab is vocab
        calls.append(token_ids.copy())
        return real_decoder(token_ids, supplied_vocab)
    decode_bpe = decoder_spy
    try:
        _check(torch.tensor([97, 256, eos]), 1, vocab, eos, "ab")
        _check(torch.tensor([97]), 1, vocab, eos, "")
    finally:
        decode_bpe = real_decoder
    assert calls == [[256], []], calls
    noncontiguous = torch.tensor([97, 99, 256, 99, eos, 99])[::2]
    _check(noncontiguous, 1, vocab, eos, "ab")
    with torch.no_grad():
        _check(torch.tensor([97, 256, eos]), 1, vocab, eos, "ab")
    print("PASS: completed decoder reused with a Python list, preserved inputs/mode/RNG, and strided IDs")

    _expect_error(KeyError, torch.tensor([97, 999]), 1, vocab, eos)
    _expect_error(UnicodeDecodeError, torch.tensor([97, 228, 189, eos]), 1, vocab, eos)
    _expect_error(UnicodeDecodeError, torch.tensor([97, 255]), 1, vocab, eos)
    print("PASS: unknown IDs and incomplete/invalid UTF-8 propagate required decoder errors")

    whole = torch.tensor([228, 189, 160])
    assert real_decoder(whole.tolist(), vocab) == "你"
    _expect_error(UnicodeDecodeError, whole, 1, vocab, None)
    _check(whole, 0, vocab, None, "你")
    print("PASS: valid full text can have an invalid independently decoded continuation at a byte boundary")

    table = torch.zeros((259, 2), dtype=torch.float64)
    table[97], table[256] = torch.tensor([1., 0.]), torch.tensor([0., 1.])
    weights = torch.zeros((2, 259), dtype=torch.float64)
    weights[0, 256], weights[1, eos] = 1000., 1000.
    bias, prompt = torch.zeros(259, dtype=torch.float64), torch.tensor([97])
    for budget, expected_ids, reason in ((1, [97, 256], "length"), (5, [97, 256, eos], "eos")):
        generated_ids, stop_reason = generate_until_eos(
            table, prompt, weights, bias, budget, 1., torch.Generator().manual_seed(19), eos)
        assert generated_ids.tolist() == expected_ids and stop_reason == reason
        _check(generated_ids, prompt.numel(), vocab, eos, "ab")
    print("PASS: EOS and length returns from generation both display only the matching-vocabulary continuation")
    print("Lesson 03.18 checks passed.")


if __name__ == "__main__":
    run_checks()
