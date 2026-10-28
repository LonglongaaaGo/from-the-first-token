# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.06: recover text from BPE token IDs using the byte vocabulary.

GOAL
    Implement decode_bpe(token_ids, vocab). This completes the reverse path:
    text -> encode_bpe -> token IDs -> decode_bpe -> the original text.
    You already built the vocabulary in train_bpe and the encoder in Lesson 02.05.
    This exercise uses the Python standard library; no new package is needed.

INPUTS
    token_ids: list[int]
        The sequence to decode, in its original order. IDs can exceed 255.
        Example: [257, 97, 99]. These are vocabulary keys, not raw byte values.
    vocab: dict[int, bytes]
        The SAME learned byte vocabulary used by the encoder's tokenizer.
        Each key is an ID; each value is that token's original byte contents.
        Examples: vocab[97] == b"a", vocab[256] == b"ab".
        The caller supplies this dictionary. Do not train or change it here.

OUTPUT
    str: the text reconstructed from the complete byte sequence.
    Empty input returns "". Keep token_ids and vocab unchanged.
    An unknown ID raises KeyError through dictionary lookup.
    Invalid or incomplete UTF-8 raises UnicodeDecodeError through strict decoding.
    These errors are required behavior; no manual error handling is needed.

WORKED EXAMPLE
    train_bpe("ababab", 258) learned these relevant vocabulary entries:
        vocab[257] = b"abab"
        vocab[97]  = b"a"
        vocab[99]  = b"c"

    Input:             token_ids = [257, 97, 99]
    After first lookup: chunks = [b"abab"]
    After next lookup:  chunks = [b"abab", b"a"]
    After final lookup: chunks = [b"abab", b"a", b"c"]
    Join the chunks:    raw_bytes = b"ababac"
    Decode UTF-8 once:  returned text = "ababac"

WHY JOIN BEFORE DECODING?
    UTF-8 stores a Chinese character or emoji in several bytes. A BPE token
    may end in the middle of those bytes. Its own contents need not be valid
    UTF-8, even when the complete sequence is valid.
    Example: the three bytes of "你" are [228, 189, 160].
        first chunk:  bytes([228, 189])  -- incomplete by itself
        second chunk: bytes([160])       -- invalid by itself
        joined bytes: bytes([228, 189, 160]) -- decodes to "你"
    Look up bytes inside the loop; decode the joined bytes after the loop.
    Also, bytes([257, 97, 99]) cannot work: one raw byte must be in 0..255.
    The vocabulary lookup is what recovers the byte contents of ID 257.

HOW TO WORK
    1. Read the worked example and the numbered hints beside the learner region.
    2. Edit only decode_bpe's marked learner-code region.
    3. Run from the repository root:
           python3 exercises/02_06_bpe_decode.py
       Each PASS line describes a checked behavior. The final line should be:
           Lesson 02.06 checks passed.
    The supplied setup and checks below prepare inputs; you do not implement them.
"""

from pathlib import Path
import runpy


# PROVIDED SETUP: reuse completed exercises without running their script checks.
_exercise_dir = Path(__file__).resolve().parent
train_bpe = runpy.run_path(str(_exercise_dir / "02_03_bpe_train.py"))["train_bpe"]
encode_bpe = runpy.run_path(str(_exercise_dir / "02_05_bpe_encode.py"))["encode_bpe"]


def decode_bpe(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """Look up each token's bytes, join in order, and strictly decode UTF-8.

    Use the vocabulary associated with the rules that encoded these IDs.
    Each value is bytes, not str. Do not use merges or create new entries.
    Dictionary lookup supplies KeyError for an unknown ID; strict decoding
    supplies UnicodeDecodeError for invalid or incomplete UTF-8.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: Collect each token's byte contents in a list named chunks.
    # Action: start with an empty list, then visit token_ids in order.
    # Syntax hint: [] creates an empty list; "for token_id in token_ids:"
    # visits one ID per iteration. Indent the lookup/append inside that loop.
    # Lookup hint: vocab[token_id] returns bytes, even when token_id is 257.
    # Append hint: list.append(value) adds ONE value to a list.
    # Append the whole bytes object; do not append the numeric ID or decode it.
    # Checkpoint for [257, 97, 99]: chunks == [b"abab", b"a", b"c"].
    #
    # STEP 2: Join chunks into one bytes object named raw_bytes AFTER the loop.
    # Syntax hint: separator.join(parts) joins a list of bytes objects.
    # Example: b"|".join([b"a", b"b"]) returns b"a|b".
    # Our separator is b"", an empty bytes object: insert no extra bytes.
    # Checkpoint: the worked example has raw_bytes == b"ababac".
    # Empty chunks join to b"" automatically; no special branch is needed.
    #
    # STEP 3: Convert the complete raw_bytes into text and return it.
    # Syntax hint: a bytes object's .decode("utf-8", errors="strict") returns str.
    # Example: b"hi".decode("utf-8", errors="strict") returns "hi".
    # Place this return AFTER the loop, so all tokens contribute to the result.
    # Do not catch or replace decoding errors: they identify broken byte streams.
    # Checkpoint: the worked example returns "ababac"; empty input returns "".
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement decode_bpe for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# PROVIDED CHECKS: these prepare inputs and verify your function's output.
def expect_error(error_type: type[Exception], function, *args) -> None:
    """Require the specified error; unrelated errors still fail the check."""
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} from {function.__name__}")


def check_decoding() -> None:
    _, merges, vocab = train_bpe("ababab", 258)
    assert vocab[256] == b"ab" and vocab[257] == b"abab"
    original_vocab = vocab.copy()

    cases = [
        ("empty input", [], ""),
        ("one base byte ID", [65], "A"),
        ("base byte sequence", [97, 98, 99], "abc"),
        ("one merged ID above 255", [257], "abab"),
        ("merged and base IDs together", [257, 97, 99], "ababac"),
        ("ordered and repeated tokens", [256, 99, 256], "abcab"),
        ("Unicode across base tokens", [228, 189, 160], "你"),
        ("emoji across base tokens", [65, 240, 159, 140, 141, 66], "A🌍B"),
    ]
    for label, token_ids, expected in cases:
        original_ids = token_ids.copy()
        actual = decode_bpe(token_ids, vocab)
        assert actual == expected, (
            f"{label}: expected {expected!r}, got {actual!r}. "
            "Look up every token's bytes, join in order, and decode once."
        )
        assert token_ids == original_ids, "Keep the input IDs unchanged"
        assert vocab == original_vocab, "Keep the vocabulary unchanged"
        print(f"PASS: {label}")

    # A real training vocabulary can contain an incomplete character's bytes.
    # With one allowed merge, our trainer joins [189, 160] into token 256.
    _, unicode_merges, unicode_vocab = train_bpe("你", 257)
    assert unicode_vocab[256] == bytes([189, 160])
    unicode_ids = encode_bpe("你", unicode_merges)
    assert unicode_ids == [228, 256]
    original_unicode_vocab = unicode_vocab.copy()
    assert decode_bpe(unicode_ids, unicode_vocab) == "你"
    assert unicode_ids == [228, 256]
    assert unicode_vocab == original_unicode_vocab
    print("PASS: join partial UTF-8 chunks before decoding")

    # Frozen rules also encode texts that were not used to train the vocabulary.
    for text in ("", "ab", "ababac", "zz", "你", "A🌍B", "café\n"):
        encoded_ids = encode_bpe(text, merges)
        original_ids = encoded_ids.copy()
        assert decode_bpe(encoded_ids, vocab) == text, f"Round trip failed: {text!r}"
        assert encoded_ids == original_ids
        assert vocab == original_vocab
    print("PASS: encode/decode round trips with the same tokenizer")

    expect_error(KeyError, decode_bpe, [999], vocab)
    print("PASS: unknown token ID raises KeyError")

    expect_error(UnicodeDecodeError, decode_bpe, [228, 189], vocab)
    expect_error(UnicodeDecodeError, decode_bpe, [255], vocab)
    expect_error(UnicodeDecodeError, decode_bpe, [256, 228, 189], vocab)
    assert vocab == original_vocab
    print("PASS: incomplete and invalid UTF-8 raise UnicodeDecodeError")
    print("Lesson 02.06 checks passed.")


if __name__ == "__main__":
    check_decoding()
