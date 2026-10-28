# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 01.01: byte tokenization and shifted next-token training windows.

Run from the repository root:
    python3 exercises/01_01_text_to_tokens.py --check encode
    python3 exercises/01_01_text_to_tokens.py --check decode
    python3 exercises/01_01_text_to_tokens.py --check example
    python3 exercises/01_01_text_to_tokens.py
"""

import argparse


class ByteTokenizer:
    """A fixed vocabulary with one token ID per byte; no learned merges yet."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        """Return the UTF-8 bytes as integer token IDs in [0, 255]."""
        # ===== START LEARNER CODE =====
        # Convert the string to UTF-8 bytes, then return a list of IDs.
        # TODO: implement the contract using the worked example and numbered hints.
        raise NotImplementedError("Implement ByteTokenizer.encode for this exercise")
        # ===== END LEARNER CODE =====

    def decode(self, token_ids: list[int]) -> str:
        """Recover text using strict UTF-8 decoding.

        Invalid byte values must raise ValueError. An invalid UTF-8 sequence
        must raise UnicodeDecodeError rather than silently dropping bytes.
        """
        # ===== START LEARNER CODE =====
        # Convert the IDs to bytes, then decode the complete sequence.
        # TODO: implement the contract using the worked example and numbered hints.
        raise NotImplementedError("Implement ByteTokenizer.decode for this exercise")
        # ===== END LEARNER CODE =====


def make_example(
    token_ids: list[int], start: int, context_length: int
) -> tuple[list[int], list[int]]:
    """Return input and next-token targets, each of length context_length.

    The input starts at start; targets start one position later. Raise
    ValueError when start is negative, context_length is not positive, or
    there are fewer than context_length + 1 tokens available from start.
    Return new lists without changing token_ids. Inputs are already integers.
    """
    # ===== START LEARNER CODE =====
    # Validate the window, then construct the two shifted slices.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement make_example for this exercise")
    # ===== END LEARNER CODE =====


def expect_error(error_type: type[Exception], function, *args) -> None:
    """Check a required error without accepting unrelated exceptions."""
    try:
        function(*args)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} from {function.__name__}")


def check_encode() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("") == []
    assert tokenizer.encode("hello") == [104, 101, 108, 108, 111]
    assert tokenizer.encode("A🌍B") == [65, 240, 159, 140, 141, 66]
    assert tokenizer.vocab_size == 256
    print("PASS encode: ASCII, Unicode, and empty input")


def check_decode() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.decode([]) == ""
    assert tokenizer.decode([104, 101, 108, 108, 111]) == "hello"
    assert tokenizer.decode([65, 240, 159, 140, 141, 66]) == "A🌍B"
    expect_error(UnicodeDecodeError, tokenizer.decode, [255])
    expect_error(UnicodeDecodeError, tokenizer.decode, [240, 159])
    expect_error(ValueError, tokenizer.decode, [256])
    expect_error(ValueError, tokenizer.decode, [-1])
    print("PASS decode: valid text and strict error handling")


def check_example() -> None:
    tokens = [10, 20, 30, 40, 50, 60]
    original = tokens.copy()
    assert make_example(tokens, start=1, context_length=3) == (
        [20, 30, 40], [30, 40, 50]
    )
    assert make_example(tokens, start=2, context_length=3) == (
        [30, 40, 50], [40, 50, 60]
    )
    assert make_example([7, 8], start=0, context_length=1) == ([7], [8])
    expect_error(ValueError, make_example, tokens, -1, 3)
    expect_error(ValueError, make_example, tokens, 0, 0)
    expect_error(ValueError, make_example, tokens, 0, -1)
    expect_error(ValueError, make_example, tokens, 3, 3)
    expect_error(ValueError, make_example, [], 0, 1)
    assert tokens == original, "The input token list must stay unchanged"
    x, y = make_example(tokens, start=0, context_length=2)
    x[0] = 99
    y[0] = 99
    assert tokens == original, "Return new lists rather than modifying the source"
    print("PASS example: shifted targets and window boundaries")


def check_pipeline() -> None:
    tokenizer = ByteTokenizer()
    for text in ("", "hello", "A🌍B", "café\nOneLong Model"):
        assert tokenizer.decode(tokenizer.encode(text)) == text
    x, y = make_example(tokenizer.encode("hello"), start=0, context_length=4)
    assert x == [104, 101, 108, 108]
    assert y == [101, 108, 108, 111]
    print(f"hello inputs:  {x}")
    print(f"hello targets: {y}")
    print("Lesson 01.01 checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", choices=("encode", "decode", "example", "all"), default="all"
    )
    choice = parser.parse_args().check
    checks = {"encode": check_encode, "decode": check_decode, "example": check_example}
    if choice == "all":
        for check in checks.values():
            check()
        check_pipeline()
    else:
        checks[choice]()


if __name__ == "__main__":
    main()
