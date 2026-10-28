# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.03: train a basic BPE vocabulary on one text string.

Run from the repository root:
    python3 exercises/02_03_bpe_train.py
"""

from pathlib import Path
import runpy


# Scaffold plumbing: reuse your helpers without running their script checks.
_exercise_dir = Path(__file__).resolve().parent
count_pairs = runpy.run_path(str(_exercise_dir / "02_01_bpe_pair_counts.py"))["count_pairs"]
merge_pair = runpy.run_path(str(_exercise_dir / "02_02_bpe_merge.py"))["merge_pair"]


def train_bpe(
    text: str, vocab_size: int
) -> tuple[list[int], dict[tuple[int, int], int], dict[int, bytes]]:
    """Return the final training IDs, learned merge rules, and byte vocabulary.

    Keep all 256 base byte entries. Repeatedly select the most frequent pair
    in the current IDs; on equal counts, choose the numerically smallest pair
    tuple. Allocate the next contiguous ID, record its concatenated bytes and
    the pair-to-ID rule, then apply merge_pair and recompute counts.

    Stop when len(vocab) reaches vocab_size or there are no adjacent pairs.
    Frequency-one pairs may be merged. This exercise has no special tokens
    or text segmentation; vocab_size is an upper bound on the number of
    entries, not a maximum ID and not a required output sequence length.
    """
    # ===== START LEARNER CODE =====
    # Count, select, record, and merge until the vocabulary budget is reached.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement train_bpe for this exercise")
    # ===== END LEARNER CODE =====


def check_state(text, requested_size, ids, merges, vocab) -> None:
    """Verify preserved bytes, contiguous IDs, and the vocabulary budget."""
    assert 256 <= len(vocab) <= requested_size
    assert len(vocab) == 256 + len(merges)
    assert set(vocab) == set(range(len(vocab)))
    assert set(merges.values()) == set(range(256, len(vocab)))
    for idx in range(256):
        assert vocab[idx] == bytes([idx]), "Keep every original byte entry"
    for pair, new_id in merges.items():
        assert vocab[new_id] == vocab[pair[0]] + vocab[pair[1]]
    assert b"".join(vocab[idx] for idx in ids) == text.encode("utf-8")


def check_training() -> None:
    ids, merges, vocab = train_bpe("ababab", 258)
    assert ids == [257, 256]
    assert merges == {(97, 98): 256, (256, 256): 257}
    assert len(vocab) == 258
    assert vocab[256] == b"ab"
    assert vocab[257] == b"abab"
    check_state("ababab", 258, ids, merges, vocab)
    print(f"ababab -> {ids}; vocabulary entries: {len(vocab)}")

    # A zero-merge budget keeps the original byte sequence.
    text = "A🌍B"
    ids, merges, vocab = train_bpe(text, 256)
    assert ids == [65, 240, 159, 140, 141, 66]
    assert merges == {}
    check_state(text, 256, ids, merges, vocab)

    # Counts overlap, but replacements must preserve the original text.
    ids, merges, vocab = train_bpe("aaa", 257)
    assert ids == [256, 97]
    assert merges == {(97, 97): 256}
    assert vocab[256] == b"aa"
    check_state("aaa", 257, ids, merges, vocab)

    # Both initial pairs occur once; (97, 99) is smaller than (98, 97).
    ids, merges, vocab = train_bpe("bac", 257)
    assert ids == [98, 256]
    assert merges == {(97, 99): 256}
    assert vocab[256] == b"ac"
    check_state("bac", 257, ids, merges, vocab)

    # The actual vocabulary may be smaller than the requested upper bound.
    for text, expected_ids, expected_merges, expected_size in (
        ("", [], {}, 256),
        ("A", [65], {}, 256),
        ("ab", [256], {(97, 98): 256}, 257),
    ):
        ids, merges, vocab = train_bpe(text, 300)
        assert ids == expected_ids
        assert merges == expected_merges
        assert len(vocab) == expected_size
        check_state(text, 300, ids, merges, vocab)

    # Tokens can contain partial UTF-8; their joined bytes must stay intact.
    ids, merges, vocab = train_bpe("A🌍B", 258)
    assert len(vocab) == 258
    check_state("A🌍B", 258, ids, merges, vocab)

    try:
        train_bpe("ab", 255)
    except ValueError:
        pass
    else:
        raise AssertionError("Reject a limit below the base vocabulary size")

    print("PASS BPE training: recomputed counts, vocabulary, ties, and stopping")
    print("Lesson 02.03 checks passed.")


if __name__ == "__main__":
    check_training()
