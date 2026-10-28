# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.01: count adjacent token pairs before implementing BPE merges.

Run from the repository root:
    python3 exercises/02_01_bpe_pair_counts.py
"""


def count_pairs(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Return the number of occurrences of every ordered adjacent pair.

    Visit every adjacent position, including overlapping occurrences.
    Empty and one-token inputs return an empty dictionary. The input is
    already a valid token sequence; IDs may exceed 255 after BPE merges.
    Leave token_ids unchanged. Do not merge anything in this function.
    """
    # ===== START LEARNER CODE =====
    # Collect adjacent pairs and count their occurrences.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement count_pairs for this exercise")
    # ===== END LEARNER CODE =====



def check_pair_counts() -> None:
    assert count_pairs([]) == {}
    assert count_pairs([97]) == {}
    assert count_pairs([97, 98]) == {(97, 98): 1}
    assert count_pairs([97, 98, 97, 98, 97, 98]) == {
        (97, 98): 3,
        (98, 97): 2,
    }
    assert count_pairs([42, 42, 42]) == {(42, 42): 2}
    assert count_pairs([256, 257, 256]) == {
        (256, 257): 1,
        (257, 256): 1,
    }
    assert count_pairs([240, 159, 140, 141]) == {
        (240, 159): 1,
        (159, 140): 1,
        (140, 141): 1,
    }
    tokens = [10, 20, 10, 20]
    original = tokens.copy()
    assert count_pairs(tokens) == {(10, 20): 2, (20, 10): 1}
    assert tokens == original, "Leave the source sequence unchanged"
    print("PASS pair counts: ordered pairs, overlap, boundaries, and merged IDs")
    print("Lesson 02.01 checks passed.")


if __name__ == "__main__":
    check_pair_counts()
