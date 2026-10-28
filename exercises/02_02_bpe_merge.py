# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.02: replace an adjacent token pair without overlapping matches.

Run from the repository root:
    python3 exercises/02_02_bpe_merge.py
"""


def merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Return a new sequence after a single left-to-right merge pass.

    Replace each non-overlapping occurrence of pair with new_id. Once two
    original tokens are consumed, neither can participate in another match.
    Preserve unmatched tokens and leave the input unchanged.

    Inputs are valid token IDs and pair contains two IDs. The caller supplies
    the replacement new_id: newly allocated during training or already learned
    during encoding. This function does not choose the pair or modify a vocab.
    """
    # ===== START LEARNER CODE =====
    # Scan the source, consume two tokens on a match and one otherwise.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement merge_pair for this exercise")
    # ===== END LEARNER CODE =====

def check_merges() -> None:
    assert merge_pair([], (97, 98), 256) == []
    assert merge_pair([97], (97, 98), 256) == [97]
    assert merge_pair([97, 98], (97, 98), 256) == [256]
    assert merge_pair([98, 97], (97, 98), 256) == [98, 97]
    assert merge_pair([97, 98, 97, 98, 97, 98], (97, 98), 256) == [256, 256, 256]
    assert merge_pair([97, 98, 97], (97, 98), 256) == [256, 97]
    assert merge_pair([97, 97, 97], (97, 97), 256) == [256, 97]
    assert merge_pair([97, 97, 97, 97], (97, 97), 256) == [256, 256]
    assert merge_pair([1, 2, 3, 4], (3, 4), 256) == [1, 2, 256]
    assert merge_pair([256, 257, 256, 257], (256, 257), 258) == [258, 258]

    source = [97, 98, 97, 98]
    original = source.copy()
    merged = merge_pair(source, (97, 98), 256)
    assert source == original, "Leave the source sequence unchanged"
    merged[0] = 999
    assert source == original, "Return an independent result list"
    source = [97]
    merged = merge_pair(source, (97, 98), 256)
    merged[0] = 999
    assert source == [97], "Return a new list even when no pair matches"
    print("PASS merges: non-overlapping matches, order, boundaries, and merged IDs")
    print("Lesson 02.02 checks passed.")


if __name__ == "__main__":
    check_merges()
