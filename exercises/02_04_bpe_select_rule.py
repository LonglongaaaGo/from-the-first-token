# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.04: select a learned BPE rule for encoding.

Run from the repository root:
    python3 exercises/02_04_bpe_select_rule.py

GOAL
Choose one existing merge rule that can be applied to the current text.
Use the numbered steps to implement the selection rule before the
complete encoding exercise.

INPUTS
pair_counts: {(left_id, right_id): current_occurrence_count}
    Its keys tell us which adjacent pairs are present now.
merges: {(left_id, right_id): learned_replacement_id}
    Its keys are the rules learned earlier. Our trainer assigns replacement
    IDs in learning order, so smaller replacement IDs have earlier priority.

OUTPUT
Return one pair, such as (98, 99). Return None when no learned rule applies.

WORKED EXAMPLE
    pair_counts = {(97, 98): 3, (98, 99): 1}
    merges      = {(97, 98): 257, (98, 99): 256}
    Step 1: candidates = [(97, 98), (98, 99)]
    Step 2: the candidate list is nonempty, so selection continues.
    Step 3: compare learned IDs 257 and 256; return the pair (98, 99).

The returned object is the pair. The complete encoder will look up its
replacement ID and call merge_pair to perform the actual replacement.
check_selection supplies independent example dictionaries to the function.
"""

from typing import Optional


def select_merge_pair(
    pair_counts: dict[tuple[int, int], int],
    merges: dict[tuple[int, int], int],
) -> Optional[tuple[int, int]]:
    """Return the present learned pair with the earliest learned rule.

    pair_counts describes adjacent pairs in the current token sequence.
    merges maps learned pairs to their allocated token IDs. Our trainer
    assigns these IDs in learning order, so a smaller new ID means an earlier
    rule. Other tokenizer formats may store a separate rank.

    Consider only pairs present in both dictionaries. Select the one with
    the smallest merges[pair], regardless of its current frequency or the
    numeric order of its child IDs. Return None when no learned rule applies.
    Leave both input dictionaries unchanged.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: Keep current pairs for which a learned rule exists.
    # An empty list is our starting collection. Each item we append is a pair.
    # .items() supplies (pair, count); count is not the priority in this step.
    # "pair in merges" checks dictionary keys, not its replacement-ID values.
    # STEP 2: Report that no rule applies when the collection is empty.
    # "not candidates" is True for []. None is the encoder's stopping signal.
    # Returning here also avoids calling min on an empty collection.
    # STEP 3: Choose according to the stored rule priority.
    # min examines the candidates and uses the key function to compare them.
    # In "lambda x: merges[x]", x is a candidate pair; merges[x] is its learned ID.
    # min returns the winning pair itself, rather than the comparison value.
    # STEP 4: Return one pair to the caller. The caller performs the merge.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement select_merge_pair for this exercise")
    # ===== END LEARNER CODE =====



def check_selection() -> None:
    """Check the selector with independent example count/rule dictionaries."""
    cases = (
        (
            "learning order overrides current frequency",
            {(97, 98): 1, (98, 99): 10},
            {(97, 98): 256, (98, 99): 257},
            (97, 98),
        ),
        (
            "unknown pairs cannot be merged",
            {(98, 97): 20, (97, 98): 1},
            {(97, 98): 256},
            (97, 98),
        ),
        (
            "rule order overrides numeric child-pair order",
            {(97, 98): 3, (98, 99): 1},
            {(97, 98): 257, (98, 99): 256},
            (98, 99),
        ),
        (
            "a later rule may apply when an earlier rule is absent",
            {(256, 256): 1, (98, 97): 5},
            {(97, 98): 256, (256, 256): 257},
            (256, 256),
        ),
        ("empty sequence", {}, {(97, 98): 256}, None),
        ("no learned pair present", {(97, 99): 2}, {(97, 98): 256}, None),
        ("no learned rules", {(97, 98): 3}, {}, None),
    )
    for label, pair_counts, merges, expected in cases:
        original_counts = pair_counts.copy()
        original_merges = merges.copy()
        actual = select_merge_pair(pair_counts, merges)
        assert actual == expected, (
            f"{label}: expected {expected!r}, received {actual!r}"
        )
        assert pair_counts == original_counts, "Keep the current counts unchanged"
        assert merges == original_merges, "Keep learned rules unchanged"
        print(f"PASS: {label}")

    print("PASS rule selection: learned order, applicability, and frozen inputs")
    print("Lesson 02.04 checks passed.")


if __name__ == "__main__":
    check_selection()
