# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 03.01: retrieve one embedding row per input token position.

Run from the repository root:
    python3 exercises/03_01_embedding_lookup.py

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

GOAL
Retrieve one embedding row for each input token position, preserving order.
This first demonstration uses Python lists and requires no tensor library.

WHAT AN EMBEDDING REPRESENTS
A vector is an ordered list of numbers. A D-dimensional embedding has D
floating-point components. The table has one row per token ID: IDs select rows;
the integer ID itself does not determine the row's values or semantic meaning.
The demonstration values below are supplied manually. In a trainable model,
the table starts from an initialization and changes through the training loss.
The vocabulary maps IDs to bytes; this table maps IDs to numeric model inputs.

INPUTS
token_ids: list[int], for example [97, 98, 97].
embedding_table: list[list[float]], one floating-point list per vocabulary ID.
    V rows represent the vocabulary. D columns represent embedding dimensions.
    embedding_table[token_id] retrieves the complete row for that token ID.

OUTPUT
A list of vector rows, with T rows and D columns, where T = len(token_ids).
The returned type is list[list[float]]. The full table has shape [V, D]; the
selected sequence has shape [T, D]. D counts components, not a numeric limit.
An empty token sequence produces an empty result list.

WORKED EXAMPLE
    token_ids = [97, 98, 97]
    embedding_table[97] = [0.2, -0.1, 0.4, 0.3]
    embedding_table[98] = [-0.3, 0.5, 0.1, -0.2]
    After position 1: one row, equal to embedding_table[97].
    After position 2: two rows, for IDs 97 then 98.
    After position 3: three rows, for IDs 97, 98, then 97 again.
    Final output shape: [3, 4]. Repeated IDs retrieve repeated rows.

HOW TO WORK
Review the completed learner implementation and its numbered hints inside
lookup_embeddings. The table and IDs are arguments supplied by the checks
below; the learner function only retrieves rows. Run the supplied checks to
review the result. The final check line is: Lesson 03.01 checks passed.
"""


def lookup_embeddings(
    token_ids: list[int], embedding_table: list[list[float]]
) -> list[list[float]]:
    """Return embedding rows in input order, preserving repeated token IDs.

    Assume a rectangular table and valid nonnegative token IDs. Return an
    empty list for an empty input. Keep the input sequence and table unchanged.
    Rows may be reused or copied; this exercise reads their values only.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: Create an empty list to hold output rows. Call it vectors.
    # Hint: [] creates an empty list. Returning this list later handles T = 0.
    #
    # STEP 2: Visit each token ID in token_ids, in its original order.
    # Hint: the loop header can be "for token_id in token_ids:".
    # Use a loop over the sequence; a set would remove repeated IDs.
    #
    # STEP 3: Retrieve the row for the current ID and append that whole row.
    # Hint: embedding_table[token_id] is a list of D floats.
    # list.append(value) adds one item to the end of a list. Here that item
    # must be a whole row, so vectors remains a list of rows.
    # Checkpoint: after the first two IDs [97, 98], vectors has two rows.
    #
    # STEP 4: After the loop finishes, return vectors.
    # Hint: keep the return statement outside the loop, or only the first
    # token will be processed. The empty input must also reach this return.
    #
    # Completed learner implementation; retain these steps for review.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement lookup_embeddings for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


def check_lookup() -> None:
    """Verify representation lookup before introducing trainable tensors."""
    table = [[0.0, 0.0, 0.0, 0.0] for _ in range(256)]
    table[97] = [0.2, -0.1, 0.4, 0.3]
    table[98] = [-0.3, 0.5, 0.1, -0.2]
    original_table = [row.copy() for row in table]
    token_ids = [97, 98, 97]
    original_ids = token_ids.copy()

    vectors = lookup_embeddings(token_ids, table)
    assert vectors == [
        [0.2, -0.1, 0.4, 0.3],
        [-0.3, 0.5, 0.1, -0.2],
        [0.2, -0.1, 0.4, 0.3],
    ], "Retrieve rows in sequence order, including repeated IDs"
    assert len(vectors) == 3
    assert all(len(row) == 4 for row in vectors)
    assert token_ids == original_ids, "Keep input token IDs unchanged"
    assert table == original_table, "Keep the embedding table unchanged"
    print("PASS: sequence order, repeated IDs, and output shape [3, 4]")
    assert lookup_embeddings([], table) == []
    assert lookup_embeddings([98], table) == [[-0.3, 0.5, 0.1, -0.2]]
    print("PASS: empty and one-token inputs")

    # BPE token IDs can exceed the range of a single byte.
    bpe_table = [[0.0, 0.0] for _ in range(258)]
    bpe_table[97] = [0.4, -0.5]
    bpe_table[256] = [1.25, -0.75]
    original_bpe_table = [row.copy() for row in bpe_table]
    assert lookup_embeddings([256, 97, 256], bpe_table) == [
        [1.25, -0.75],
        [0.4, -0.5],
        [1.25, -0.75],
    ], "Use the supplied vocabulary size and embedding dimension"
    assert bpe_table == original_bpe_table
    print("PASS: vocabulary IDs above 255 and a different embedding dimension")

    print(f"aba -> {token_ids} -> embedding output shape [3, 4]")
    print("PASS embedding lookup: order, repetition, dimensions, and preserved inputs")
    print("Lesson 03.01 checks passed.")


if __name__ == "__main__":
    check_lookup()
