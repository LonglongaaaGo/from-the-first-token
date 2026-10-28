# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.05: encode a new text with fixed learned BPE rules.

Run after filling the learner-code region:
    python3 exercises/02_05_bpe_encode.py
Expected final line:
    Lesson 02.05 checks passed.

GOAL
Connect your existing helpers into a complete text-to-token-ID encoder.
The vocabulary was learned earlier. This function applies those fixed rules
to another input and returns the encoded sequence.

INPUTS
text: the new Python string to encode, for example "ababac".
merges: a dictionary mapping a learned pair to its replacement token ID.
    Example: {(97, 98): 256, (256, 256): 257}.
    This dictionary is supplied by the caller. Keep it unchanged.

OUTPUT
A list of integer token IDs. For the example above, return [257, 97, 99].
There is no vocabulary-size budget here: encoding uses already learned IDs.

PROVIDED HELPERS
count_pairs(ids) -> dictionary of current adjacent pairs and their counts.
select_merge_pair(pair_counts, merges) -> one applicable pair, or None.
merge_pair(ids, pair, replacement_id) -> a NEW list after replacement.
The imports below load your completed functions without running their checks.
train_bpe is loaded only to prepare tokenizer rules in the grading code.

COMPLETE WORKED EXAMPLE
    text = "ababac"
    merges = {(97, 98): 256, (256, 256): 257}

    Starting IDs: [97, 98, 97, 98, 97, 99]
    Round 1:
        pair_counts = {(97, 98): 2, (98, 97): 2, (97, 99): 1}
        selected pair = (97, 98)
        replacement ID = merges[(97, 98)] = 256
        new IDs = [256, 256, 97, 99]
    Round 2:
        pair_counts = {(256, 256): 1, (256, 97): 1, (97, 99): 1}
        selected pair = (256, 256)
        replacement ID = 257
        new IDs = [257, 97, 99]
    Round 3:
        pair_counts = {(257, 97): 1, (97, 99): 1}
        selected pair = None; neither current pair has a learned rule.
        Stop and return [257, 97, 99].

HOW TO WORK
1. Read the example and identify what changes after a merge: ids.
2. Read the numbered hints inside encode_bpe.
3. Edit only its marked learner-code region; the checks supply the inputs.
4. Run the command above after replacing the placeholder.
5. Check the named PASS messages. Acceptance cases use independent rule sets.
"""

from pathlib import Path
import runpy


# PROVIDED SETUP: reuse the exercises you already completed.
_exercise_dir = Path(__file__).resolve().parent
count_pairs = runpy.run_path(str(_exercise_dir / "02_01_bpe_pair_counts.py"))["count_pairs"]
merge_pair = runpy.run_path(str(_exercise_dir / "02_02_bpe_merge.py"))["merge_pair"]
select_merge_pair = runpy.run_path(str(_exercise_dir / "02_04_bpe_select_rule.py"))["select_merge_pair"]
train_bpe = runpy.run_path(str(_exercise_dir / "02_03_bpe_train.py"))["train_bpe"]


def encode_bpe(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Encode text using existing rules; leave merges unchanged.

    Assumes valid learned rules from the tokenizer trainer: each rule replaces
    a two-token pair with its allocated ID. Original byte IDs are 0 through 255.
    New text can include any UTF-8 bytes covered by that base vocabulary.
    """
    # ===== START LEARNER CODE =====
    # PROVIDED START: use the byte encoding operation from Lesson 01.
    # STEP 1: Start a loop that repeats the count/select/replace process.
    # Hint: "while True:" repeats until a break statement is reached.
    # Put steps 2 through 5 inside this loop; step 6 belongs after the loop.
    # STEP 2: Count pairs in the CURRENT ids and store the result in pairs.
    # Hint: your existing count_pairs function takes the current list of IDs.
    # Recompute these counts every round because replacement changes the list.
    # STEP 3: Ask select_merge_pair to choose an applicable pair.
    # Hint: pass pairs and the unchanged merges dictionary. Store its
    # result in pair. A tuple identifies a rule; None means no rule applies.
    # STEP 4: If pair is None, leave the loop with break.
    # Hint: "if pair is None:" checks this stopping signal.
    # This must happen before looking up merges[pair]. There can still be
    # several IDs left when no rule applies, as shown by the worked example.
    # "break" leaves the loop; "return" leaves the entire function.
    # STEP 5: Look up the replacement ID and apply merge_pair.
    # Hint: merges[pair] supplies the already learned replacement ID.
    # merge_pair takes (ids, pair, replacement_id) and returns a new list.
    # Replace ids with that returned list so the next round sees the new state.
    # Checkpoint: round 1 of the example ends with [256, 256, 97, 99].
    # STEP 6: Return the final ids after the loop has stopped.
    # Hint: keep this return outside the loop. Returning inside the loop would
    # finish the function before later learned rules can be applied.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement encode_bpe for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE =====


# PROVIDED CHECKS: these prepare inputs and verify your function's output.
def check_encoding() -> None:
    """Check fixed-rule encoding, byte preservation, and stopping behavior."""
    _, merges, vocab = train_bpe("ababab", 258)
    assert merges == {(97, 98): 256, (256, 256): 257}
    original_merges = merges.copy()

    cases = (
        ("empty text", "", []),
        ("one byte", "A", [65]),
        ("one learned merge", "ab", [256]),
        ("a rule using already merged IDs", "abab", [257]),
        ("an unmatched final token", "ababa", [257, 97]),
        ("unlearned pairs remain", "abac", [256, 97, 99]),
        ("two rounds followed by no applicable rule", "ababac", [257, 97, 99]),
        ("the original training text", "ababab", [257, 256]),
        ("only unknown pairs", "zz", [122, 122]),
        ("Unicode bytes outside the learned pairs", "A🌍B", [65, 240, 159, 140, 141, 66]),
    )
    for label, text, expected in cases:
        actual = encode_bpe(text, merges)
        assert actual == expected, (
            f"{label}: expected {expected!r}, received {actual!r}. "
            "Use current IDs, fixed rule priority, and the no-rule stopping signal."
        )
        assert b"".join(vocab[idx] for idx in actual) == text.encode("utf-8")
        assert merges == original_merges, "Keep learned rules unchanged"
        print(f"PASS: {label}")

    assert encode_bpe("aba", {}) == [97, 98, 97], "An empty rule set keeps byte IDs"
    assert encode_bpe("aaa", {(97, 97): 256}) == [256, 97], "Use non-overlapping replacement"

    # Separate example tokenizers verify rule priority, rather than new counts.
    priority_rules = {(97, 98): 256, (98, 99): 257}
    original_priority_rules = priority_rules.copy()
    assert encode_bpe("abcbc", priority_rules) == [256, 99, 257], (
        "Use learned priority even when a later pair occurs more often"
    )
    assert priority_rules == original_priority_rules
    assert encode_bpe("abc", {(97, 98): 257, (98, 99): 256}) == [97, 256], (
        "Compare learned IDs rather than numeric pair order"
    )

    # A learned token may contain part of a Unicode character. Joined bytes
    # must still preserve the complete text, using the supplied vocabulary.
    unicode_text = "A🌍B"
    training_ids, unicode_rules, unicode_vocab = train_bpe(unicode_text, 258)
    original_unicode_rules = unicode_rules.copy()
    encoded_ids = encode_bpe(unicode_text, unicode_rules)
    assert encoded_ids == training_ids
    assert b"".join(unicode_vocab[idx] for idx in encoded_ids) == unicode_text.encode("utf-8")
    assert unicode_rules == original_unicode_rules
    print("PASS: empty rules, overlap, learned priority, and Unicode byte preservation")
    print("Lesson 02.05 checks passed.")


if __name__ == "__main__":
    check_encoding()
