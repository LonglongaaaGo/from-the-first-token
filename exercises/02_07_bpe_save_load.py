# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Lesson 02.07: save and reload the exact tokenizer vocabulary and rules.

STATUS: PUBLIC EXERCISE SCAFFOLD; IMPLEMENTATION TODO.

WHY SAVE?
    Variables live in a running program. A saved file lets a later program
    recover the SAME token meanings and rule priorities without training again.
    Saving a tokenizer does not save neural network parameters.

JSON AND OUR FORMAT
    JSON is a text format for data: objects, arrays, strings, numbers, booleans,
    and null. Python dictionaries/lists correspond to JSON objects/arrays.
    Our tuple dictionary keys and bytes values cannot be dumped directly.
    Store each merge as [left_id, right_id, new_id].
    Store each vocabulary entry as [token_id, list_of_byte_values].
    All IDs keep their original values. Save ALL vocabulary entries, including
    the 256 base entries. Individual token bytes need not be valid UTF-8 text.

WORKED EXAMPLE
    train_bpe("ababab", 258) learned:
        merges[(97, 98)]   = 256
        merges[(256, 256)] = 257
        vocab[256] = b"ab"; vocab[257] = b"abab"

    Stage A merge records:
        [[97, 98, 256], [256, 256, 257]]
    Relevant vocabulary records (excerpt, not the complete saved list):
        [97, [97]], [98, [98]], [256, [97, 98]], [257, [97, 98, 97, 98]]
    b"ab" becomes [97, 98] via list(...); the original bytes remain unchanged.
    Stage B reverses the records:
        [97, 98, 256] -> dictionary key (97, 98), value 256
        [256, [97, 98]] -> dictionary key 256, value b"ab"
    After loading, encoding "ababac" still returns [257, 97, 99], and decoding
    those IDs still returns "ababac".

RUN FROM THE REPOSITORY ROOT
    python3 exercises/02_07_bpe_save_load.py --check all
    Expected final line: Lesson 02.07 checks passed.
    Individual stage checks remain available with --check pack or --check unpack.
    The completed learner regions are available for review; setup/checks are supplied.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory


# PROVIDED SETUP: import completed exercises without running their checks.
_exercise_dir = Path(__file__).resolve().parent
train_bpe = runpy.run_path(str(_exercise_dir / "02_03_bpe_train.py"))["train_bpe"]
encode_bpe = runpy.run_path(str(_exercise_dir / "02_05_bpe_encode.py"))["encode_bpe"]
decode_bpe = runpy.run_path(str(_exercise_dir / "02_06_bpe_decode.py"))["decode_bpe"]
FORMAT_NAME = "onelong-byte-bpe"
FORMAT_VERSION = 1


def pack_tokenizer(merges: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> dict:
    """Return a JSON-compatible payload without changing either input.

    merges: learned (left_id, right_id) -> new_id rules from train_bpe.
    vocab: learned token_id -> original bytes entries from that same trainer.
    payload: format/version metadata plus merge and vocabulary record lists.
    Preserve IDs, byte contents, every entry, and dictionary iteration order.
    Assume a valid matching rule set and vocabulary from our trainer.
    """
    # ===== START LEARNER CODE =====
    # PROVIDED START: this is the complete outer shape of our saved format.
    # STEP 1: Collect one record for every learned rule in out_merges.
    # "for item in merges.items():" visits a key/value pair as one tuple.
    # For the first rule, item == ((97, 98), 256).
    # item[0] is the child pair; item[1] is the existing replacement ID.
    # item[0][0] and item[0][1] retrieve the left and right child IDs.
    # Build a THREE-INTEGER LIST and append it to out_merges.
    # Checkpoint for the first rule: out_merges == [[97, 98, 256]].
    # Add the second rule as another record; do not allocate new IDs.
    # STEP 2: Collect one record for EVERY vocabulary entry in out_vocab.
    # Loop hint: "for token_id, token_bytes in vocab.items():" gives the ID
    # and its bytes. list(token_bytes) converts bytes to their integer values.
    # Example: list(b"ab") is [97, 98]. This operation does not decode text.
    # Build [token_id, list(token_bytes)] and append it to out_vocab.
    # The byte-value list is nested inside the two-item record.
    # Checkpoint for ID 256: its record is [256, [97, 98]].
    # Do not call .decode() on token_bytes; it may contain partial UTF-8.
    # STEP 3: Put the collected lists in payload and return it AFTER both loops.
    # Keep this return outside the loops so every entry is included.
    # With no learned merges, payload["merges"] remains []; the base vocab
    # must still be collected. No special branch is needed.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement pack_tokenizer for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE A =====


def unpack_tokenizer(payload: dict) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Return (merges, vocab) from a valid payload, leaving payload unchanged.

    Stage A passed its review. payload uses the format shown above.
    The file reader checks the format/version before calling this function.
    Create new dictionaries, retaining the saved IDs and original byte values.
    """
    # ===== START LEARNER CODE =====
    # STEP 1: Create two empty dictionaries named merges and vocab.
    # Syntax hint: {} creates a dictionary.
    # STEP 2: Visit the records in payload["merges"]. Each record contains
    # exactly three integers. The learner names them a, b, and c:
    # a is the left ID, b is the right ID, and c is the saved replacement ID.
    # payload["merges"] is a LIST of records; loop over it directly.
    # .items() is for dictionaries, not for this record list.
    # Unpacking hint: "a, b, c = [1, 2, 3]" assigns one item to each name.
    # Rebuild the TUPLE key (a, b) and associate it with c.
    # The parentheses create a tuple; a list cannot be used as a dictionary key.
    # Dictionary assignment reference: some_dict[key] = value.
    # Checkpoint: record [97, 98, 256] restores {(97, 98): 256}.
    # STEP 3: Visit payload["vocab"]. Unpack each record into token_id and
    # byte_list. bytes(byte_list) restores the bytes object for that ID.
    # Example: bytes([97, 98]) is b"ab"; store that under the original ID.
    # Checkpoint: record [256, [97, 98]] restores vocab[256] == b"ab".
    # STEP 4: After both loops, return merges and vocab in that order.
    # A comma-separated return supplies a tuple: the caller can unpack it.
    # TODO: implement the contract using the worked example and numbered hints.
    raise NotImplementedError("Implement unpack_tokenizer for this exercise")
    # ===== END LEARNER CODE =====
    # ===== END LEARNER CODE: STAGE B =====


# PROVIDED FILE HELPERS: your current task is data conversion, not file I/O.
def save_tokenizer(path: Path, merges: dict, vocab: dict) -> None:
    payload = pack_tokenizer(merges, vocab)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_tokenizer(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["format"] != FORMAT_NAME or payload["version"] != FORMAT_VERSION:
        raise ValueError("Unsupported tokenizer format or version")
    return unpack_tokenizer(payload)


# PROVIDED CHECKS: temporary files are removed automatically after each check.
def check_packing() -> None:
    for label, text, budget in (
        ("learned rules and merged byte contents", "ababab", 258),
        ("zero merges with the complete base vocabulary", "", 256),
        ("partial UTF-8 token bytes", "你", 257),
    ):
        _, merges, vocab = train_bpe(text, budget)
        original_merges, original_vocab = merges.copy(), vocab.copy()
        payload = pack_tokenizer(merges, vocab)
        assert payload["format"] == FORMAT_NAME and payload["version"] == 1
        assert len(payload["merges"]) == len(merges), "Save every learned rule"
        assert len(payload["vocab"]) == len(vocab), "Save every vocabulary entry"
        assert [tuple(row[:2]) for row in payload["merges"]] == list(merges)
        assert [row[0] for row in payload["vocab"]] == list(vocab)
        for row in payload["merges"]:
            assert len(row) == 3
            assert row[2] == merges[tuple(row[:2])], "Preserve the learned ID"
        for token_id, byte_values in payload["vocab"]:
            assert bytes(byte_values) == vocab[token_id], "Preserve exact bytes"
        if text == "ababab":
            assert payload["merges"] == [[97, 98, 256], [256, 256, 257]]
            assert payload["vocab"][256:] == [
                [256, [97, 98]], [257, [97, 98, 97, 98]]
            ]
        if text == "你":
            assert payload["vocab"][256] == [256, [189, 160]]
        assert json.loads(json.dumps(payload)) == payload, (
            "Use JSON-compatible record lists, not tuple keys or bytes objects"
        )
        assert merges == original_merges and vocab == original_vocab
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            save_tokenizer(path, merges, vocab)
            assert json.loads(path.read_text(encoding="utf-8")) == payload
        print(f"PASS: {label}")
    print("PASS: JSON file writing and unchanged inputs")
    print("Lesson 02.07 stage A checks passed.")


def example_payload() -> dict:
    """A supplied saved-data fixture; stage B does not depend on stage A code."""
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "merges": [[97, 98, 256], [256, 256, 257]],
        "vocab": [[idx, [idx]] for idx in range(256)] + [
            [256, [97, 98]], [257, [97, 98, 97, 98]]
        ],
    }


def check_unpacking() -> None:
    payload = example_payload()
    # Explicit saved IDs preserve priority even when records are listed later first.
    payload["merges"].reverse()
    original_payload = deepcopy(payload)
    merges, vocab = unpack_tokenizer(payload)
    _, expected_merges, expected_vocab = train_bpe("ababab", 258)
    assert merges == expected_merges and vocab == expected_vocab
    assert payload == original_payload
    assert encode_bpe("ababac", merges) == [257, 97, 99]
    assert decode_bpe([257, 97, 99], vocab) == "ababac"
    print("PASS: restored IDs, bytes, and learned priority")

    payload = example_payload()
    payload["merges"] = []
    payload["vocab"] = payload["vocab"][:256]
    original_payload = deepcopy(payload)
    merges, vocab = unpack_tokenizer(payload)
    assert merges == {} and len(vocab) == 256
    assert decode_bpe(encode_bpe("A🌍B", merges), vocab) == "A🌍B"
    assert payload == original_payload
    print("PASS: restored zero-merge tokenizer")

    payload = example_payload()
    payload["merges"] = [[189, 160, 256]]
    payload["vocab"] = payload["vocab"][:256] + [[256, [189, 160]]]
    original_payload = deepcopy(payload)
    merges, vocab = unpack_tokenizer(payload)
    assert vocab[256] == bytes([189, 160])
    assert decode_bpe(encode_bpe("你", merges), vocab) == "你"
    assert payload == original_payload
    print("PASS: restored partial UTF-8 byte contents")

    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_tokenizer(path) == (merges, vocab)
    print("PASS: JSON file reading and unchanged payload")
    print("Lesson 02.07 stage B checks passed.")


def check_round_trip() -> None:
    _, merges, vocab = train_bpe("A🌍B abab", 260)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        save_tokenizer(path, merges, vocab)
        restored_merges, restored_vocab = load_tokenizer(path)
        assert restored_merges == merges and restored_vocab == vocab
        assert restored_merges is not merges and restored_vocab is not vocab
        for text in ("", "ababac", "你", "A🌍B", "café\n"):
            before = encode_bpe(text, merges)
            after = encode_bpe(text, restored_merges)
            assert after == before, "Loading must preserve encoded IDs"
            assert decode_bpe(after, restored_vocab) == text
        saved_payload = json.loads(path.read_text(encoding="utf-8"))
        for field, unsupported_value in (("version", 2), ("format", "other-format")):
            payload = deepcopy(saved_payload)
            payload[field] = unsupported_value
            path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                load_tokenizer(path)
            except ValueError:
                pass
            else:
                raise AssertionError("Reject an unsupported format/version")
    print("PASS: save/load preserves encoding and decoding; version is checked")
    print("Lesson 02.07 checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("pack", "unpack", "all"), default="all")
    choice = parser.parse_args().check
    if choice in ("pack", "all"):
        check_packing()
    if choice in ("unpack", "all"):
        check_unpacking()
    if choice == "all":
        check_round_trip()


if __name__ == "__main__":
    main()
