# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Recreate the original, synthetic teaching fixture and its provenance manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import random


HERE = Path(__file__).resolve().parent
DATA_VERSION = "synthetic-bilingual-bigram-v1"
SEED = 20261028


def sha256(content):
    """Return the hexadecimal SHA-256 of a bytes object."""
    return hashlib.sha256(content).hexdigest()


def build_records():
    """Return 64 original string records with a fixed 48/16 document split."""
    english = [f"{subject} {verb} {obj}."
               for subject in ("a cat", "a dog", "a fox", "a hen")
               for verb in ("sees", "likes")
               for obj in ("the sun", "the moon", "the rain", "the pond")]
    chinese = [f"{subject}{verb}{obj}。"
               for subject in ("小猫", "小狗", "小鸟", "小鹿")
               for verb in ("看见", "喜欢")
               for obj in ("月亮", "太阳", "小河", "花朵")]
    texts = english + chinese
    random.Random(SEED).shuffle(texts)
    records = [{"id": f"toy-{index:03d}", "text": text,
                "split": "train" if index < 48 else "validation"}
               for index, text in enumerate(texts)]
    train_chars = set("".join(item["text"] for item in records if item["split"] == "train"))
    validation_chars = set("".join(item["text"] for item in records
                                   if item["split"] == "validation"))
    if not validation_chars <= train_chars:
        raise ValueError("This fixture must not introduce validation-only characters")
    return records


def generated_files():
    """Return filename-to-bytes mappings; no external data is read or fetched."""
    records = build_records()
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records).encode("utf-8")
    manifest = {
        "format": "teaching-data-provenance-v1",
        "dataset_version": DATA_VERSION,
        "file": "toy_sentences.jsonl",
        "sha256": sha256(payload),
        "bytes": len(payload),
        "records": len(records),
        "split_counts": {"train": 48, "validation": 16},
        "source": "Original synthetic English and Simplified Chinese sentence templates",
        "external_data_used": False,
        "generator": "examples/generate_teaching_data.py",
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "generation_seed": SEED,
        "split_policy": "Shuffle unique documents with the recorded seed; first 48 train, remaining 16 validation; form token pairs only after splitting",
        "record_fields": {"id": "unique string document identifier", "text": "original string sentence", "split": "train or validation"},
        "license": "PolyForm-Noncommercial-1.0.0",
        "required_notice": "Copyright 2026 Wanglong Lu (卢望龙).",
        "license_scope": "Only this generated fixture; no rights to external or private datasets",
        "purpose": "Small, deliberately repetitive CPU integration demonstration; not a language-quality benchmark",
    }
    return {"toy_sentences.jsonl": payload,
            "manifest.json": (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare bundled bytes without rewriting files")
    args = parser.parse_args()
    data = HERE / "data"
    for filename, content in generated_files().items():
        target = data / filename
        if args.check:
            if not target.is_file() or target.read_bytes() != content:
                raise SystemExit(f"Teaching data mismatch: {filename}")
        else:
            data.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print("Teaching data reproducibility check passed." if args.check else "Teaching data generated: 64 documents (48 train, 16 validation).")


if __name__ == "__main__":
    main()
