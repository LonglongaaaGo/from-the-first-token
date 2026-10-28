# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Wanglong Lu (卢望龙).
"""Run a self-contained CPU character-bigram demonstration, without exercise answers."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
BOS, EOS = "<BOS>", "<EOS>"


def load_fixture():
    """Validate provenance and return document-disjoint train/validation strings."""
    data = HERE / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    generator_digest = hashlib.sha256((HERE / "generate_teaching_data.py").read_bytes()).hexdigest()
    if generator_digest != manifest["generator_sha256"]:
        raise ValueError("Teaching data generator checksum differs from the supplied manifest")
    content = (data / "toy_sentences.jsonl").read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise ValueError("Teaching data checksum differs from the supplied manifest")
    records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    if len(records) != manifest["records"] or len({item["id"] for item in records}) != len(records):
        raise ValueError("Unexpected document count or duplicate identifiers")
    if any(item["split"] not in ("train", "validation") or not isinstance(item["text"], str)
           or not item["text"] for item in records):
        raise ValueError("Expected nonempty string documents with declared splits")
    train = [item["text"] for item in records if item["split"] == "train"]
    validation = [item["text"] for item in records if item["split"] == "validation"]
    if {"train": len(train), "validation": len(validation)} != manifest["split_counts"]:
        raise ValueError("Unexpected document split counts")
    if set(train) & set(validation) or len(set(train + validation)) != len(records):
        raise ValueError("The document splits must be unique and disjoint")
    return train, validation, manifest


def make_pairs(texts, vocabulary):
    """Map each string to BOS/character/EOS IDs and return flat input/target tensors."""
    inputs, targets = [], []
    for text in texts:
        ids = [vocabulary[BOS]] + [vocabulary[character] for character in text] + [vocabulary[EOS]]
        inputs.extend(ids[:-1])
        targets.extend(ids[1:])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def generate(logits, symbols, vocabulary, prefix, max_new_characters=48):
    """Return a fixed-seed sample conditioned only on the preceding character."""
    generator = torch.Generator(device="cpu").manual_seed(17)
    result = prefix
    current = vocabulary[prefix[-1]] if prefix else vocabulary[BOS]
    with torch.no_grad():
        for _ in range(max_new_characters):
            scores = logits[current].clone()
            scores[vocabulary[BOS]] = -float("inf")
            current = int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator))
            symbol = symbols[current]
            if symbol == EOS:
                break
            result += symbol
    return result


def run(steps=120, seed=17):
    """Train a one-character lookup table and return a portable result dictionary."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    train, validation, manifest = load_fixture()
    symbols = [BOS, EOS] + sorted(set("".join(train)))
    vocabulary = {symbol: index for index, symbol in enumerate(symbols)}
    missing = set("".join(validation)) - vocabulary.keys()
    if missing:
        raise ValueError("Validation contains characters absent from the frozen training vocabulary")
    train_x, train_y = make_pairs(train, vocabulary)
    validation_x, validation_y = make_pairs(validation, vocabulary)
    logits = torch.nn.Parameter(torch.zeros(len(symbols), len(symbols), device="cpu"))
    optimizer = torch.optim.Adam([logits], lr=0.15)
    with torch.no_grad():
        initial_train = float(F.cross_entropy(logits[train_x], train_y))
        initial_validation = float(F.cross_entropy(logits[validation_x], validation_y))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits[train_x], train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_train = float(F.cross_entropy(logits[train_x], train_y))
        final_validation = float(F.cross_entropy(logits[validation_x], validation_y))
    finite = all(torch.isfinite(torch.tensor(value)).item()
                 for value in (initial_train, initial_validation, final_train, final_validation))
    passed = finite and final_train < initial_train * 0.75 and final_validation < initial_validation
    return {
        "format": "teaching-bigram-check-v1",
        "model": "character-bigram-logit-table",
        "device": "cpu",
        "data_version": manifest["dataset_version"],
        "data_sha256": manifest["sha256"],
        "entrypoint_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_sha256": manifest["generator_sha256"],
        "tokenizer": "Unicode code points; sorted training-only vocabulary plus BOS/EOS",
        "vocabulary_size": len(symbols),
        "seed": seed,
        "optimizer": "Adam",
        "learning_rate": 0.15,
        "steps": steps,
        "train_documents": len(train),
        "validation_documents": len(validation),
        "train_pairs": len(train_x),
        "validation_pairs": len(validation_x),
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "loss_unit": "mean natural-log cross entropy per next-token target",
        "check_passed": passed,
        "samples": {"english_prefix_a": generate(logits, symbols, vocabulary, "a"),
                    "chinese_prefix_small": generate(logits, symbols, vocabulary, "小")},
        "interpretation": "A repetitive synthetic fixture verifies data-to-update-to-generation wiring. This is neither a Transformer nor evidence of language understanding or chatbot quality.",
        "python_version": __import__("platform").python_version(),
        "torch_version": torch.__version__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Require finite losses and specified loss reductions")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--report", type=Path, help="Optionally write this run's portable JSON report")
    args = parser.parse_args()
    result = run(steps=args.steps)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Documents: {result['train_documents']} train / {result['validation_documents']} validation")
    print(f"Vocabulary: {result['vocabulary_size']} IDs (training characters + BOS/EOS)")
    print(f"Train loss: {result['initial_train_loss']:.4f} -> {result['final_train_loss']:.4f}")
    print(f"Validation loss: {result['initial_validation_loss']:.4f} -> {result['final_validation_loss']:.4f}")
    print("Generated sample (a): " + result["samples"]["english_prefix_a"])
    print("Generated sample (小): " + result["samples"]["chinese_prefix_small"])
    if args.check:
        if not result["check_passed"]:
            raise SystemExit("Teaching bigram check failed: expected finite losses and loss reductions")
        print("Teaching bigram check passed.")


if __name__ == "__main__":
    main()
