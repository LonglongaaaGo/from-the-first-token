# Runnable teaching example: a character bigram

[简体中文](README.zh-CN.md)

This supplied demonstration pairs a tiny original dataset with a complete CPU run. It is separate from the numbered exercises: their learner functions remain TODOs. All example inputs are supplied in `data/`. The code and the explicitly identified synthetic fixture follow [PolyForm Noncommercial 1.0.0](../licenses/teaching-code-PolyForm-Noncommercial-1.0.0.md); see [data/DATA_LICENSE.md](data/DATA_LICENSE.md) for the precise data scope.

## Read first

The [English reader](../book/en/manuscript.md#section-03-19) explains shifted next-token pairs in Lesson 03.19; [Lesson 03.20](../book/en/manuscript.md#section-03-20) connects scores to loss, and [Lesson 06.02](../book/en/manuscript.md#section-06-02) explains document-level splitting. The [Chinese reader](../book/zh-CN/manuscript.md#section-03-19) presents the same prerequisites. Familiarity with Python lists, dictionaries and a PyTorch tensor is sufficient to run this example.

The model is a trainable table `logits[current_id, next_id]`. A row contains one score for every possible next ID; softmax turns that row into probabilities. Cross entropy compares those probabilities with the observed next ID. Adam updates the table to reduce that loss. The table sees **one preceding character**, so it cannot remember a whole word or a sentence. This small baseline makes the one-character prediction rule easy to inspect.

## Data and a complete small trace

[data/toy_sentences.jsonl](data/toy_sentences.jsonl) contains 64 distinct synthetic sentences: 32 English and 32 Simplified Chinese. A fixed shuffle assigns 48 documents to training and 16 to validation **before** token pairs are created. All validation characters occur in training; the sorted vocabulary is built from training text only and then frozen. [data/manifest.json](data/manifest.json) records the source, seed, version, split and hashes. These deliberately repetitive templates are an integration fixture, not a broad language-quality benchmark.

For an independent tiny trace, take `texts = ["ab", "ba"]` and `vocabulary = {"<BOS>": 0, "<EOS>": 1, "a": 2, "b": 3}`. `BOS` marks a document beginning and `EOS` its end:

```text
document "ab": IDs [0, 2, 3, 1] -> inputs [0, 2, 3], targets [2, 3, 1]
document "ba": IDs [0, 3, 2, 1] -> inputs [0, 3, 2], targets [3, 2, 1]
combined: inputs [0, 2, 3, 0, 3, 2], targets [2, 3, 1, 3, 2, 1]
```

Each position predicts the next token inside the same document. No pair crosses from the end of `"ab"` to the beginning of `"ba"`. For the bundled fixture the vocabulary has 40 IDs, including `BOS` and `EOS`; the score table has shape `[40, 40]`. Initially every score is zero, so all 40 next-token probabilities equal `1/40` and mean cross entropy is `ln(40) = 3.688879...` natural-log units per target.

## Run from the release-package root

Use the environment installed from the package's `requirements.txt`. No graphics device or network request is needed. Run:

```sh
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

The first command compares the bundled dataset and provenance manifest against the deterministic generator; it does not rewrite them. Its success line is:

```text
Teaching data reproducibility check passed.
```

The second command checks the dataset hash, disjoint documents, frozen vocabulary, finite losses, a training loss below 75% of its initial value, and validation loss below its initial value. It then prints two fixed-seed generated samples and ends with:

```text
Teaching bigram check passed.
```

A verified 120-step run reduced mean training loss from about `3.6889` to `0.8390` and validation loss from about `3.6889` to `0.9054`. Exact final decimals and generated text may differ between supported numerical-library versions. Generated samples can contain malformed words or mixed languages because each new character depends only on the preceding character.

To save a portable result report:

```sh
python examples/train_bigram.py --check --report outputs/teaching-bigram.json
```

The optional JSON includes data and code hashes, vocabulary policy, step budget, optimizer, seed, environment versions, train/validation losses and interpretation. `--steps` changes the default budget of 120 full-training-batch updates; very small budgets can intentionally fail `--check`.

## Script interfaces

| Function | Inputs | Returned value |
| --- | --- | --- |
| `load_fixture()` | No arguments; reads only the adjacent supplied data and manifest | `(train, validation, manifest)`: two lists of document strings and the provenance dictionary |
| `make_pairs(texts, vocabulary)` | `texts`: list of document strings; `vocabulary`: dictionary mapping symbol strings to integer IDs | Two one-dimensional `torch.long` tensors, with aligned current/next IDs |
| `generate(logits, symbols, vocabulary, prefix, max_new_characters=48)` | `logits`: square CPU score tensor; `symbols`: ID-ordered string list; `vocabulary`: symbol-to-ID dictionary; `prefix`: string whose characters are known; `max_new_characters`: integer upper bound | String containing the prefix and sampled continuation; stops at `EOS` or the bound |
| `run(steps=120, seed=17)` | `steps`: positive integer update count; `seed`: integer random seed | Dictionary containing provenance, settings, losses, checks and generated samples |

The demonstration supplies its implementation. For your own implementation practice, continue with the numbered exercise files and their guided learner regions.
