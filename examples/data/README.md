# Original synthetic teaching data

[简体中文](README.zh-CN.md)

`toy_sentences.jsonl` is a deliberately small bilingual integration fixture. Each UTF-8 line is a JSON object with string fields `id`, `text`, and `split`. There are 64 unique documents: 32 English and 32 Simplified Chinese sentences. The fixed split contains 48 training and 16 validation documents. These short templates are original teaching examples; no external corpus, personal learning submission or private research result is included.

`manifest.json` records dataset version `synthetic-bilingual-bigram-v1`, generation seed `20261028`, file size and SHA-256, generator SHA-256, document split policy, and explicit data license. [DATA_LICENSE.md](DATA_LICENSE.md) grants the scoped teaching-fixture permission under PolyForm Noncommercial 1.0.0. Its complete legal text is included in the release's `licenses/` directory.

Run from the release root:

```sh
python examples/generate_teaching_data.py --check
```

Expected final line: `Teaching data reproducibility check passed.` The command regenerates expected bytes in memory and compares them with both bundled files. Without `--check`, it rewrites those two generated files from the same templates and seed. The original generator must be unchanged for its own recorded hash to match.

Follow [the runnable example](../README.md) for the exact paired training command. Data version, source checksum and code checksum accompany the optional result report. Low validation loss on these similar templates does not establish broad language capability or contamination-free natural-language generalization.
