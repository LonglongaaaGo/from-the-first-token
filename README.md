# From the First Token

**Building Language Models from Scratch · Wanglong Lu · First public teaching edition**

[简体中文](README.zh-CN.md)

This independent teaching package contains ten chapters, 71 bilingual sections, release links to two reader PDFs, the required illustrations, and 70 numbered exercise scaffolds with supplied checks. It carries no research repository or Git history.

The book prose, figures and Markdown reading snapshots use **CC BY-NC-ND 4.0**: share unchanged copies noncommercially with attribution; distributing altered or translated editions needs separate permission. The teaching Python files and author-owned executable examples use **PolyForm Noncommercial 1.0.0**, permitting modification and sharing for its defined permitted purposes. See [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md). These noncommercial materials are not advertised as OSI open-source software. Third-party terms remain separate.

## Read and implement

Start with the [English reading snapshot](book/en/manuscript.md), with the [Chinese edition](book/zh-CN/manuscript.md) alongside it. Follow [EXERCISES.md](EXERCISES.md) in numerical order. Section 03.04 is conceptual and has no separate exercise. Every assigned function is a fresh TODO, so this is not a complete solution package. Supplied drivers test the illustrated cases, not universal correctness. Later drivers can stop in an unfinished prerequisite.

The exercise baseline is Python 3.11, PyTorch 2.0.0 and NumPy 1.x. CPU execution is sufficient. From this package's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python exercises/01_01_text_to_tokens.py --check encode
```

The first run raises `NotImplementedError`, the expected starting point. Read the contract and hints, implement the region marked `START LEARNER CODE`, and rerun the check. Later files may report an unfinished prerequisite through the course helper. No external model or dataset download, cluster account or credentials are required. Training and distributed-system exercises use small CPU fixtures and do not establish a completed large-scale run.

## Runnable teaching example and paired data

The reader PDFs retain version 0.2.4. This companion release adds a self-contained example and original synthetic teaching data. Read [the English walkthrough](examples/README.md), with [Simplified Chinese](examples/README.zh-CN.md) alongside it. These commands verify the package, import every exercise, and run CPU data loading, training, validation and generation:

```sh
python tools/check_package.py
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

The full package check prints JSON with `status: PASS`; the standalone training success output is documented in the walkthrough. The 76 assigned functions remain TODOs and are recorded separately from the runnable demonstration. The demonstration has a one-character context; it is not a complete Transformer or chatbot. Its tiny original synthetic data has a version, generator, split and checksums in [the data manifest](examples/data/manifest.json). Only the explicitly bundled teaching fixtures are covered by the teaching-software license.

The inventory check applies to unchanged distributed files. Local `.git/`, `.venv/`, `outputs/` and `__pycache__/` runtime directories are outside that inventory. After implementing an exercise, run its supplied driver; the inventory check will correctly report your changed file. For a dependency-free download check, run `python tools/check_package.py --inventory-only`.

## Version and corrections

This remains a teaching preview; independent technical review is ongoing. AI tools assisted with drafting, revision, translation and polishing, with the author responsible for the released material. `content-manifest.json` and `SHA256SUMS` identify the package files. Include version, section and a minimal reproduction when reporting a correction; omit personal data. Public project address: https://github.com/LonglongaaaGo/from-the-first-token. Building a local package does not confirm an upload. No private research material, private training datasets or weights are licensed by this package, and it makes no normal-chat competence claim for an experimental model.

## Reader PDF downloads

This GitHub source tree keeps large PDFs in the versioned release assets. The complete ZIP and both PDFs belong to the same companion release.

- [English PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-English.pdf)
- [简体中文 PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-Chinese.pdf)
