# From the First Token

**Building Language Models from Scratch · Wanglong Lu · Public reading edition**

[简体中文](README.zh-CN.md)

The book contains ten chapters, 71 bilingual sections, two reader PDFs, illustrations, and 70 numbered exercises with supplied checks.

The book and illustrations use **CC BY-NC-ND 4.0**; teaching code uses **PolyForm Noncommercial 1.0.0**. See [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md) for the scope of each license.

## Read and implement

Start with the [English reading snapshot](book/en/manuscript.md), with the [Chinese edition](book/zh-CN/manuscript.md) alongside it. Follow [EXERCISES.md](EXERCISES.md) in numerical order. Section 03.04 is conceptual and has no separate exercise. Every assigned function is a fresh TODO, so this is not a complete solution package. Supplied drivers test the illustrated cases, not universal correctness. Later drivers can stop in an unfinished prerequisite.

The exercise baseline is Python 3.11, PyTorch 2.0.0 and NumPy 1.x. CPU execution is sufficient. From this package's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python exercises/01_01_text_to_tokens.py --check encode
```

The first run raises `NotImplementedError`, the exercise's starting point. Read the contract and hints, implement the region marked `START LEARNER CODE`, and rerun the check. If a later exercise reports an unfinished prerequisite, complete that earlier assignment first. All supplied examples run on CPU.

## Runnable teaching example and paired data

The package also includes a complete CPU example and paired synthetic data. Read [the English walkthrough](examples/README.md), with [Simplified Chinese](examples/README.zh-CN.md) alongside it, then run:

```sh
python tools/check_package.py
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

The package check prints `status: PASS` on success. The 76 assigned functions remain TODOs; the complete example supplies its own implementation. This example uses one character of context. See [the data guide](examples/data/README.md) for its source, generator and fixed split.

The integrity check applies to the files as downloaded. After implementing an exercise, use that exercise's supplied check driver.

## Questions and corrections

For corrections or questions, include the relevant chapter or exercise.

[Project and feedback](https://github.com/LonglongaaaGo/from-the-first-token/issues)

## Reader PDF downloads

This GitHub source tree keeps large PDFs in the versioned release assets. The complete ZIP and both PDFs belong to the same companion release.

- [English PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-English.pdf)
- [简体中文 PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-Chinese.pdf)
