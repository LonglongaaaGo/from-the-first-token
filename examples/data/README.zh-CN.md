# 原创合成教学数据

[English](README.md)

`toy_sentences.jsonl` 是有意保持微型的双语集成测试数据（integration fixture）。每个 UTF-8 行是 JSON 对象，包含字符串字段 `id`、`text` 和 `split`。共 64 篇不同文档：英文 32 句、简体中文 32 句。固定划分为训练文档 48 篇、验证文档 16 篇。这些短模板是原创教学示例，不包含外部语料、个人学习作业或私有研究结果。

`manifest.json` 记录数据版本 `synthetic-bilingual-bigram-v1`、生成种子 `20261028`、文件大小与 SHA-256、生成器 SHA-256、文档划分策略以及明确的数据许可。[DATA_LICENSE.md](DATA_LICENSE.md)根据 PolyForm Noncommercial 1.0.0 授予限定于本教学数据的使用权限。发布包的 `licenses/` 目录包含完整法律文本。

在发布包根目录运行：

```sh
python examples/generate_teaching_data.py --check
```

预期成功末行：`Teaching data reproducibility check passed.` 此命令在内存中重新生成预期字节，与随包附带的两个文件比较。不加 `--check` 时，会按相同模板与种子改写这两个生成文件。原生成器必须保持不变，其记录的自身哈希才会一致。

对应的准确训练命令见[可运行示例](../README.zh-CN.md)。可选结果报告会同时记录数据版本、数据校验和与代码校验和。这些相似模板上的低验证损失不能证明广泛语言能力，也不能证明不存在自然语言泛化中的数据污染。
