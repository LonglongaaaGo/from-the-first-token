# 可运行教学示例：字符二元模型

[English](README.md)

本配套演示把微型原创数据与完整的 CPU 运行流程放在一起。它独立于编号练习；练习中由学习者完成的函数仍保留 TODO。示例所需输入均随附在 `data/` 中。代码与明确列出的合成数据采用 [PolyForm Noncommercial 1.0.0](../licenses/teaching-code-PolyForm-Noncommercial-1.0.0.md)；数据授权范围见 [data/DATA_LICENSE.md](data/DATA_LICENSE.md)。

## 先读哪些内容

[英文读者版](../book/en/manuscript.md#section-03-19)的 Lesson 03.19 解释错位的下一个词元配对；[Lesson 03.20](../book/en/manuscript.md#section-03-20)把分数接到损失，[Lesson 06.02](../book/en/manuscript.md#section-06-02)解释按文档划分数据。[中文读者版](../book/zh-CN/manuscript.md#section-03-19)提供相同的前置知识。运行本示例只需熟悉 Python 列表、字典和 PyTorch 张量（tensor）。

模型是一张可训练的表 `logits[current_id, next_id]`。每行包含所有候选下一个编号的分数，softmax 把这一行转成概率。交叉熵（cross entropy）将这些概率与真实的下一个编号进行比较，Adam 更新表中的数值以降低损失。这张表只能看到**前一个字符**，不能记住整个单词或句子。这个微型基线（baseline）便于直接检查只依赖前一个字符的预测规则。

## 数据与完整微型轨迹

[data/toy_sentences.jsonl](data/toy_sentences.jsonl)包含 64 条互不相同的合成句子：英文 32 条、简体中文 32 条。固定随机打乱后，先将 48 篇文档分到训练集（training set），16 篇文档分到验证集（validation set），**然后**才生成词元对。验证集的每个字符均在训练集中出现；词表（vocabulary）只根据训练文本排序构建，随后冻结。[data/manifest.json](data/manifest.json)记录来源、种子、版本、划分方式与哈希。这些模板有意重复，用于检查各环节连接，不是广泛语言质量的基准测试。

取一个独立的微型算例：`texts = ["ab", "ba"]`，`vocabulary = {"<BOS>": 0, "<EOS>": 1, "a": 2, "b": 3}`。`BOS` 表示文档开始，`EOS` 表示文档结束：

```text
文档 "ab"：编号 [0, 2, 3, 1] -> 输入 [0, 2, 3]，目标 [2, 3, 1]
文档 "ba"：编号 [0, 3, 2, 1] -> 输入 [0, 3, 2]，目标 [3, 2, 1]
合并：输入 [0, 2, 3, 0, 3, 2]，目标 [2, 3, 1, 3, 2, 1]
```

每个位置都在同一篇文档内预测下一个词元（token）。不会产生从 `"ab"` 的末尾跨到 `"ba"` 开头的配对。随包附带数据的词表含 40 个编号，包括 `BOS` 与 `EOS`；分数表的形状是 `[40, 40]`。初始分数全部为零，因此每个下一个词元的概率均为 `1/40`，平均交叉熵是 `ln(40) = 3.688879...`，单位为每个目标的自然对数损失。

## 在发布包根目录运行

使用按发布包 `requirements.txt` 安装的环境，无需图形计算设备或网络请求。运行：

```sh
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

第一条命令比较随包数据与来源清单是否和确定性生成器完全一致，不会改写文件。成功末行是：

```text
Teaching data reproducibility check passed.
```

第二条命令检查数据哈希、文档互不重叠、冻结词表、损失为有限值、训练损失低于初始值的 75%，以及验证损失低于初始值。随后打印两个固定种子的生成样例，成功末行是：

```text
Teaching bigram check passed.
```

一次已验证的 120 步运行将平均训练损失从约 `3.6889` 降至 `0.8390`，验证损失从约 `3.6889` 降至 `0.9054`。不同受支持数值库版本的最后小数位和生成文本可能不同。生成样例可能包含拼错的词或混合语言，因为每个新字符的预测只依赖前一个字符。

若需保存可移植的结果报告：

```sh
python examples/train_bigram.py --check --report outputs/teaching-bigram.json
```

可选 JSON 包含数据与代码哈希、词表策略、训练步数预算、优化器、种子、环境版本、训练／验证损失及解释。`--steps` 可调整默认 120 次全训练批次更新的预算；极小预算可能故意导致 `--check` 失败。

## 脚本接口

| 函数 | 输入 | 返回值 |
| --- | --- | --- |
| `load_fixture()` | 无参数，只读取旁边随包附带的数据与清单 | `(train, validation, manifest)`：两个文档字符串列表与来源字典 |
| `make_pairs(texts, vocabulary)` | `texts`：文档字符串列表；`vocabulary`：将符号字符串映射到整数编号的字典 | 两个一维 `torch.long` 张量，分别为对齐的当前编号与下一个编号 |
| `generate(logits, symbols, vocabulary, prefix, max_new_characters=48)` | `logits`：CPU 方形分数张量；`symbols`：按编号排列的符号字符串列表；`vocabulary`：符号到编号的字典；`prefix`：所有字符均已知的前缀字符串；`max_new_characters`：整数上限 | 包含前缀及采样续文的字符串；遇到 `EOS` 或达到上限时停止 |
| `run(steps=120, seed=17)` | `steps`：正整数更新次数；`seed`：整数随机种子 | 包含来源、设置、损失、检查结果及生成样例的字典 |

本演示提供实现。需要亲自练习实现时，请继续使用编号练习文件及其带提示的学习者代码区域。
