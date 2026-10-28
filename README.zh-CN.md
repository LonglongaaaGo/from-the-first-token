# 从第一个 Token 开始

**作者：卢望龙（Wanglong Lu） · 公开阅读版**

[English](README.md)

本书包含十章、71节的中英文书稿、两本 PDF、配套插图，以及70个编号练习和检查程序。

书稿和插图采用 **CC BY-NC-ND 4.0**，教学代码采用 **PolyForm Noncommercial 1.0.0**。具体使用范围见 [LICENSE.md](LICENSE.md) 和 [NOTICE.md](NOTICE.md)。

## 阅读与练习

先阅读 [英文阅读快照](book/en/manuscript.md) 或 [中文阅读快照](book/zh-CN/manuscript.md)，再查看 [练习目录](EXERCISES.md)。03.04 只讲概念，无独立代码作业。所有核心函数已重置为明确 TODO；这不是完整解答包。检查程序覆盖书中案例，不能保证实现的普遍正确性。前置练习未完成时，后面的检查可能先在前置函数停止。

环境基线：Python 3.11、PyTorch 2.0.0、NumPy 1.x；使用 CPU 即可。请在包根目录运行：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python exercises/01_01_text_to_tokens.py --check encode
```

首次运行会报告 `NotImplementedError`，这是作业的起点。阅读函数约定和提示，填写 `START LEARNER CODE` 区域后重跑。后续练习如提示前置函数未完成，请先完成对应作业。所有配套样例均可在 CPU 上运行。

## 可直接运行的教学示例与数据

另附一个完整的 CPU 教学示例和配套合成数据。阅读[英文示例说明](examples/README.md)或[中文示例说明](examples/README.zh-CN.md)，然后运行：

```sh
python tools/check_package.py
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

完整包检查通过时输出 `status: PASS`。76个练习核心函数仍为作业 TODO；完整示例的实现另行提供。该示例使用一个字符的上下文；数据来源、生成方法和固定划分见[数据说明](examples/data/README.md)。

完整性检查适用于刚下载的文件；填写作业后，请使用对应练习的检查程序。

## 反馈与勘误

勘误与问题反馈请注明对应章节或练习。

[项目与反馈](https://github.com/LonglongaaaGo/from-the-first-token/issues)

## 书籍 PDF 下载

此 GitHub 源码目录不重复保存大型 PDF；完整 ZIP 包与两本 PDF 在同一版本的发布附件中提供。

- [English PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-English.pdf)
- [简体中文 PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-Chinese.pdf)
