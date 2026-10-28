# 从第一个 Token 开始

**作者：卢望龙（Wanglong Lu） · 首次公开教学版**

[English](README.md)

本独立教学包包含十章、71节的中英文阅读快照、两本 PDF 的发布下载入口、必要插图，以及70个编号练习和检查程序。它不携带原研究仓库或历史。

书稿、插图及 Markdown 阅读快照采用 **CC BY-NC-ND 4.0**：可以署名、原样、非商业分享；公开分发修改版或翻译版须另行授权。教学 Python 文件及作者有权授权的可执行示例采用 **PolyForm Noncommercial 1.0.0**：允许在许可规定的非商业等许可用途内修改和分享。详见 [LICENSE.md](LICENSE.md) 和 [NOTICE.md](NOTICE.md)。这些限非商业用途的材料不称为 OSI 标准开源。第三方条款另行保留。

## 阅读与练习

先阅读 [英文阅读快照](book/en/manuscript.md) 或 [中文阅读快照](book/zh-CN/manuscript.md)，再查看 [练习目录](EXERCISES.md)。03.04 只讲概念，无独立代码作业。所有核心函数已重置为明确 TODO；这不是完整解答包。检查程序覆盖书中案例，不能保证实现的普遍正确性。前置练习未完成时，后面的检查可能先在前置函数停止。

环境基线：Python 3.11、PyTorch 2.0.0、NumPy 1.x；使用 CPU 即可。请在包根目录运行：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python exercises/01_01_text_to_tokens.py --check encode
```

首次运行会报告 `NotImplementedError`，这是正常的作业起点。阅读函数约定和提示，填写 `START LEARNER CODE` 区域后重跑。后续练习可能使用项目检查辅助程序报告未完成的前置作业。无需额外下载模型或外部数据集，也不需要集群账号或配置密钥。大型训练与分布式章节的练习采用小型 CPU 示例，不能据此宣称已完成大规模训练。

## 可直接运行的教学示例与数据

书籍 PDF 保留 v0.2.4；本次教学包将配套示例与原创合成数据一起发布。阅读[英文示例说明](examples/README.md)或[中文示例说明](examples/README.zh-CN.md)。运行以下命令可验证包内文件、导入全部练习，并执行 CPU 数据读取、训练、验证与生成示例：

```sh
python tools/check_package.py
python examples/generate_teaching_data.py --check
python examples/train_bigram.py --check
```

完整包检查输出 JSON，其中 `status` 为 `PASS`；单独训练命令的成功输出见示例说明。练习里的76个核心函数仍为作业 TODO，与可运行示例分别记录。示例只有一个字符的上下文，不是完整 Transformer 或聊天机器人。数据是版本化的小型原创合成教学素材，来源、生成方法、划分和校验值见 [数据清单](examples/data/manifest.json)，仅这些附带教学样例适用包内的教学软件许可。

校验命令适用于未修改的发布文件；本地 `.git/`、`.venv/`、`outputs/` 和 `__pycache__/` 运行目录不属于发布文件清单。填写练习之后，请运行该练习的检查驱动，此时完整性校验会按设计提示文件已变化。只检查下载完整性且不导入依赖时，运行 `python tools/check_package.py --inventory-only`。

## 版本与反馈

本书仍为教学预览，独立技术审读未完成。人工智能辅助了起草、修订、双语翻译和润色，发布内容由作者负责。`content-manifest.json` 与 `SHA256SUMS` 标识本包文件。报告问题时提供版本、章节及最小复现，不要提交个人数据。公开项目地址：https://github.com/LonglongaaaGo/from-the-first-token。本地包生成不代表远程上传已完成。本包未授予私有研究资料、私人训练数据或权重的权利，也不宣称实验模型已具备正常聊天能力。

## 书籍 PDF 下载

此 GitHub 源码目录不重复保存大型 PDF；完整 ZIP 包与两本 PDF 在同一版本的发布附件中提供。

- [English PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-English.pdf)
- [简体中文 PDF](https://github.com/LonglongaaaGo/from-the-first-token/releases/download/v0.2.4-teaching.1/From-the-First-Token-Chinese.pdf)
