# 四个现有方法的正式复现入口

此目录是后续实现的真实模型实验链条；与 `external/` 中保留的历史改编、简化和词法代理入口分开。
实现了流程不代表正式实验已结束，完成状态必须查看实际运行目录中的 `complete.json`、模型指纹和逐样本输出。

## 本轮范围

每种方法先完成一个可核验设置，不宣称复现论文全部模型、数据集、基线和消融表。

| 方法 | 本轮设置 | 实现链条与差异 |
| --- | --- | --- |
| Neighbour | GPT-2 base / BERT / AG News；60,000 条目标训练样本，独立 200/200 校准、1000/1000 测试；100 个邻居 | embedding dropout p=0.7；p(candidate)/(1-p(original)) 排名；单 token 替换；平均邻居 loss 减原文 loss。修复特殊 token、padding 与单 GPU；学习率等未公开项明确写入配置；最多 10 epoch，达到验证 PPL 30 后停止 |
| SPV | GPT-2 XL 1.5B / AG News；10,000 个 128-token 块；target 10 epoch、reference 4 epoch；batch 16、lr 1e-4 | target 全参数微调 → 用未参与 target 训练的同域文本前 16 tokens 自提示生成 10,000 条 128-token 参考记录 → 从原始预训练模型微调 reference → 10 对 ±Gaussian embedding 噪声、sigma=0.05 → target/reference 概率变动校准。采用论文附录 Algorithm 1，非默认的 T5 semantic 版本；自提示来源跟随发布脚本的同域块，非论文默认的独立 TLDR News |
| IA | NFCorpus；30 问题/文档；k=3；1000/1000 测试 + 200/200 校准 | 生成 retrieval summary 和具体 Yes/No 问题 → 独立参考回答 → 目标 RAG 检索并回答 → 正确性加分、UNK 惩罚 5。去除 TF-IDF cosine ≥0.95 的非成员近重复；vanilla RAG（论文 Appendix D） |
| DCMI | BEIR TREC-COVID；80% 真实 datastore；1000/1000 reference + 1000/1000 test；Flan-T5-large / BM25 / k=4 | 构造 verification query → 局部反义替换 → 真正两次查询 RAG 得到 P(Yes) → 差分校准；只在 reference 上搜索 theta/gamma；另报告 reference 非成员 Q3 和黑盒 decision reversal 两种设置 |

**用户已同意无 API 时替换生成模型。** IA 的问题/参考答案模型、DCMI 的扰动模型均使用本地 Qwen3-1.7B instruction，
不能声称使用了 GPT-4o/GPT-4o-mini。IA 的目标生成器同时替换为可获取的 Flan-T5-large。
DCMI 用论文扰动模板让本地模型生成改写，只回填可对齐的词替换；保存原始输出、实际替换数量和未满足的编辑预算。被模型误删的句子和无关变更不会直接写回样本。
DCMI 的 BM25 分词和 Lucene IDF 参数可见，未声称完全等同 FlashRAG 默认分词器。

## 获取资产与运行

使用单独的已记录环境，依赖 Torch / Transformers / NumPy / SciPy / scikit-learn / PyArrow / Sentence Transformers。
本轮可用的 Python 3.10 依赖固定在 [`requirements.txt`](requirements.txt)，CUDA Torch 轮子需按机器安装。
运行配置中所有路径可编辑；第一次修改后应使用新的输出目录，工具拒绝混入另一份配置。

```bash
python -m reproductions.assets --root /root/mia-paper-assets --models \
  openai-community/gpt2-xl Qwen/Qwen3-1.7B google/flan-t5-large

python -m reproductions.run --config reproductions/configs/neighbour_agnews.json --output runs/neighbour
python -m reproductions.run --config reproductions/configs/spv_agnews_gpt2xl.json --output runs/spv
python -m reproductions.run --config reproductions/configs/ia_nfcorpus_local.json --output runs/ia
python -m reproductions.run --config reproductions/configs/dcmi_trec_covid_local.json --output runs/dcmi
```

公开模型和数据会固定获取时的 revision，并记录所有资产 SHA-256。TREC-COVID 仅下载公开 BEIR 语料，记录 URL 和内容哈希。
训练使用 FP32 参数、梯度与 AdamW 状态，前向采用 BF16 autocast，避免直接 BF16 更新吞掉小幅变化。
SPV XL 的 FP32 AdamW 主权重和状态放到 CPU，以保持单张 24GB GPU 上的完整训练；有效 batch 仍为 16。
用于推理的 checkpoint 保存为 BF16，并记录训练精度；它不包含可恢复训练的优化器状态。
SPV 保留生成的原始 token IDs 作为 reference 训练输入，避免 GPT 字节级解码、再编码改变样本。
训练以真实验证 loss 选 checkpoint，测试记录不参与优化器更新或攻击阈值选择。
中断后会复用完成的训练阶段和逐样本攻击记录；未完成训练阶段会从原始预训练模型重新开始，**不声称恢复了优化器状态**。

## 输出与检查

- `config.json`、`environment.json`、`invocation_sources/`：配置、环境与每次启动的源码快照。
- `data/`：训练输入、划分及真实成员定义；RAG 使用实际索引文档 ID。
- `target/`、`reference/`：训练历史、最优 epoch、完整 checkpoint 指纹。
- `self_prompt/generated.jsonl`：实际生成的参考数据、prompt IDs、采样种子与生成 IDs。
- `records.jsonl` / `attack/records.jsonl`：每篇文档的真实 score、标签、校准/测试划分。
- DCMI 的原始 reference 记录在参数选定前将 `score` 留空；最终选定 theta 后的全部分数保存为 `scored_records.json`，不能把未选参数的占位值当作 0 分。
- RAG 的 `query_records.jsonl`：每次原始/重写查询、检索文档、完整 prompt、目标回复、P(Yes)、截断情况。
- `evaluation/result.json` 或 `attack/result.json`：校准阈值、AUC、accuracy、低 FPR ROC 统计及其经验分辨率。
- `complete.json`：仅在所有配置的阶段成功后写入；错误另写 `failure.json`，失败不能当作成功复现。

1000 个非成员的经验 FPR 分辨率为 0.1%；不将这一样本量说成可靠验证了 0.01% FPR 效果。
GPU 长任务可用 `python -m reproductions.queue --root /absolute/run/root --detach` 顺序执行后续 SPV、IA、DCMI。
本次已有 Neighbour 独立训练时，队列增加 `--wait-neighbour`，会等待其完成后再运行后续方法。

## 论文与原作者来源

- Neighbour：[ACL 2023 论文](https://aclanthology.org/2023.findings-acl.719/)，[原作者代码](https://github.com/justusmattern27/neighbour-mia)。
- SPV：[NeurIPS 2024 论文](https://arxiv.org/abs/2311.06062)，[原作者代码](https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA)。本实现保留明确的概率变动定义与两模型校准，并补上发布脚本中未实现的对称 embedding 版本。
- IA：[Riddle Me This 论文](https://arxiv.org/abs/2502.00306)，[原作者代码](https://github.com/ali7naseh/RAG_MIA)。
- DCMI：[CCS 2025 论文](https://arxiv.org/abs/2509.06026)，[原作者代码](https://github.com/Xinyu140203/RAG_MIA)。本实现补齐原作者 `MIA.py` 的四组信号占位所需的真实采集链。

固定的原作者 commit 与原始关键文件仍见 `paper_reproductions/`；新实现文件不能标成原作者逐行原码。

## IA / DCMI 生成修复与后台续跑

IA 按小批次补齐每篇文档的 30 个问题，重试采用有固定文档种子的采样，
拒绝开放式问题、重复文本及未知参考答案；每批保存 `.partial.json`，不再反复执行同一个贪心生成任务。
DCMI 先按上下文词性排除名词和动词，使用 WordNet 的直接形容词/副词反义词关系作为候选白名单，
再由本地模型选择符合上下文的词义和位置。
保存词典版本、关系来源、每个替换的位置、原词和新词，不接受模型任意编造的替换，也不采用整段改写造成的删除。
这个词典约束是原论文 GPT-4o 生成器的替代实现，需要在报告中注明；它不等价于人工语义核验。

安装 `reproductions/requirements.txt` 后，先执行 `python -m nltk.downloader wordnet averaged_perceptron_tagger_eng`。
Linux 后台运行：

```bash
tmux new-session -d -s mia-repair 'python -u -m reproductions.repair_queue --root /absolute/run/root'
tmux attach -t mia-repair
```

修复队列先分别验证 4 篇文档的实际模型链路（IA 仍要求每篇 30 问），
只有对应检查完整通过，才启动该方法的正式设置。正式顺序为 DCMI、IA。
输出使用新的 `*_repair_v2` 目录，仅复制原来的成员划分与检索向量，保留全部旧结果。
阶段状态见 `repair_queue_status.json`，每个阶段的日志为根目录下同名 `.log`。

文档无法生成合格问题或无适当反义词时，将原因写入 `unavailable_samples.jsonl`，不赋假分数、不补换样本。
继续处理其他文档，并按成员标签和校准/测试划分报告覆盖率。
有排除样本时，完成状态为 `completed_with_exclusions`，指标仅针对可评分文档，不能称为原论文全样本复现。
未满足请求扰动比例的查询仍单独统计真实替换比例；不得把较少的替换数量说成满足了原论文 theta。

IA 后续修复使用 `python -m reproductions.ia_queue --root /absolute/run/root`，
可放在独立的 `mia-ia` tmux 会话中。它使用新的 `*_repair_v3` 目录，
只导入旧检查中已生成的问题和原始证据，保留 27/30、25/30 等部分进度；
不导入旧的失败样本排除清单或目标模型分数，从而真正重新处理失败文档。
问题生成最多尝试 72 批，交替关注原句事实、明确矛盾的备选陈述和事实比较，
参考答案仍根据全文生成，仍要求恰好 30 个文本不同且格式合格的问题。
这不保证问题在语义上完全不同，也不等价于人工核验；保留每一条提示词、回复和参考答案供审查。

该队列在有至少 11 GB 空闲显存时执行四文档推理检查，之后等待现有 GPU 队列释放锁，
再运行正式 IA 实验。状态见 `ia_queue_status.json`，不覆盖 DCMI 的队列状态。
生成失败后的覆盖率会在计算指标前写入 `coverage.json`；缺少成员类别时明确报错，不能生成无效指标。
