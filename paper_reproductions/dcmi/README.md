# DCMI

**2026-10-22 当前结果：** 已完成本地替代生成器版本；AUC 0.817316，测试准确率 75.72%，3281/4000 篇可评分。719 篇被排除，多数扰动未满足请求预算。

本轮使用 `reproductions/` 新管线，详见[结果与实跑配置](../../reproductions/results/2026-10-21/README.md)。下文保留历史适配入口的核查，不代表新管线仍缺少同样的阶段，也不表示严格完整论文复现已获验证。

论文：[DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation](https://arxiv.org/abs/2509.06026)。
原作者：[Xinyu140203/RAG_MIA](https://github.com/Xinyu140203/RAG_MIA)，固定版本
`ef961a1f23d09c62c5fbf69d798e325fb0362e55`。

## 历史适配代码入口

`external/DCMI/MIA.py` 是本地 NFCorpus 词法代理；它用 query/document 的词法信号，
不是生成式 RAG 的真实输出概率。已有 100/100 的结果只是该代理实验的记录。
`experiments/min_dcmi_nfcorpus.py` 另有检索分数实验，也不能替代论文的输出校准。

```bash
python run_attack.py --attack dcmi --config configs/dcmi.yml
```

该命令当前仍运行词法代理，结果已标明 `lexical_proxy`。根据测试标签翻转方向和
真实数据失败后自动切换 synthetic 的逻辑已经删除，但这不会让代理变成原方法。

## 原作者代码与方法

论文针对实际 RAG 检索库成员，比较原始查询与轻微扰动查询产生的 RAG 输出信号：
概率可见场景使用对应生成概率差；仅文本可见场景需要明确采用论文相应信号。

`example/run_exp.py` 和 `flashrag/` 提供 RAG 基础设施；`perturb.py` 调用模型生成
扰动。原作者 `MIA.py` 的四组输入还是 `[...]` 占位，没有完整的分数采集链。
该原始入口保存在 `../reference_entries/dcmi_original.py`，不能把占位脚本直接当作
开箱可运行的复现实验。

## 需要实现/核验的接口

建立真实检索库与明确的成员划分 → 为候选样本准备验证查询及扰动查询 → 两次查询
同一 RAG → 保存实际输出信号 → 做 differential calibration → 独立校准和评估。
成员标签必须来自文档是否在目标检索库中，不能用 qrels 的相关/不相关代替。
公开 NFCorpus 文档不自动等于某个私有/子集 RAG 索引的真实非成员。
