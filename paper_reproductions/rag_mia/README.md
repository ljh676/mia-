# RAG-MIA / Interrogation Attack

论文：[Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation](https://arxiv.org/abs/2502.00306)。
原作者：[ali7naseh/RAG_MIA](https://github.com/ali7naseh/RAG_MIA)，固定版本
`a53e8fda4f1492f204d968d7afb53ce8287c193c`。

## 当前自己的代码

`external/RAG_MIA/` 是本地适配版；统一调用为：

```bash
python run_attack.py --attack rag_mia --config configs/rag_mia.yml
```

这个命令需要原服务器布局、数据、索引、问题和模型；当前不能宣称开箱即复现。
实际配置将模型/检索器换成 Phi3/BGE，且启用了 checkpoint，因此也不等于论文原设置。

论文攻击查询目标 RAG，根据参考答案与 RAG 回答的一致性推断目标文档是否在检索库中。
检索命中率不能替代 reference answers 或直接作为同一攻击的成功证据。

## 核查发现

已有 M100/N10 文件包含 200 条文档、2,000 条回答和检索结果，但没有 `answers`。
旧本地 scorer 曾在答案缺失时使用检索命中构造 Yes/No；该替代已经删除。
现在源码 scorer 和统一解析器都会拒绝这种不完整结果。
配置中的 `/root/autodl-tmp/models/phi3-mini-4k-instruct` 当前未找到。

桌面 `成员推断攻击.docx` 另记录了 10 篇文档（5 member / 5 non-member）、每篇 3 个问题的
小实验，并分别统计了回答正确与检索命中。它与上述 200 篇文档缓存不是同一份记录。
目前只确认笔记中存在这份总结，尚未定位与之匹配的原始逐问题输出、配置和日志，
因此既不将笔记数字当作已验证复现指标，也不否认这次小实验曾经运行。

原作者关键评分文件保存在 `../reference_entries/rag_mia_original.py`。
模型换型、运行路径、回答保存和评分修复应在复现报告中明确说明。

## 要补齐的链条

1. 恢复可加载模型，使用与其兼容的独立环境，并记录权重指纹。
2. 按原仓库准备数据与 `selected_indices.json`，核对实际索引中的 member/non-member。
3. 生成或恢复每个问题的参考答案，同时保存来源与问题对应关系。
4. 生成目标 RAG 回答，再按严格的答案一致性计算分数。
5. 保存逐样本结果、AUC/低 FPR 指标；如报告 accuracy/F1，阈值来自独立校准集。

原项目准备顺序：`prepare_dataset.py` → `generate_index.py --config ...` →
`main_mia.py --config ...`。配置中的模型、问题缓存和检索库必须匹配同一次实验。
