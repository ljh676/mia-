# 桌面 mia 文件夹：论文与代码对应

核对日期：2026-10-19。来源为用户桌面 `mia` 文件夹的 27 篇 PDF、3 份 DOCX。
按 PDF 标题页和正文识别论文，不将中文重命名视为正式标题；此清单不代表 27 篇均已复现。
本仓库只保存对应关系，不收录用户的论文 PDF 或 Word 原文。

## 四个现有攻击入口

| 本地 PDF 文件名 | 正式论文 / 方法 | 代码位置 | 当前需要补齐的部分 |
| --- | --- | --- | --- |
| mia隐秘成员推断攻击（有码）.pdf | Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation；Interrogation Attack (IA) | `external/RAG_MIA/` | Phi3/BGE 改编；M100/N10 缓存缺参考答案；需恢复模型并重新评分 |
| 大模型mia针对句子生成.pdf | Membership Inference Attacks against Language Models via Neighbourhood Comparison | `external/Neighbour_MIA/` | 当前邻域构造是改编版；需与原版 embedding dropout 方法对齐 |
| 大模型mia针对模型输出概率波动.pdf | Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration；SPV-MIA | `external/SPV_MIA/` | 简化入口缺完整 target / self-prompt / reference 训练和校准链条 |
| dcmi差分校准成员攻击（有码）.pdf | DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation | `external/DCMI/` | 当前为词法代理；缺真实 RAG 原始及扰动输出信号、实际检索库成员标签 |

注意：本地目录名 `RAG_MIA` 对应的是 Riddle Me This 的 IA。该论文也比较了名为 RAG-MIA 的其他基线，不能把目录名当作方法正式名称。

## 其他 MIA、记忆与隐私攻击资料

| 文件名 | 论文 / 范围 | 与现有交接代码的关系 |
| --- | --- | --- |
| 大模型mia针对预训练数据.pdf | Do Membership Inference Attacks Work on Large Language Models? | 预训练数据 MIA 的系统评估与 benchmark；四入口交接包尚未覆盖其完整评估 |
| 2606.30081v1.pdf | Discard the Dross and Select the Essential: Pre-query Sample Selection for Black-box Membership Inference Attacks | 查询前样本选择；四入口包没有对应实现 |
| janus.pdf | The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks | 微调放大隐私风险；主仓库另有 `external/Janus/`，不属于四入口包 |
| 2510.19811v1.pdf | Hubble: a Model Suite to Advance the Study of LLM Memorization | 记忆、MIA、遗忘的模型与评估资源；不能等同于一个现有攻击入口 |
| 2411.14110v2.pdf | Feedback-Guided Extraction of Knowledge Base from Retrieval-Augmented LLM Applications | 知识库提取，CopyBreakRAG；不是上述四种成员推断方法 |
| 知识不对称攻击.pdf | Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation | RAG 隐私提取 |
| 良性查询攻击.pdf | Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries | RAG 隐式知识提取 |
| llp攻击.pdf | Loss Landscape Poisoning: Targeted Extraction of Unseen Training Data from LLMs | 投毒与定向数据提取 |

## 防御、遗忘与其他背景资料

| 文件名 | 正式论文 / 范围 |
| --- | --- |
| hamp防御.pdf | Overconfidence is a Dangerous Thing: Mitigating Membership Inference Attacks by Enforcing Less Confident Prediction |
| 查询防御mia.pdf | MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members |
| ragmia防御.pdf | Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks |
| mia防御dp.pdf | Differentially Private Fine-tuning of Language Models |
| 差分私有关键词防御.pdf | Differentially Private Retrieval-Augmented Generation |
| 加密防御.pdf | Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval |
| 隐私保护联邦学习（有码）.pdf | Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation |
| 检索增强生成中的隐私保护技术综述_刘晓迁.pdf | 检索增强生成中的隐私保护技术综述 |
| answer-token-only gradient ascent（unlearning）.pdf | Large Language Model Unlearning |
| Negative Preference Optimization From Catastrophic Collapse to Effective Unlearning.pdf | Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning |
| Simplicity Prevails Rethinking Negative Preference Optimization for LLM Unlearning.pdf | Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning |
| 投毒检索干扰攻击jamming（有码）.pdf | Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents；可用性攻击 |
| 2607.26099v1.pdf | Lilith: Backdoor Generalization under Training–Inference Trigger Shift；后门攻击 |
| 2602.00428v2.pdf | When Agents “Misremember” Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems |
| ICLR-2024-mechanistically-analyzing-the-effects-of-fine-tuning-on-procedurally-defined-tasks-Paper-Conference.pdf | Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks |

## 三份 Word 笔记

- `成员推断攻击.docx`：综述、DCMI / IA 论文总结及一次小规模 RAG 实验表。实验表是待对应原始文件的记录，不是已经核验的复现指标。
- `新建 Microsoft Word 文档.docx`：HAMP、MIAShield、Neighbour、SPV 等阅读总结。
- `janus关于lora与mia的相互影响.docx`：Janus / LoRA 与 MIA 结合的研究草稿，属于扩展工作；草稿数字需要独立核验，不能作为四篇论文的复现效果。

## 给老师的准确表述

“这是我整理的 MIA 实验代码，包含四个现有方法入口、固定版本的原作者代码来源、运行说明及逐篇待补项。
部分是改编或简化实验，目前还不能声明四篇论文均已完成效果复现。文件夹中其他论文的覆盖情况见对应清单。”
