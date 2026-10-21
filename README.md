# Unified Janus-MIA Framework

## 本轮 MIA 实验结果（2026-10-22 更新）

四种方法各自当前配置的实验已结束，代码和轻量结果可作为阶段交付。
**尚未完成四篇论文的严格完整复现验证。**

| 方法 | AUC | 测试准确率 | 可评分/配置（含校准与测试） |
| --- | ---: | ---: | ---: |
| Neighbour | 0.5343 | 52.50% | 2400/2400 |
| SPV | 0.8657 | 81.20% | 2400/2400 |
| IA | 0.9298 | 85.98% | 2378/2400 |
| DCMI | 0.8173 | 75.72% | 3281/4000 |

完整测试分母、低误报指标、模型替换、样本排除、DCMI 编辑预算及实际配置见
[本轮结果与交付说明](reproductions/results/2026-10-21/README.md)。
这些是不同设置的结果，不是统一受控 benchmark；IA/DCMI 的指标只针对可评分子集。
历史简化入口与本轮正式实验的状态分别保留，请以本轮报告为当前交付依据。

**正式复现的新入口：[`reproductions/`](reproductions/README.md)。** 包含真实目标训练、
SPV self-prompt/reference 训练、Neighbour embedding dropout、IA 与 DCMI 的真实 RAG 信号采集。
历史 `run_attack.py` 配置保留原实验用途；本轮正式实验请使用 `python -m reproductions.run`。
实验是否完成以运行目录实际 `complete.json` 和逐样本证据为准；IA/DCMI 的生成模型替换已明确标注。

**给老师交接 RAG-MIA、Neighbour-MIA、SPV-MIA、DCMI 代码，请先看
[四篇 MIA 交接入口](paper_reproductions/README.md)。** 其中按论文逐篇标明实际代码、
原作者版本、已运行记录与尚未完成的环节，避免把简化实验作为完整论文复现。

## 项目简介

Unified Janus-MIA Framework 是一个面向 LLM privacy attack / defense / benchmark 的统一实验框架，用于组织和运行隐私攻击评估、防御方法验证、attack × defense benchmark，以及 privacy-utility tradeoff 分析。

框架支持统一入口：

```bash
python run_attack.py --attack <attack_name> --config <config.yml>
python run_defense.py --defense <defense_name> --config <config.yml>
python run_benchmark.py --attack <attack_name> --attack-config <attack_config.yml> --defense <defense_name> --defense-config <defense_config.yml>
```

## 源码交接

现已将服务器上的 Janus、SPV-MIA、DCMI、Neighbour-MIA、RAG-MIA 实际脚本及
源码依赖收集到 [`external/`](external/README.md)，并记录上游版本与文件哈希。
数据、模型及实验输出需单独准备；现有配置仍使用原服务器路径。
部署步骤、依赖、实现状态与评估限制见[源码交接说明](docs/source_handoff.md)。

```bash
python -m pip install -r requirements.txt
python scripts/restore_external_sources.py
```

## 评估修复

标签泄漏、过滤统计、无效指标和旧结果误读等问题已修复，变更与运行方式见
[评估修复说明](docs/evaluation_repairs.md)。HAMP/MIAShield 当前仍为明确标注的
分数模拟；完整论文复现与历史结论需要重新验证。

```bash
python -m unittest discover -s tests -v
```

## 已接入攻击方法

- `rag_mia`
- `neighbour_mia`
- `dcmi`
- `spv_mia`
- `janus`
- `janus_mia`
- `janus_lora_transfer`

## 已接入 Defense

- `query_sanitization`
- `epd`
- `hamp`
- `miashield`
- `pii_output_filter`

## Janus-LoRA 主线

当前 Janus-LoRA 研究主线聚焦于 true held-out recovery，不再做旧的训练集记忆实验。核心流程是：

1. `synthetic PII pretraining`
2. `general text continual training`
3. `S1 known-pair Janus tuning`
4. `S2 hidden held-out recovery`
5. `LoRA-Janus hidden memorization activation`

实验目标是研究：只用 S1 known synthetic pairs 进行 Janus/LoRA tuning，是否可以激活模型中已经被遗忘或隐藏的 S2 held-out synthetic associations。

## 历史实验记录（修复后需重新验证）

以下为旧实验记录中的结论，尚未用修复后的完整流程重新验证，不应直接作为
当前版本已经证明的研究结论。

- `LoRA-Janus can recover hidden S2 associations`：stronger LoRA-Janus 可以在 true held-out setting 下恢复 hidden S2 synthetic associations。
- `hard non-member controls show stronger membership distinguishability`：在 shuffled_A、near_neighbor 等 harder non-member controls 下，LoRA-Janus hidden recovery 会增强 membership distinguishability，主要体现为 member loss 降低、loss gap 和 calibrated gap 扩大。
- `pii_output_filter blocks direct output leakage but not teacher-forced score-based MIA signal`：`pii_output_filter` 可以阻断直接生成输出中的 synthetic email 泄漏，但不会消除 teacher-forced / score-based MIA signal。

## 运行示例

运行攻击：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py   --attack janus_mia   --config ./configs/janus_mia.yml
```

运行防御：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_defense.py   --defense pii_output_filter   --config ./configs/pii_output_filter.yml
```

运行 benchmark：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py --attack rag_mia --attack-config ./configs/rag_mia.yml --defense query_sanitization --defense-config ./configs/query_sanitization.yml
```

## 项目结构

```text
unified_mia_framework/
├── attacks/
│   ├── rag_mia.py
│   ├── neighbour_mia.py
│   ├── dcmi.py
│   ├── spv_mia.py
│   ├── janus.py
│   ├── janus_mia.py
│   └── janus_lora_transfer.py
├── defenses/
│   ├── query_sanitization.py
│   ├── epd.py
│   ├── hamp.py
│   ├── miashield.py
│   └── pii_output_filter.py
├── configs/
├── docs/
├── external/  # 原项目源码与本地实验脚本快照
├── scripts/   # 源码校验与目录恢复工具
├── utils/
├── run_attack.py
├── run_defense.py
└── run_benchmark.py
```

## 数据与模型说明

Janus/LoRA-Janus 的合成标识符实验使用以下约定；四种 MIA 本轮实验另外使用公开的
AG News、NFCorpus、BEIR TREC-COVID 数据，详见本轮结果和配置。

- 只使用 synthetic identifiers 和 `example-synthetic.com` synthetic emails。
- 不使用真实姓名、真实邮箱、真实电话或真实地址。
- `datasets/`、`data/`、`models/`、`outputs/` 等目录不上传 GitHub。

## GitHub 上传注意事项

`.gitignore` 已配置忽略大文件、模型、数据和实验输出，包括：

- `datasets/`
- `data/`
- `outputs/`
- `logs/`
- `models/`
- `checkpoints/`
- `*.pt`
- `*.pth`
- `*.bin`
- `*.safetensors`
- `.cache/`

如果需要上传到 GitHub，请只提交框架代码、配置文件、文档和轻量级脚本。
