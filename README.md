# Unified Janus-MIA Framework

## 项目简介

Unified Janus-MIA Framework 是一个面向 LLM privacy attack / defense / benchmark 的统一实验框架，用于组织和运行隐私攻击评估、防御方法验证、attack × defense benchmark，以及 privacy-utility tradeoff 分析。

框架支持统一入口：

```bash
python run_attack.py --attack <attack_name> --config <config.yml>
python run_defense.py --defense <defense_name> --config <config.yml>
python run_benchmark.py --config <benchmark_config.yml>
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

## 关键实验结论

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
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py   --config ./configs/benchmark_standard.yml
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
├── utils/
├── run_attack.py
├── run_defense.py
└── run_benchmark.py
```

## 数据与模型说明

本项目实验必须保持 synthetic-safe：

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
