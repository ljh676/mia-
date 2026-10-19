# SPV-MIA

论文：[Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration](https://arxiv.org/abs/2311.06062)。
原作者：[tsinghua-fib-lab/NeurIPS2024_SPV-MIA](https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA)，
固定版本 `ee9ebaad837c3798ad9f2c920b01868d9e62478d`。

## 当前自己的代码

`external/SPV_MIA/run_spv_minimal.py` 比较单个 GPT-2 上原文与人工扰动文本的 loss，
已有 20/20 的结果。它没有完整 self-prompt reference model 校准链，
应称为简化扰动实验，不能把该入口产生的结果当作 SPV 论文复现。

当前统一命令仍用于这个简化入口：

```bash
python run_attack.py --attack spv_mia --config configs/spv_mia.yml
```

## 正式方法所需的代码

归档中也保留了原作者的完整方法组件：

- `ft_llms/llms_finetune.py`：target 与 reference 模型微调。
- `ft_llms/refer_data_generate.py`：对 target 做 self-prompt 生成参考数据。
- `attack/attack_model.py`：target/reference 的原始与扰动 loss 计算及校准。
- `data/prepare.py`、`configs/config.yaml`、`attack.py`：数据准备、配置和正式攻击入口。

原作者主入口对照为 `../reference_entries/spv_mia_original.py`。
两个辅助脚本中原先写死的代理认证配置已移除；代理由进程环境变量设置。

## 需要补齐的记录

target checkpoint、self-prompt 生成数据、reference checkpoint，以及各自训练/验证
划分、训练参数和指纹。当前 `configs/config.yaml` 仍包含原作者机器路径，需要
替换为本次实验真实资产。不能拿同一个未经微调的 GPT-2 同时充当两个角色。

顺序为 target fine-tuning → self-prompt reference dataset → reference fine-tuning
→ 正式 `attack.py`。每一步分别保存命令、日志、输入输出指纹；依赖版本参照原项目
`requirements.txt`，不要直接与其他三篇混装。
