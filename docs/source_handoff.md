# 源码交接与复现准备

## 本次收集范围

本次从研究服务器收集现有配置实际调用的脚本及其源码依赖，保存到
`external/`。模型权重、训练/评估数据、索引、缓存和实验输出未上传。
服务器原有工作区以及未提交的新实验未被修改。

| 方法 | 主要源码入口 | 当前实现说明 |
| --- | --- | --- |
| Janus / LoRA-Janus | `external/Janus/run_janus_minimal.py` | 本地 synthetic association 训练与恢复实验 |
| Janus held-out | `external/Janus/run_janus_lora_true_pipeline.py` | 预训练、持续训练、S1 微调与 S2 恢复；需另外准备数据和模型 |
| Janus 参数实验 | `external/Janus/run_janus_lora_sweep.py`、`run_janus_lora_boundary.py` | 参数扫描及边界实验 |
| Janus-MIA | `external/Janus/run_janus_mia_eval.py` | 本地 hard controls、loss scoring 与防御分析入口 |
| Janus-LoRA transfer | `external/Janus/run_janus_lora_cross_model_transfer.py` | 本地跨模型 adapter 转移实验 |
| SPV-MIA | `external/SPV_MIA/run_spv_minimal.py` | 本地简化的文本扰动与 loss 差实验；不等同于原论文完整 self-prompt/reference-model 流程 |
| Neighbour-MIA | `external/Neighbour_MIA/attack.py` | 本地 GPT-2/BERT 邻域比较适配版；两个 finetune 脚本也已收集 |
| DCMI | `external/DCMI/MIA.py` | 当前配置调用本地 NFCorpus 词法/扰动代理实验；不能据此声称完整 LLM 攻击复现 |
| RAG-MIA | `external/RAG_MIA/main_mia.py` | 上游代码及服务器修改版，含回答保存/计分修改与 Phi3 adapter |

上游 README、可用的 LICENSE、源码依赖和配置一并保留。每个文件的哈希、
上游地址及基准 commit 见 `external/source_manifest.json`。
`server_added` / `server_modified` 表示本地研究新增/修改，不是上游原版。
这份清单记录来源与完整性，不代表论文结论已经独立验证。

SPV 上游两个辅助脚本内嵌的带认证信息代理配置已移除，代理应由进程环境变量
提供。这两处归档修改及原服务器哈希在 manifest 的 `archive_changes` /
`server_sha256` 中单独记录。

## 环境

统一框架入口只需 Python 3.10+ 和 PyYAML：

```bash
python -m pip install -r requirements.txt
python scripts/restore_external_sources.py
python run_attack.py --help
python run_benchmark.py --help
```

模型实验另需对应的 PyTorch、Transformers、PEFT 等依赖；RAG 还需要检索器
及其依赖。服务器 `dcmi` 环境的实际版本记录在
`docs/server_runtime_versions.json`，它是环境记录，不是保证跨平台可安装的
lockfile。`external/*/requirements.txt` 和 RAG 的 `environment.yml` 是原项目
环境材料，其中 Janus 原始 `torch<2.0` 约束与本地新增实验所用环境不同，
不要把所有项目的 requirements 无差别安装进一个环境。

服务器 Phi3 还使用 `/root/autodl-tmp/phi3_compat_pkgs` 兼容依赖目录。
该目录是第三方安装包，不属于本次源码快照；迁移环境时需重新准备兼容环境。
本次检查发现，把该兼容目录全局加入 `PYTHONPATH` 会使当前 PEFT 导入失败
（`huggingface_hub.errors.LocalEntryNotFoundError` 缺失）。普通 `dcmi` 环境
下 Janus-MIA、transfer 和 RAG 入口导入正常；不要对 Janus 全局启用 Phi3
兼容目录。原 RAG 配置中的 `extra_pythonpath` 也需要结合独立兼容环境重新核对。

## 保持原服务器目录布局

现有 `configs/*.yml` 保留了原服务器绝对路径，`external/` 是源码归档目录，
不会自动改变 wrapper 的 `project_path`。在新机器按原路径部署时：

```bash
git clone https://github.com/ljh676/mia-.git /root/autodl-tmp/unified_mia_framework
cd /root/autodl-tmp/unified_mia_framework
python -m pip install -r requirements.txt
python scripts/restore_external_sources.py --target-root /root/autodl-tmp
```

恢复工具先验证全部哈希和语法，再检查全部目标文件；只复制缺失源码。
已有文件内容不同时会退出，不覆盖已有研究代码。原研究服务器已有这些项目，
无需再次恢复。自定义路径需要同时修改 YAML、脚本中的默认路径和 RAG 的模型
JSON；当前并未宣称做到任意目录的一键运行。

数据与模型仍需单独准备，例如：

- `/root/autodl-tmp/models/gpt2`、`bert-base-uncased`、`phi3-mini-4k-instruct`。
- Janus `data/` 下配置指定的 synthetic association JSONL，以及训练生成的模型/adapter。
- SPV 的 member/non-member JSONL。
- Neighbour 的成员/非成员文本、`membership_eval_scale.json` 和微调模型。
- RAG 的 NFCorpus 数据、成员划分、索引和生成问题等缓存。

生成或取得这些材料时，参照对应项目 README 和配置；本次源码补齐未恢复被排除
的数据资产，也未重新生成任何实验结论。

## 命令核对

运行攻击：

```bash
python run_attack.py --attack janus_mia --config configs/janus_mia.yml
```

运行一个 attack × defense 组合：

```bash
python run_benchmark.py --attack rag_mia --attack-config configs/rag_mia.yml --defense query_sanitization --defense-config configs/query_sanitization.yml
```

`run_benchmark.py` 不接受 `--config configs/benchmark_standard.yml`；该文件目前
只是实验设置摘要，不是可执行的组合调度配置。
`run_defense.py` 当前仅 prepare 并报告可用状态，不会读取输入并实际执行防御；
实际调用链在 benchmark 的 `defense.apply()`。

## 评估边界

本次主要补齐源码与交接信息，未改变攻击、防御或指标算法。现有 HAMP、MIAShield
包含 score-level 简化/模拟，PII filter 的恢复率统计、utility 代理指标和外部
进程失败后的结果读取仍需单独整改。不能把源码归档完成视为这些问题已经修复。
训练集成员、RAG 检索库成员以及 synthetic association 恢复是不同评估目标，
报告结果时应分开说明。

## 本次验证

- 200 个归档文件 SHA-256 校验通过，其中 160 个 Python 源文件在服务器
  Python 3.10 环境中通过语法解析。
- 统一框架三个 CLI，以及 Janus 六个入口、SPV 最小入口、Neighbour attack、
  DCMI 入口的 `--help` 检查通过。
- RAG `main_mia.py` 在普通 `dcmi` 环境中导入通过；没有加载模型或执行生成。
- DCMI `--mode real --sample-n 2` 在 CPU 上完成极小数据流程，输出留在服务器
  临时目录，未作为论文有效性或攻击效果验证。
- 恢复工具的首次复制、重复执行不重写，以及文件冲突时拒绝覆盖均已检查。

没有重新训练目标模型、运行完整 GPU 攻击或验证 README 的研究结论。
