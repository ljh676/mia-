# 四个已有 MIA 入口与论文对应

本目录按现有四个攻击入口组织论文、原作者版本和已有实验记录。
已对照桌面 `mia` 文件夹的 PDF 首页核实论文身份；该文件夹共有 27 篇 PDF 和 3 份 Word 笔记，
范围大于这四个入口，完整分类见[论文文件对应表](paper_folder_index.md)。
不能据此声称老师只要求这四篇，或已经覆盖文件夹中的全部 MIA 方法。
这是当前实际工作的交接状态，不将简化实验或代码收集写成论文复现完成。

| 论文 | 当前自己的入口 | 核对后的状态 | 详细说明 |
| --- | --- | --- | --- |
| Riddle Me This! / RAG-MIA | `external/RAG_MIA/main_mia.py` | 上游框架的 Phi3/BGE 改编；旧 200 条记录缺少 reference answers，配置的 Phi3 目录当前缺失 | [RAG-MIA](rag_mia/README.md) |
| Neighbourhood Comparison | `external/Neighbour_MIA/attack.py` | 有 200/200 本地实验及 target 训练记录；邻域生成改成 masked MLM 和混合扰动，未验证与原版一致 | [Neighbour-MIA](neighbour_mia/README.md) |
| SPV-MIA | `external/SPV_MIA/run_spv_minimal.py` | 只有 20/20 的单模型简化实验；未找到 self-prompt reference model 的训练/评估记录 | [SPV-MIA](spv_mia/README.md) |
| DCMI | `external/DCMI/MIA.py` | 100/100 的词法代理实验；尚非生成式 RAG 输出的 differential calibration 复现 | [DCMI](dcmi/README.md) |

表中的数量来自服务器已有结果的 schema 核查，不代表这些结果已通过论文一致性验证。
[证据摘要](server_evidence_summary.json)只包含文件哈希、数量及结构，不含样本文本。
上一轮的回归测试、单样本生成和小规模流程测试不能替代论文效果验证。
Word 笔记另记录了一次 10 篇文档、30 次问题的 RAG 小实验；其原始输出与配置尚未对应确认。
这份笔记与上述 M100/N10 缓存分别登记，不能由一个文件的问题推断全部既有实验均无效。

## 代码在哪里

- `external/`：服务器上实际使用的源码和本地修改。
- `attacks/`、`configs/`、`run_attack.py`：自己的统一调用与结果适配。
- 本目录的 `reference_entries/`：从固定上游 commit 取出的四个原作者关键文件，
  供直接比较；它们依赖各自完整原仓库，不是四个可独立执行的脚本。
- [reference_entries.json](reference_entries.json)：原作者 commit、文件路径和 SHA-256。
- [复现清单](manifest.json)：论文链接、源码地址、方法依赖和待补环节。

## 获取固定版本的原作者完整代码

```bash
python scripts/checkout_paper_source.py --method all --destination ./work/paper_upstreams
```

工具逐篇检出清单中的固定 commit，不执行模型训练、API 调用或数据下载。
已有目录会被保留；新检出的 SPV 上游辅助脚本中写死的代理设置会被移除，并记录为本地修改。
原作者仓库本身仍可能有数据、模型路径和占位内容，不能把 clone 成功当作实验成功。

## 独立导出给老师

```bash
python scripts/export_four_paper_code.py --output /path/to/mia_four_papers_code.zip
```

导出包包含四篇相关源码、统一攻击入口、原作者关键文件、依赖材料和交接说明。
不包含 Janus/LoRA-Janus、防御 benchmark、模型权重、数据或未经核验的效果表。
保留这些状态说明一起交接，避免将代理实验当作论文复现结果。

## 接下来的工作顺序

1. Neighbour：先恢复/实现原版 embedding-dropout 邻居选择，关闭额外混合扰动，
   保留同一目标模型和成员划分做对照。
2. RAG-MIA：恢复可用的目标模型、补齐 reference answers、核查检索库实际成员，
   再重新计分；严格评分现在会拒绝缺少答案的旧缓存。
3. SPV：按原作者顺序完成 target fine-tuning → self-prompt 数据生成 → reference
   fine-tuning → target/reference 概率变动校准。
4. DCMI：补齐原始/扰动查询与真实 RAG 输出信号的采集，使用实际检索库成员标签，
   再计算原始信号与扰动信号之差。

每篇以独立的配置、日志、模型/数据指纹和结果作交付，不混成一个已完成的结论。
