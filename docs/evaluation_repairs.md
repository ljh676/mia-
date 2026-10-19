# 评估逻辑修复

这些改动修复评估可信度问题，不代表完成 HAMP、MIAShield 等论文的完整复现。
以前的 JSON、图和 README 结论必须重新运行核对，不能直接沿用旧数值。

## 已修复

1. **MIAShield 标签泄漏**：排除决策不再读取真实成员标签，不修改传入的 details。
   缺少文本时，exact-signature 不再构造带标签的假文本，而是报告无匹配。
2. **HAMP 人工扰动**：移除依赖样本序号的噪声和 rank-mixing；旧非零配置会报错。
   HAMP、MIAShield 都明确返回 `implementation_kind: score_simulation`，benchmark
   标为 `simulation_only`，不再把分数变化称为已证明的防御有效性。
3. **PII 过滤**：实际过滤 synthetic 域名的输出并逐条重新匹配，支持 User_A、
   User_B 等邮箱。只保存少量 examples 或缺少完整生成文本时，全量恢复率为
   `null`；不会从原恢复数量编造过滤命中数量，也不修改 teacher-forced MIA 分数。
   Janus 新运行会保存完整 generation_records，展示样例仍只保留前五条。
4. **无效指标**：空样本、单类别 AUC 和缺少已校准预测的分类指标为 `null`，
   Markdown 显示 `N/A`。长度不一致、非二元标签或非有限分数会报错。
   Janus association recovery 与 MIA 分类指标分开；恢复率不再填入 accuracy。
5. **阈值和方向**：统一攻击入口不采用在评估标签上优化出来的预测。默认报告 AUC，
   accuracy/F1 为未校准；只有显式 `decision_threshold` 才报告分类指标。
   阈值必须由独立校准集或预先规定的规则确定。DCMI 移除了根据测试 AUC 翻转分数
   的逻辑，真实数据失败时也不再悄悄切换到 synthetic 模式。
6. **utility**：无 gold answer 时不再默认 Yes，无法解析的模型回答计错，不从
   分母中剔除。移除把攻击分数方差混入 utility 的综合分；未定义可靠组合指标时
   `utility_score` 为 `null`。检索目标文档命中率明确标为攻击相关诊断，不能替代
   一般任务效用。只在明确没有改变任务输出时继承 baseline，并标明没有重新测量。
7. **执行与结果来源**：外部命令失败或超时直接报错；成功返回后也必须验证结果文件
   是本次新建/更新的文件。禁止自动扫描旧文件或仅从 stdout 指标拼成成功结果。
   RAG 防御使用本次攻击的文档结果，前后样本身份/顺序不一致时停止比较。
8. **入口与报告**：`run_defense.py` 现在真正读取输入并执行 apply。
   `--check` 只返回 prepared，不能冒充已执行。benchmark 使用唯一 run_id，
   每次汇总仅包含本次结果，不混入目录里的历史实验。

## 使用方式

```bash
python -m unittest discover -s tests -v
python scripts/restore_external_sources.py
python run_defense.py --defense pii_output_filter --config configs/pii_output_filter.yml --input /path/to/result.json
python run_defense.py --defense hamp --config configs/hamp.yml --check
```

HAMP/MIAShield 默认不输出未经校准的 accuracy/F1。需要报告时，在各自配置中
加入从独立校准实验获得的 `decision_threshold`；不能再拿测试标签搜索最佳阈值。
当前仍是分数模拟，不会因此变成原论文防御。

PII 输入可以显式提供完整记录：

```json
{
  "eval_size": 1,
  "generation_records_complete": true,
  "generation_records": [
    {
      "identifier": "User_B001",
      "expected_synthetic_pii": "user_b001@example-synthetic.com",
      "generated": "The email is user_b001@example-synthetic.com"
    }
  ]
}
```

`eval_size` 必须是完整评估样本数；缺少记录时只能报告已观察到的过滤次数，不能
宣称全量恢复率归零。历史只有展示样例的结果，需要重新生成以补齐完整记录。

## 验证与边界

回归测试覆盖标签独立性、无输入修改、过滤真实命中/未命中/部分记录、缺少 gold、
无效回答、单类别与并列 AUC、失败/超时/旧输出、真实数据失败禁止 synthetic
回退，以及实际 CLI/benchmark 调用链。测试不训练模型、不需要 GPU。
`external/source_manifest.json` 同步记录了修改后的哈希，并用 `server_sha256`
保留原服务器版本的哈希。

已在本机和服务器上通过 30 个回归测试；100 组随机并列分数样本的指标与
scikit-learn 一致。服务器还完成每类 2 条的真实 NFCorpus DCMI 流程，以及
本地 GPT-2 不训练、单个 synthetic 样本的生成记录和过滤检查。

源码补齐和评估修复不等于研究结论复现：仍需重新运行模型实验，检查数据成员定义、
独立校准集、攻击者可见信息、统计不确定性，并对真正的防御重新测量任务效用。
归档脚本中的历史单独报表可能仍包含探索性阈值结果；统一入口已排除这些预测，
正式汇报应使用 schema_version=2 的 benchmark 输出。
