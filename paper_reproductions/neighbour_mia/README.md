# Neighbourhood Comparison

论文：[Membership Inference Attacks against Language Models via Neighbourhood Comparison](https://aclanthology.org/2023.findings-acl.719/)。
原作者：[justusmattern27/neighbour-mia](https://github.com/justusmattern27/neighbour-mia)，
固定版本 `c73e29960206ad65798a67bb853065721a402d2f`。

## 当前自己的代码

- `external/Neighbour_MIA/attack.py`：本地 GPT-2/BERT 攻击适配。
- `finetune_gpt2_membership.py`、`finetune_gpt2_membership_scaled.py`：本地目标模型训练。
- `attacks/neighbour_mia.py`、`configs/neighbour_mia.yml`：统一调用和结果解析。

```bash
python run_attack.py --attack neighbour_mia --config configs/neighbour_mia.yml
```

服务器存在 200/200 分数及目标模型训练记录，训练元信息记录 200 条样本、5 个 epoch，
使用 NFCorpus。它是已运行的本地改编实验，尚未验证论文数据/模型设置和结果一致性。

## 和原版的差异

原作者代码对 token embedding 做 dropout，用候选替换概率相对原 token 概率排序，
再比较原文本和邻居在目标模型上的分数。
当前本地代码对 token 做 `[MASK]` 替换，并扩展 synonym、phrase、deletion、sentence
扰动；直接拿这些结果不能声称复现了原版的邻居生成过程。

原版 `attack.py` 存于 `../reference_entries/neighbour_mia_original.py`，它还包含
目标模型路径占位、双 GPU 假设以及 `one_word_neighbours` 未定义等运行问题。
应以保留邻居算法的最小工程修复替代整个算法重写。

## 下一轮验证

先记录训练成员/非成员划分与模型指纹，再恢复原版 dropout 邻居生成、候选排序与
邻居数量设置。使用同一目标模型比较“原版算法适配”和“本地混合扰动”，分别报告。
验证 member 确实参与 target fine-tuning，non-member 未参与；不能仅按文件顺序
或名字推定成员身份。
