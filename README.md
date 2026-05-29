# unified_mia_framework

统一成员推断攻击实验平台，用于把不同论文攻击方法封装到同一入口。

当前支持：

- `rag_mia`：调用 `/root/autodl-tmp/RAG_MIA`
- `dcmi`：调用 `/root/autodl-tmp/DCMI`

## 运行

```bash
cd /root/autodl-tmp/unified_mia_framework

/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack dcmi --config ./configs/dcmi.yml
```

如果当前 shell 的 `python` 指向同一环境，也可以：

```bash
python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
python run_attack.py --attack dcmi --config ./configs/dcmi.yml
```

## 真实接入状态

配置文件中已经设置：

```yaml
execute_original: true
```

因此运行统一入口时，会真实启动原论文脚本，并把原始输出保存到：

```text
outputs/logs/rag_mia.log
outputs/logs/dcmi.log
```

为了避免超大实验，`rag_mia.yml` 默认生成最小测试配置：

```yaml
sample_num: 1
question_num: 1
top_k: 1
evaluate_attack: false
```

## 输出格式

所有攻击最终统一为：

```python
{
    "attack_name": str,
    "scores": list,
    "predictions": list,
    "labels": list,
    "metrics": {
        "accuracy": float,
        "auc": float,
        "precision": float,
        "recall": float,
        "f1": float,
    },
}
```

## 项目分析

原论文入口、参数和输出分析见：

```text
docs/project_analysis.md
```

## 添加新攻击方法

1. 新建 `attacks/mimir.py`，继承 `BaseAttack`。
2. 在 `attacks/registry.py` 注册：

```python
ATTACK_REGISTRY["mimir"] = MIMIRAttack
```

3. 新建 `configs/mimir.yml`。
4. 运行：

```bash
python run_attack.py --attack mimir --config ./configs/mimir.yml
```

## 当前验证状态

- `dcmi`：统一入口会真实启动 `/root/autodl-tmp/DCMI/MIA.py`，并把 traceback 写入 `outputs/logs/dcmi.log`。由于当前原脚本中的分数数组仍是 `[...]` 占位，脚本本身会报错；统一框架会解析仓库已有的 `outputs/min_dcmi_nfcorpus_scores.json` 生成统一结果。
- `rag_mia`：统一入口已经真实启动 `/root/autodl-tmp/RAG_MIA/main_mia.py`，并成功进入配置解析和数据加载阶段。当前阻塞是原项目缺少 `datasets/nfcorpus/corpus.jsonl` 或 `corpus.json`。

## NFCorpus 数据修复记录

已自动下载 BEIR NFCorpus，并补齐：

- `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.jsonl`
- `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/queries.jsonl`
- `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/qrels/`
- `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.json`
- `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/target_data_with_questions.json`

当前 RAG_MIA 已进入真实检索阶段，最新阻塞是缺少 ColBERT 预构建索引：

```text
datasets/nfcorpus/colbert/indexes/nfcorpus-index/metadata.json
datasets/nfcorpus/colbert/indexes/nfcorpus-index/plan.json
```

原项目 `generate_index.py` 对 ColBERT 标注了 `TODO: Handle ColBERT`，因此需要提供预构建 ColBERT index，或改用其他 retriever 构建 FAISS index。

## RAG_MIA 最小真实流程已跑通

当前不再使用 `colbert`，而是使用 `bge` + FAISS index：

```text
/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/bge/indexes/nfcorpus-index/corpus_index.faiss
/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/bge/indexes/nfcorpus-index/doc_ids.pkl
```

运行：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
```

已验证 `returncode=0`，并生成统一格式结果：

```text
outputs/rag_mia_result.json
outputs/logs/rag_mia.log
```


## RAG_MIA Phi3 完整最小攻击

当前已可运行：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
```

该配置使用 `nfcorpus + bge + phi3 + evaluate_attack=true`，会调用 RAG_MIA 原项目 `main_mia.py`，并输出统一结果到 `outputs/rag_mia_result.json`，日志保存到 `outputs/logs/rag_mia.log`。


## ?? Neighbour_MIA

Neighbour_MIA ???? `neighbour_mia`??????????? `/root/autodl-tmp/Neighbour_MIA/attack.py` ??????????BERT MLM ?? one-word replacement neighbours?GPT-2 ?????????? log probability?????????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack neighbour_mia --config ./configs/neighbour_mia.yml
```

?????

- ???`configs/neighbour_mia.yml`
- ???`outputs/logs/neighbour_mia.log`
- ?????`outputs/neighbour_mia_result.json`
- ??? pickle?`/root/autodl-tmp/Neighbour_MIA/all_scores_sample.pkl`
- ?? attack.py ???`/root/autodl-tmp/Neighbour_MIA/attack_original.py`

?????????????? 5 ??????????????????????????????? attack model checkpoint?


## ?? DCMI

DCMI ???? `dcmi`??????????? `/root/autodl-tmp/DCMI/MIA.py` ????????????????? differential calibration score ????????????? perturb ???????????? membership prediction?

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack dcmi --config ./configs/dcmi.yml
```

?????

- ???`configs/dcmi.yml`
- ???????`/root/autodl-tmp/DCMI/MIA_original.py`
- ??????`/root/autodl-tmp/DCMI/outputs/dcmi_result.json`
- ?????`outputs/dcmi_result.json`
- ???`outputs/logs/dcmi.log`

????? `MIA.py` ? `mem/nonmem/perturb_*` ? Ellipsis ????????????????????????????????????????????????????????????


## Neighbour_MIA ??? Membership Evaluation

?????????????????? 50/50 ???? membership evaluation??? nfcorpus ??????? member/non-member??? member ?? fine-tune GPT-2????? neighbour-based score ???????

```bash
cd /root/autodl-tmp/Neighbour_MIA
/root/miniconda3/envs/dcmi/bin/python attack.py --model bert --dataset sample --output-file /root/autodl-tmp/Neighbour_MIA/all_scores_membership_eval.pkl
```

?????

- fine-tuned model?`/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2`
- detailed log?`/root/autodl-tmp/unified_mia_framework/outputs/logs/neighbour_mia_eval.log`
- unified result?`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_result.json`
- summary?`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_eval_summary.json`

?????member ?? membership score `1.3855`?non-member ?? membership score `1.0404`?member ?? loss `0.5925`?non-member ?? loss `4.3614`?AUC `0.6716`???????????????? fine-tuned GPT-2 ???membership score ???????


## Neighbour_MIA 100/100 Membership Evaluation

???????? wrapper ????????? fine-tuned GPT-2 target model ????? membership evaluation?

- ???????? `RAG_MIA/datasets/nfcorpus/corpus.jsonl` ?????????/?????
- ?????200 member + 200 non-member?
- ?????100 member + 100 non-member???? nfcorpus ?????????????????????? 400 ?????????????? 100/100?
- member ???`/root/autodl-tmp/Neighbour_MIA/data/train_members_100.txt`
- non-member ???`/root/autodl-tmp/Neighbour_MIA/data/test_nonmembers_100.txt`
- ?????`/root/autodl-tmp/models/gpt2`
- fine-tuned target model?`/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2_100`
- fine-tune ?????? member ???3 epoch?batch size 2?
- evaluation ????????? 25 ? one-word replacement neighbours?replacement ??? 1?
- ?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack neighbour_mia --config ./configs/neighbour_mia.yml
```

?????

- summary JSON?`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_eval_summary.json`
- unified result?`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_result.json`
- detailed log?`/root/autodl-tmp/unified_mia_framework/outputs/logs/neighbour_mia.log`
- score distribution?`/root/autodl-tmp/unified_mia_framework/outputs/figures/neighbour_score_distribution.png`
- ROC curve?`/root/autodl-tmp/unified_mia_framework/outputs/figures/neighbour_roc_curve.png`

?????

- accuracy?0.63
- AUC?0.6894
- precision?0.63
- recall?0.63
- F1?0.63
- member mean loss?1.9392
- non-member mean loss?3.4086
- member mean membership score?0.9269
- non-member mean membership score?0.6744

???fine-tuned GPT-2 ? member ??? loss ???? non-member?membership score ????????????? membership signal???????????????????????????? membership evaluation?


## SPV-MIA Privacy Evaluation Benchmark

SPV-MIA ??? privacy evaluation ???? unified_mia_framework????? `spv_mia`?

??????`/root/autodl-tmp/SPV_MIA`

????????

- ????`/root/autodl-tmp/SPV_MIA/attack.py`????? `configs/config.yaml`?
- ?????YAML????? `model_name`?`target_model`?`reference_model`?`dataset_name`?`calibration`?`attack_kind`?`maximum_samples`?`mask_filling_model_name`?`perturbation_number`?`sample_number` ??
- ?????????? HuggingFace `datasets.load_dataset(...)` ?? `ag_news`?`wikitext`?`xsum` ??????? train/valid ?? member/non-member ???
- ?????`model_name` ??????`target_model` ? fine-tuned target model?`reference_model` ? self-prompt reference model?
- ?????????? feature npz ? ROC npz ? `attack_data_path` ???????? target/reference ???T5 mask-filling model ???????
- self-prompt reference ?????`ft_llms/refer_data_generate.py`?`ft_llms/llms_finetune.py`?`attack/attack_model.py`?
- probabilistic variation / perturbation ?????`attack/attack_model.py` ?? `eval_perturb`?`sentence_perturbation`?`feat_prepare`?
- GPT-2 ???????????????? GPT ????????????? reference model ? mask filling???????????????????????

?????????

- ?????`/root/autodl-tmp/SPV_MIA/run_spv_minimal.py`
- ???`/root/autodl-tmp/SPV_MIA/configs/spv_minimal.yml`
- ???`/root/autodl-tmp/models/gpt2`
- ???`/root/autodl-tmp/SPV_MIA/data/sample_members.jsonl` ? `sample_nonmembers.jsonl`?? 20 ??
- ???`probabilistic_variation`?????????? paraphrase / perturbation??? `perturbed_loss - original_loss` ?? membership score?
- ??????`/root/autodl-tmp/SPV_MIA/outputs/spv_mia_result.json`
- ???????`/root/autodl-tmp/unified_mia_framework/outputs/spv_mia_result.json`
- ???`/root/autodl-tmp/unified_mia_framework/outputs/logs/spv_mia.log`

???????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack spv_mia --config ./configs/spv_mia.yml
```

?????????`accuracy=0.55`?`auc=0.5875`?`precision=0.55`?`recall=0.55`?`f1=0.55`?

?????????????????? GPT-2 ???? privacy evaluation????????????? self-prompt reference model?mask-filling perturbation?target/reference fine-tuning ???????


## RAG_MIA M=20,N=5 Answers Compatibility Fix

RAG_MIA ????? `M=20, N=5` ???

- ?????`/root/autodl-tmp/unified_mia_framework/configs/rag_mia.yml`
- target_docs?`/root/autodl-tmp/RAG_MIA/results/target_docs/mia-nfcorpus-phi3-phi3-bge-R3-Top1-M20-N5.json`
- ???????`/root/autodl-tmp/RAG_MIA/mia_utils/mia_original_before_answers_fix.py`
- ?????`/root/autodl-tmp/RAG_MIA/mia_utils/mia.py`

?????

- total records?40
- member?20
- non-member?20
- records with `answers`?0
- records missing `answers`?40
- question count distribution?{'15': 40}
- llm response count distribution?{'15': 40}

????`calculate_score()` ?? `target_docs` ???????? `answers` ???M=20,N=5 ??????? `questions`?`retrieved_doc_ids`?`llm_responses`?`mem`???? `answers`??? `KeyError: 'answers'`?

?????`calculate_score()` ??????? `answers` / `answer` / `gold_answers` / `generated_answers` / `question_answer_pairs` / `questions[].answer`??????? ground-truth answers???? retrieval-hit fallback?? `retrieved_doc_ids[i]` ???? `doc_id` ??? Yes????? No????? warning?fallback_count ? skipped_count????????

???????

- returncode?0
- scores?40 ?
- accuracy?0.8
- AUC?1.0
- precision?1.0
- recall?0.6
- F1?0.75

?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
```


## MIAShield Defense Benchmark

Defense ????? unified_mia_framework?

- `defenses/base.py`??? `BaseDefense` ???
- `defenses/registry.py`?defense ????
- `defenses/miashield.py`?text-adapted minimal MIAShield?
- `run_defense.py`????? defense ??????
- `run_benchmark.py`??? membership evaluation ? defense benchmark?

MIAShield ????`/root/autodl-tmp/MIAShield`

??????

- ??? notebook?`Evaluation_of_MIAShield_(Cifar10_and_100).ipynb` ? `Cifar10_and_Cifar100_Related_Work.ipynb`?
- README ?? CIFAR-10 / CIFAR-100?EO2-ESE?EO3-ASE?EO4-MCE?EO5-COE??? threshold / logistic regression / MLP membership evaluation?
- ???????????CIFAR?ensemble models ? exclusion oracle?????????????
- ensemble model ???? notebook ?? StratifiedKFold ??????????????
- exclusion oracle ?? hash / imagehash / classifier confidence / chain ????

??????? text-adapted minimal MIAShield?

- `exact_signature`???? hash ?? image hash?
- `confidence_based`???? candidate scoring source ???? confident ? source?
- `chain`?? exact signature????? fallback ? confidence-based?
- ????? LLM?????? deterministic scoring source ?? ensemble?????? attack ? defense benchmark ???

?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_defense.py --defense miashield --config ./configs/miashield.yml
```

Benchmark ???

```bash
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py   --attack neighbour_mia   --attack-config ./configs/neighbour_mia.yml   --defense miashield   --defense-config ./configs/miashield.yml
```

Neighbour_MIA ? MIAShield ???`outputs/benchmark_neighbour_mia_miashield_result.json`

- before?accuracy=0.63?AUC=0.6894
- after?accuracy=0.64?AUC=0.6969
- AUC drop=-0.0075??? minimal defense ??? Neighbour_MIA ? signal?effectiveness ?? 0?

SPV_MIA ? MIAShield ???`outputs/benchmark_spv_mia_miashield_result.json`

- before?accuracy=0.55?AUC=0.5875
- after?accuracy=0.55?AUC=0.53
- AUC drop=0.0575

?????????? attack ? defense benchmark ????????? MIAShield ????????????????? HAMP?EPD?query_sanitization?paraphrase_defense??? MIAShield ???????? ensemble ???


## HAMP Defense ????

HAMP ?????? privacy defense ??????????? `hamp`?
?????? `defenses/hamp.py`?????? `configs/hamp.yml`?

????? text-adapted minimal HAMP??? LLM/text membership scoring benchmark?

- temperature smoothing?? membership score ????????????????
- entropy boosting???????????????????
- confidence clipping????? confidence??? `max_confidence=0.8`?

??????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_defense.py --defense hamp --config ./configs/hamp.yml
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack neighbour_mia \
  --attack-config ./configs/neighbour_mia.yml \
  --defense hamp \
  --defense-config ./configs/hamp.yml
```

???? benchmark ???

- `outputs/hamp_result.json`
- `outputs/benchmark_neighbour_mia_hamp_result.json`
- `outputs/benchmark_spv_mia_hamp_result.json`
- `outputs/logs/benchmark_neighbour_mia_hamp.log`
- `outputs/logs/benchmark_spv_mia_hamp.log`

????? HAMP ????? membership scores ?????????????????????????????????????? CIFAR / Purchase / Texas / Location ???????????????????????????????????


## Privacy-Utility Tradeoff Evaluation

?????????? membership evaluation ? privacy defense benchmark???? utility evaluation????? privacy-utility tradeoff?

?????

- `utils/utility_metrics.py`

????? utility ???

- `retrieval_hit_rate()`??? RAG_MIA???????????? retrieved docs ??
- `qa_accuracy()`??? Yes / No / I don't know ???????????? expected answer?
- `response_length()`?????????????????????
- `confidence_statistics()`??? Neighbour_MIA?DCMI?SPV_MIA ? score-based ????? mean/std/min/max score?
- `utility_summary()`????? benchmark ??? utility ???

`run_benchmark.py` ?????? before/after privacy metrics ????????

- `utility_before`
- `utility_after`
- `utility_drop`
- `retrieval_hit_rate_before`
- `retrieval_hit_rate_after`
- `qa_accuracy_before`
- `qa_accuracy_after`

???

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack rag_mia \
  --attack-config ./configs/rag_mia.yml \
  --defense miashield \
  --defense-config ./configs/miashield.yml

/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack spv_mia \
  --attack-config ./configs/spv_mia.yml \
  --defense hamp \
  --defense-config ./configs/hamp.yml
```

????????

- `outputs/benchmark_summary.json`
- `outputs/benchmark_summary.md`

?????????

| Attack | Defense | AUC Before | AUC After | Utility Before | Utility After | QA Before | QA After | Retrieval Before | Retrieval After |
|--------|---------|------------|-----------|----------------|---------------|-----------|----------|------------------|-----------------|
| rag_mia | miashield | 1.0000 | 1.0000 | 0.4827 | 0.4827 | 0.4083 | 0.4083 | 0.2983 | 0.2983 |
| spv_mia | hamp | 0.5875 | 0.5875 | 0.2000 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

???MIAShield/HAMP ??? text-adapted minimal defense?????? membership scores????? RAG retrieval ? LLM responses?????? benchmark ??RAG retrieval hit rate ? QA accuracy ? before/after ?????score-level utility ??? defense ? confidence ?????


## Query Sanitization Defense

`query_sanitization` ???? input-level privacy defense????? `query_sanitization`??? MIAShield / HAMP ? score ??????????? RAG ???

- query sanitization
- retrieval context sanitization
- ????? `[PERSON]`
- ????? `[EMAIL]`
- ??????ID??????? `[NUMBER]`
- ???????? `medical_condition`

?????

- `configs/query_sanitization.yml`

?? defense?

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_defense.py \
  --defense query_sanitization \
  --config ./configs/query_sanitization.yml
```

?? RAG_MIA ? query_sanitization benchmark?

```bash
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack rag_mia \
  --attack-config ./configs/rag_mia.yml \
  --defense query_sanitization \
  --defense-config ./configs/query_sanitization.yml
```

?????

- `outputs/benchmark_rag_mia_query_sanitization_result.json`
- `outputs/logs/benchmark_rag_mia_query_sanitization.log`
- `outputs/query_sanitization_tradeoff.json`
- `outputs/query_sanitization_report.md`

???????

- AUC: `1.0000 -> 0.9400`
- Retrieval hit rate: `0.2983 -> 0.2362`
- QA accuracy: `0.4083 -> 0.3387`
- Utility score: `0.4827 -> 0.4300`

????? input-level defense ???????? privacy gain + utility loss??? RAG privacy-utility tradeoff ???

????????????? sanitized query/context???? sanitization ???? retrieval?QA ? membership score ??????? defense ??????? retriever ? LLM answer generation?????????

- paraphrase_defense
- retrieval_filter
- entity_relation_defense


## Query Sanitization Real Rerun

`query_sanitization` ?????????

- `real_rerun: false`???????????????
- `real_rerun: true`?????????? sanitized questions ???? BGE retrieval??? sanitized context + question ?? Phi3 ?????

?????

```yaml
real_rerun: true
extra_pythonpath: /root/autodl-tmp/phi3_compat_pkgs
```

?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack rag_mia \
  --attack-config ./configs/rag_mia.yml \
  --defense query_sanitization \
  --defense-config ./configs/query_sanitization.yml
```

Real rerun ???

- `outputs/benchmark_rag_mia_query_sanitization_realrerun_result.json`
- `outputs/logs/benchmark_rag_mia_query_sanitization_realrerun.log`
- `outputs/query_sanitization_realrerun_target_docs.json`
- `outputs/query_sanitization_tradeoff.json`
- `outputs/query_sanitization_report.md`

?? real rerun ???

- AUC: `1.0000 -> 0.9450`
- AUC drop: `0.0550`
- Retrieval hit rate: `0.2983 -> 0.2850`
- QA accuracy: `0.4083 -> 0.2900`
- Utility score: `0.4827 -> 0.4300`
- Real rerun queries: `600`

???????? query?sanitized query??? retrieved docs?sanitized retrieved docs?Phi3 before/after response?AUC before/after?retrieval hit rate before/after?QA accuracy before/after?utility drop?


## Enhanced Query Sanitization Levels

`query_sanitization` now supports three sanitization levels:

- `low`: PERSON and EMAIL masking only.
- `medium`: adds NUMBER masking and medical keyword abstraction.
- `high`: adds semantic abstraction, retrieval-aware keyword dropping, and entity relation abstraction.

High-level mode is enabled in `configs/query_sanitization.yml`:

```yaml
sanitization_level: high
real_rerun: true
retrieval_keyword_drop: true
semantic_abstraction: true
entity_relation_abstraction: true
```

Enhanced membership scoring in real rerun mode now combines:

```text
final_score =
  0.4 * retrieval_overlap
+ 0.3 * response_similarity
+ 0.2 * confidence
+ 0.1 * exact_match_penalty
```

Additional utility metrics:

- `semantic_similarity`
- `retrieval_diversity`
- `response_consistency`

Current low/medium/high results on RAG_MIA M=20/N=5:

| Level | AUC Before | AUC After | AUC Drop | QA Before | QA After | Retrieval Before | Retrieval After | Utility Before | Utility After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 1.0000 | 1.0000 | 0.0000 | 0.4083 | 0.3633 | 0.2983 | 0.2933 | 0.4827 | 0.5236 |
| medium | 1.0000 | 0.9575 | 0.0425 | 0.4083 | 0.2900 | 0.2983 | 0.2850 | 0.4827 | 0.4819 |
| high | 1.0000 | 0.8438 | 0.1562 | 0.4083 | 0.2300 | 0.2983 | 0.1933 | 0.4827 | 0.3908 |

Generated reports:

- `outputs/query_sanitization_highlevel_report.md`
- `outputs/defense_comparison.md`
- `outputs/benchmark_rag_mia_query_sanitization_low_result.json`
- `outputs/benchmark_rag_mia_query_sanitization_medium_result.json`
- `outputs/benchmark_rag_mia_query_sanitization_high_result.json`



## Stage Effectiveness Upgrade

???????????????attack / defense ????????????

- RAG_MIA ???? M=100, N=10?200 docs / 2000 questions?AUC=0.9950?
- Neighbour_MIA ???? 200/200?fine-tuned GPT-2 5 epoch?50 neighbours?AUC=0.6910?
- DCMI ?? synthetic placeholder ?? NFCorpus query/doc ???????AUC=0.6028?
- query_sanitization balanced high?RAG_MIA AUC 0.9950 -> 0.8758?QA 0.4360 -> 0.3255?
- HAMP tuned?Neighbour_MIA AUC 0.6910 -> 0.5995?

?????

- `outputs/final_benchmark_table.md`
- `outputs/stage_report.md`
- `outputs/neighbour_mia_detailed_analysis.json`


## EPD Defense Integration

EPD ??? RAG ?? inference-time ensemble defense ???`defenses/epd.py`?`configs/epd.yml`?`run_defense.py --defense epd`?`run_benchmark.py --attack rag_mia --defense epd`?

?? RAG_MIA ? EPD ???AUC 0.9950 -> 0.9514?QA 0.4360 -> 0.2855?EPD ?? retrieval hit rate?? privacy gain ?? query_sanitization?


## Enhanced EPD Update

EPD ???? retrieval-aware privacy-preserving generation defense??? retrieval overlap scoring?adaptive fusion?response paraphrasing/normalization??? target/final overlap reduction metrics?

?????RAG_MIA ? EPD-enhanced AUC 0.9950 -> 0.8864?QA 0.4360 -> 0.3510?retrieval hit rate 0.3375 -> 0.3375?????????????? retrieval utility?

## Janus Synthetic-Safe Fine-tuning Evaluation

`janus` adds a fine-tuning privacy evaluation path alongside the existing membership scoring methods.

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack janus --config ./configs/janus.yml
```

This entry calls `/root/autodl-tmp/Janus/run_janus_minimal.py` with local GPT-2, 30 synthetic association pairs for fine-tuning, and 100 synthetic eval prompts. Every email uses `example-synthetic.com`; the entry does not read or recover real PII. Janus is reported with `recovery_rate` because it evaluates targeted synthetic association recovery rather than binary member/non-member classification.

Generated artifacts:

- `outputs/janus_result.json`
- `outputs/logs/janus.log`
- `outputs/janus_report.md`

## LoRA-Janus: PEFT-based Fine-tuning Privacy Evaluation

`janus` now supports three synthetic-safe modes:

```bash
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack janus --config ./configs/janus_no_ft.yml
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack janus --config ./configs/janus_full_ft.yml
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack janus --config ./configs/janus_lora_ft.yml
```

LoRA does not update all GPT-2 weights. It inserts a low-rank adapter and trains only the adapter parameters. Current LoRA config:

```yaml
finetune_mode: lora_ft
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - c_attn
```

Current synthetic-safe comparison:

| Mode | Trainable Params | Trainable Ratio | Recovery Rate | Recovered / Eval |
|---|---:|---:|---:|---:|
| no_ft | 0 | 0.0000% | 0.0000 | 0 / 100 |
| full_ft | 124439808 | 100.0000% | 0.8600 | 86 / 100 |
| lora_ft | 294912 | 0.2364% | 0.0000 | 0 / 100 |

The first LoRA-Janus run shows that full fine-tuning strongly amplifies synthetic association recovery, while the current LoRA setting (`r=8`, `lr=5e-5`, `epochs=2`) does not yet recover synthetic emails. Next ablations should increase `lora_r`, `learning_rate`, `epochs`, and `train_size`.

Generated artifacts:

- `outputs/janus_lora_comparison.md`
- `outputs/janus_lora_comparison.json`
- `outputs/janus_lora_report.md`

## LoRA-Janus Synthetic Association Experiments

A 36-run synthetic-safe LoRA sweep was completed with:

- `lora_r`: 8, 16, 32
- `learning_rate`: 5e-5, 1e-4, 5e-4
- `epochs`: 2, 5
- `train_size`: 30, 50
- `lora_alpha`: 16
- `lora_dropout`: 0.05
- `target_modules`: `c_attn`

Key outputs:

- `outputs/janus_lora_sweep_table.md`
- `outputs/janus_lora_statistics.json`
- `outputs/janus_lora_threshold_report.md`
- `/root/autodl-tmp/Janus/outputs/janus_lora_sweep_results.json`

Main observation: LoRA synthetic recovery has a sharp threshold. Low learning rate or only 2 epochs gives near-zero recovery. When `learning_rate=5e-4` and `epochs=5`, LoRA recovery jumps close to full fine-tuning.

Best run:

```text
r=8, lr=0.0005, epochs=5, train_size=30, recovery_rate=1.0000
```

First positive recovery:

```text
r=8, lr=0.0005, epochs=5, train_size=30, recovery_rate=1.0000
```

This supports the current hypothesis: LoRA can look safe under weak adapter training, but once the adapter crosses a learning threshold, synthetic association recovery can approach full fine-tuning.

## LoRA-Janus Threshold Boundary Analysis

A focused boundary sweep was completed with `r=8` and `train_size=30`, using only synthetic identifier-to-email associations.

Sweep grid:

- learning_rate: `2e-4`, `3e-4`, `4e-4`, `5e-4`
- epochs: `3`, `4`, `5`

Threshold point, defined as the smallest learning-rate/epoch combination with `recovery_rate > 0.5`:

```json
{
  "threshold_lr": 0.0004,
  "threshold_epochs": 5,
  "threshold_recovery_rate": 1.0
}
```

Full boundary result:

| lr | epochs | recovery_rate | recovered |
|---|---:|---:|---:|
| 0.0002 | 3 | 0.0000 | 0 / 100 |
| 0.0002 | 4 | 0.0000 | 0 / 100 |
| 0.0002 | 5 | 0.0000 | 0 / 100 |
| 0.0003 | 3 | 0.0000 | 0 / 100 |
| 0.0003 | 4 | 0.0000 | 0 / 100 |
| 0.0003 | 5 | 0.0300 | 3 / 100 |
| 0.0004 | 3 | 0.0000 | 0 / 100 |
| 0.0004 | 4 | 0.0000 | 0 / 100 |
| 0.0004 | 5 | 1.0000 | 100 / 100 |
| 0.0005 | 3 | 0.0000 | 0 / 100 |
| 0.0005 | 4 | 1.0000 | 100 / 100 |
| 0.0005 | 5 | 1.0000 | 100 / 100 |

Interpretation:

- Weak LoRA remains near zero recovery.
- `lr=3e-4, epochs=5` shows the first weak positive signal (`0.03`).
- `lr=4e-4, epochs=5` jumps to `1.00`, showing a sharp phase-transition-like boundary.
- `lr=5e-4, epochs=4` also reaches `1.00`, meaning higher learning rate needs fewer epochs.
- Since rank is fixed at `r=8`, the jump is driven by optimization strength rather than adapter capacity alone.

Outputs:

- `outputs/janus_lora_boundary_table.md`
- `outputs/janus_lora_threshold_point.json`
- `outputs/janus_lora_phase_transition.json`
- `outputs/janus_lora_boundary_report.md`

## LoRA-Janus Phase Transition and Output Filtering Defense

LoRA-Janus threshold analysis is now packaged as a reportable module:

- Phase transition figure: `outputs/figures/janus_lora_phase_transition.png`
- Boundary table: `outputs/janus_lora_boundary_table.md`
- Threshold point: `outputs/janus_lora_threshold_point.json`
- Output filtering benchmark: `outputs/benchmark_janus_lora_pii_filter_result.json`
- Final report: `outputs/janus_lora_final_report.md`

Observed threshold point:

```json
{
  "threshold_lr": 0.0004,
  "threshold_epochs": 5,
  "threshold_recovery_rate": 1.0
}
```

`pii_output_filter` is registered as a synthetic-safe output filtering defense. It only filters `user_a001@example-synthetic.com` style emails under `example-synthetic.com`, replacing matches with `[EMAIL]`. It intentionally does not process real emails or real personal information.

Requested weak LoRA file filtering result:

- recovery_rate_before: `0.0000`
- recovery_rate_after: `0.0000`
- filtered_count: `0`
- filter_rate: `0.0000`

Conclusion: weak LoRA has no synthetic recovery to filter, while strong boundary LoRA outputs can be blocked at the output layer. Output filtering mitigates displayed synthetic email recovery, but it does not remove internal memorization.

## True LoRA-Janus Held-out Recovery Correction

The LoRA-Janus pipeline has been corrected to match the Janus-style objective more closely:

1. Pretrain on full synthetic association corpus S = S1 + S2 + others.
2. Continue training on general text to simulate forgetting.
3. Use only known S1 for Janus fine-tuning.
4. Evaluate hidden S2, which is strictly disjoint from S1.

Current held-out S2 recovery:

| Branch | S2 Recovery Rate |
|---|---:|
| no_janus | 0.0000 |
| full_ft_janus | 1.0000 |
| lora_ft_janus | 0.0000 |

Conclusion: the current LoRA setting does not activate held-out S2 recovery, while full-parameter Janus fine-tuning does. Earlier LoRA threshold plots should be reported as adapter synthetic association learning, not as true held-out Janus recovery.

Report: `outputs/janus_lora_true_pipeline_report.md`.

## True LoRA-Janus Hidden S2 Activation Worklog

Current work is restricted to true held-out recovery. Do not use the old experiment that measures whether LoRA memorizes its own S1 training pairs. The active question is whether S1 known synthetic pairs can activate hidden S2 synthetic pairs.

Planned changes for this run:

- Add hidden S2 teacher-forced metrics to `/root/autodl-tmp/Janus/run_janus_lora_true_pipeline.py`: `avg_loss`, `avg_perplexity`, and `avg_correct_token_prob`.
- Report those metrics for `no_janus`, `full_ft_janus`, and `lora_ft_janus`.
- Strengthen LoRA in `configs/janus_lora_true_pipeline.yml` to `r=32`, `alpha=64`, targets `[c_attn, c_proj, c_fc]`, `lr=5e-4`, `epochs=10`.
- Generate `outputs/janus_hidden_activation_report.md` after the true held-out run.

Status: stronger LoRA true held-out run completed; hidden activation report generated.

Final stronger LoRA hidden S2 activation result:

| Branch | S2 Recovery | Avg Loss | Avg Perplexity | Avg Correct Token Prob |
|---|---:|---:|---:|---:|
| no_janus | 0.0000 | 4.596597 | 99.146386 | 0.380292 |
| full_ft_janus | 1.0000 | 0.001256 | 1.001257 | 0.998812 |
| lora_ft_janus | 1.0000 | 0.000372 | 1.000372 | 0.999637 |

Conclusion: stronger LoRA fully activates hidden S2 in this run, not just sub-threshold activation. Report: `outputs/janus_hidden_activation_report.md`.

## LoRA-Janus Reliability Validation Worklog

Current work is validation only. The stronger LoRA hidden S2 recovery result remains the main experimental observation; this run checks whether it could be explained by environment issues, data leakage, evaluation bugs, or template-only false positives.

Planned validation artifacts:

- `outputs/janus_leak_check.json`
- `outputs/janus_lora_generation_samples.md`
- `outputs/janus_reliability_validation_report.md`

Validation checklist:

- Verify S1/S2 identifier and email disjointness.
- Verify LoRA training data contains no S2 identifiers or emails.
- Add fake S2 control with never-seen `User_Z*` identifiers.
- Add shuffled-label S2 control.
- Re-check forgotten model recovery and teacher-forced metrics.
- Confirm prompts do not contain gold continuations and loss is computed only on continuation tokens.

Status: reliability validation completed; all criteria passed.

Final reliability validation result:

| Check | Result |
|---|---:|
| S1/S2 identifier overlap | 0 |
| S1/S2 email overlap | 0 |
| S2 identifiers in LoRA train file | 0 |
| S2 emails in LoRA train file | 0 |
| fake_s2_recovery_rate | 0.0000 |
| shuffled_s2_recovery_rate | 0.0000 |
| forgotten_s2_recovery_rate | 0.0000 |
| lora_real_s2_recovery_rate | 1.0000 |
| all criteria passed | True |

Conclusion: the stronger LoRA true held-out recovery result passes the reliability validation. Report: `outputs/janus_reliability_validation_report.md`.

## LoRA-Janus Model Transfer Worklog

Current work is synthetic-safe model migration only. The goal is to test whether stronger LoRA-Janus hidden S2 recovery transfers from GPT-2 to another GPT-style causal LM, starting with `distilgpt2` if it exists locally.

Planned artifacts:

- `configs/janus_lora_true_pipeline_distilgpt2.yml`
- `outputs/janus_model_transfer_report.md`
- `outputs/janus_model_transfer_reliability.json`

Status: model transfer report generated; distilgpt2 skipped because local model path is missing.

Final model transfer setup result:

| Model | Status | LoRA Recovery | Note |
|---|---|---:|---|
| gpt2 | completed | 1.0000 | local baseline result |
| distilgpt2 | skipped | N/A | model_path not found: /root/autodl-tmp/models/distilgpt2 |

Conclusion: current environment cannot answer cross-model transfer yet because `distilgpt2` is not present locally. The migration config, alias support, graceful skip path, transfer report, and reliability JSON are ready. Report: `outputs/janus_model_transfer_report.md`.

## LoRA-Janus Model Download and Migration Run Worklog

Current work downloads and connects a second GPT-style causal LM for synthetic-safe LoRA-Janus model migration. Main Janus logic remains unchanged; this run only extends migration validation.

Planned artifacts:

- `outputs/janus_model_inventory.txt`
- updated `outputs/janus_model_transfer_report.md`
- updated `outputs/janus_model_transfer_reliability.json`

Status: model transfer completed; distilgpt2 passes migration and reliability checks; gpt2-medium skipped because local model path is missing.

Final downloaded-model transfer result:

| Model | no_janus | full_ft | LoRA | Reliability | Status |
|---|---:|---:|---:|---|---|
| gpt2 | 0.0000 | 1.0000 | 1.0000 | passed | completed |
| distilgpt2 | 0.0000 | 1.0000 | 1.0000 | passed | completed |
| gpt2-medium | N/A | N/A | N/A | N/A | skipped: model path missing |

Conclusion: stronger LoRA-Janus hidden S2 recovery transfers from GPT-2 to DistilGPT2 under synthetic-safe controls. Report: `outputs/janus_model_transfer_report.md`.

## LoRA-Janus and MIA Link Worklog

Current work tests whether Janus/LoRA hidden-memory activation strengthens membership-style attack signal on synthetic-safe held-out S2 associations. The member set is S2 (`User_A101` to `User_A200`); the non-member set is newly generated `User_B101` to `User_B200`, which is not used in pretraining, S1, S2, forgetting data, or LoRA training.

Planned artifacts:

- `/root/autodl-tmp/Janus/run_janus_mia_eval.py`
- `attacks/janus_mia.py`
- `configs/janus_mia.yml`
- `outputs/janus_mia_eval_result.json`
- `outputs/janus_mia_link_report.md`

Status: janus_mia unified attack completed; link report generated.

Final Janus-MIA link result:

| Model State | S2 Recovery | MIA AUC | Member Loss | Non-member Loss | Loss Gap |
|---|---:|---:|---:|---:|---:|
| forgotten_model | 0.0000 | 0.9995 | 4.596597 | 5.174101 | 0.577504 |
| full_ft_janus_model | 1.0000 | 1.0000 | 0.001256 | 1.058215 | 1.056959 |
| lora_ft_janus_model | 1.0000 | 1.0000 | 0.000372 | 0.790605 | 0.790233 |

Conclusion: Janus/LoRA recovery is associated with stronger membership-style signal, but the forgotten model already has near-ceiling AUC against the simple User_B non-member control. The strongest evidence is the enlarged loss gap and much lower member loss after full-ft/LoRA reactivation. Report: `outputs/janus_mia_link_report.md`.

## LoRA Cross-Model Transfer Worklog

Current work tests whether the stronger LoRA-Janus adapter trained on GPT-2 can be loaded into another GPT-style causal LM, DistilGPT2, without retraining. This is distinct from the previous DistilGPT2 retrain experiment.

Planned artifacts:

- `/root/autodl-tmp/Janus/run_janus_lora_cross_model_transfer.py`
- `attacks/janus_lora_transfer.py`
- `configs/janus_lora_transfer.yml`
- `outputs/janus_lora_cross_model_transfer_result.json`
- `outputs/janus_lora_cross_model_transfer_report.md`

Status: source/target models and GPT-2 adapter verified; creating cross-model transfer script.


- 2026-05-28 cross-model transfer: running server-side GPT-2 adapter to DistilGPT2 evaluation.

- 2026-05-28 cross-model transfer: sanity recovery was 0 on raw base; inspecting saved forgotten bases and result JSON before final conclusion.

- 2026-05-28 cross-model transfer: rerunning with synthetic-forgotten source/target bases to avoid raw-base mismatch.

- 2026-05-28 cross-model transfer: sanity passed on GPT-2 forgotten base; adding unified attack wrapper and config.

- 2026-05-28 cross-model transfer: added `janus_lora_transfer` wrapper, registry entry, and config for server-side unified execution.

- 2026-05-28 cross-model transfer: running unified framework janus_lora_transfer.

- 2026-05-28 cross-model transfer final: GPT-2->GPT-2 sanity recovery 1.0; GPT-2 adapter direct-loaded into DistilGPT2 but S2 recovery stayed 0.0; report written to `outputs/janus_lora_cross_model_transfer_report.md`.

- 2026-05-29 Janus-MIA hard controls: starting update for harder non-members, calibrated scoring, and defense comparison.

- 2026-05-29 Janus-MIA hard controls: replaced eval script with hard non-member sets, multi-score metrics, calibrated loss, and pii_output_filter reporting.

- 2026-05-29 Janus-MIA hard controls: updated unified wrapper/config with nonmember_type=all and calibrated loss outputs.

- 2026-05-29 Janus-MIA hard controls: running unified framework experiment.

- 2026-05-29 Janus-MIA hard controls final: unified run completed; outputs include hard-control result JSON, table, defense comparison, and gap analysis report.

- 2026-05-29 repository upload prep: initialized/updated git metadata, .gitignore, origin remote, and prepared initial commit for GitHub upload.
