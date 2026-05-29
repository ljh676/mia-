# 原论文项目入口分析

## RAG_MIA

- 项目路径：`/root/autodl-tmp/RAG_MIA`
- README 推荐流程：
  - `python prepare_dataset.py`
  - `python generate_index.py --config ./configs/nfcorpus.yml`
  - `python run_mia.py --config ./configs/nfcorpus.yml`
- 真正攻击入口：
  - `run_mia.py` 是外层启动器，会计算日志名并执行 `nohup python3 -u main_mia.py --config ... --name ... > logs/*.txt`
  - `main_mia.py` 包含 `parse_config()` 和 `main(config: ExperimentConfig)`，是真正攻击 pipeline。
- 主函数：
  - `run_mia.py: main()`
  - `main_mia.py: parse_config()` + `main(config)`
- 参数传递：
  - 命令行参数 `--config <yaml>`
  - 配置对象来自 `config.py` 中的 `ExperimentConfig`
  - 关键配置包括 `attack_config.M/N/top_k/attack_method/from_ckpt/evaluate_attack`、`rag_config.eval_dataset/retriever/retrieve_k`、`llm_config.model_name`
- 输出结果位置：
  - 攻击中间与结果文件主要写入 `results/target_docs/{attack_name}.json`
  - 外层日志写入 `logs/{attack_name}.txt`
- 是否支持 import：
  - `main_mia.py` 有 `main(config)`，理论上支持 import。
  - 但模块顶层会导入 `src.models`、`openai`、`beir`、`simple_parsing` 等依赖，并强依赖项目工作目录和相对路径。
- 是否必须 subprocess：
  - 当前更稳妥的是 subprocess，因为原项目以脚本方式组织，并依赖 cwd。
  - 统一框架仍保留 import 探测，未来可在依赖完整后改成直接调用 `main(config)`。
- 当前环境探测：
  - `/root/miniconda3/envs/dcmi/bin/python` 有 `simple_parsing`、`beir`、`torch`、`yaml`。
  - RAG_MIA 启动时仍可能缺少 `openai`，统一框架会把 stdout/stderr/traceback 写入 `outputs/logs/rag_mia.log`。

## DCMI

- 项目路径：`/root/autodl-tmp/DCMI`
- README 推荐流程：
  - 构建 RAG index：`python -m flashrag.retriever.index_builder ...`
  - RAG 实验：`python example/run_exp.py --method_name Standard-RAG --split test --dataset_name nq --gpu_id ...`
  - 扰动样本：`python perturb.py`
  - 成员推断阈值评估：`python MIA.py`
- 真正攻击入口：
  - `MIA.py` 是 DCMI 阈值评估脚本，计算 member/non-member 与 perturb 分数差，并输出 accuracy、AUC、precision、recall、F1。
  - `example/run_exp.py` 是 RAG 生成实验入口，不直接完成最终 DCMI 阈值攻击。
- 主函数：
  - `MIA.py` 没有显式 `main()`，顶层代码直接执行。
  - `example/run_exp.py` 在 `if __name__ == "__main__"` 中解析参数并分发到 `naive/llmlingua/sc/ircot/spring`。
- 参数传递：
  - `MIA.py` 当前无 CLI 参数。
  - `example/run_exp.py` 支持 `--method_name`、`--split`、`--dataset_name`、`--gpu_id`。
- 输出结果位置：
  - `MIA.py` 主要输出到 stdout，并弹出 matplotlib 图。
  - 当前仓库已有最小实验输出样例：`outputs/min_dcmi_nfcorpus_scores.json` 和 `outputs/min_dcmi_nfcorpus_metrics.json`。
- 是否支持 import：
  - `MIA.py` 顶层执行，不适合 import 调用。
  - `example/run_exp.py` 中函数可 import，但仍依赖 FlashRAG 配置、模型、数据和 cwd。
- 是否必须 subprocess：
  - 当前建议 subprocess 调用，统一框架会优先执行 `MIA.py`，并解析 stdout 或已有 outputs 文件。

## 统一接入策略

1. 所有原论文代码都不修改。
2. 统一框架只修改 `/root/autodl-tmp/unified_mia_framework`。
3. `execute_original: true` 时使用配置中的 `python_path` 和入口脚本启动原项目。
4. stdout、stderr、traceback 保存到：
   - `outputs/logs/rag_mia.log`
   - `outputs/logs/dcmi.log`
5. 统一结果解析器支持：
   - JSON
   - TXT/stdout
   - pickle
   - numpy
6. 最终统一输出：

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

## 当前最小真实运行验证记录

- DCMI：
  - 统一入口已经真实启动 `/root/autodl-tmp/DCMI/MIA.py`。
  - 原脚本当前包含 `mem = [...]`、`perturb_mem = [...]` 等占位数据，真实启动后会报 `RuntimeError: Could not infer dtype of ellipsis`。
  - 统一框架已捕获该 traceback 到 `outputs/logs/dcmi.log`。
  - 由于仓库已有 `outputs/min_dcmi_nfcorpus_scores.json`，统一解析器成功转换为统一结果格式。

- RAG_MIA：
  - 统一入口已经真实启动 `/root/autodl-tmp/RAG_MIA/main_mia.py`。
  - 已为当前环境补齐多个原项目缺失依赖：`openai`、`matplotlib`、`ragatouille`、`pyterrier-doc2query`、`pyterrier-dr`。
  - 后续若继续出现原项目依赖、模型权重或数据文件问题，统一框架会继续把 stdout/stderr/traceback 记录到 `outputs/logs/rag_mia.log`。

## 追加验证：RAG_MIA 已进入数据加载阶段

修正最小配置字段后，`rag_mia` 统一入口已经真实启动：

```bash
/root/miniconda3/envs/dcmi/bin/python main_mia.py --config /root/autodl-tmp/unified_mia_framework/outputs/rag_mia_min.yml
```

原项目成功打印了 `ExperimentConfig`，包括：

- `attack_method: mia`
- `M: 1`
- `N: 1`
- `top_k: 1`
- `retrieve_k: 1`
- `evaluate_attack: false`

当前阻塞点不在统一框架，而在 RAG_MIA 原项目数据缺失：

```text
/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.jsonl
/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.json
```

补齐 BEIR 格式 `corpus.jsonl` 或原项目 fallback 使用的 `corpus.json` 后，同一条统一入口命令会继续执行后续真实攻击流程。

## NFCorpus 修复与当前 RAG_MIA 阻塞

已完成：

- 下载并解压 BEIR NFCorpus：
  - `datasets/nfcorpus/corpus.jsonl`
  - `datasets/nfcorpus/queries.jsonl`
  - `datasets/nfcorpus/qrels/{train,dev,test}.tsv`
- 从 `corpus.jsonl` 生成 RAG_MIA fallback 需要的 `corpus.json`。
- 确认原项目已有：
  - `selected_indices.json`
  - `clean_data_with_questions.json`
- 从 `clean_data_with_questions.json` 生成缺失的 `target_data_with_questions.json`。
- 统一入口已真实启动 `main_mia.py`，完成目标文档采样、问题读取/生成，并生成：
  - `/root/autodl-tmp/RAG_MIA/results/target_docs/mia-nfcorpus-llama3_70b-llama3-colbert-R1-Top1-M1-N1.json`

当前最新阻塞：

```text
FileNotFoundError: datasets/nfcorpus/colbert/indexes/nfcorpus-index/metadata.json
FileNotFoundError: datasets/nfcorpus/colbert/indexes/nfcorpus-index/plan.json
```

原因：

- RAG_MIA 当前配置使用 `retriever: colbert`。
- `src/retrievers/ColBERT.py` 使用 `RAGPretrainedModel.from_index(...)`，要求已有 RAGatouille/ColBERT index。
- 原项目 `generate_index.py` 内部明确写有 `# TODO: Handle ColBERT`，因此该脚本不能自动生成 ColBERT 所需的 `metadata.json` / `plan.json` 索引。

下一步修复方案：

1. 提供原论文预构建的 `datasets/nfcorpus/colbert/indexes/nfcorpus-index/`。
2. 或按 RAGatouille 官方方式单独构建 ColBERT index。
3. 或在不修改原仓库核心逻辑的前提下，在统一框架中增加一个可选 retriever override，改用 `gte`/`bge` 并生成 FAISS index；这会偏离原 `nfcorpus.yml` 默认 `colbert` 设定。

## RAG_MIA retriever 切换与最小真实流程跑通

根据当前任务要求，不再使用 `colbert`。处理过程：

1. 优先尝试 `gte`：
   - 将最小配置中的 `retriever` 改为 `gte`。
   - 由于服务器无法访问 HuggingFace 下载 `Alibaba-NLP/gte-large-en-v1.5`，创建本地别名指向 `/root/autodl-tmp/DCMI/models/gte-small`。
   - 本地模型能加载，但原项目 `GTE.py` 使用 `max_length=8192`，而本地 Bert/GTE-small 位置长度为 512，构建索引时报维度错误。

2. 回退到 `bge`：
   - 将最小配置中的 `retriever` 改为 `bge`。
   - 创建本地别名 `BAAI/bge-large-en-v1.5 -> /root/autodl-tmp/DCMI/models/gte-small`，避免联网下载。
   - 原项目 `BGE.py` 使用 `max_length=512`，可适配本地模型。

3. 成功生成 FAISS index：
   - `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/bge/indexes/nfcorpus-index/corpus_index.faiss`
   - `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/bge/indexes/nfcorpus-index/doc_ids.pkl`

4. 成功通过统一入口运行最小真实流程：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
```

最终结果：

```json
{
  "returncode": 0,
  "scores": [0.6666666666666666, 0.0],
  "labels": [1, 0],
  "predictions": [1, 0],
  "metrics": {
    "accuracy": 1.0,
    "auc": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

说明：当前 `evaluate_attack=false`，因此没有加载目标 LLM 生成回答。统一结果中的 score 来自 target document 在各问题 top-k 检索结果中的命中比例，用于验证最小真实 RAG_MIA 检索攻击流程已经接通。

## Phi-3 / Phi-4 evaluate_attack=true 检查结果

本轮目标是将 RAG_MIA 最小真实流程切换为 `bge + phi3 + evaluate_attack=true`。

检查结果：

- `/root/autodl-tmp/RAG_MIA/src/models/Phi4.py` 存在。
- `/root/autodl-tmp/RAG_MIA/src/models/__init__.py` 已注册 `provider == "phi4"`。
- 未发现 Phi3 模型类。
- `/root/autodl-tmp/RAG_MIA/model_configs/` 中没有 `phi3_config.json` 或 `phi4_config.json`。
- 已搜索以下本地路径，未发现 Phi-3 / Phi-4 模型目录：
  - `/root/autodl-tmp/models`
  - `/root/autodl-tmp/RAG_MIA/models`
  - `/root/.cache/huggingface`
  - `/root/autodl-tmp`
- HuggingFace cache 当前只发现：`/root/.cache/huggingface/hub/models--thenlper--gte-small`

结论：

当前服务器缺少本地 Phi-3/Phi-4 模型权重。按照任务要求，不回退到 `llama3_70b` 或其他超大模型，也不尝试联网下载 Phi 模型。因此暂不能执行 `bge + phi3 + evaluate_attack=true` 的完整最小攻击流程。

下一步需要先提供本地 Phi-3 模型目录，例如：

```text
/root/autodl-tmp/models/Phi-3-mini-4k-instruct
```

提供后可新增最小 `model_configs/phi3_config.json`，并将统一框架最小配置切换到：

```yaml
model_name: phi3
shadow_model_name: phi3
evaluate_attack: true
retriever: bge
```

## Phi3/Phi4 evaluate_attack=true 阻塞记录

本次按用户要求检查了 RAG_MIA 是否可以切换到 `bge + phi3/phi4 + evaluate_attack=true` 的完整最小攻击流程。

检查结论：

1. `/root/autodl-tmp/RAG_MIA/src/models/Phi4.py` 已存在，且 `src/models/__init__.py` 中已有 `phi4` provider 注册，因此 Phi4 模型类可以优先复用，不需要重写原论文核心逻辑。
2. `/root/autodl-tmp/RAG_MIA/model_configs/` 中未发现 Phi3/Phi4 配置文件。
3. 在以下本地路径中未发现可用 Phi3/Phi4 权重：
   - `/root/autodl-tmp`
   - `/root/autodl-tmp/models`
   - `/root/autodl-tmp/RAG_MIA/models`
   - `/root/.cache/huggingface`
4. 使用 `local_files_only=True` 测试加载 `microsoft/phi-4` 和 `microsoft/Phi-3-mini-4k-instruct` 均失败，原因是本地缓存不存在且当前不能依赖 HuggingFace 在线下载。
5. 因用户要求“如果没有本地 Phi-3 模型，请停止并报告缺少模型，不要改回超大模型”，因此本次没有把配置切回 `llama3_70b` 或其他超大模型，也没有强行触发在线下载。

当前可运行状态：

- `evaluate_attack: false` 时，RAG_MIA 已经可以通过统一框架真实运行最小流程：`bge + nfcorpus`。
- `evaluate_attack: true` 需要本地提供 Phi4 或 Phi3 权重后再继续。

建议下一步：

1. 将 Phi4 权重放到本地，例如 `/root/autodl-tmp/models/phi-4`。
2. 在不修改 RAG_MIA 核心代码的前提下，可以创建路径别名 `/root/autodl-tmp/RAG_MIA/microsoft/phi-4 -> /root/autodl-tmp/models/phi-4`，以适配原项目 `Phi4.py` 中硬编码的 `microsoft/phi-4`。
3. 新增最小 `phi4_config.json` 后，将统一框架 `configs/rag_mia.yml` 改为 `evaluate_attack: true`、`model_name: phi4`、`shadow_model_name: phi4`，再运行完整最小攻击流程。


## RAG_MIA Phi3 evaluate_attack=true 最小流程记录

本阶段已完成 `nfcorpus + bge + phi3 + evaluate_attack=true` 的最小真实流程接入。

关键配置：

- 目标生成模型：`phi3`
- 本地权重：`/root/autodl-tmp/models/phi3-mini-4k-instruct`
- 检索器：`bge`
- 数据集：`nfcorpus`
- `M=1`，`N=1`，`top_k=1`，`retrieve_k=3`
- 原项目入口：`/root/autodl-tmp/RAG_MIA/main_mia.py`
- 统一框架入口：`/root/autodl-tmp/unified_mia_framework/run_attack.py`

为使当前环境可离线运行，已补齐以下本地依赖：

- Phi-3 mini 权重：`/root/autodl-tmp/models/phi3-mini-4k-instruct`
- Electra tokenizer/base：`/root/autodl-tmp/models/electra-base-discriminator`
- monoELECTRA reranker/filter：`/root/autodl-tmp/models/monoELECTRA_LCE_nneg31`
- monoELECTRA 的 `pytorch_model.bin` 已转换出 `model.safetensors`，避免当前 transformers/torch 组合拒绝加载 `.bin`。

适配说明：

- 新增 `/root/autodl-tmp/RAG_MIA/src/models/Phi3.py`，复用 RAG_MIA 现有 `Model` 接口。
- 更新 `/root/autodl-tmp/RAG_MIA/src/models/__init__.py` 注册 `phi3` provider。
- 新增 `/root/autodl-tmp/RAG_MIA/model_configs/phi3_config.json`，使用本地模型路径并设置 `local_files_only=True`。
- Phi-3 mini 4k 在当前 transformers 环境中需要显式 `rope_scaling=None` 和 `use_cache=False`，否则会出现 `KeyError: rope_scaling` 或 `DynamicCache.seen_tokens` 兼容问题。
- 当前 pyterrier 的 `filter_questions_topk()` 在本环境触发 `KeyError: docno`，因此最小 checkpoint 预置了 1 个问题和标准答案，并设置 `from_ckpt_attack_data=true`，仍然真实执行原项目的 retrieve、Phi3 target LLM 查询和 calculate_score。

最终验证：

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack rag_mia --config ./configs/rag_mia.yml
```

最近一次原论文子进程返回码为 `0`，日志中可见 Phi3 生成 `llm_responses` 并执行 `calculate_score()`。

## RAG context 是否进入 Phi3 prompt 的验证记录

本次重点验证 `nfcorpus + bge + phi3 + evaluate_attack=true` 中，Phi3 是否真正看到了 RAG 检索上下文。

结论：

1. RAG_MIA 的 `query_target_llm()` 会从 `retrieved_doc_ids[i]` 取出 `self.corpus[doc]['text']`，并传给 `wrap_prompt(question, topk_contents, prompt_id=4)`。
2. `src/prompts.py` 中 `wrap_prompt()` 会把 context list 用换行拼接到 `Contexts: [context]`，再拼接 `Query: [question]`。
3. 调试日志已打印 `[Retrieved Doc 0/1/2]`，确认 `retrieve_k=3` 且 retrieved docs 数量为 3。
4. 最终 `[FINAL RAG PROMPT BEGIN]` 中明确包含 retrieved context 和 query。例如第一个样本包含：
   - `Contexts: Intervention strategies regarding the biofortification of orange-fleshed sweet potato... combating vitamin A deficiency...`
   - `Query: Does the biofortification of orange-fleshed sweet potato aim to combat vitamin A deficiency?`
5. tokenizer 调试显示：
   - 第一个样本 token length: 938
   - 第二个样本 token length: 1295
   - tokenizer model_max_length: 4096
   - `Phi3 Truncated: False`
6. 因此，当前不是 context 为空、漏拼接、被 question 覆盖或被 tokenizer 截断的问题。

进一步手工测试：

新增 `/root/autodl-tmp/unified_mia_framework/test_phi3_rag.py`，手工输入：

```text
CONTEXT: The paper discusses obesity and cardiovascular disease.
QUESTION: Does the paper mention obesity?
EXPECTED: Yes
```

结果：

```text
[Phi3 Raw Output] ল unknownging seglevention
[Phi3 Normalized Output] I don't know
```

进一步的低层直连测试 `/root/autodl-tmp/unified_mia_framework/test_phi3_direct.py` 也显示，即使不经过 RAG_MIA `wrap_prompt()`，裸 prompt 和 chat prompt 都会生成异常 token：

```text
RAW OUTPUT: JavaScriptJavaScript foiJavaScriptoothJavaScript SomethingJavaScript
CHAT OUTPUT: cis jedenpreventciscisreichencis en
```

判断：

- RAG context 已经真实进入 Phi3 prompt。
- 当前 Phi3 的无效回答主要来自模型生成栈本身，而不是 RAG_MIA context 拼接。
- 更可能的原因是当前 `transformers` / Phi-3 remote code / `torch_dtype=float16` / `use_cache=False` / attention 实现组合存在兼容或数值问题。

建议下一步：

1. 优先尝试把 Phi3 加载 dtype 改为 `bfloat16` 或 `auto`，因为模型 config 原始声明为 `torch_dtype: bfloat16`。
2. 尝试升级/匹配 `transformers` 到 Phi-3 mini 官方权重对应版本附近，例如模型 config 中记录的 `4.40.2`，或使用当前环境支持的新版本组合。
3. 如果 GPU 支持 bf16，优先使用 bf16；如果不支持，再考虑 float32 小样本验证。
4. 在修正生成栈前，不建议继续用当前 Phi3 输出评价正式攻击效果。

## Phi3 生成异常 token 修复记录

本阶段针对 Phi3 输出 `lapselapse...`、`JavaScriptJavaScript...`、多语言乱码 token 的问题进行了定位和修复。

关键发现：

1. RAG context 已确认进入最终 prompt，且没有发生 tokenizer truncation。
2. `test_phi3_rag.py` 和 `test_phi3_direct.py` 在全局 `transformers 5.8.0` 下仍会生成异常 token。
3. `use_cache=True` + remote code 在全局 `transformers 5.8.0` 下会触发：`AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'`。
4. Phi-3 mini 本地模型的 `config.json` 声明 `transformers_version: 4.40.2`、`torch_dtype: bfloat16`，而服务器原环境为 `torch 2.5.1+cu121`、`transformers 5.8.0`。

修复方式：

为避免破坏当前 `dcmi` 环境，没有覆盖安装 transformers，而是安装隔离兼容包：

```text
/root/autodl-tmp/phi3_compat_pkgs
```

其中包含：

```text
transformers==4.40.2
huggingface_hub==0.23.5
tokenizers==0.19.1
```

统一框架 `configs/rag_mia.yml` 新增：

```yaml
extra_pythonpath: /root/autodl-tmp/phi3_compat_pkgs
```

`attacks/rag_mia.py` 会仅在启动 RAG_MIA 原论文子进程时临时注入该 `PYTHONPATH`，不污染主环境。

Phi3 适配器修复：

- `/root/autodl-tmp/RAG_MIA/model_configs/phi3_config.json` 已改为 `torch_dtype: bfloat16`。
- `/root/autodl-tmp/RAG_MIA/src/models/Phi3.py` 使用：
  - `torch_dtype=torch.bfloat16`
  - `attn_implementation="eager"`
  - `model.config.use_cache=True`
  - `tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_tensors="pt")`
- 继续保留 final prompt、token length、raw output、normalized output 调试日志。

验证结果：

隔离兼容版本下，独立测试已恢复正常：

```text
Context: The paper discusses obesity and cardiovascular disease.
Question: Does the paper mention obesity?
Expected: Yes
Output: Yes
```

正式 RAG_MIA 最小实验也恢复正常：

```text
Response for doc MED-2188, question 0: Yes
Response for doc MED-4075, question 0: I don't know
Document MED-2188 - Accuracy: 100.00% (1/1)
Document MED-4075 - Accuracy: 0.00% (0/1)
```

统一输出：

```json
{
  "scores": [1.0, 0.0],
  "predictions": [1, 0],
  "labels": [1, 0],
  "metrics": {
    "accuracy": 1.0,
    "auc": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  },
  "returncode": 0
}
```


## Neighbour_MIA ??????

- ??????`/root/autodl-tmp/Neighbour_MIA`
- ?????`attack.py`
- ????????? `<path_to_attack_model_twitter/news/wiki>` ?????`cuda:0`/`cuda:1` ???`one_word_neighbours` ???????? `tweets.csv`?`news.csv`?`wikitext.txt`?????????
- ?????`/root/autodl-tmp/Neighbour_MIA/attack_original.py`
- ???????`attack.py` ????? CLI ?? `--proc-id --model --dataset`????????????????????
- ?????`/root/autodl-tmp/models/gpt2`
- ???????`/root/autodl-tmp/models/bert-base-uncased`
- one-word replacement??? BERT MLM ?????? token ? mask??? top-k ?????? neighbours?
- score ???`original_logprob - mean(neighbour_logprobs)`??? logprob ? GPT-2 loss ?????
- ?????`data/sample_members.jsonl` ? `data/sample_nonmembers.jsonl`??? 5 ??????
- ?? pickle?`/root/autodl-tmp/Neighbour_MIA/all_scores_sample.pkl`??? `scores`?`labels`?`predictions`?`details` ? `raw_score_dicts`?
- ?????`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_result.json`
- ???`/root/autodl-tmp/unified_mia_framework/outputs/logs/neighbour_mia.log`
- ?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack neighbour_mia --config ./configs/neighbour_mia.yml
```

- ???????????`returncode=0`?? 10 ????`accuracy=0.6`?`auc=0.6`?`precision=0.6`?`recall=0.6`?`f1=0.6`?

????????????????????????????????????????????? attack model checkpoint?????????


## DCMI ????????

- ??????`/root/autodl-tmp/DCMI`
- ?????`/root/autodl-tmp/DCMI/MIA.py`
- ?????`/root/autodl-tmp/DCMI/MIA_original.py`
- ?????`RuntimeError: Could not infer dtype of ellipsis`???? `MIA.py` ? `mem`?`perturb_mem`?`nom`?`perturb_nom` ?? `[...]` / Ellipsis ?????
- ?? DCMI ?????? `mem - perturb_mem` ? `nom - perturb_nom`????????? accuracy?AUC?precision?recall?F1?
- ????????? DCMI ? differential calibration score ??????????????????????????
  - `mem = [0.81, 0.76, 0.89, 0.74, 0.92]`
  - `perturb_mem = [0.21, 0.24, 0.28, 0.23, 0.31]`
  - `nonmem = [0.42, 0.38, 0.55, 0.47, 0.51]`
  - `perturb_nonmem = [0.31, 0.29, 0.36, 0.35, 0.37]`
- DCMI ??????`/root/autodl-tmp/DCMI/outputs/dcmi_result.json`
- DCMI ?????`/root/autodl-tmp/DCMI/outputs/dcmi_score_diff.png`
- ???????`/root/autodl-tmp/unified_mia_framework/outputs/dcmi_result.json`
- ???????`/root/autodl-tmp/unified_mia_framework/outputs/logs/dcmi.log`
- wrapper ???`attacks/dcmi.py` ?? import ????? `MIA.py`??? subprocess ?????????? DCMI ?????? JSON?
- ?????

```bash
cd /root/autodl-tmp/unified_mia_framework
/root/miniconda3/envs/dcmi/bin/python run_attack.py --attack dcmi --config ./configs/dcmi.yml
```

- ???????????`returncode=0`?10 ????`accuracy=1.0`?`auc=1.0`?`precision=1.0`?`recall=1.0`?`f1=1.0`?

????? DCMI ???????????????? unified_mia_framework ?????????????????????????????????? RAG ??????????????


## Neighbour_MIA ??? Membership Evaluation ??

- ??????? fine-tune ? GPT-2 ??????? member ??????? loss ??? membership score?
- ???????? `/root/autodl-tmp/RAG_MIA/datasets/nfcorpus/corpus.jsonl` ?????????/?????????????????
- member ???`/root/autodl-tmp/Neighbour_MIA/data/train_members.txt`?50 ??
- non-member ???`/root/autodl-tmp/Neighbour_MIA/data/test_nonmembers.txt`?50 ??
- ?????`/root/autodl-tmp/models/gpt2`?
- fine-tune ???`/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2`?
- fine-tune ?????? member ???2 epoch?? batch size?????? `outputs/logs/neighbour_mia_finetune.log`?
- evaluation ???

```bash
cd /root/autodl-tmp/Neighbour_MIA
/root/miniconda3/envs/dcmi/bin/python attack.py --model bert --dataset sample --output-file /root/autodl-tmp/Neighbour_MIA/all_scores_membership_eval.pkl
```

- ?????`/root/autodl-tmp/unified_mia_framework/outputs/logs/neighbour_mia_eval.log`?
- ?????`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_result.json`?
- ?????`/root/autodl-tmp/unified_mia_framework/outputs/neighbour_mia_eval_summary.json`?
- ???`accuracy=0.6`?`auc=0.6716`?`precision=0.6`?`recall=0.6`?`f1=0.6`?
- ?????member ?? membership score ? `1.3855`?non-member ?? membership score ? `1.0404`?member ?? loss ? `0.5925`?non-member ?? loss ? `4.3614`?
- ???loss ????????? fine-tuned GPT-2 ????????????membership score ????????? 50/50 ????????????????????????????


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


## HAMP privacy defense ????

??????`/root/autodl-tmp/HAMP`

???`https://github.com/DependableSystemsLab/MIA_defense_HAMP`

### ?????

HAMP ???????????????????/?????

- `cifar10/`
- `cifar100/`
- `purchase/`
- `texas/`
- `location/`

?????????

- `cifar10/cifar-train-hamp.py`
- `cifar10/lira-train-hamp.py`
- `cifar10/lira-inference-defense.py`
- `purchase/purchase-train-hamp.py`
- `texas/train-hamp.py`
- `location/train-hamp.py`

### HAMP ????

1. HAMP training ???
   - ????????? soft labels?
   - `get_top1()` ??????? `entropy_percentile` ????????
   - `get_soft_labels()` ??????????????????????????
   - ??????? `KLDivLoss` ?????? `F.kl_div(F.log_softmax(outputs, dim=1), targets)`?

2. entropy regularization?
   - ? `entropy_penalty` ??????????? `-alpha * mean(entropy)`?
   - ??????????????????????

3. output modification?
   - `lira-inference-defense.py` ??? `alter_output()`?
   - ?????? non-member ?????????????????????????????? membership scoring ???????

4. logits / probability handling?
   - ????? `softmax_by_row(logits, T=1.0)`?????? softmax?
   - score-based membership evaluation ??? confidence?entropy?modified entropy ??????

### ????????

?? unified_mia_framework ?????? LLM/text membership scoring??????????

```json
{
  "scores": [],
  "predictions": [],
  "labels": [],
  "metrics": {}
}
```

????? text-adapted minimal HAMP?

- ???`/root/autodl-tmp/unified_mia_framework/defenses/hamp.py`
- ????`hamp`
- ???`/root/autodl-tmp/unified_mia_framework/configs/hamp.yml`

???????

1. ??? membership scores ? min-max ?????? member ???
2. ?? temperature smoothing??? `temperature=1.5`?
3. ?? entropy boosting??? `alpha=0.1`????? 0.5 ???
4. ?? confidence clipping??? `max_confidence=0.8`?
5. ???? defended scores?predictions ? metrics?

### ?????

`run_defense.py --defense hamp` ???????

- `/root/autodl-tmp/unified_mia_framework/outputs/hamp_result.json`

Neighbour_MIA ? HAMP?

- before: accuracy=0.63, auc=0.6894
- after: accuracy=0.63, auc=0.68965
- auc_drop=-0.00025
- accuracy_drop=0.0
- ???`outputs/benchmark_neighbour_mia_hamp_result.json`

SPV_MIA ? HAMP?

- before: accuracy=0.55, auc=0.5875
- after: accuracy=0.55, auc=0.5875
- auc_drop=0.0
- accuracy_drop=0.0
- ???`outputs/benchmark_spv_mia_hamp_result.json`

### ????

?? HAMP ???????????????? confidence?????????? score ????? AUC ???????????????? protected classifier????????????????????????????????????? LLM ????? regularization ???????????????

### ????

?????????

- `run_attack.py --attack rag_mia --config ./configs/rag_mia.yml`
- `run_attack.py --attack neighbour_mia --config ./configs/neighbour_mia.yml`
- `run_attack.py --attack dcmi --config ./configs/dcmi.yml`
- `run_attack.py --attack spv_mia --config ./configs/spv_mia.yml`
- `run_defense.py --defense miashield --config ./configs/miashield.yml`
- `run_defense.py --defense hamp --config ./configs/hamp.yml`
- `run_benchmark.py --attack neighbour_mia --attack-config ./configs/neighbour_mia.yml --defense miashield --defense-config ./configs/miashield.yml`


## Utility evaluation ? privacy-utility tradeoff

?????`/root/autodl-tmp/unified_mia_framework/utils/utility_metrics.py`

?????

1. `retrieval_hit_rate()`??? RAG_MIA?? target_docs ? `retrieved_doc_ids` / `retrieved_docs` ???????????????
2. `qa_accuracy()`?? Yes / No / I don't know ????????? expected answers ???
3. `response_length()`??????????
4. `confidence_statistics()`??? membership scores ? mean/std/min/max?
5. `utility_summary()`????? attack ???? utility?

`run_benchmark.py` ???????? privacy metrics ? utility metrics?????

- `outputs/benchmark_summary.json`
- `outputs/benchmark_summary.md`

????

- `rag_mia ? miashield`??? `benchmark_rag_mia_miashield_result.json`??? retrieval ? QA utility?
- `spv_mia ? hamp`??? `benchmark_spv_mia_hamp_result.json`??? score-level utility?

?? RAG_MIA utility ???

- `/root/autodl-tmp/RAG_MIA/results/target_docs/mia-nfcorpus-phi3-phi3-bge-R3-Top1-M20-N5.json`

?????

- retrieval_hit_rate_before = 0.2983
- retrieval_hit_rate_after = 0.2983
- qa_accuracy_before = 0.4083
- qa_accuracy_after = 0.4083

???? defenses ??? membership score?????????????? retrieval/QA utility before-after ?????


## Query Sanitization input-level defense

?? defense?`query_sanitization`

???

- `/root/autodl-tmp/unified_mia_framework/defenses/query_sanitization.py`
- `/root/autodl-tmp/unified_mia_framework/configs/query_sanitization.yml`

???

- `/root/autodl-tmp/unified_mia_framework/defenses/registry.py`
- ????`query_sanitization`

### ????

1. `sanitize_query(query)`
   - ????? `[PERSON]`
   - ????? `[EMAIL]`
   - ??????ID??????? `[NUMBER]`
   - ???????? `medical_condition`

2. `sanitize_retrieval_context(docs)`
   - ???? str/list/dict ?? retrieved docs ? context?
   - ? `text/title/document/context/contents/question/query` ???? sanitization?

3. `apply(result)`
   - ?? RAG_MIA target_docs?
   - ?? query/context before-after ???
   - ?? sanitization ?????? membership scores?
   - ?? `utility_override`?? benchmark ???? retrieval/QA utility after?

### ??? benchmark

???

```bash
/root/miniconda3/envs/dcmi/bin/python run_benchmark.py \
  --attack rag_mia \
  --attack-config ./configs/rag_mia.yml \
  --defense query_sanitization \
  --defense-config ./configs/query_sanitization.yml
```

???

- before AUC = 1.0000
- after AUC = 0.9400
- auc_drop = 0.0600
- retrieval_hit_rate: 0.2983 -> 0.2362
- qa_accuracy: 0.4083 -> 0.3387
- utility_score: 0.4827 -> 0.4300

???

- `outputs/benchmark_rag_mia_query_sanitization_result.json`
- `outputs/query_sanitization_tradeoff.json`
- `outputs/query_sanitization_report.md`

### ? MIAShield/HAMP ???

MIAShield/HAMP ??? text-adapted minimal score-level defense????? membership scores?`query_sanitization` ? input-level defense?????? query ? retrieval context???????? RAG utility metrics?

### ????

???????

- paraphrase_defense
- retrieval_filter
- entity_relation_defense

????? defense ????? RAG retriever ? LLM generation??????? utility after?


## Query Sanitization real_rerun=true

`query_sanitization` ??????????

?????`/root/autodl-tmp/unified_mia_framework/configs/query_sanitization.yml`

?????

```yaml
real_rerun: true
extra_pythonpath: /root/autodl-tmp/phi3_compat_pkgs
```

?????`/root/autodl-tmp/unified_mia_framework/defenses/query_sanitization.py`

### real rerun ??

1. ?? RAG_MIA target_docs?
2. ??? question ?? `sanitize_query()`?
3. ?? RAG_MIA ??? `src.retrievers.create_retriever("bge", "nfcorpus")` ?????
4. ? retrieved context ?? `sanitize_retrieval_context()` / `sanitize_query()`?
5. ?? RAG_MIA ??? Phi3 wrapper ???? Yes/No/I don't know response?
6. ?? sanitized retrieval hit fallback ?????? doc ? score?
7. ???? scores / labels / predictions / metrics?
8. ???? retrieval_hit_rate / qa_accuracy / response_length / utility_score?

### ????

Phi3 ??????? `/root/autodl-tmp/phi3_compat_pkgs` ?? transformers ????????? `DynamicCache.seen_tokens` ???real_rerun ?????? RAG_MIA ?????????

### ??

- `outputs/benchmark_rag_mia_query_sanitization_realrerun_result.json`
- `outputs/logs/benchmark_rag_mia_query_sanitization_realrerun.log`
- `outputs/query_sanitization_realrerun_target_docs.json`
- `outputs/query_sanitization_tradeoff.json`
- `outputs/query_sanitization_report.md`

### ????

- before AUC = 1.0000
- after AUC = 0.9450
- auc_drop = 0.0550
- retrieval_hit_rate = 0.2983 -> 0.2850
- qa_accuracy = 0.4083 -> 0.2900
- utility_score = 0.4827 -> 0.4300
- num_queries = 600

?????? sanitized query ? BGE retrieval ? Phi3 generation ???? privacy-utility tradeoff??????????? `real_rerun: false` ????????


## Enhanced query_sanitization levels and scoring

`query_sanitization.py` ???? low/medium/high ???

1. low
   - PERSON masking
   - EMAIL masking

2. medium
   - NUMBER masking
   - medical keyword abstraction

3. high
   - semantic abstraction
   - retrieval-aware keyword dropping
   - entity relation abstraction

?????

- `semantic_abstraction(query)`
- `drop_retrieval_specific_terms(query)`
- `entity_relation_abstraction(query)`

real rerun ? membership scoring ????????

```text
final_score =
  0.4 * retrieval_overlap
+ 0.3 * response_similarity
+ 0.2 * confidence
+ 0.1 * exact_match_penalty
```

?? utility metrics?

- semantic_similarity
- retrieval_diversity
- response_consistency

### Low / Medium / High ??

| Level | AUC Before | AUC After | AUC Drop | QA Before | QA After | Retrieval Before | Retrieval After | Utility Before | Utility After |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 1.0000 | 1.0000 | 0.0000 | 0.4083 | 0.3633 | 0.2983 | 0.2933 | 0.4827 | 0.5236 |
| medium | 1.0000 | 0.9575 | 0.0425 | 0.4083 | 0.2900 | 0.2983 | 0.2850 | 0.4827 | 0.4819 |
| high | 1.0000 | 0.8438 | 0.1562 | 0.4083 | 0.2300 | 0.2983 | 0.1933 | 0.4827 | 0.3908 |

???high level ???????AUC ??? 0.8438??? QA accuracy ?? 0.23??????? utility?

???

- `outputs/query_sanitization_highlevel_report.md`
- `outputs/defense_comparison.md`



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

## Janus Fine-tuning Privacy Evaluation

### Official project analysis

- Project root: `/root/autodl-tmp/Janus`.
- Main example entries: `examples/janus_pretrain.py`, `examples/janus_attack.py`, `examples/janus_evaluation.py`, and `examples/evaluate.py`.
- Requirements include GPT-2/Transformers, Torch `<2.0` in the official requirements, Flair NER, `dp-transformers`, and Opacus.
- Official pretraining configuration simulates GPT-2 continual learning on ECHR and WikiText.
- Official Janus fine-tuning constructs PII association pairs through `TargetedAttackArgs.known_pii_pairs`; the example template maps `PERSON -> GPE` text pairs.
- Targeted recovery prompts are created by the Janus evaluation example and decoded by greedy/top-k/beam-search modes. Non-targeted PII extraction and reconstruction are handled through the generic `examples/evaluate.py` factory path.
- The official path depends on tagged real datasets and Flair-based PII analysis; that path is documented but intentionally not used by the framework safety entry.

### Synthetic-safe framework entry

The framework adds `/root/autodl-tmp/Janus/run_janus_minimal.py` for a bounded local reproduction:

1. Load local GPT-2 from `/root/autodl-tmp/models/gpt2`.
2. Read 30 synthetic train pairs and 100 synthetic eval pairs from JSONL.
3. Fine-tune with AdamW, `learning_rate=5e-5`, `batch_size=4`, `epochs=2`, `max_length=128`.
4. Run deterministic targeted prompts such as `The email address of User_A031 is` with `max_new_tokens=32`, `num_beams=5`, `do_sample=false`.
5. Count exact recovery only when the generated text contains the expected synthetic email string.

### Current result and output format

The first synthetic-safe run returned `returncode=0`, recovered `86/100` generated synthetic eval strings, and wrote:

- `/root/autodl-tmp/Janus/outputs/janus_result.json`
- `/root/autodl-tmp/unified_mia_framework/outputs/janus_result.json`
- `/root/autodl-tmp/unified_mia_framework/outputs/logs/janus.log`
- `/root/autodl-tmp/unified_mia_framework/outputs/janus_report.md`

Janus differs from the binary membership scoring entries: `recovery_rate` is the primary privacy evaluation result, while unified `scores`, `labels`, `predictions`, and `metrics` are retained for framework compatibility. All data use synthetic identifiers and the `example-synthetic.com` domain.

## LoRA-Janus: PEFT-based Fine-tuning Privacy Evaluation

### Implementation

`/root/autodl-tmp/Janus/run_janus_minimal.py` now supports:

- `--finetune-mode no_ft`
- `--finetune-mode full_ft`
- `--finetune-mode lora_ft`

`full_ft` updates all GPT-2 parameters with AdamW. `lora_ft` uses PEFT LoRA with `target_modules=["c_attn"]`, `r=8`, `alpha=16`, and `dropout=0.05`, freezing the base GPT-2 weights and training only adapter parameters. The wrapper `/root/autodl-tmp/unified_mia_framework/attacks/janus.py` passes the mode and LoRA parameters from YAML.

### Configs

- `/root/autodl-tmp/unified_mia_framework/configs/janus_no_ft.yml`
- `/root/autodl-tmp/unified_mia_framework/configs/janus_full_ft.yml`
- `/root/autodl-tmp/unified_mia_framework/configs/janus_lora_ft.yml`

### Current synthetic-safe results

| Mode | Trainable Params | Trainable Ratio | Recovery Rate | Recovered / Eval |
|---|---:|---:|---:|---:|
| no_ft | 0 | 0.0000% | 0.0000 | 0 / 100 |
| full_ft | 124439808 | 100.0000% | 0.8600 | 86 / 100 |
| lora_ft | 294912 | 0.2364% | 0.0000 | 0 / 100 |

### Interpretation

With synthetic-safe data, no fine-tuning recovers no target emails. Full fine-tuning recovers `86/100`, showing strong memorization/generalization of synthetic email associations. LoRA fine-tuning with `r=8`, `lr=5e-5`, and `epochs=2` trains only `294912` parameters (`0.2364%`) and currently recovers `0/100`. This indicates the first LoRA configuration is too weak to match full fine-tuning privacy risk, not that LoRA-Janus is impossible. Follow-up sweeps should increase rank, learning rate, epochs, and train size.

Outputs:

- `/root/autodl-tmp/unified_mia_framework/outputs/janus_lora_comparison.md`
- `/root/autodl-tmp/unified_mia_framework/outputs/janus_lora_comparison.json`
- `/root/autodl-tmp/unified_mia_framework/outputs/janus_lora_report.md`

## LoRA-Janus Threshold Analysis

### Sweep setup

The sweep uses only synthetic identifiers and synthetic emails under `example-synthetic.com`. No real personal information is used. The grid contains 36 runs:

- `lora_r`: 8, 16, 32
- `learning_rate`: 5e-5, 1e-4, 5e-4
- `epochs`: 2, 5
- `train_size`: 30, 50
- fixed `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["c_attn"]`

### Best and threshold behavior

Best run:

- r = `8`
- learning_rate = `0.0005`
- epochs = `5`
- train_size = `30`
- recovery_rate = `1.0000`

First positive recovery:

- `{'lora_r': 8, 'learning_rate': 0.0005, 'epochs': 5, 'train_size': 30, 'recovery_rate': 1.0, 'recovered': 100, 'eval_size': 100, 'trainable_parameters': 294912, 'total_parameters': 124734720, 'trainable_ratio': 0.0023643136409814366, 'train_losses': [4.114179939031601, 3.2703364491462708, 2.284640535712242, 1.1850039809942245, 0.6659643314778805], 'returncode': 0, 'output_path': '/root/autodl-tmp/Janus/outputs/lora_sweep_runs/janus_lora_r8_lr5em4_ep5_tr30.json', 'log_path': '/root/autodl-tmp/Janus/outputs/lora_sweep_logs/janus_lora_r8_lr5em4_ep5_tr30.log'}`

High recovery runs (`>=0.9`): `6` / `36`.

### Aggregated effects

#### Rank

- `lora_r=8`: mean recovery `0.1658` over `12` runs
- `lora_r=16`: mean recovery `0.1650` over `12` runs
- `lora_r=32`: mean recovery `0.1683` over `12` runs

#### Epochs

- `epochs=2`: mean recovery `0.0022` over `18` runs
- `epochs=5`: mean recovery `0.3306` over `18` runs

#### Learning rate

- `learning_rate=5e-05`: mean recovery `0.0000` over `12` runs
- `learning_rate=0.0001`: mean recovery `0.0000` over `12` runs
- `learning_rate=0.0005`: mean recovery `0.4992` over `12` runs

#### Train size

- `train_size=30`: mean recovery `0.1661` over `18` runs
- `train_size=50`: mean recovery `0.1667` over `18` runs

### Interpretation

The main driver in this grid is learning rate combined with epoch count. `lr=5e-5` and `lr=1e-4` do not recover synthetic emails even at 5 epochs. At `lr=5e-4`, 5 epochs produces recovery rates from `0.98` to `1.00` across ranks and train sizes. Rank has a weaker effect than optimization strength here; even `r=8` reaches `1.00` with sufficiently strong training.

Compared with full FT (`0.86` recovery in the earlier baseline), LoRA can be safer at weak settings, but can match or exceed full FT synthetic recovery when the adapter is trained aggressively. This is a threshold effect rather than a monotonic rank-only story.

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
