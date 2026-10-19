"""统一结果解析器。

该模块负责把原论文可能产生的 json / txt / stdout / pickle / numpy 输出，
尽量转换成统一成员推断攻击结果格式。
"""

import json
import os
import pickle
import re
from typing import Any, Dict, Iterable, List, Optional

from utils.metrics import compute_metrics


def empty_result(attack_name):
    return {"attack_name": attack_name, "status": "not_run", "scores": [], "predictions": [], "labels": [],
            "metrics": compute_metrics([], [], [])}


def _as_float_list(values):
    import math
    output = [float(value) for value in ([] if values is None else values)]
    if any(not math.isfinite(value) for value in output):
        raise ValueError("Result contains non-finite scores")
    return output


def _as_int_list(values):
    output = _as_float_list(values)
    if any(value not in (0, 1) for value in output):
        raise ValueError("Result labels/predictions must be binary")
    return [int(value) for value in output]


def _threshold_predictions(scores: List[float]) -> List[int]:
    """当原结果没有 predictions 时生成二分类预测。

    对 RAG_MIA 的回答正确率/命中率这类 0-1 分数，固定使用 0.5
    更符合直觉；其他连续分数仍使用中位数阈值。
    """
    if not scores:
        return []
    if all(0.0 <= score <= 1.0 for score in scores):
        return [1 if score >= 0.5 else 0 for score in scores]
    ordered = sorted(scores)
    threshold = ordered[len(ordered) // 2]
    return [1 if score >= threshold else 0 for score in scores]


def _looks_like_rag_mia_target_docs(data: Dict[str, Any]) -> bool:
    """判断 JSON 是否是 RAG_MIA 的 target_docs 结果文件。"""
    if not data:
        return False
    first_value = next(iter(data.values()))
    return isinstance(first_value, dict) and "mem" in first_value and "retrieved_doc_ids" in first_value


def _extract_yes_no(text: Any) -> str:
    """把自由文本回答归一化为 Yes / No / I don't know / Unknown。"""
    value = re.sub(r"\s+", " ", str(text)).strip()
    lowered = value.lower()
    if "i don't know" in lowered or "i do not know" in lowered or "unknown" in lowered:
        return "I don't know"
    # 支持 Yes. / Yes, ... / The answer is yes 等常见格式。
    if re.search(r"\byes\b", lowered):
        return "Yes"
    if re.search(r"\bno\b", lowered):
        return "No"
    return "Unknown"


def _parse_rag_mia_target_docs(attack_name, data):
    result = empty_result(attack_name)
    scores, labels = [], []
    for doc_id, row in data.items():
        membership = str(row.get("mem", "")).lower()
        if membership not in {"yes", "no"}:
            raise ValueError(f"Missing/invalid membership annotation for {doc_id}")
        gold, responses = row.get("answers"), row.get("llm_responses")
        if row.get("score_answer_source") == "retrieval_hit_fallback":
            raise ValueError(f"RAG artifact {doc_id} uses retrieval-hit pseudo answers; regenerate real reference answers")
        if not isinstance(gold, list) or not isinstance(responses, list) or not gold or len(gold) != len(responses):
            raise ValueError(f"RAG artifact {doc_id} lacks aligned reference answers and model responses; retrieval hits cannot replace the paper's answer-agreement score")
        expected = [_extract_yes_no(value) for value in gold]
        if any(value == "Unknown" for value in expected):
            raise ValueError(f"RAG artifact {doc_id} has unrecognized reference answers")
        scores.append(sum(_extract_yes_no(response) == answer for response, answer in zip(responses, expected)) / len(expected))
        labels.append(int(membership == "yes"))
    result.update({"scores": scores, "labels": labels, "predictions": [], "sample_ids": list(data),
                   "metrics": compute_metrics(labels, [], scores), "score_source": "reference_answer_agreement"})
    return result


def _normalize_result(attack_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """把常见字段名归一化为统一格式。"""
    result = empty_result(attack_name)

    if _looks_like_rag_mia_target_docs(data):
        return _parse_rag_mia_target_docs(attack_name, data)

    if {"mem", "perturb_mem", "nom", "perturb_nom"}.issubset(data):
        mem = _as_float_list(data.get("mem", []))
        perturb_mem = _as_float_list(data.get("perturb_mem", []))
        nom = _as_float_list(data.get("nom", []))
        perturb_nom = _as_float_list(data.get("perturb_nom", []))
        if len(mem) != len(perturb_mem) or len(nom) != len(perturb_nom):
            raise ValueError("Original and perturbed score arrays have different lengths")
        member_diff = [a - b for a, b in zip(mem, perturb_mem)]
        nonmember_diff = [a - b for a, b in zip(nom, perturb_nom)]
        scores = member_diff + nonmember_diff
        labels = [1] * len(member_diff) + [0] * len(nonmember_diff)
        predictions = _threshold_predictions(scores)
        result.update({"scores": scores, "labels": labels, "predictions": predictions})
    else:
        scores = data.get("scores", data.get("score", data.get("all_diff", [])))
        labels = data.get("labels", data.get("y_true", []))
        predictions = data.get("predictions", data.get("preds", data.get("y_pred", [])))
        result["scores"] = _as_float_list(scores)
        result["labels"] = _as_int_list(labels)
        result["predictions"] = _as_int_list(predictions)
        if result["scores"] and result["labels"] and not result["predictions"]:
            result["predictions"] = []

    # Recompute from raw aligned samples; do not trust external aggregate metrics.
    result["metrics"] = compute_metrics(result["labels"], result["predictions"], result["scores"])
    return result


def parse_json_file(attack_name: str, path: str) -> Dict[str, Any]:
    """解析 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return _normalize_result(attack_name, data)
    return empty_result(attack_name)


def parse_pickle_file(attack_name: str, path: str) -> Dict[str, Any]:
    """解析 pickle 文件。"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return _normalize_result(attack_name, data)
    return empty_result(attack_name)


def parse_numpy_file(attack_name: str, path: str) -> Dict[str, Any]:
    """解析 numpy 文件。"""
    try:
        import numpy as np
    except Exception:
        return empty_result(attack_name)
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "item"):
        try:
            obj = data.item()
            if isinstance(obj, dict):
                return _normalize_result(attack_name, obj)
        except Exception:
            pass
    return empty_result(attack_name)


def parse_stdout_metrics(attack_name: str, text: str) -> Dict[str, Any]:
    """从 stdout/stderr 文本中提取常见指标。"""
    result = empty_result(attack_name)
    patterns = {
        "accuracy": r"(?:Best Accuracy|Accuracy)\s*[:=]\s*([0-9.]+)%?",
        "auc": r"AUC\s*[:=]\s*([0-9.]+)",
        "precision": r"Precision\s*[:=]\s*([0-9.]+)",
        "recall": r"Recall\s*[:=]\s*([0-9.]+)",
        "f1": r"(?:F1-score|F1)\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if key == "accuracy" and value > 1:
                value = value / 100.0
            result["metrics"][key] = value
    return result


def parse_result_sources(attack_name, output_path=None, stdout_text="", candidate_dirs=None, fresh_outputs=None):
    if fresh_outputs is None:
        raise ValueError("Fresh output provenance is required; old directories cannot be scanned implicitly")
    allowed = {os.path.abspath(path) for path in fresh_outputs}
    if output_path:
        preferred = os.path.abspath(output_path)
        if preferred not in allowed:
            raise RuntimeError(f"Configured output was not refreshed: {preferred}")
        candidates = [preferred]
    else:
        candidates = sorted(allowed, key=lambda path: (not path.endswith('.json'), path))
    for path in candidates:
        if path.endswith('.json'):
            parsed = parse_json_file(attack_name, path)
        elif path.endswith(('.pkl', '.pickle')):
            parsed = parse_pickle_file(attack_name, path)
        elif path.endswith('.npy'):
            parsed = parse_numpy_file(attack_name, path)
        else:
            continue
        if parsed['scores'] or parsed['labels']:
            parsed.update({'parsed_output_path': path, 'status': 'success'})
            return parsed
    raise RuntimeError("External method produced no fresh per-sample result; stdout-only metrics are not accepted")
