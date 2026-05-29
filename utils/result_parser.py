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


def empty_result(attack_name: str) -> Dict[str, Any]:
    """生成统一空结果。"""
    return {
        "attack_name": attack_name,
        "scores": [],
        "predictions": [],
        "labels": [],
        "metrics": {
            "accuracy": 0.0,
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    }


def _as_float_list(values: Iterable[Any]) -> List[float]:
    """把列表元素尽量转换成 float。"""
    output: List[float] = []
    for value in values or []:
        try:
            output.append(float(value))
        except Exception:
            continue
    return output


def _as_int_list(values: Iterable[Any]) -> List[int]:
    """把列表元素尽量转换成 int。"""
    output: List[int] = []
    for value in values or []:
        try:
            output.append(int(value))
        except Exception:
            continue
    return output


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


def _parse_rag_mia_target_docs(attack_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """把 RAG_MIA target_docs 转换为统一 scores/labels。

    evaluate_attack=true 时，优先根据 answers 与 llm_responses 的 Yes/No
    一致性计算分数；如果还没有 LLM 回答，则退回检索命中率。
    """
    result = empty_result(attack_name)
    scores: List[float] = []
    labels: List[int] = []
    for doc_id, doc_data in data.items():
        # evaluate_attack=true 后优先使用回答正确率。原项目会打印 accuracy，
        # 但当前版本没有 save_target_docs，所以这里从 answers/llm_responses 复算。
        answers = doc_data.get("answers", []) or []
        responses = doc_data.get("llm_responses", []) or []
        if answers and responses:
            total = min(len(answers), len(responses))
            correct = 0
            for answer, response in zip(answers[:total], responses[:total]):
                if _extract_yes_no(answer) != "Unknown" and _extract_yes_no(answer) == _extract_yes_no(response):
                    correct += 1
            scores.append(float(correct / total if total else 0.0))
            labels.append(1 if str(doc_data.get("mem", "")).lower() == "yes" else 0)
            continue
        if "accuracy" in doc_data and doc_data.get("accuracy") is not None:
            try:
                score = float(doc_data.get("accuracy", 0.0))
                scores.append(score / 100.0 if score > 1 else score)
                labels.append(1 if str(doc_data.get("mem", "")).lower() == "yes" else 0)
                continue
            except Exception:
                pass
        retrieved = doc_data.get("retrieved_doc_ids", []) or []
        total = len(retrieved)
        hits = 0
        for ids in retrieved:
            if isinstance(ids, list) and str(doc_id) in [str(x) for x in ids]:
                hits += 1
        scores.append(float(hits / total if total else 0.0))
        labels.append(1 if str(doc_data.get("mem", "")).lower() == "yes" else 0)
    result["scores"] = scores
    result["labels"] = labels
    result["predictions"] = _threshold_predictions(scores)
    result["metrics"] = compute_metrics(labels, result["predictions"], scores)
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
            result["predictions"] = _threshold_predictions(result["scores"])

    metrics = data.get("metrics", {})
    for key in ["accuracy", "auc", "precision", "recall", "f1"]:
        if key in data:
            metrics[key] = data[key]
    if metrics:
        result["metrics"].update({k: float(v) for k, v in metrics.items() if k in result["metrics"]})
    else:
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


def parse_result_sources(
    attack_name: str,
    output_path: Optional[str] = None,
    stdout_text: str = "",
    candidate_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """按优先级解析原论文输出。"""
    candidates: List[str] = []
    if output_path:
        candidates.append(output_path)
    for directory in candidate_dirs or []:
        if os.path.isdir(directory):
            for root, _, files in os.walk(directory):
                for name in files:
                    if name.endswith((".json", ".pkl", ".pickle", ".npy", ".txt")):
                        candidates.append(os.path.join(root, name))
    candidates = [p for p in candidates if p and os.path.isfile(p)]
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for path in candidates:
        try:
            if path.endswith(".json"):
                parsed = parse_json_file(attack_name, path)
            elif path.endswith((".pkl", ".pickle")):
                parsed = parse_pickle_file(attack_name, path)
            elif path.endswith(".npy"):
                parsed = parse_numpy_file(attack_name, path)
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    parsed = parse_stdout_metrics(attack_name, f.read())
            if parsed["scores"] or parsed["labels"] or any(parsed["metrics"].values()):
                return parsed
        except Exception:
            continue

    return parse_stdout_metrics(attack_name, stdout_text)
