"""Utility evaluation ?????

?????? privacy benchmark ?????????? privacy defense ???
??????????????????????RAG_MIA target_docs???
??????? retrieval?QA?response length ? confidence ???
"""

import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    """???????? float??????????"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flatten(values: Any) -> List[Any]:
    """??????????????? questions/responses?"""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _normalize_answer(text: Any) -> Optional[str]:
    """????????? Yes / No / I don't know?

    ?????????????????????????????? None?
    """
    if text is None:
        return None
    value = str(text).strip().lower()
    value = value.replace("?", "'").replace(".", " ").replace(",", " ")
    value = " ".join(value.split())
    if not value:
        return None
    if "i don't know" in value or "i do not know" in value or "unknown" in value:
        return "I don't know"
    if value.startswith("yes") or "answer is yes" in value:
        return "Yes"
    if value.startswith("no") or "answer is no" in value:
        return "No"
    return None


def _extract_records(container: Any) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    """? list/dict ??? target_docs ??? (doc_id, record) ???"""
    if isinstance(container, dict):
        return [(str(key), value) for key, value in container.items() if isinstance(value, dict)]
    if isinstance(container, list):
        records = []
        for index, value in enumerate(container):
            if isinstance(value, dict):
                doc_id = value.get("doc_id") or value.get("id") or value.get("_id") or str(index)
                records.append((str(doc_id), value))
        return records
    return []


def retrieval_hit_rate(retrieved_docs: Any, target_docs: Any) -> Dict[str, float]:
    """?? RAG retrieval hit rate?

    ????? RAG_MIA ? target_docs dict/list???????? retrieved_docs
    ? target_docs???? target doc id ????????????? retrieved doc
    ?????? target text?
    """
    records = _extract_records(target_docs)
    if not records and isinstance(retrieved_docs, (dict, list)):
        records = _extract_records(retrieved_docs)

    num_queries = 0
    hit_count = 0
    retrieved_count = 0

    for doc_id, record in records:
        target_text = str(record.get("text") or record.get("document") or record.get("target_text") or "")
        per_query_docs = record.get("retrieved_docs") or record.get("retrieved_doc_ids") or record.get("contexts") or []
        if not isinstance(per_query_docs, list):
            per_query_docs = [per_query_docs]

        for docs in per_query_docs:
            docs_list = docs if isinstance(docs, list) else [docs]
            docs_list = [doc for doc in docs_list if doc is not None]
            if not docs_list:
                num_queries += 1
                continue
            num_queries += 1
            retrieved_count += len(docs_list)

            found = False
            for doc in docs_list:
                if isinstance(doc, dict):
                    rid = str(doc.get("id") or doc.get("doc_id") or doc.get("_id") or "")
                    rtext = str(doc.get("text") or doc.get("document") or doc.get("contents") or "")
                else:
                    rid = str(doc)
                    rtext = str(doc)
                if doc_id and rid == doc_id:
                    found = True
                elif target_text and target_text[:160].lower() in rtext.lower():
                    found = True
                if found:
                    break
            if found:
                hit_count += 1

    return {
        "hit_rate": float(hit_count / num_queries) if num_queries else None,
        "avg_retrieved_docs": float(retrieved_count / num_queries) if num_queries else 0.0,
        "num_queries": int(num_queries),
    }


def qa_accuracy(responses, expected_answers):
    responses = _flatten(responses)
    expected = _flatten(expected_answers)
    references = [_normalize_answer(v) for v in expected]
    if not references or len(responses) != len(references) or any(v is None for v in references):
        return {"qa_accuracy": None, "correct": None, "total": 0, "status": "missing_or_incomplete_gold_answers"}
    correct = sum(_normalize_answer(response) == answer for response, answer in zip(responses, references))
    return {"qa_accuracy": correct / len(references), "correct": correct, "total": len(references), "status": "measured"}


def response_length(responses: Any) -> Dict[str, float]:
    """?? response ????????????? token/word ??"""
    values = [str(v).strip() for v in _flatten(responses) if str(v).strip()]
    lengths = [len(value.split()) for value in values]
    if not lengths:
        return {"avg_response_length": 0.0, "min_response_length": 0.0, "max_response_length": 0.0, "num_responses": 0}
    return {
        "avg_response_length": float(sum(lengths) / len(lengths)),
        "min_response_length": float(min(lengths)),
        "max_response_length": float(max(lengths)),
        "num_responses": int(len(lengths)),
    }


def confidence_statistics(scores: Iterable[float]) -> Dict[str, float]:
    """?? membership scores ???????????"""
    values = [_safe_float(v) for v in (scores or [])]
    if not values:
        return {"mean_score": 0.0, "std_score": 0.0, "max_score": 0.0, "min_score": 0.0}
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {
        "mean_score": float(mean_value),
        "std_score": float(math.sqrt(variance)),
        "max_score": float(max(values)),
        "min_score": float(min(values)),
    }


def _latest_rag_target_docs(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """???????? RAG_MIA target_docs ???"""
    config = config or {}
    explicit = config.get("target_docs_path") or config.get("target_docs_file")
    if explicit and os.path.exists(explicit):
        return explicit
    project_path = config.get("project_path", "/root/autodl-tmp/RAG_MIA")
    sample_num = config.get("sample_num")
    question_num = config.get("question_num")
    pattern = os.path.join(project_path, "results", "target_docs", "*.json")
    candidates = glob.glob(pattern)
    if sample_num and question_num:
        preferred = [p for p in candidates if f"M{sample_num}-N{question_num}" in os.path.basename(p)]
        if preferred:
            candidates = preferred
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _load_json(path: str) -> Any:
    """?? JSON ???"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rag_responses_and_expected(records):
    responses, expected = [], []
    for _, record in records:
        current = _flatten(record.get("llm_responses") or record.get("model_responses") or record.get("responses") or [])
        gold = _flatten(record.get("answers") or record.get("gold_answers") or record.get("expected_answers") or [])
        responses.extend(current)
        # Keep alignment across records; never invent a Yes reference.
        expected.extend(gold if len(gold) == len(current) else [None] * len(current))
    return responses, expected


def utility_summary(result, attack_name=None, config=None, fallback_utility=None):
    attack_name = attack_name or result.get("attack_name") or "unknown"
    responses = result.get("responses") or result.get("llm_responses") or []
    expected = result.get("expected_answers") or result.get("answers") or []
    summary = {
        "attack_name": attack_name,
        "confidence": confidence_statistics(result.get("scores", [])),
        "response_length": response_length(responses),
        "retrieval": {"hit_rate": None, "avg_retrieved_docs": None, "num_queries": 0},
        "qa": qa_accuracy(responses, expected),
        "utility_source": "current_result",
        "utility_score": None,
        "utility_score_status": "no_validated_composite_defined",
        "retrieval_metric_scope": "target_document_hit_rate; attack diagnostic, not general task accuracy",
    }
    override = result.get("utility_override")
    if isinstance(override, dict) and result.get("real_rerun") is True:
        if (override.get("retrieval") or {}).get("num_queries", 0) > 0:
            summary["retrieval"] = override["retrieval"]
        if override.get("qa_reference_source") == "gold_answers":
            summary["qa"] = override["qa"]
        summary["response_length"] = override.get("response_length", summary["response_length"])
        summary["utility_source"] = "measured_rerun"
    elif attack_name == "rag_mia" and not result.get("defense_name"):
        # Only consume this run's explicit artifact, never search for a latest old file.
        path = result.get("parsed_output_path") or result.get("target_docs_path")
        if path and os.path.isfile(path):
            target_docs = _load_json(path)
            records = _extract_records(target_docs)
            if records and any("retrieved_doc_ids" in record for _, record in records):
                responses, expected = _extract_rag_responses_and_expected(records)
                summary["target_docs_path"] = path
                summary["retrieval"] = retrieval_hit_rate(None, target_docs)
                summary["qa"] = qa_accuracy(responses, expected)
                summary["response_length"] = response_length(responses)
                summary["qa_expected_source"] = "gold_answers_only"
    if fallback_utility is not None and result.get("task_output_changed") is False:
        for key in ("retrieval", "qa", "response_length", "target_docs_path", "qa_expected_source"):
            if key in fallback_utility:
                summary[key] = fallback_utility[key]
        summary["utility_source"] = "inherited_unchanged_task_output; not remeasured"
    summary["retrieval_hit_rate"] = summary["retrieval"].get("hit_rate")
    summary["qa_accuracy"] = summary["qa"].get("qa_accuracy")
    summary["avg_response_length"] = summary["response_length"].get("avg_response_length") if summary["response_length"].get("num_responses", 0) else None
    summary["mean_score"] = summary["confidence"].get("mean_score") if result.get("scores") else None
    return summary


def utility_drop(before, after):
    from utils.metrics import metric_delta
    keys = ["retrieval_hit_rate", "qa_accuracy", "avg_response_length", "mean_score", "utility_score"]
    return {f"{key}_drop": metric_delta(before.get(key), after.get(key)) for key in keys}
