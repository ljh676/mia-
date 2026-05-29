"""Text-adapted MIAShield minimal defense.

??? MIAShield ??????? ensemble model??????????
membership scoring benchmark???? hash ? score source perturbation ??
exclusion oracle??????? attack x defense benchmark ???
"""

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional

from .base import BaseDefense
from utils.metrics import compute_metrics


class MIAShieldDefense(BaseDefense):
    """MIAShield ? text-adapted minimal ???"""

    defense_name = "miashield"

    def prepare(self):
        """????????????"""
        self.project_path = self.config.get("project_path", "/root/autodl-tmp/MIAShield")
        self.oracle = self.config.get("oracle", "chain")
        self.num_subsets = int(self.config.get("num_subsets", 4))
        self.aggregation = self.config.get("aggregation", "average")
        self.hash_method = self.config.get("hash_method", "sha1")
        self.output_path = self.config.get("output_path", "./outputs/miashield_result.json")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        return self

    def _hash_text(self, text: str) -> int:
        """???????? subset id?"""
        normalized = " ".join(str(text).lower().split())
        if self.hash_method == "md5":
            digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        else:
            digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _exact_signature(self, item: Dict[str, Any], index: int) -> Optional[int]:
        """exact text hash oracle??????????????? subset id?"""
        text = item.get("text") or item.get("original_text") or item.get("query") or item.get("document")
        if not text:
            # benchmark result ???? scores/labels???? index ???????
            text = f"sample-{index}-label-{item.get('label', 'unknown')}"
        return self._hash_text(str(text)) % self.num_subsets

    def _candidate_scores(self, score: float, index: int) -> List[float]:
        """???? subset model ? membership scores?

        ?? MIAShield ?? ensemble model????????????? LLM?
        ????? score ????? candidate sources???? exclusion pipeline?
        """
        candidates = []
        for subset in range(self.num_subsets):
            phase = (index + 1) * (subset + 1)
            offset = 0.08 * math.sin(phase) + 0.03 * math.cos(phase * 0.7)
            candidates.append(float(score + offset))
        return candidates

    def _confidence_based(self, candidates: List[float]) -> List[float]:
        """??? confident ? source?"""
        if len(candidates) <= 1:
            return candidates
        center = sum(candidates) / len(candidates)
        remove_index = max(range(len(candidates)), key=lambda i: abs(candidates[i] - center))
        return [v for i, v in enumerate(candidates) if i != remove_index]

    def _aggregate(self, candidates: List[float]) -> float:
        """???? source?"""
        if not candidates:
            return 0.0
        if self.aggregation == "majority_vote":
            votes = [1 if v >= 0.5 else 0 for v in candidates]
            return float(sum(votes) / len(votes))
        return float(sum(candidates) / len(candidates))

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """? attack/evaluation ????? MIAShield-style exclusion?"""
        scores = [float(v) for v in data.get("scores", [])]
        labels = [int(v) for v in data.get("labels", [])]
        details = data.get("details", []) or []
        defended_scores: List[float] = []
        excluded_subsets: List[int] = []

        for index, score in enumerate(scores):
            item = details[index] if index < len(details) and isinstance(details[index], dict) else {}
            item["label"] = labels[index] if index < len(labels) else None
            candidates = self._candidate_scores(score, index)

            excluded = None
            if self.oracle in {"exact_signature", "chain"}:
                excluded = self._exact_signature(item, index)
            if self.oracle == "confidence_based" or (self.oracle == "chain" and excluded is None):
                kept = self._confidence_based(candidates)
            elif excluded is not None:
                kept = [v for i, v in enumerate(candidates) if i != excluded]
                if self.oracle == "chain" and kept:
                    kept = self._confidence_based(kept)
            else:
                kept = candidates
            defended_scores.append(self._aggregate(kept))
            excluded_subsets.append(-1 if excluded is None else int(excluded))

        if defended_scores:
            threshold = sorted(defended_scores)[len(defended_scores) // 2]
            predictions = [1 if score >= threshold else 0 for score in defended_scores]
        else:
            threshold = 0.0
            predictions = []
        metrics = compute_metrics(labels, predictions, defended_scores) if labels else {}
        return {
            "defense_name": self.defense_name,
            "status": "success",
            "oracle": self.oracle,
            "aggregation": self.aggregation,
            "scores": defended_scores,
            "predictions": predictions,
            "labels": labels,
            "metrics": metrics,
            "threshold": threshold,
            "excluded_subsets": excluded_subsets,
            "notes": "minimal text-adapted MIAShield defense is available",
        }

    def evaluate(self, before_result, after_result):
        """?? before/after ?????"""
        before_metrics = before_result.get("metrics", {}) or {}
        after_metrics = after_result.get("metrics", {}) or {}
        auc_before = float(before_metrics.get("auc", 0.0))
        auc_after = float(after_metrics.get("auc", 0.0))
        acc_before = float(before_metrics.get("accuracy", 0.0))
        acc_after = float(after_metrics.get("accuracy", 0.0))
        return {
            "auc_drop": auc_before - auc_after,
            "accuracy_drop": acc_before - acc_after,
            "defense_effectiveness": max(0.0, auc_before - auc_after),
        }
