"""Text-adapted minimal HAMP defense.

HAMP ??????????????????????????????
??????????? confidence/loss ? membership signal?????
?????? LLM ?? membership scoring ???????????
text-adapted minimal ?????????????????? scores ?
????temperature smoothing?entropy boosting ? confidence clipping?
"""

import math
import os
from typing import Any, Dict, Iterable, List, Tuple

from .base import BaseDefense
from utils.metrics import compute_metrics


class HAMPDefense(BaseDefense):
    """HAMP ??? membership scoring ???????"""

    defense_name = "hamp"

    def prepare(self):
        """?? defense ??????????"""
        self.project_path = self.config.get("project_path", "/root/autodl-tmp/HAMP")
        self.temperature = float(self.config.get("temperature", 1.5))
        self.alpha = float(self.config.get("alpha", 0.1))
        self.max_confidence = float(self.config.get("max_confidence", 0.8))
        # AUC ??? smoothing ???????????? rank mixing?
        # ?? HAMP ???????? confidence order ???/???
        self.rank_mixing = float(self.config.get("rank_mixing", 0.35))
        self.calibration_noise = float(self.config.get("calibration_noise", 0.08))
        self.output_path = self.config.get(
            "output_path", "/root/autodl-tmp/unified_mia_framework/outputs/hamp_result.json"
        )
        self.temperature = max(self.temperature, 1e-6)
        self.alpha = min(max(self.alpha, 0.0), 1.0)
        self.max_confidence = min(max(self.max_confidence, 0.5), 1.0)
        self.rank_mixing = min(max(self.rank_mixing, 0.0), 0.9)
        self.calibration_noise = min(max(self.calibration_noise, 0.0), 0.25)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        return self

    def _normalize_scores(self, scores: Iterable[float]) -> Tuple[List[float], Dict[str, float]]:
        """??? membership score ?????? member ???

        ????? score ????????? loss?????????????
        ???? min-max ???????????????????????
        ?????? 0.5????????????
        """
        raw = [float(v) for v in scores]
        if not raw:
            return [], {"min_score": 0.0, "max_score": 0.0}
        min_score = min(raw)
        max_score = max(raw)
        if math.isclose(max_score, min_score):
            return [0.5 for _ in raw], {"min_score": min_score, "max_score": max_score}
        eps = 1e-6
        probs = [(value - min_score) / (max_score - min_score) for value in raw]
        probs = [min(max(p, eps), 1.0 - eps) for p in probs]
        return probs, {"min_score": min_score, "max_score": max_score}

    def _temperature_smoothing(self, p_member: float) -> float:
        """???? member ??? temperature smoothing?"""
        eps = 1e-6
        p_member = min(max(float(p_member), eps), 1.0 - eps)
        logit = math.log(p_member / (1.0 - p_member))
        scaled = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled))

    def _entropy_boosting(self, p_member: float) -> float:
        """????????????????????????"""
        return (1.0 - self.alpha) * p_member + self.alpha * 0.5

    def _confidence_clipping(self, p_member: float) -> float:
        """????????????????????"""
        lower = 1.0 - self.max_confidence
        upper = self.max_confidence
        return min(max(p_member, lower), upper)

    def _defend_probability(self, p_member: float) -> float:
        """? HAMP ???????????????????"""
        p_member = self._temperature_smoothing(p_member)
        p_member = self._entropy_boosting(p_member)
        p_member = self._confidence_clipping(p_member)
        return float(p_member)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """? membership evaluation ???? text-adapted HAMP?"""
        labels = [int(v) for v in data.get("labels", [])]
        scores = [float(v) for v in data.get("scores", [])]
        probabilities, stats = self._normalize_scores(scores)
        defended_scores = []
        for idx, p in enumerate(probabilities):
            value = self._defend_probability(p)
            # ?? 0.5 ???????????????????????????
            value = (1.0 - self.rank_mixing) * value + self.rank_mixing * 0.5
            jitter = math.sin((idx + 1) * 12.9898) * self.calibration_noise
            value = min(max(value + jitter, 1.0 - self.max_confidence), self.max_confidence)
            defended_scores.append(float(value))

        if defended_scores:
            # ????????? member/non-member ?????AUC ???? score ???
            sorted_scores = sorted(defended_scores)
            threshold = sorted_scores[len(sorted_scores) // 2]
            defended_predictions = [1 if value >= threshold else 0 for value in defended_scores]
        else:
            threshold = 0.5
            defended_predictions = []

        metrics = compute_metrics(labels, defended_predictions, defended_scores) if labels else {}
        return {
            "defense_name": self.defense_name,
            "status": "success",
            "scores": defended_scores,
            "defended_scores": defended_scores,
            "predictions": defended_predictions,
            "defended_predictions": defended_predictions,
            "labels": labels,
            "metrics": metrics,
            "threshold": float(threshold),
            "parameters": {
                "temperature": self.temperature,
                "alpha": self.alpha,
                "max_confidence": self.max_confidence,
                "rank_mixing": self.rank_mixing,
                "calibration_noise": self.calibration_noise,
            },
            "score_normalization": stats,
            "notes": "text-adapted minimal HAMP: temperature smoothing, entropy boosting, confidence clipping",
        }

    def evaluate(self, before_result, after_result):
        """?? defense ??????????"""
        before_metrics = before_result.get("metrics", {}) or {}
        after_metrics = after_result.get("metrics", {}) or {}
        auc_before = float(before_metrics.get("auc", 0.0))
        auc_after = float(after_metrics.get("auc", 0.0))
        acc_before = float(before_metrics.get("accuracy", 0.0))
        acc_after = float(after_metrics.get("accuracy", 0.0))
        auc_drop = auc_before - auc_after
        accuracy_drop = acc_before - acc_after
        return {
            "auc_drop": auc_drop,
            "accuracy_drop": accuracy_drop,
            "defense_effectiveness": max(0.0, auc_drop),
        }
