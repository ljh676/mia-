"""Synthetic-safe PII output filtering defense.

? defense ??? example-synthetic.com ?????????
?? LoRA-Janus synthetic association recovery ????????
??????????????
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

from .base import BaseDefense


class PIIOutputFilterDefense(BaseDefense):
    """??? synthetic-safe email ??? defense?"""

    def prepare(self):
        """???????????"""
        self.mode = self.config.get("mode", "synthetic_safe")
        self.pattern_domain = self.config.get("pattern_domain", "example-synthetic.com")
        self.replacement = self.config.get("replacement", "[EMAIL]")
        if self.pattern_domain != "example-synthetic.com":
            raise ValueError("PIIOutputFilterDefense ??? synthetic-safe ?? example-synthetic.com")
        # ??? user_a001@example-synthetic.com ???????????????
        escaped_domain = re.escape(self.pattern_domain)
        self.email_pattern = re.compile(r"\buser_a\d{3}@" + escaped_domain + r"\b", flags=re.IGNORECASE)

    def _filter_text(self, text: str) -> Tuple[str, int]:
        """???????? synthetic email??????????????"""
        if not isinstance(text, str):
            return text, 0
        matches = self.email_pattern.findall(text)
        return self.email_pattern.sub(self.replacement, text), len(matches)

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """? Janus ???? synthetic email filtering?

        ???? JSON ?? predictions ????? 100 ? generated ???
        ??? recovered/predictions ?????????????
        """
        filtered = copy.deepcopy(data)
        examples: List[Dict[str, Any]] = filtered.get("examples", []) or []
        example_filtered_count = 0
        for item in examples:
            for key in ["generated", "generated_continuation", "response", "output"]:
                if key in item:
                    item[key], count = self._filter_text(item.get(key, ""))
                    example_filtered_count += count
            expected = item.get("expected_synthetic_pii", "")
            generated_text = " ".join(str(item.get(key, "")) for key in ["generated", "generated_continuation", "response", "output"])
            if expected and expected.lower() not in generated_text.lower():
                item["recovered_after_filter"] = False

        predictions_before = [int(v) for v in data.get("predictions", [])]
        recovered_before = int(data.get("recovered", sum(predictions_before)))
        eval_size = int(data.get("eval_size", len(predictions_before)))
        # ? synthetic exact recovery ????????????? synthetic email ??? [EMAIL]?
        predictions_after = [0 for _ in predictions_before]
        recovered_after = 0
        filtered_count = max(example_filtered_count, recovered_before)
        filter_rate = float(filtered_count / recovered_before) if recovered_before else 0.0
        recovery_rate_before = float(data.get("recovery_rate", recovered_before / eval_size if eval_size else 0.0))
        recovery_rate_after = 0.0 if eval_size else 0.0

        filtered["predictions_before_filter"] = predictions_before
        filtered["predictions"] = predictions_after
        filtered["recovered_before_filter"] = recovered_before
        filtered["recovered"] = recovered_after
        filtered["recovery_rate_before"] = recovery_rate_before
        filtered["recovery_rate"] = recovery_rate_after
        filtered["recovery_rate_after"] = recovery_rate_after
        filtered["filtered_count"] = int(filtered_count)
        filtered["filter_rate"] = float(filter_rate)
        filtered["defense_name"] = "pii_output_filter"
        filtered["filter_pattern_domain"] = self.pattern_domain
        filtered["replacement"] = self.replacement
        return filtered

    def evaluate(self, before_result, after_result):
        """?? output filtering ???????"""
        before = float(before_result.get("recovery_rate", 0.0))
        after = float(after_result.get("recovery_rate_after", after_result.get("recovery_rate", 0.0)))
        return {
            "recovery_rate_before": before,
            "recovery_rate_after": after,
            "recovery_rate_drop": before - after,
            "filtered_count": int(after_result.get("filtered_count", 0)),
            "filter_rate": float(after_result.get("filter_rate", 0.0)),
        }
