"""Redact synthetic email outputs and measure recovery from actual generations."""

import copy
import re

from .base import BaseDefense
from utils.metrics import metric_delta


OUTPUT_FIELDS = ("generated", "generated_continuation", "response", "output")


class PIIOutputFilterDefense(BaseDefense):
    def prepare(self):
        self.pattern_domain = self.config.get("pattern_domain", "example-synthetic.com")
        self.replacement = self.config.get("replacement", "[EMAIL]")
        if self.pattern_domain != "example-synthetic.com":
            raise ValueError("This filter supports example-synthetic.com only")
        self.email_pattern = re.compile(
            r"(?<![\w.+-])[\w.+-]+@" + re.escape(self.pattern_domain) + r"(?![\w.-])", re.IGNORECASE
        )

    def _filter_text(self, text):
        if not isinstance(text, str):
            return text, 0
        return self.email_pattern.subn(self.replacement, text)

    def apply(self, data):
        filtered = copy.deepcopy(data)
        replacement_count = 0

        def redact(value):
            nonlocal replacement_count
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in OUTPUT_FIELDS and isinstance(item, str):
                        value[key], count = self._filter_text(item)
                        replacement_count += count
                    elif isinstance(item, (dict, list)):
                        redact(item)
            elif isinstance(value, list):
                for item in value:
                    redact(item)

        redact(filtered)
        key = "generation_records" if "generation_records" in data else "examples"
        original_records = data.get(key) or []
        records = filtered.get(key) or []
        before_flags, after_flags = [], []
        for original, record in zip(original_records, records):
            expected = original.get("expected_synthetic_pii")
            outputs = [field for field in OUTPUT_FIELDS if isinstance(original.get(field), str)]
            if not isinstance(expected, str) or not expected or not outputs:
                continue
            before = any(expected.casefold() in original[field].casefold() for field in outputs)
            after = any(expected.casefold() in record[field].casefold() for field in outputs)
            record["recovered_before_filter"] = before
            record["recovered_after_filter"] = after
            before_flags.append(int(before))
            after_flags.append(int(after))

        total = data.get("eval_size")
        if total is None and data.get("generation_records_complete") is True:
            total = len(original_records)
        complete = isinstance(total, int) and total > 0 and len(before_flags) == len(original_records) == total
        observed_before, observed_after = sum(before_flags), sum(after_flags)
        before_rate = observed_before / total if complete else None
        after_rate = observed_after / total if complete else None
        filtered.update({
            "defense_name": "pii_output_filter", "status": "success",
            "implementation_kind": "output_redaction", "task_output_changed": True,
            "recovery_evaluation_status": "complete" if complete else "insufficient_generation_records",
            "recovery_evaluated_count": len(before_flags), "recovery_expected_count": total,
            "recovery_rate_before": before_rate, "recovery_rate_after": after_rate,
            "recovery_rate": after_rate, "recovered": observed_after if complete else None,
            "recovered_before_filter": observed_before if complete else None,
            "recovery_predictions_after_filter": after_flags if complete else None,
            "observed_recovered_before": observed_before, "observed_recovered_after": observed_after,
            "filtered_count": replacement_count,
            "filter_rate": (observed_before - observed_after) / observed_before if complete and observed_before else None,
            "filter_pattern_domain": self.pattern_domain, "replacement": self.replacement,
        })
        # scores / predictions / labels / metrics describe membership inference;
        # output redaction cannot change that teacher-forced scoring channel.
        return filtered

    def evaluate(self, before_result, after_result):
        before = after_result.get("recovery_rate_before")
        after = after_result.get("recovery_rate_after")
        return {"recovery_rate_before": before, "recovery_rate_after": after,
                "recovery_rate_drop": metric_delta(before, after),
                "filtered_count": after_result.get("filtered_count"),
                "filter_rate": after_result.get("filter_rate"),
                "recovery_evaluation_status": after_result["recovery_evaluation_status"]}
