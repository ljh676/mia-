"""Janus-MIA link attack wrapper.

Evaluates whether hidden-memory recovery strengthens membership-style signal,
including hard non-member controls and calibrated loss scoring.
"""

import json
import os
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result
from utils.subprocess_utils import run_with_log

JANUS_PROJECT_PATH = "/root/autodl-tmp/Janus"


class JanusMIAAttack(BaseAttack):
    """Synthetic-safe Janus recovery -> MIA signal evaluation."""

    attack_name = "janus_mia"

    def prepare(self) -> None:
        self.project_path = self.config.get("project_path", JANUS_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.entry_path = os.path.join(self.project_path, "run_janus_mia_eval.py")
        self.output_path = self.config.get("output_path", "/root/autodl-tmp/unified_mia_framework/outputs/janus_mia_hard_controls_result.json")
        self.output_file = self.config.get("output_file", "./outputs/janus_mia_hard_controls_result.json")
        self.table_path = self.config.get("table_path", "/root/autodl-tmp/unified_mia_framework/outputs/janus_mia_hard_controls_table.md")
        self.defense_report_path = self.config.get("defense_report_path", "/root/autodl-tmp/unified_mia_framework/outputs/janus_mia_defense_comparison.md")
        self.gap_report_path = self.config.get("gap_report_path", "/root/autodl-tmp/unified_mia_framework/outputs/janus_mia_gap_analysis_report.md")
        self.log_path = self.config.get("log_path", "./outputs/logs/janus_mia.log")
        self.timeout = int(self.config.get("timeout", 3600))
        self.base_model_path = self.config.get("base_model_path", "/root/autodl-tmp/models/gpt2")
        self.s2_data = self.config.get("s2_data", os.path.join(self.project_path, "data", "synthetic_janus_hidden_s2.jsonl"))
        self.forgotten_model = self.config.get("forgotten_model", os.path.join(self.project_path, "models", "gpt2_synthetic_forgotten"))
        self.full_ft_model = self.config.get("full_ft_model", os.path.join(self.project_path, "models", "gpt2_full_ft_janus_s1"))
        self.lora_base_model = self.config.get("lora_base_model", self.forgotten_model)
        self.lora_adapter = self.config.get("lora_adapter", os.path.join(self.project_path, "models", "gpt2_lora_ft_janus_s1"))
        self.nonmember_type = self.config.get("nonmember_type", "all")
        self.use_calibrated_loss = bool(self.config.get("use_calibrated_loss", True))
        self.use_hard_controls = bool(self.config.get("use_hard_controls", True))
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        required = [
            (self.entry_path, "Janus-MIA eval script"),
            (self.base_model_path, "base tokenizer/model path"),
            (self.s2_data, "S2 data"),
            (self.forgotten_model, "forgotten model"),
            (self.full_ft_model, "full-ft Janus model"),
            (self.lora_base_model, "LoRA base model"),
            (self.lora_adapter, "LoRA adapter"),
        ]
        for path, name in required:
            exists = os.path.isdir(path) if name.endswith("model") or "adapter" in name or "path" in name else os.path.isfile(path)
            if not exists:
                raise FileNotFoundError(f"missing {name}: {path}")

    def _command(self) -> List[str]:
        cmd = [
            self.python_path, self.entry_path,
            "--base-model-path", str(self.base_model_path),
            "--s2-data", str(self.s2_data),
            "--forgotten-model", str(self.forgotten_model),
            "--full-ft-model", str(self.full_ft_model),
            "--lora-base-model", str(self.lora_base_model),
            "--lora-adapter", str(self.lora_adapter),
            "--output-path", str(self.output_path),
            "--table-path", str(self.table_path),
            "--defense-report-path", str(self.defense_report_path),
            "--gap-report-path", str(self.gap_report_path),
            "--nonmember-type", str(self.nonmember_type),
            "--max-new-tokens", str(self.config.get("max_new_tokens", 32)),
            "--num-beams", str(self.config.get("num_beams", 5)),
        ]
        if self.use_calibrated_loss:
            cmd.append("--use-calibrated-loss")
        if self.use_hard_controls:
            cmd.append("--use-hard-controls")
        return cmd

    def _load_result(self) -> Dict[str, Any]:
        result = empty_result(self.attack_name)
        if not os.path.isfile(self.output_path):
            return result
        with open(self.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        first_control = (data.get("nonmember_controls") or ["user_b"])[0]
        lora_branch = ((data.get("branches") or {}).get("lora_ft_janus_model") or {}).get(first_control, {})
        result["scores"] = [float(v) for v in data.get("scores", lora_branch.get("scores", []))]
        result["labels"] = [int(v) for v in data.get("labels", lora_branch.get("labels", []))]
        result["predictions"] = [int(v) for v in data.get("predictions", lora_branch.get("predictions", []))]
        for key in ["mode", "member_set", "nonmember_type", "nonmember_controls", "overlap_checks", "model_paths", "summary_table", "branches", "conclusion", "safety_notes", "use_calibrated_loss", "use_hard_controls"]:
            if key in data:
                result[key] = data[key]
        if data.get("metrics"):
            result["metrics"].update({k: float(v) for k, v in data["metrics"].items() if k in result["metrics"]})
        result["parsed_output_path"] = self.output_path
        result["table_path"] = self.table_path
        result["defense_report_path"] = self.defense_report_path
        result["gap_report_path"] = self.gap_report_path
        return result

    def run(self) -> Dict[str, Any]:
        self.result = empty_result(self.attack_name)
        if not bool(self.config.get("execute_original", True)):
            return self.result
        run_info = run_with_log(self._command(), cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        self.result = self._load_result()
        self.result["returncode"] = run_info.get("returncode")
        self.result["command"] = run_info.get("command")
        self.result["log_path"] = self.log_path
        if run_info.get("returncode") != 0:
            self.result["stderr"] = run_info.get("stderr", "")[-2000:]
        return self.result

    def evaluate(self) -> Dict[str, Any]:
        if self.result.get("labels"):
            self.result["metrics"] = compute_metrics(self.result.get("labels", []), self.result.get("predictions", []), self.result.get("scores", []))
        return self.result
