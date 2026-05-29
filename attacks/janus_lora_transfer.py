"""Janus LoRA cross-model transfer attack wrapper.

Loads a LoRA-Janus adapter trained on one GPT-style model into another model
and evaluates synthetic-safe hidden S2 recovery.
"""

import json
import os
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.result_parser import empty_result
from utils.subprocess_utils import run_with_log

JANUS_PROJECT_PATH = "/root/autodl-tmp/Janus"


class JanusLoRATransferAttack(BaseAttack):
    """Synthetic-safe cross-model LoRA-Janus transfer evaluation."""

    attack_name = "janus_lora_transfer"

    def prepare(self) -> None:
        self.project_path = self.config.get("project_path", JANUS_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.entry_path = os.path.join(self.project_path, "run_janus_lora_cross_model_transfer.py")
        self.source_model_alias = self.config.get("source_model_alias", "gpt2")
        self.target_model_alias = self.config.get("target_model_alias", "distilgpt2")
        self.source_model_path = self.config.get("source_model_path", os.path.join(self.project_path, "models", "gpt2_synthetic_forgotten"))
        self.target_model_path = self.config.get("target_model_path", os.path.join(self.project_path, "models", "gpt2_synthetic_forgotten_distilgpt2"))
        self.source_lora_adapter = self.config.get("source_lora_adapter", os.path.join(self.project_path, "models", "gpt2_lora_ft_janus_s1"))
        self.s2_data = self.config.get("s2_data", os.path.join(self.project_path, "data", "synthetic_janus_hidden_s2.jsonl"))
        self.output_path = self.config.get("output_path", "/root/autodl-tmp/unified_mia_framework/outputs/janus_lora_cross_model_transfer_result.json")
        self.output_file = self.config.get("output_file", "./outputs/janus_lora_cross_model_transfer_result.json")
        self.log_path = self.config.get("log_path", "./outputs/logs/janus_lora_transfer.log")
        self.timeout = int(self.config.get("timeout", 3600))
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        required = [
            (self.entry_path, False, "cross-model transfer script"),
            (self.source_model_path, True, "source/sanity base model"),
            (self.target_model_path, True, "target base model"),
            (self.source_lora_adapter, True, "source LoRA adapter"),
            (self.s2_data, False, "S2 data"),
        ]
        for path, is_dir, name in required:
            exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
            if not exists:
                raise FileNotFoundError(f"missing {name}: {path}")

    def _command(self) -> List[str]:
        return [
            self.python_path,
            self.entry_path,
            "--source-model-alias", str(self.source_model_alias),
            "--source-model-path", str(self.source_model_path),
            "--source-lora-adapter", str(self.source_lora_adapter),
            "--target-model-alias", str(self.target_model_alias),
            "--target-model-path", str(self.target_model_path),
            "--s2-data", str(self.s2_data),
            "--output-path", str(self.output_path),
        ]

    def _load_result(self) -> Dict[str, Any]:
        result = empty_result(self.attack_name)
        if not os.path.isfile(self.output_path):
            return result
        with open(self.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result.update(data)
        cross = (data.get("cases") or {}).get("gpt2_to_distilgpt2", {})
        result["metrics"].update({
            "auc": 0.0,
            "accuracy": float(cross.get("recovery_rate", 0.0)),
            "f1": 0.0,
        })
        result["parsed_output_path"] = self.output_path
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
        return self.result
