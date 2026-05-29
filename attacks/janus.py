
"""Janus / LoRA-Janus synthetic-safe privacy evaluation wrapper?"""

import json
import os
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result
from utils.subprocess_utils import run_with_log


JANUS_PROJECT_PATH = "/root/autodl-tmp/Janus"
if JANUS_PROJECT_PATH not in sys.path:
    sys.path.append(JANUS_PROJECT_PATH)


class JanusAttack(BaseAttack):
    """?? synthetic-safe Janus??? minimal ? true held-out recovery ?????"""

    attack_name = "janus"

    def prepare(self) -> None:
        """?? Janus ????????"""
        self.project_path = self.config.get("project_path", JANUS_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.janus_mode = self.config.get("janus_mode", "minimal")
        self.model_path = self.config.get("model_path", "/root/autodl-tmp/models/gpt2")
        self.model_alias = self.config.get("model_alias", os.path.basename(str(self.model_path).rstrip(os.sep)) or "gpt2")
        self.skip_if_model_missing = bool(self.config.get("skip_if_model_missing", False))
        self.skip_reason = ""
        self.output_path = self.config.get("output_path", os.path.join(self.project_path, "outputs", "janus_result.json"))
        self.latest_output_path = self.config.get("latest_output_path", os.path.join(self.project_path, "outputs", "janus_result.json"))
        self.output_dir = self.config.get("output_dir", "./outputs")
        self.log_path = self.config.get("log_path", "./outputs/logs/janus.log")
        self.timeout = int(self.config.get("timeout", 3600))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if self.janus_mode == "true_heldout_recovery":
            self.entry_path = os.path.join(self.project_path, "run_janus_lora_true_pipeline.py")
            self.pretrain_pii_data = self.config.get("pretrain_pii_data")
            self.known_s1_data = self.config.get("known_s1_data")
            self.hidden_s2_data = self.config.get("hidden_s2_data")
            required = [
                (self.entry_path, "true Janus ??"),
                (self.pretrain_pii_data, "pretrain PII JSONL"),
                (self.known_s1_data, "known S1 JSONL"),
                (self.hidden_s2_data, "hidden S2 JSONL"),
            ]
        else:
            self.entry_path = os.path.join(self.project_path, self.config.get("entry_script", "run_janus_minimal.py"))
            self.train_data = self.config.get("train_data", os.path.join(self.project_path, "data", "synthetic_pii_train.jsonl"))
            self.eval_data = self.config.get("eval_data", os.path.join(self.project_path, "data", "synthetic_pii_eval.jsonl"))
            self.finetune_mode = self.config.get("finetune_mode", "full_ft")
            required = [(self.entry_path, "Janus ??"), (self.train_data, "train JSONL"), (self.eval_data, "eval JSONL")]
        if self.janus_mode == "true_heldout_recovery" and self.skip_if_model_missing and not os.path.isdir(str(self.model_path)):
            self.skip_reason = f"model_path not found: {self.model_path}"
            return
        for path, name in required:
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"??? {name}: {path}")

    def _load_result(self) -> Dict[str, Any]:
        """?? Janus JSON???? recovery ????"""
        result = empty_result(self.attack_name)
        load_path = self.output_path if os.path.isfile(self.output_path) else self.latest_output_path
        if not os.path.isfile(load_path):
            return result
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result["scores"] = [float(v) for v in data.get("scores", [])]
        result["labels"] = [int(v) for v in data.get("labels", [])]
        result["predictions"] = [int(v) for v in data.get("predictions", [])]
        for key in [
            "mode", "janus_mode", "model_alias", "model_path", "finetune_mode", "pii_type", "train_size", "eval_size",
            "recovered", "recovery_rate", "fine_tuning", "lora", "generation", "examples",
            "safety_notes", "pipeline", "paths", "s1_size", "s2_eval_size", "pretrain_size",
            "strict_disjoint_s1_s2", "no_janus_recovery_rate", "full_ft_janus_recovery_rate",
            "lora_ft_janus_recovery_rate", "hidden_s2_metrics", "branches", "training",
        ]:
            if key in data:
                result[key] = data[key]
        result["parsed_output_path"] = load_path
        if data.get("metrics"):
            result["metrics"].update({k: float(v) for k, v in data["metrics"].items() if k in result["metrics"]})
        return result

    def _true_pipeline_command(self) -> List[str]:
        """?? true held-out recovery pipeline ???"""
        cmd: List[str] = [
            self.python_path, self.entry_path,
            "--model-path", str(self.model_path),
            "--model-alias", str(self.model_alias),
            "--pretrain-pii-data", str(self.pretrain_pii_data),
            "--known-s1-data", str(self.known_s1_data),
            "--hidden-s2-data", str(self.hidden_s2_data),
            "--output-path", str(self.output_path),
            "--lora-r", str(self.config.get("lora_r", 8)),
            "--lora-alpha", str(self.config.get("lora_alpha", 16)),
            "--lora-dropout", str(self.config.get("lora_dropout", 0.05)),
            "--lora-learning-rate", str(self.config.get("lora_learning_rate", 4e-4)),
            "--lora-epochs", str(self.config.get("lora_epochs", 5)),
            "--full-ft-learning-rate", str(self.config.get("full_ft_learning_rate", self.config.get("lora_learning_rate", 4e-4))),
            "--full-ft-epochs", str(self.config.get("full_ft_epochs", self.config.get("lora_epochs", 5))),
            "--pretrain-learning-rate", str(self.config.get("pretrain_learning_rate", 5e-4)),
            "--pretrain-epochs", str(self.config.get("pretrain_epochs", 5)),
            "--forget-learning-rate", str(self.config.get("forget_learning_rate", 5e-4)),
            "--forget-epochs", str(self.config.get("forget_epochs", 2)),
            "--general-text-count", str(self.config.get("general_text_count", 600)),
            "--batch-size", str(self.config.get("batch_size", 4)),
            "--max-length", str(self.config.get("max_length", 128)),
            "--max-new-tokens", str(self.config.get("max_new_tokens", 32)),
            "--num-beams", str(self.config.get("num_beams", 5)),
            "--lora-target-modules",
        ]
        cmd.extend([str(v) for v in self.config.get("lora_target_modules", ["c_attn"])])
        return cmd

    def _minimal_command(self) -> List[str]:
        """??? minimal synthetic Janus ???"""
        cmd: List[str] = [
            self.python_path, self.entry_path,
            "--model-path", str(self.model_path),
            "--train-data", str(self.train_data),
            "--eval-data", str(self.eval_data),
            "--output-path", str(self.output_path),
            "--latest-output-path", str(self.latest_output_path),
            "--finetune-mode", str(self.finetune_mode),
            "--learning-rate", str(self.config.get("learning_rate", 5e-5)),
            "--batch-size", str(self.config.get("batch_size", 4)),
            "--epochs", str(self.config.get("epochs", 2)),
            "--max-length", str(self.config.get("max_length", 128)),
            "--max-new-tokens", str(self.config.get("max_new_tokens", 32)),
            "--num-beams", str(self.config.get("num_beams", 5)),
            "--lora-r", str(self.config.get("lora_r", 8)),
            "--lora-alpha", str(self.config.get("lora_alpha", 16)),
            "--lora-dropout", str(self.config.get("lora_dropout", 0.05)),
            "--lora-target-modules",
        ]
        cmd.extend([str(v) for v in self.config.get("lora_target_modules", ["c_attn"])])
        return cmd

    def run(self) -> Dict[str, Any]:
        """?? subprocess ?? Janus?"""
        self.result = empty_result(self.attack_name)
        if self.skip_reason:
            self.result.update({
                "attack_name": self.attack_name,
                "janus_mode": self.janus_mode,
                "model_alias": self.model_alias,
                "model_path": self.model_path,
                "status": "skipped",
                "skip_reason": self.skip_reason,
                "returncode": 0,
            })
            return self.result
        if not bool(self.config.get("execute_original", False)):
            return self.result
        command = self._true_pipeline_command() if self.janus_mode == "true_heldout_recovery" else self._minimal_command()
        command.extend([str(arg) for arg in self.config.get("extra_args", []) or []])
        run_info = run_with_log(command, cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        self.result = self._load_result()
        self.result["returncode"] = run_info.get("returncode")
        self.result["command"] = run_info.get("command")
        self.result["log_path"] = self.log_path
        if not self.result.get("scores"):
            self.result["stderr"] = run_info.get("stderr", "")[-2000:]
        return self.result

    def evaluate(self) -> Dict[str, Any]:
        """???????Janus ???? recovery_rate?"""
        if not self.result:
            self.result = empty_result(self.attack_name)
        if self.result.get("labels"):
            self.result["metrics"] = compute_metrics(self.result.get("labels", []), self.result.get("predictions", []), self.result.get("scores", []))
        return self.result
