"""SPV-MIA privacy evaluation benchmark ????????"""

import json
import os
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result
from utils.subprocess_utils import run_with_log


SPV_MIA_PROJECT_PATH = "/root/autodl-tmp/SPV_MIA"
if SPV_MIA_PROJECT_PATH not in sys.path:
    sys.path.append(SPV_MIA_PROJECT_PATH)


class SPVMIAAttack(BaseAttack):
    """SPV-MIA ?? privacy evaluation ????"""

    attack_name = "spv_mia"

    def prepare(self) -> None:
        """??????????????????????"""
        self.project_path = self.config.get("project_path", SPV_MIA_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.config_path = self.config.get("config_path", os.path.join(self.project_path, "configs", "spv_minimal.yml"))
        self.entry_script = self.config.get("entry_script", "run_spv_minimal.py")
        self.entry_path = os.path.join(self.project_path, self.entry_script)
        self.output_path = self.config.get("output_path", os.path.join(self.project_path, "outputs", "spv_mia_result.json"))
        self.output_dir = self.config.get("output_dir", "./outputs")
        self.log_path = self.config.get("log_path", "./outputs/logs/spv_mia.log")
        self.timeout = int(self.config.get("timeout", 1800))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.isdir(self.project_path):
            raise FileNotFoundError(f"??? SPV-MIA ????: {self.project_path}")
        if not os.path.isfile(self.entry_path):
            raise FileNotFoundError(f"??? SPV-MIA ?? evaluation ??: {self.entry_path}")

    def _load_original_result(self) -> Dict[str, Any]:
        """?? SPV-MIA ??? JSON????????????"""
        result = empty_result(self.attack_name)
        if not os.path.isfile(self.output_path):
            return result
        with open(self.output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result["scores"] = [float(v) for v in data.get("scores", [])]
        result["predictions"] = [int(v) for v in data.get("predictions", [])]
        result["labels"] = [int(v) for v in data.get("labels", [])]
        metrics = data.get("metrics") or {}
        if metrics:
            result["metrics"].update({k: float(v) for k, v in metrics.items() if k in result["metrics"]})
        elif result["labels"]:
            result["metrics"] = compute_metrics(result["labels"], result["predictions"], result["scores"])
        result["method"] = data.get("method", self.config.get("method", "probabilistic_variation"))
        result["benchmark_name"] = data.get("benchmark_name", "spv_mia")
        result["threshold"] = data.get("threshold")
        result["model_path"] = data.get("model_path", self.config.get("model_path"))
        result["parsed_output_path"] = self.output_path
        return result

    def run(self) -> Dict[str, Any]:
        """?? subprocess ?? SPV-MIA ?? membership scoring?"""
        self.result = empty_result(self.attack_name)
        if not bool(self.config.get("execute_original", False)):
            return self.result

        command: List[str] = [
            self.python_path,
            self.entry_path,
            "--config", self.config_path,
            "--method", str(self.config.get("method", "probabilistic_variation")),
            "--output-path", self.output_path,
        ]
        if self.config.get("model_path"):
            command.extend(["--model-path", str(self.config.get("model_path"))])
        command.extend([str(arg) for arg in self.config.get("extra_args", []) or []])

        run_info = run_with_log(command, cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        self.result = self._load_original_result()
        self.result["returncode"] = run_info.get("returncode")
        self.result["command"] = run_info.get("command")
        self.result["log_path"] = self.log_path
        if not self.result.get("scores"):
            self.result["stderr"] = run_info.get("stderr", "")[-2000:]
        return self.result

    def evaluate(self) -> Dict[str, Any]:
        """???????"""
        if not self.result:
            self.result = empty_result(self.attack_name)
        if self.result.get("labels"):
            self.result["metrics"] = compute_metrics(
                self.result.get("labels", []),
                self.result.get("predictions", []),
                self.result.get("scores", []),
            )
        return self.result
