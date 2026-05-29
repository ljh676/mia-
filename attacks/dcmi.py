"""DCMI 攻击方法真实接入封装。"""

import os
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result, parse_result_sources
from utils.subprocess_utils import run_with_log


DCMI_PROJECT_PATH = "/root/autodl-tmp/DCMI"
if DCMI_PROJECT_PATH not in sys.path:
    sys.path.append(DCMI_PROJECT_PATH)


class DCMIAttack(BaseAttack):
    """DCMI 的统一适配器。"""

    attack_name = "dcmi"

    def prepare(self) -> None:
        """???? DCMI ??????????"""
        self.project_path = self.config.get("project_path", DCMI_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.output_path = self.config.get("output_path")
        self.output_dir = self.config.get("output_dir", "./outputs")
        self.log_path = self.config.get("log_path", "./outputs/logs/dcmi.log")
        self.timeout = int(self.config.get("timeout", 1800))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.isdir(self.project_path):
            raise FileNotFoundError(f"??? DCMI ?????: {self.project_path}")

        entry_candidates = self.config.get("entry_candidates") or ["MIA.py", "attack.py", "main.py", "train.py", "example/run_exp.py"]
        self.entry_script = None
        for candidate in entry_candidates:
            if os.path.isfile(os.path.join(self.project_path, candidate)):
                self.entry_script = candidate
                break
        if self.entry_script is None:
            raise FileNotFoundError(f"???? DCMI ???????: {entry_candidates}")
        self.entry_path = os.path.join(self.project_path, self.entry_script)

        # DCMI ? MIA.py ???????????????????????
        # ??? import ??????? subprocess ?????
        self.original_module = None

    def run(self) -> Dict[str, Any]:
        """真实执行 DCMI 原论文入口。"""
        self.result = empty_result(self.attack_name)
        if not bool(self.config.get("execute_original", False)):
            return self.result

        command: List[str] = [self.python_path, self.entry_path]
        if self.entry_script == "example/run_exp.py":
            command.extend([
                "--method_name", str(self.config.get("method_name", "Standard-RAG")),
                "--split", str(self.config.get("split", "test")),
                "--dataset_name", str(self.config.get("dataset_name", "nq")),
                "--gpu_id", str(self.config.get("gpu_id", "-1")),
            ])
        else:
            command.extend([str(arg) for arg in self.config.get("extra_args", []) or []])

        run_info = run_with_log(command, cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        combined_text = f"{run_info.get('stdout', '')}\n{run_info.get('stderr', '')}"
        # ???? DCMI ?????????? JSON??????? fallback ???????
        original_json = self.config.get("original_output_path") or os.path.join(self.project_path, "outputs", "dcmi_result.json")
        preferred_output = original_json if os.path.isfile(original_json) else self.output_path
        candidate_dirs = [
            os.path.join(self.project_path, "outputs"),
            os.path.join(self.project_path, "output"),
            os.path.join(self.project_path, "logs"),
            self.output_dir,
        ]
        self.result = parse_result_sources(self.attack_name, preferred_output, combined_text, candidate_dirs)
        self.result["returncode"] = run_info.get("returncode")
        self.result["command"] = run_info.get("command")
        self.result["log_path"] = self.log_path
        return self.result

    def evaluate(self) -> Dict[str, Any]:
        """补齐统一指标。"""
        if not self.result:
            self.result = empty_result(self.attack_name)
        self.result["metrics"] = compute_metrics(
            self.result.get("labels", []),
            self.result.get("predictions", []),
            self.result.get("scores", []),
        ) if self.result.get("labels") else self.result.get("metrics", empty_result(self.attack_name)["metrics"])
        return self.result
