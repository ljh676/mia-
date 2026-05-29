"""Neighbour_MIA 攻击方法统一接入封装。

原论文仓库：/root/autodl-tmp/Neighbour_MIA
入口脚本：attack.py

注意：原始 attack.py 依赖用户提供 attack model、数据文件和两张 GPU，
且源码中仍有 <path_to_attack_model_*> 与 one_word_neighbours 占位问题。
本 wrapper 不修改原论文核心代码，只做统一调用、日志保存和结果解析。
"""

import os
import pickle
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result
from utils.subprocess_utils import run_with_log


NEIGHBOUR_MIA_PROJECT_PATH = "/root/autodl-tmp/Neighbour_MIA"
if NEIGHBOUR_MIA_PROJECT_PATH not in sys.path:
    sys.path.append(NEIGHBOUR_MIA_PROJECT_PATH)


class NeighbourMIAAttack(BaseAttack):
    """Neighbourhood Comparison MIA 的统一适配器。"""

    attack_name = "neighbour_mia"

    def prepare(self) -> None:
        """检查原项目路径、入口脚本、输出和日志目录。"""
        self.project_path = self.config.get("project_path", NEIGHBOUR_MIA_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.entry_script = self.config.get("entry_script", "attack.py")
        self.entry_path = os.path.join(self.project_path, self.entry_script)
        self.output_dir = self.config.get("output_dir", "./outputs")
        self.output_path = self.config.get("output_path")
        self.log_path = self.config.get("log_path", "./outputs/logs/neighbour_mia.log")
        self.timeout = int(self.config.get("timeout", 3600))
        self.proc_id = int(self.config.get("proc_id", 0))
        self.model = str(self.config.get("model", "bert"))
        self.dataset = str(self.config.get("dataset", "wiki"))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.isdir(self.project_path):
            raise FileNotFoundError(f"找不到 Neighbour_MIA 原项目目录: {self.project_path}")
        if not os.path.isfile(self.entry_path):
            raise FileNotFoundError(f"找不到 Neighbour_MIA 入口脚本: {self.entry_path}")

    def _expected_pickle_paths(self) -> List[str]:
        """返回原论文 attack.py 可能写出的 pickle 文件路径。"""
        candidates: List[str] = []
        if self.output_path:
            candidates.append(self.output_path)
        candidates.extend([
            os.path.join(self.project_path, f"all_scores_{self.dataset}_{self.model}_{self.proc_id}.pkl"),
            os.path.join(self.project_path, f"all_scores{self.proc_id}.pkl"),
        ])
        return candidates

    def _score_one_item(self, item: Dict[Any, Any]) -> float:
        """把原论文单条 scores dict 转成统一 membership score。

        原始 pickle 中通常包含原文本 logprob 和 neighbour logprob。
        这里使用 original_logprob - mean(neighbour_logprob) 作为可解释分数。
        """
        original_score = None
        neighbour_scores: List[float] = []
        for key, value in item.items():
            try:
                if isinstance(value, (list, tuple)):
                    numeric_values = [float(v) for v in value]
                    score_value = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
                else:
                    score_value = float(value)
            except Exception:
                continue
            if isinstance(key, str) and key.startswith("<original_text>:"):
                original_score = score_value
            else:
                neighbour_scores.append(score_value)
        if original_score is None:
            return 0.0
        if not neighbour_scores:
            return float(original_score)
        return float(original_score - (sum(neighbour_scores) / len(neighbour_scores)))

    def _parse_pickle_result(self) -> Dict[str, Any]:
        """解析 Neighbour_MIA 原论文 pickle 输出为统一结果。"""
        result = empty_result(self.attack_name)
        existing = [p for p in self._expected_pickle_paths() if p and os.path.isfile(p)]
        if not existing:
            return result
        latest = max(existing, key=os.path.getmtime)
        with open(latest, "rb") as f:
            data = pickle.load(f)

        scores: List[float] = []
        parsed_labels: List[int] = []
        parsed_predictions: List[int] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    scores.append(self._score_one_item(item))
        elif isinstance(data, dict):
            if "scores" in data:
                scores = [float(v) for v in data.get("scores", [])]
                # ???????? labels/predictions ???? pickle??????????? wrapper ???
                parsed_labels = [int(v) for v in data.get("labels", [])][:len(scores)]
                parsed_predictions = [int(v) for v in data.get("predictions", [])][:len(scores)]
            elif "raw_score_dicts" in data:
                for item in data.get("raw_score_dicts", []):
                    if isinstance(item, dict):
                        scores.append(self._score_one_item(item))
                parsed_labels = [int(v) for v in data.get("labels", [])][:len(scores)]
                parsed_predictions = [int(v) for v in data.get("predictions", [])][:len(scores)]
            else:
                scores = [self._score_one_item(data)]

        result["scores"] = scores
        labels = self.config.get("labels")
        if parsed_labels:
            result["labels"] = parsed_labels
        elif labels:
            result["labels"] = [int(v) for v in labels][:len(scores)]
        else:
            member_count = self.config.get("member_count")
            if member_count is not None:
                m = min(int(member_count), len(scores))
                result["labels"] = [1] * m + [0] * (len(scores) - m)

        if parsed_predictions:
            result["predictions"] = parsed_predictions
        elif scores:
            ordered = sorted(scores)
            threshold = ordered[len(ordered) // 2]
            result["predictions"] = [1 if score >= threshold else 0 for score in scores]
        if result.get("labels"):
            result["metrics"] = compute_metrics(result["labels"], result.get("predictions", []), scores)
        result["parsed_output_path"] = latest
        return result

    def _write_preflight_log(self, message: str) -> None:
        """在不真实执行原论文时，把原因写入日志。"""
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write("[Neighbour_MIA preflight]\n")
            log.write(message.rstrip() + "\n")

    def run(self) -> Dict[str, Any]:
        """执行 Neighbour_MIA 原论文入口或返回可解释的占位结果。"""
        self.result = empty_result(self.attack_name)
        if not bool(self.config.get("execute_original", False)):
            self._write_preflight_log(
                "execute_original=false，已完成统一框架接入但未启动原论文重实验。\n"
                "如需真实运行，请准备 attack model 和数据文件，并将 execute_original 设为 true。"
            )
            return self.result

        command: List[str] = [
            self.python_path,
            self.entry_path,
            "--proc-id", str(self.proc_id),
            "--model", self.model,
            "--dataset", self.dataset,
        ]
        command.extend([str(arg) for arg in self.config.get("extra_args", []) or []])

        run_info = run_with_log(command, cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        self.result = self._parse_pickle_result()
        self.result["returncode"] = run_info.get("returncode")
        self.result["command"] = run_info.get("command")
        self.result["log_path"] = self.log_path
        if not self.result.get("scores"):
            self.result["stderr"] = run_info.get("stderr", "")[-2000:]
        return self.result

    def evaluate(self) -> Dict[str, Any]:
        """补齐统一指标。"""
        if not self.result:
            self.result = empty_result(self.attack_name)
        if self.result.get("labels"):
            self.result["metrics"] = compute_metrics(
                self.result.get("labels", []),
                self.result.get("predictions", []),
                self.result.get("scores", []),
            )
        return self.result
