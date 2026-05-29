"""RAG_MIA 攻击方法真实接入封装。"""

import importlib
import os
import shutil
import sys
from typing import Any, Dict, List

from .base import BaseAttack
from utils.metrics import compute_metrics
from utils.result_parser import empty_result, parse_result_sources
from utils.subprocess_utils import run_with_log


RAG_MIA_PROJECT_PATH = "/root/autodl-tmp/RAG_MIA"
if RAG_MIA_PROJECT_PATH not in sys.path:
    sys.path.append(RAG_MIA_PROJECT_PATH)


class RAGMIAAttack(BaseAttack):
    """RAG_MIA 的统一适配器。

    优先尝试 import 原项目模块；实际执行时默认走 subprocess，
    因为原项目入口依赖工作目录、相对路径和命令行参数。
    """

    attack_name = "rag_mia"

    def prepare(self) -> None:
        """读取配置并检查原项目入口。"""
        self.project_path = self.config.get("project_path", RAG_MIA_PROJECT_PATH)
        self.python_path = self.config.get("python_path") or sys.executable
        self.config_path = self.config.get("config_path") or os.path.join(self.project_path, "configs", "nfcorpus.yml")
        self.output_path = self.config.get("output_path")
        self.output_dir = self.config.get("output_dir", "./outputs")
        self.log_path = self.config.get("log_path", "./outputs/logs/rag_mia.log")
        self.timeout = int(self.config.get("timeout", 3600))
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.isdir(self.project_path):
            raise FileNotFoundError(f"找不到 RAG_MIA 原项目目录: {self.project_path}")

        self.entry_script = self.config.get("entry_script", "run_mia.py")
        self.entry_path = os.path.join(self.project_path, self.entry_script)
        if not os.path.isfile(self.entry_path):
            self.entry_script = "main_mia.py"
            self.entry_path = os.path.join(self.project_path, self.entry_script)
        if not os.path.isfile(self.entry_path):
            raise FileNotFoundError(f"找不到 RAG_MIA 入口脚本: {self.entry_path}")

        self.original_module = None
        try:
            module_name = os.path.splitext(os.path.basename(self.entry_script))[0]
            self.original_module = importlib.import_module(module_name)
        except Exception as exc:
            # import 失败不阻断，后续 subprocess 仍可真实启动原项目。
            print(f"[TODO] RAG_MIA import 调用不可用，改用 subprocess: {exc}")

    def _make_minimal_config(self) -> str:
        """复制并缩小原配置，避免启动超大实验。"""
        minimal_config = self.config.get("minimal_config_path", "./outputs/rag_mia_min.yml")
        if not os.path.isabs(minimal_config):
            minimal_config = os.path.abspath(minimal_config)
        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            attack_cfg = data.setdefault("attack_config", {})
            # 原始 nfcorpus.yml 中存在 repeat_times 等额外字段，但 config.AttackConfig
            # dataclass 不接受这些字段；最小运行配置只保留原项目实际声明的字段。
            allowed_attack_keys = {
                "attack_method", "M", "N", "top_k", "seed", "name", "from_ckpt",
                "post_filter", "evaluate_attack", "proxy_lm_mba",
                "from_ckpt_model_responses", "from_ckpt_attack_data",
            }
            for key in list(attack_cfg.keys()):
                if key not in allowed_attack_keys:
                    attack_cfg.pop(key, None)
            attack_cfg["M"] = int(self.config.get("sample_num", 1))
            attack_cfg["N"] = int(self.config.get("question_num", 1))
            attack_cfg["top_k"] = int(self.config.get("top_k", 1))
            attack_cfg["post_filter"] = None
            attack_cfg["from_ckpt"] = bool(self.config.get("from_ckpt", True))
            attack_cfg["evaluate_attack"] = bool(self.config.get("evaluate_attack", False))
            attack_cfg["from_ckpt_model_responses"] = bool(self.config.get("from_ckpt_model_responses", False))
            attack_cfg["from_ckpt_attack_data"] = bool(self.config.get("from_ckpt_attack_data", True))
            rag_cfg = data.setdefault("rag_config", {})
            rag_cfg["retrieve_k"] = int(self.config.get("retrieve_k", 1))
            rag_cfg["retriever"] = str(self.config.get("retriever", "gte"))
            llm_cfg = data.setdefault("llm_config", {})
            llm_cfg["model_name"] = str(self.config.get("model_name", "gemma2_2b"))
            llm_cfg["shadow_model_name"] = str(self.config.get("shadow_model_name", "gemma2_2b"))
            llm_cfg["model_config_path"] = None
            with open(minimal_config, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            return minimal_config
        except Exception as exc:
            # 如果 YAML 处理失败，退回原配置，并把原因写入日志。
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(f"[WARN] 生成最小配置失败，使用原配置: {exc}\n")
            return self.config_path

    def run(self) -> Dict[str, Any]:
        """真实执行 RAG_MIA 原论文入口。"""
        self.result = empty_result(self.attack_name)
        if not bool(self.config.get("execute_original", False)):
            return self.result

        run_config = self._make_minimal_config() if bool(self.config.get("use_minimal_config", True)) else self.config_path
        if os.path.basename(self.entry_script) == "run_mia.py":
            # run_mia.py 内部会再调用 python3，并把输出重定向到原项目 logs；
            # 为了让统一框架完整捕获 stdout/stderr，这里直接调用真正入口 main_mia.py。
            command: List[str] = [self.python_path, "main_mia.py", "--config", run_config]
        else:
            command = [self.python_path, "main_mia.py", "--config", run_config]

        # Phi-3 官方权重与当前全局 transformers 版本不兼容；
        # 允许通过配置给原论文子进程临时注入隔离 PYTHONPATH，不污染 dcmi 环境。
        extra_pythonpath = self.config.get("extra_pythonpath")
        old_pythonpath = os.environ.get("PYTHONPATH")
        if extra_pythonpath:
            os.environ["PYTHONPATH"] = str(extra_pythonpath) + ((":" + old_pythonpath) if old_pythonpath else "")
        try:
            run_info = run_with_log(command, cwd=self.project_path, log_path=self.log_path, timeout=self.timeout)
        finally:
            if extra_pythonpath:
                if old_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = old_pythonpath
        combined_text = f"{run_info.get('stdout', '')}\n{run_info.get('stderr', '')}"
        candidate_dirs = [
            os.path.join(self.project_path, "results", "target_docs"),
            os.path.join(self.project_path, "logs"),
        ]
        self.result = parse_result_sources(self.attack_name, self.output_path, combined_text, candidate_dirs)
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
