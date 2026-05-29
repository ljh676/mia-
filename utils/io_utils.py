"""输入输出工具函数。"""

import json
import os
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。

    优先使用 PyYAML；如果环境缺少 PyYAML，会给出清晰报错。
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("请先安装 PyYAML: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(data: Dict[str, Any], path: str) -> None:
    """把结果保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
