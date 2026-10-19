"""统一成员推断攻击运行入口。"""

import argparse
import os
from typing import Any, Dict

from attacks.registry import get_attack
from utils.io_utils import load_yaml, save_json
from utils.metrics import compute_metrics


def run_attack(attack_name: str, config_path: str = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """以统一方式运行指定成员推断攻击。

    Args:
        attack_name: 攻击方法名称，例如 rag_mia、dcmi。
        config_path: YAML 配置文件路径。
        config: 也可以直接传入配置字典，便于未来被其他 Python 脚本调用。

    Returns:
        统一格式结果字典。
    """
    if config is None:
        if config_path is None:
            config_path = os.path.join("configs", f"{attack_name}.yml")
        config = load_yaml(config_path)

    attack = get_attack(attack_name, config)
    attack.prepare()
    attack.run()
    result = attack.evaluate()
    if result.get("status") != "skipped" and result.get("returncode") == 0:
        if not (result.get("scores") or result.get("branches") or result.get("cases") or result.get("recovery_rate") is not None):
            raise RuntimeError("Attack completed without per-sample scores or recovery measurements")
        result["status"] = "success"
        result["source_freshness"] = "verified_new_or_changed_output"
        if attack_name not in {"janus", "janus_lora_transfer"}:
            threshold = config.get("decision_threshold")
            result["predictions"] = [int(score >= float(threshold)) for score in result.get("scores", [])] if threshold is not None else []
            result["threshold"] = threshold
            result["threshold_source"] = "configured_independently" if threshold is not None else "not_calibrated"
            result["metrics"] = compute_metrics(result.get("labels", []), result["predictions"], result.get("scores", []))
        result["implementation_kind"] = {"dcmi": "lexical_proxy", "spv_mia": "simplified_score_attack"}.get(attack_name, "local_research_adaptation")
        result["paper_reproduction_status"] = "not_independently_validated"
    elif result.get("status") != "skipped":
        result["status"] = "not_run"

    output_file = config.get("output_file")
    if output_file:
        save_json(result, output_file)
        print(f"结果已保存到: {output_file}")

    return result


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="统一成员推断攻击实验平台")
    parser.add_argument("--attack", required=True, help="攻击方法名称，例如 rag_mia 或 dcmi")
    parser.add_argument("--config", required=True, help="攻击配置文件路径")
    return parser


def main() -> None:
    """命令行入口函数。"""
    parser = build_parser()
    args = parser.parse_args()
    result = run_attack(args.attack, args.config)
    print(result)


if __name__ == "__main__":
    main()
