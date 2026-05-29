"""?? defense ?????"""

import argparse
import json
import os
from defenses.registry import get_defense
from utils.io_utils import load_yaml, save_json


def run_defense(defense_name, config_path=None, config=None):
    """???? defense ???????"""
    if config is None:
        config = load_yaml(config_path or os.path.join("configs", f"{defense_name}.yml"))
    cls = get_defense(defense_name)
    defense = cls(config)
    defense.prepare()
    result = {
        "defense_name": defense_name,
        "status": "success",
        "oracle": config.get("oracle", config.get("method", "available")),
        "parameters": {
            key: value
            for key, value in config.items()
            if key not in {"project_path", "output_path", "execute_original"}
        },
        "notes": (
            "minimal text-adapted MIAShield defense is available"
            if defense_name == "miashield"
            else "text-adapted minimal HAMP defense is available"
            if defense_name == "hamp"
            else "defense is available"
        ),
    }
    output_path = config.get("output_path")
    if output_path:
        save_json(result, output_path)
        print(f"??????: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="?? privacy defense benchmark ??")
    parser.add_argument("--defense", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_defense(args.defense, args.config))


if __name__ == "__main__":
    main()
