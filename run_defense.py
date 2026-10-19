"""Apply a defense to a saved result, or explicitly check preparation only."""

import argparse
import json
import os

from defenses.registry import get_defense
from utils.io_utils import load_yaml, save_json


def run_defense(defense_name, config_path=None, config=None, input_path=None, check_only=False):
    if config is None:
        config = load_yaml(config_path or os.path.join("configs", f"{defense_name}.yml"))
    defense = get_defense(defense_name)(config)
    defense.prepare()
    if check_only:
        return {"defense_name": defense_name, "status": "prepared", "applied": False}
    input_path = input_path or config.get("input_path")
    if not input_path:
        raise ValueError("Supply --input or config.input_path to actually apply a defense; use --check for preparation only")
    with open(input_path, encoding="utf-8") as stream:
        before = json.load(stream)
    result = defense.apply(before)
    result["defense_effect"] = defense.evaluate(before, result)
    result["applied"] = True
    if config.get("output_path"):
        save_json(result, config["output_path"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defense", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", help="Saved attack/generation JSON")
    parser.add_argument("--check", action="store_true", help="Prepare only, without reporting an applied defense")
    args = parser.parse_args()
    print(json.dumps(run_defense(args.defense, args.config, input_path=args.input, check_only=args.check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
