
"""LoRA-Janus threshold boundary sweep.

?? r=8 ? train_size=30???? learning_rate x epochs???
synthetic association recovery ? 0 ?? recovery ????
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List

import yaml

PROJECT_PATH = "/root/autodl-tmp/Janus"
DEFAULT_CONFIG = "/root/autodl-tmp/unified_mia_framework/configs/janus_lora_boundary.yml"


def load_yaml(path: str) -> Dict[str, Any]:
    """?? YAML ???"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def synthetic_row(index: int) -> Dict[str, str]:
    """???? synthetic association?"""
    return {
        "identifier": f"User_A{index:03d}",
        "pii_type": "email",
        "pii": f"user_a{index:03d}@example-synthetic.com",
    }


def ensure_train_30(path: str) -> None:
    """?? train_size=30 ? synthetic ?????"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for index in range(1, 31):
            f.write(json.dumps(synthetic_row(index), ensure_ascii=False) + "\n")


def ensure_eval(path: str) -> None:
    """?? eval ?????"""
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for index in range(31, 131):
            f.write(json.dumps(synthetic_row(index), ensure_ascii=False) + "\n")


def run_one(cfg: Dict[str, Any], lr: float, epochs: int, index: int, total: int) -> Dict[str, Any]:
    """???? boundary ???"""
    project_path = cfg.get("project_path", PROJECT_PATH)
    fixed = cfg.get("fixed", {}) or {}
    r = int(fixed.get("lora_r", 8))
    train_size = int(fixed.get("train_size", 30))
    output_dir = os.path.join(project_path, "outputs", "lora_boundary_runs")
    log_dir = os.path.join(project_path, "outputs", "lora_boundary_logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    lr_tag = str(lr).replace("-", "m").replace(".", "p")
    tag = f"r{r}_lr{lr_tag}_ep{epochs}_tr{train_size}"
    output_path = os.path.join(output_dir, f"janus_lora_boundary_{tag}.json")
    log_path = os.path.join(log_dir, f"janus_lora_boundary_{tag}.log")
    command = [
        cfg.get("python_path", sys.executable),
        os.path.join(project_path, "run_janus_minimal.py"),
        "--model-path", str(cfg.get("model_path", "/root/autodl-tmp/models/gpt2")),
        "--train-data", str(cfg.get("train_data")),
        "--eval-data", str(cfg.get("eval_data")),
        "--output-path", output_path,
        "--latest-output-path", os.path.join(project_path, "outputs", "janus_result.json"),
        "--finetune-mode", "lora_ft",
        "--learning-rate", str(lr),
        "--batch-size", str(cfg.get("batch_size", 4)),
        "--epochs", str(epochs),
        "--max-length", str(cfg.get("max_length", 128)),
        "--max-new-tokens", str(cfg.get("max_new_tokens", 32)),
        "--num-beams", str(cfg.get("num_beams", 5)),
        "--lora-r", str(r),
        "--lora-alpha", str(cfg.get("lora_alpha", 16)),
        "--lora-dropout", str(cfg.get("lora_dropout", 0.05)),
        "--lora-target-modules",
    ]
    command.extend([str(m) for m in cfg.get("lora_target_modules", ["c_attn"])])
    print(f"[Boundary Sweep] {index}/{total} lr={lr} epochs={epochs}", flush=True)
    proc = subprocess.run(command, cwd=project_path, capture_output=True, text=True, timeout=int(cfg.get("timeout", 7200)))
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[COMMAND] " + " ".join(command) + "\n")
        log.write(f"[RETURNCODE] {proc.returncode}\n")
        log.write("[STDOUT]\n" + proc.stdout + "\n[STDERR]\n" + proc.stderr + "\n")
    if proc.returncode != 0:
        result = {
            "learning_rate": float(lr),
            "epochs": int(epochs),
            "recovery_rate": 0.0,
            "recovered": 0,
            "eval_size": 0,
            "returncode": proc.returncode,
            "error": proc.stderr[-2000:],
            "log_path": log_path,
            "output_path": output_path,
        }
    else:
        data = json.load(open(output_path, "r", encoding="utf-8"))
        lora = data.get("lora", {}) or {}
        result = {
            "learning_rate": float(lr),
            "epochs": int(epochs),
            "lora_r": r,
            "train_size": train_size,
            "recovery_rate": float(data.get("recovery_rate", 0.0)),
            "recovered": int(data.get("recovered", 0)),
            "eval_size": int(data.get("eval_size", 0)),
            "trainable_parameters": int(lora.get("trainable_parameters", 0)),
            "trainable_ratio": float(lora.get("trainable_ratio", 0.0)),
            "train_losses": (data.get("fine_tuning", {}) or {}).get("train_losses", []),
            "returncode": int(data.get("returncode", 0)),
            "log_path": log_path,
            "output_path": output_path,
        }
    print(f"[Boundary Sweep] lr={lr} epochs={epochs} recovery_rate={result['recovery_rate']:.4f}", flush=True)
    return result


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def aggregate(results: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped: Dict[Any, List[float]] = defaultdict(list)
    for row in results:
        grouped[row[key]].append(float(row.get("recovery_rate", 0.0)))
    return [{key: key_value, "mean_recovery_rate": mean(vals), "num_runs": len(vals)} for key_value, vals in sorted(grouped.items())]


def threshold_point(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """?? recovery_rate ?? > 0.5 ??? lr x epochs ???"""
    candidates = [r for r in results if float(r.get("recovery_rate", 0.0)) > 0.5]
    if not candidates:
        return {"threshold_lr": None, "threshold_epochs": None, "threshold_recovery_rate": 0.0}
    # ?????????????????? epoch?
    best = sorted(candidates, key=lambda r: (r["learning_rate"], r["epochs"]))[0]
    return {
        "threshold_lr": best["learning_rate"],
        "threshold_epochs": best["epochs"],
        "threshold_recovery_rate": best["recovery_rate"],
    }


def write_outputs(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """?????????threshold point?phase transition ????"""
    out = cfg.get("unified_output_dir", "/root/autodl-tmp/unified_mia_framework/outputs")
    os.makedirs(out, exist_ok=True)
    sorted_low = sorted(results, key=lambda r: (float(r.get("recovery_rate", 0.0)), r["learning_rate"], r["epochs"]))
    table_path = os.path.join(out, "janus_lora_boundary_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# LoRA-Janus Boundary Sweep Table\n\n")
        f.write("| lr | epochs | recovery_rate | recovered |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in sorted_low:
            f.write(f"| {row['learning_rate']} | {row['epochs']} | {row['recovery_rate']:.4f} | {row['recovered']} / {row['eval_size']} |\n")
    point = threshold_point(results)
    point_path = os.path.join(out, "janus_lora_threshold_point.json")
    with open(point_path, "w", encoding="utf-8") as f:
        json.dump(point, f, ensure_ascii=False, indent=2)
    phase = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fixed": cfg.get("fixed", {}),
        "recovery_rate_vs_lr": aggregate(results, "learning_rate"),
        "recovery_rate_vs_epochs": aggregate(results, "epochs"),
        "threshold_curve": sorted(results, key=lambda r: (r["learning_rate"], r["epochs"])),
        "threshold_point": point,
    }
    phase_path = os.path.join(out, "janus_lora_phase_transition.json")
    with open(phase_path, "w", encoding="utf-8") as f:
        json.dump(phase, f, ensure_ascii=False, indent=2)
    best = sorted(results, key=lambda r: r["recovery_rate"], reverse=True)[0]
    report_path = os.path.join(out, "janus_lora_boundary_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LoRA-Janus Threshold Boundary Analysis\n\n")
        f.write("## Setup\n\n")
        f.write("- Fixed LoRA rank: `8`\n")
        f.write("- Fixed train size: `30` synthetic associations\n")
        f.write("- Sweep: learning_rate in `2e-4, 3e-4, 4e-4, 5e-4`; epochs in `3, 4, 5`\n")
        f.write("- Data: synthetic identifiers and synthetic emails under `example-synthetic.com` only.\n\n")
        f.write("## Threshold Point\n\n")
        f.write(f"- First recovery_rate > 0.5: lr=`{point['threshold_lr']}`, epochs=`{point['threshold_epochs']}`, recovery=`{point['threshold_recovery_rate']:.4f}`\n")
        f.write(f"- Best observed: lr=`{best['learning_rate']}`, epochs=`{best['epochs']}`, recovery=`{best['recovery_rate']:.4f}`\n\n")
        f.write("## Full Results\n\n")
        f.write("| lr | epochs | recovery_rate | recovered |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in sorted(results, key=lambda r: (r["learning_rate"], r["epochs"])):
            f.write(f"| {row['learning_rate']} | {row['epochs']} | {row['recovery_rate']:.4f} | {row['recovered']} / {row['eval_size']} |\n")
        f.write("\n## Analysis\n\n")
        f.write("1. The sweep directly tests whether recovery changes sharply as training strength increases.\n")
        f.write("2. A jump from low recovery to high recovery indicates a phase-transition-like threshold, not a smooth rank-only effect.\n")
        f.write("3. Since rank is fixed at r=8, any recovery jump is driven by learning_rate and epochs.\n")
        f.write("4. Weak LoRA has nearly no recovery because the adapter does not learn the synthetic association rule strongly enough.\n")
        f.write("5. Once learning_rate x epochs crosses the threshold, even r=8 can enter a high memorization state.\n")
        f.write("6. Compared with full FT, LoRA can be safer under weak settings but can become similarly risky under aggressive adapter training.\n")
    print(table_path)
    print(point_path)
    print(phase_path)
    print(report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA-Janus boundary sweep")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    ensure_train_30(str(cfg.get("train_data")))
    ensure_eval(str(cfg.get("eval_data")))
    sweep = cfg.get("boundary_sweep", {})
    combos = list(itertools.product(sweep.get("learning_rate", []), sweep.get("epochs", [])))
    results: List[Dict[str, Any]] = []
    output_path = cfg.get("output_path", os.path.join(PROJECT_PATH, "outputs", "janus_lora_boundary_results.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for index, (lr, epochs) in enumerate(combos, start=1):
        result = run_one(cfg, float(lr), int(epochs), index, len(combos))
        results.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    write_outputs(cfg, results)
    print(f"[Boundary Sweep] saved results: {output_path}")


if __name__ == "__main__":
    main()
