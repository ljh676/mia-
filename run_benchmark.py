"""Run one attack/defense comparison with explicit measurement provenance."""

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
import traceback
import uuid

from defenses.registry import get_defense
from run_attack import run_attack
from utils.io_utils import load_yaml, save_json
from utils.metrics import METRIC_NAMES, format_metric
from utils.utility_metrics import utility_drop, utility_summary


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _metric_block(metrics):
    return {key: metrics.get(key) for key in METRIC_NAMES}


def _summary_entry(path):
    data = _load_json(path)
    return {"attack": data["attack"], "defense": data["defense"],
            "evidence_status": data.get("evidence_status", "legacy_unvalidated"),
            "run_id": data.get("run_id"),
            "auc_before": (data.get("before") or {}).get("auc"),
            "auc_after": (data.get("after") or {}).get("auc"),
            "qa_before": (data.get("utility_before") or {}).get("qa_accuracy"),
            "qa_after": (data.get("utility_after") or {}).get("qa_accuracy"),
            "recovery_after": (data.get("after_result") or {}).get("recovery_rate_after"),
            "source": path}


def _write_summary_files(paths):
    # Summarize this invocation only. Do not silently mix old experiments.
    rows = [_summary_entry(path) for path in paths]
    save_json(rows, "outputs/benchmark_summary.json")
    lines = ["| Attack | Defense | Evidence | AUC before | AUC after | QA before | QA after | Recovery after |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        values = [row["attack"], row["defense"], row["evidence_status"]]
        values += [format_metric(row[key]) for key in ("auc_before", "auc_after", "qa_before", "qa_after", "recovery_after")]
        lines.append("| " + " | ".join(values) + " |")
    with open("outputs/benchmark_summary.md", "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n\nN/A means unavailable, not zero. Score simulations do not establish protection.\n")


def _write_query_sanitization_report(output):
    report = {"run_id": output["run_id"], "evidence_status": output["evidence_status"],
              "score_change": output["defense_effect"], "utility_before": output["utility_before"],
              "utility_after": output["utility_after"], "utility_drop": output["utility_drop"]}
    save_json(report, "outputs/query_sanitization_tradeoff.json")
    with open("outputs/query_sanitization_report.md", "w", encoding="utf-8") as stream:
        stream.write("# Query sanitization measurements\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n")


def run_benchmark(attack_name, attack_config, defense_name, defense_config):
    attack_cfg, defense_cfg = load_yaml(attack_config), load_yaml(defense_config)
    run_id = uuid.uuid4().hex
    log_path = f"outputs/logs/benchmark_{run_id}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            before_result = run_attack(attack_name, config=attack_cfg)
            if before_result.get("status") in {"not_run", "skipped", "failed"}:
                raise RuntimeError("Benchmark requires an executed attack, not an empty/preflight result")
            if attack_name == "rag_mia" and defense_name in {"query_sanitization", "epd"}:
                path = before_result.get("parsed_output_path")
                if not path:
                    raise RuntimeError("RAG defense requires this run's target-doc artifact")
                defense_cfg["target_docs_path"] = path
            defense = get_defense(defense_name)(defense_cfg)
            defense.prepare()
            after_result = defense.apply(before_result)
            if before_result.get("labels") != after_result.get("labels"):
                raise ValueError("Before/after labels differ; comparisons require the same evaluation samples")
            if before_result.get("sample_ids") is not None and before_result["sample_ids"] != after_result.get("sample_ids"):
                raise ValueError("Before/after sample identities or order differ")
            effect = defense.evaluate(before_result, after_result)
        except Exception:
            traceback.print_exc()
            raise
    kind = after_result.get("implementation_kind", "unvalidated_adaptation")
    evidence = "simulation_only" if kind == "score_simulation" else "no_defense_applied" if kind == "no_op" else kind
    before_metrics = _metric_block(before_result.get("metrics") or {})
    after_metrics = _metric_block(after_result.get("metrics") or {})
    utility_before = utility_summary(before_result, attack_name, attack_cfg)
    utility_after = utility_summary(after_result, attack_name, attack_cfg, utility_before)
    output = {"schema_version": 2, "run_id": run_id,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "attack": attack_name, "defense": defense_name, "evidence_status": evidence,
              "attack_implementation": before_result.get("implementation_kind"),
              "paper_reproduction_status": before_result.get("paper_reproduction_status"),
              "before": before_metrics, "after": after_metrics, "defense_effect": effect,
              "utility_before": utility_before, "utility_after": utility_after,
              "utility_drop": utility_drop(utility_before, utility_after),
              "auc_before": before_metrics["auc"], "auc_after": after_metrics["auc"],
              "after_result": after_result, "before_result_path": attack_cfg.get("output_file"),
              "log_path": log_path,
              "measurement_note": "AUC changes are descriptive; simulations and output redaction do not establish model privacy."}
    output_path = f"outputs/benchmark_{attack_name}_{defense_name}_{run_id}_result.json"
    save_json(output, output_path)
    _write_summary_files([output_path])
    if defense_name == "query_sanitization":
        _write_query_sanitization_report(output)
    print(f"benchmark result saved to: {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--attack-config", required=True)
    parser.add_argument("--defense", required=True)
    parser.add_argument("--defense-config", required=True)
    args = parser.parse_args()
    run_benchmark(args.attack, args.attack_config, args.defense, args.defense_config)


if __name__ == "__main__":
    main()
