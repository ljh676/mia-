"""Attack x Defense benchmark ???

?????? membership evaluation???? defense???????
privacy ??? utility ????? privacy-utility tradeoff ???
"""

import argparse
import contextlib
import glob
import json
import os
import subprocess
import sys

from defenses.registry import get_defense
from utils.io_utils import load_yaml, save_json
from utils.utility_metrics import utility_drop, utility_summary


def _load_json(path):
    """?? JSON ???"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _result_path_for_attack(attack_name):
    """?? attack ???????????"""
    return os.path.join("outputs", f"{attack_name}_result.json")


def _metric_block(metrics):
    """???? metrics ???"""
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "auc": float(metrics.get("auc", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
    }


def _summary_entry(path):
    """??? benchmark JSON ?????????

    ? benchmark ?????? utility_before/utility_after???????
    before_result_path ? after_result ???? score-level utility??????
    ???????????????? 0?
    """
    data = _load_json(path)
    attack_name = data.get("attack", "")
    utility_before = data.get("utility_before", {}) or {}
    utility_after = data.get("utility_after", {}) or {}

    if not utility_before:
        before_path = data.get("before_result_path")
        if before_path and os.path.exists(before_path):
            utility_before = utility_summary(_load_json(before_path), attack_name=attack_name)
    if not utility_after:
        after_result = data.get("after_result") or {}
        if after_result:
            utility_after = utility_summary(after_result, attack_name=attack_name, fallback_utility=utility_before)

    display_defense = data.get("defense", "")
    after_result = data.get("after_result") or {}
    if after_result.get("real_rerun"):
        level = after_result.get("sanitization_level")
        display_defense = f"{display_defense}_{level}" if level else f"{display_defense}_realrerun"

    return {
        "attack": attack_name,
        "defense": display_defense,
        "auc_before": float((data.get("before") or {}).get("auc", 0.0)),
        "auc_after": float((data.get("after") or {}).get("auc", 0.0)),
        "accuracy_before": float((data.get("before") or {}).get("accuracy", 0.0)),
        "accuracy_after": float((data.get("after") or {}).get("accuracy", 0.0)),
        "retrieval_hit_rate_before": float(utility_before.get("retrieval_hit_rate", 0.0)),
        "retrieval_hit_rate_after": float(utility_after.get("retrieval_hit_rate", 0.0)),
        "qa_before": float(utility_before.get("qa_accuracy", 0.0)),
        "qa_after": float(utility_after.get("qa_accuracy", 0.0)),
        "utility_before": float(utility_before.get("utility_score", 0.0)),
        "utility_after": float(utility_after.get("utility_score", 0.0)),
        "source": path,
    }




def _write_query_sanitization_report(output):
    """? query_sanitization ?? tradeoff JSON ? Markdown ???"""
    after_result = output.get("after_result", {}) or {}
    examples = after_result.get("examples", []) or []
    tradeoff = {
        "attack": output.get("attack"),
        "defense": output.get("defense"),
        "real_rerun": bool(after_result.get("real_rerun", False)),
        "privacy_gain": {
            "auc_drop": output.get("defense_effect", {}).get("auc_drop", 0.0),
            "accuracy_drop": output.get("defense_effect", {}).get("accuracy_drop", 0.0),
        },
        "utility_loss": {
            "retrieval_hit_rate_drop": output.get("utility_drop", {}).get("retrieval_hit_rate_drop", 0.0),
            "qa_accuracy_drop": output.get("utility_drop", {}).get("qa_accuracy_drop", 0.0),
            "utility_score_drop": output.get("utility_drop", {}).get("utility_score_drop", 0.0),
        },
        "examples": examples,
    }
    save_json(tradeoff, os.path.join("outputs", "query_sanitization_tradeoff.json"))

    before = output.get("before", {}) or {}
    after = output.get("after", {}) or {}
    utility_before = output.get("utility_before", {}) or {}
    utility_after = output.get("utility_after", {}) or {}
    first = examples[0] if examples else {}
    retrieved_before = json.dumps(first.get("retrieved_docs_before", []), ensure_ascii=False, indent=2)
    retrieved_after = json.dumps(
        first.get("sanitized_retrieved_docs_after") or first.get("retrieved_docs_after", []),
        ensure_ascii=False,
        indent=2,
    )
    report = f"""# Query Sanitization Privacy-Utility Report

## 1. Query Sanitization ??

`query_sanitization` ? input-level privacy defense??? RAG ????? query ? retrieved context ???????????????????????????????????????????????

Real rerun: `{bool(after_result.get('real_rerun', False))}`

## 2. ?? Query ??

```text
{first.get('query_before', '')}
```

## 3. Sanitized Query ??

```text
{first.get('query_after', '')}
```

## 4. ?? Retrieved Docs ??

```json
{retrieved_before[:2500]}
```

## 5. Sanitized Retrieved Docs ??

```json
{retrieved_after[:2500]}
```

## 6. Retrieval Context Before/After ??

**Before**

```text
{first.get('context_before', '')[:1200]}
```

**After**

```text
{first.get('context_after', '')[:1200]}
```

## 7. Phi3 Response ??

**Before**

```text
{first.get('response_before', '')}
```

**After**

```text
{first.get('response_after', '')}
```

## 8. Privacy Metrics

- AUC before: {before.get('auc', 0.0):.4f}
- AUC after: {after.get('auc', 0.0):.4f}
- Accuracy before: {before.get('accuracy', 0.0):.4f}
- Accuracy after: {after.get('accuracy', 0.0):.4f}

## 9. Utility Metrics

- Retrieval hit rate before: {utility_before.get('retrieval_hit_rate', 0.0):.4f}
- Retrieval hit rate after: {utility_after.get('retrieval_hit_rate', 0.0):.4f}
- QA accuracy before: {utility_before.get('qa_accuracy', 0.0):.4f}
- QA accuracy after: {utility_after.get('qa_accuracy', 0.0):.4f}
- Utility score before: {utility_before.get('utility_score', 0.0):.4f}
- Utility score after: {utility_after.get('utility_score', 0.0):.4f}

## 10. Utility Drop

- Retrieval hit rate drop: {output.get('utility_drop', {}).get('retrieval_hit_rate_drop', 0.0):.4f}
- QA accuracy drop: {output.get('utility_drop', {}).get('qa_accuracy_drop', 0.0):.4f}
- Utility score drop: {output.get('utility_drop', {}).get('utility_score_drop', 0.0):.4f}

## 11. ????

?? real rerun ?????? sanitized question ? BGE retrieval ? Phi3 generation?????? scores ? utility????????????????target_docs ? fallback answer ?????? RAG_MIA M=20/N=5 ??????????? sanitized target_docs ????? pipeline??????????????????
"""
    with open(os.path.join("outputs", "query_sanitization_report.md"), "w", encoding="utf-8") as f:
        f.write(report)


def _write_summary_files():
    """?? benchmark ????? JSON/Markdown ????"""
    os.makedirs("outputs", exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join("outputs", "benchmark_*_result.json"))):
        try:
            rows.append(_summary_entry(path))
        except Exception as exc:  # noqa: BLE001 - ???????????????????
            rows.append({"source": path, "error": str(exc)})
    save_json(rows, os.path.join("outputs", "benchmark_summary.json"))

    header = "| Attack | Defense | AUC Before | AUC After | Utility Before | Utility After | QA Before | QA After | Retrieval Before | Retrieval After |\n"
    sep = "|--------|---------|------------|-----------|----------------|---------------|-----------|----------|------------------|-----------------|\n"
    lines = [header, sep]
    for row in rows:
        if "error" in row:
            continue
        lines.append(
            "| {attack} | {defense} | {auc_before:.4f} | {auc_after:.4f} | {utility_before:.4f} | {utility_after:.4f} | {qa_before:.4f} | {qa_after:.4f} | {retrieval_hit_rate_before:.4f} | {retrieval_hit_rate_after:.4f} |\n".format(**row)
        )
    with open(os.path.join("outputs", "benchmark_summary.md"), "w", encoding="utf-8") as f:
        f.writelines(lines)


def run_benchmark(attack_name, attack_config, defense_name, defense_config):
    """?? membership evaluation ? defense benchmark???? utility ???"""
    defense_cfg = load_yaml(defense_config)
    real_rerun = bool(defense_cfg.get("real_rerun", False))
    suffix = "_realrerun" if defense_name == "query_sanitization" and real_rerun else ""
    log_path = os.path.join("outputs", "logs", f"benchmark_{attack_name}_{defense_name}{suffix}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    attack_cmd = [sys.executable, "run_attack.py", "--attack", attack_name, "--config", attack_config]
    proc = subprocess.run(attack_cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=3600)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("[ATTACK COMMAND] " + " ".join(attack_cmd) + "\n")
        log.write(f"[ATTACK RETURNCODE] {proc.returncode}\n")
        log.write("[STDOUT]\n" + proc.stdout + "\n[STDERR]\n" + proc.stderr + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"membership evaluation failed: {attack_name}, see {log_path}")

    attack_cfg = load_yaml(attack_config)
    # ???? attack config ??? output_file??????????
    # rag_mia_large_result.json ? benchmark ????? rag_mia_result.json?
    configured_output = attack_cfg.get("output_file")
    if configured_output:
        before_path = configured_output if os.path.isabs(configured_output) else os.path.abspath(configured_output)
    else:
        before_path = _result_path_for_attack(attack_name)
    before_result = _load_json(before_path)

    defense_cls = get_defense(defense_name)
    defense = defense_cls(defense_cfg)
    defense.prepare()
    with open(log_path, "a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print("[DEFENSE APPLY BEGIN]")
        after_result = defense.apply(before_result)
        print("[DEFENSE APPLY END]")
    effect = defense.evaluate(before_result, after_result)

    before_metrics = _metric_block(before_result.get("metrics", {}) or {})
    after_metrics = _metric_block(after_result.get("metrics", {}) or {})
    utility_before = utility_summary(before_result, attack_name=attack_name, config=attack_cfg)
    utility_after = utility_summary(after_result, attack_name=attack_name, config=attack_cfg, fallback_utility=utility_before)
    utility_delta = utility_drop(utility_before, utility_after)

    output = {
        "attack": attack_name,
        "defense": defense_name,
        "before": before_metrics,
        "after": after_metrics,
        "defense_effect": effect,
        "utility_before": utility_before,
        "utility_after": utility_after,
        "utility_drop": utility_delta,
        # ?????????????????????
        "auc_before": before_metrics["auc"],
        "auc_after": after_metrics["auc"],
        "retrieval_hit_rate_before": utility_before.get("retrieval_hit_rate", 0.0),
        "retrieval_hit_rate_after": utility_after.get("retrieval_hit_rate", 0.0),
        "qa_accuracy_before": utility_before.get("qa_accuracy", 0.0),
        "qa_accuracy_after": utility_after.get("qa_accuracy", 0.0),
        "after_result": after_result,
        "before_result_path": before_path,
        "log_path": log_path,
    }
    out_path = os.path.join("outputs", f"benchmark_{attack_name}_{defense_name}{suffix}_result.json")
    save_json(output, out_path)
    if defense_name == "query_sanitization":
        _write_query_sanitization_report(output)
    _write_summary_files()

    with open(log_path, "a", encoding="utf-8") as log:
        log.write("[BENCHMARK RESULT]\n" + json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"benchmark result saved to: {out_path}")
    print(json.dumps({
        "attack": output["attack"],
        "defense": output["defense"],
        "before": output["before"],
        "after": output["after"],
        "defense_effect": output["defense_effect"],
        "utility_drop": output["utility_drop"],
    }, ensure_ascii=False, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Attack x Defense privacy-utility benchmark")
    parser.add_argument("--attack", required=True)
    parser.add_argument("--attack-config", required=True)
    parser.add_argument("--defense", required=True)
    parser.add_argument("--defense-config", required=True)
    args = parser.parse_args()
    run_benchmark(args.attack, args.attack_config, args.defense, args.defense_config)


if __name__ == "__main__":
    main()
