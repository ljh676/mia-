"""Regression checks for measurement integrity; no models or GPU required."""

import copy
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from defenses.hamp import HAMPDefense
from defenses.miashield import MIAShieldDefense
from defenses.pii_output_filter import PIIOutputFilterDefense
from run_attack import run_attack
from run_benchmark import run_benchmark
from run_defense import run_defense
from utils.metrics import auc, compute_metrics
from utils.result_parser import parse_result_sources, parse_json_file
from utils.subprocess_utils import require_fresh_output, run_with_log
from utils.utility_metrics import (_extract_rag_responses_and_expected, qa_accuracy,
                                   retrieval_hit_rate, utility_drop, utility_summary)


REPO_ROOT = Path(__file__).resolve().parents[1]


def archived_function(path, name, namespace=None):
    # Exercise pure adapter/report functions without importing model libraries.
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {} if namespace is None else dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


@contextmanager
def working_directory(path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def defense(self, cls, **kwargs):
        obj = cls({"output_path": str(self.root / "result.json"), **kwargs})
        obj.prepare()
        return obj

    def test_miashield_labels_cannot_change_decisions(self):
        defense = self.defense(MIAShieldDefense)
        scores = [0.5] * 8
        a = defense.apply({"scores": scores, "labels": [0] * 8})
        b = defense.apply({"scores": scores, "labels": [1] * 8})
        self.assertEqual(a["scores"], b["scores"])
        self.assertEqual(a["excluded_subsets"], [-1] * 8)
        self.assertEqual(a["implementation_kind"], "score_simulation")

    def test_miashield_does_not_mutate_details(self):
        data = {"scores": [0.2, 0.8], "labels": [0, 1], "details": [{"text": "alpha"}, {"text": "beta"}]}
        original = copy.deepcopy(data)
        self.defense(MIAShieldDefense).apply(data)
        self.assertEqual(data, original)

    def test_invalid_subset_count_rejected(self):
        with self.assertRaises(ValueError):
            self.defense(MIAShieldDefense, num_subsets=0)

    def test_hamp_is_permutation_equivariant(self):
        defense = self.defense(HAMPDefense)
        scores = [0.1, 0.3, 0.7, 0.9]
        a = defense.apply({"scores": scores, "labels": [0, 0, 1, 1]})
        b = defense.apply({"scores": scores[::-1], "labels": [1, 1, 0, 0]})
        self.assertEqual(a["scores"], b["scores"][::-1])
        self.assertEqual(a["metrics"]["auc"], 1.0)
        self.assertIsNone(a["metrics"]["accuracy"])

    def test_legacy_hamp_rank_noise_rejected(self):
        with self.assertRaises(ValueError):
            self.defense(HAMPDefense, calibration_noise=0.14)

    def test_empty_and_single_class_auc_unavailable(self):
        self.assertTrue(all(value is None for value in compute_metrics([], [], []).values()))
        self.assertIsNone(auc([1, 1], [0.2, 0.8]))

    def test_tied_and_reversed_auc_are_real_values(self):
        self.assertEqual(auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(auc([0, 1], [0.9, 0.1]), 0.0)

    def test_invalid_metric_inputs_rejected(self):
        for labels, predictions, scores in [([0, 1], [0], [0.2, 0.9]),
                                             ([0, 1], [0, 1], [0.2]),
                                             ([0, 0.5], [0, 1], [0.2, 0.9]),
                                             ([0, 1], [0, 1], [float("nan"), 0.9])]:
            with self.subTest(labels=labels, predictions=predictions, scores=scores):
                with self.assertRaises(ValueError):
                    compute_metrics(labels, predictions, scores)

    def recovery_payload(self, expected="user_b001@example-synthetic.com"):
        return {"scores": [0.9], "labels": [1], "predictions": [1], "metrics": {"auc": None},
                "eval_size": 1, "recovered": 1, "recovery_rate": 1.0,
                "examples": [{"generated": expected, "expected_synthetic_pii": expected}]}

    def test_filter_uses_actual_outputs_and_preserves_mia(self):
        data = self.recovery_payload()
        original = copy.deepcopy(data)
        result = self.defense(PIIOutputFilterDefense).apply(data)
        self.assertEqual(result["examples"][0]["generated"], "[EMAIL]")
        self.assertEqual(result["filtered_count"], 1)
        self.assertEqual(result["recovery_rate_after"], 0.0)
        self.assertEqual(result["predictions"], original["predictions"])
        self.assertEqual(result["metrics"], original["metrics"])
        self.assertEqual(data, original)

    def test_unmatched_output_does_not_report_blocking(self):
        result = self.defense(PIIOutputFilterDefense).apply(self.recovery_payload("unmatched@example.org"))
        self.assertEqual(result["filtered_count"], 0)
        self.assertEqual(result["recovery_rate_after"], 1.0)
        self.assertEqual(result["filter_rate"], 0.0)

    def test_filter_does_not_match_a_different_domain_suffix(self):
        result = self.defense(PIIOutputFilterDefense).apply(self.recovery_payload("user_a001@example-synthetic.com.evil"))
        self.assertEqual(result["filtered_count"], 0)
        self.assertEqual(result["recovery_rate_after"], 1.0)

    def test_preview_examples_do_not_establish_global_recovery(self):
        data = self.recovery_payload()
        data["eval_size"] = 100
        result = self.defense(PIIOutputFilterDefense).apply(data)
        self.assertIsNone(result["recovery_rate_after"])
        self.assertIsNone(result["recovered"])
        self.assertEqual(result["recovery_evaluated_count"], 1)

    def test_missing_generations_are_unassessed(self):
        result = self.defense(PIIOutputFilterDefense).apply({"eval_size": 100, "recovered": 100})
        self.assertIsNone(result["recovery_rate_after"])
        self.assertEqual(result["filtered_count"], 0)

    def test_qa_requires_gold_and_counts_invalid_responses(self):
        self.assertIsNone(qa_accuracy(["Yes"], [])["qa_accuracy"])
        self.assertEqual(qa_accuracy(["Yes", "unparseable"], ["Yes", "No"])["qa_accuracy"], 0.5)

    def test_missing_gold_does_not_shift_later_records(self):
        responses, gold = _extract_rag_responses_and_expected([
            ("a", {"llm_responses": ["No"]}),
            ("b", {"llm_responses": ["Yes"], "answers": ["Yes"]})])
        self.assertEqual(gold, [None, "Yes"])
        self.assertIsNone(qa_accuracy(responses, gold)["qa_accuracy"])

    def test_empty_retrieval_counts_as_a_miss(self):
        result = retrieval_hit_rate(None, {"a": {"retrieved_doc_ids": [[], ["a"]]}})
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["hit_rate"], 0.5)

    def test_attack_score_variance_is_not_task_utility(self):
        summary = utility_summary({"scores": [0.1, 0.9]})
        self.assertIsNone(summary["utility_score"])
        self.assertIsNone(summary["qa_accuracy"])
        self.assertIsNone(utility_drop(summary, summary)["utility_score_drop"])

    def test_utility_is_only_inherited_for_unchanged_task_output(self):
        before = utility_summary({"responses": ["Yes"], "answers": ["Yes"]})
        unknown = utility_summary({}, fallback_utility=before)
        unchanged = utility_summary({"task_output_changed": False}, fallback_utility=before)
        self.assertIsNone(unknown["qa_accuracy"])
        self.assertEqual(unchanged["qa_accuracy"], 1.0)
        self.assertIn("not remeasured", unchanged["utility_source"])

    def external(self, code, output=None, timeout=10):
        return run_with_log([sys.executable, "-c", code], str(self.root), str(self.root / "process.log"),
                            timeout=timeout, output_paths=[str(output)] if output else [])

    def test_failed_process_cannot_use_old_result(self):
        output = self.root / "old.json"
        output.write_text('{"scores":[1]}')
        with self.assertRaises(RuntimeError):
            self.external("raise SystemExit(7)", output)

    def test_success_without_new_output_rejected(self):
        output = self.root / "old.json"
        output.write_text('{"scores":[1]}')
        info = self.external("pass", output)
        with self.assertRaises(RuntimeError):
            require_fresh_output(output, info)
        with self.assertRaises(RuntimeError):
            parse_result_sources("dcmi", str(output), fresh_outputs=info["fresh_outputs"])

    def test_fresh_zero_auc_is_accepted(self):
        output = self.root / "new.json"
        code = "import json; json.dump({'scores':[0.9,0.1],'labels':[0,1],'predictions':[1,0]},open('new.json','w'))"
        info = self.external(code, output)
        result = parse_result_sources("dcmi", str(output), fresh_outputs=info["fresh_outputs"])
        self.assertEqual(result["metrics"]["auc"], 0.0)

    def test_timeout_is_a_failure(self):
        with self.assertRaises(RuntimeError):
            self.external("import time; time.sleep(1)", timeout=0.05)

    def attack_config(self, code):
        (self.root / "MIA.py").write_text(code, encoding="utf-8")
        output = str(self.root / "raw.json")
        return {"project_path": str(self.root), "python_path": sys.executable, "entry_candidates": ["MIA.py"],
                "execute_original": True, "output_dir": str(self.root), "output_path": output,
                "original_output_path": output, "output_file": str(self.root / "unified.json"),
                "log_path": str(self.root / "attack.log"), "extra_args": []}

    def test_wrapper_stops_when_child_fails(self):
        cfg = self.attack_config("raise SystemExit(2)")
        Path(cfg["output_path"]).write_text('{"scores":[1],"labels":[1]}')
        with self.assertRaises(RuntimeError):
            run_attack("dcmi", config=cfg)
        self.assertFalse(Path(cfg["output_file"]).exists())

    def test_benchmark_reports_simulation_and_na(self):
        code = "import json; json.dump({'scores':[0.2,0.8],'labels':[0,1],'predictions':[0,1]},open('raw.json','w'))"
        cfg = self.attack_config(code)
        attack_cfg, defense_cfg = self.root / "attack.yml", self.root / "defense.yml"
        # JSON is also valid YAML; avoids test-only serialization dependencies.
        attack_cfg.write_text(json.dumps(cfg))
        defense_cfg.write_text(json.dumps({"output_path": str(self.root / "defended.json")}))
        with working_directory(self.root):
            result = run_benchmark("dcmi", str(attack_cfg), "hamp", str(defense_cfg))
            self.assertEqual(result["evidence_status"], "simulation_only")
            self.assertIsNone(result["after"]["accuracy"])
            self.assertIsNone(result["utility_after"]["utility_score"])
            self.assertIn("N/A", Path("outputs/benchmark_summary.md").read_text())

    def test_defense_entry_actually_applies_filter(self):
        source = self.root / "input.json"
        source.write_text(json.dumps(self.recovery_payload()))
        result = run_defense("pii_output_filter", config={"output_path": str(self.root / "output.json")}, input_path=str(source))
        self.assertTrue(result["applied"])
        self.assertEqual(result["recovery_rate_after"], 0.0)

    def test_preparation_cannot_report_applied_success(self):
        result = run_defense("pii_output_filter", config={}, check_only=True)
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["applied"])
        with self.assertRaises(ValueError):
            run_defense("pii_output_filter", config={})

    def test_janus_transfer_does_not_invent_generations(self):
        function = archived_function("external/Janus/run_janus_mia_eval.py", "evaluate_pii_output_filter_transfer")
        result = function(self.defense(PIIOutputFilterDefense), {"s2_recovery_rate": 1.0, "examples": []})
        self.assertIsNone(result["recovery"])
        self.assertEqual(result["recovery_evaluation_status"], "insufficient_generation_records")

    def test_janus_transfer_filters_complete_records(self):
        function = archived_function("external/Janus/run_janus_mia_eval.py", "evaluate_pii_output_filter_transfer")
        data = self.recovery_payload()
        result = function(self.defense(PIIOutputFilterDefense), {
            "generation_records": data["examples"], "generation_records_complete": True})
        self.assertEqual(result["recovery"], 0.0)

    def test_real_dcmi_failure_never_switches_to_synthetic(self):
        from types import SimpleNamespace

        def fail_real(_):
            raise FileNotFoundError("missing real dataset")

        def forbidden_fallback():
            self.fail("Synthetic fallback was invoked")

        function = archived_function("external/DCMI/MIA.py", "main", {
            "parse_args": lambda: SimpleNamespace(mode="real", sample_n=2),
            "run_real_dcmi": fail_real, "run_synthetic_dcmi": forbidden_fallback})
        with self.assertRaises(FileNotFoundError):
            function()

    def test_dcmi_score_orientation_does_not_depend_on_evaluation_labels(self):
        from types import SimpleNamespace
        # Force reversed label/score separation. It must remain AUC=0, not be flipped.
        scope = {"load_corpus": lambda: {"member": {"text": "m"}, "non": {"text": "n"}},
                 "load_queries": lambda: {"q": "query"}, "load_qrels": lambda: {"q": {"member": 1}},
                 "random": SimpleNamespace(shuffle=lambda _: None, choice=lambda items: items[0]),
                 "score_pair": lambda query, text: {"score": 0.1 if text == "m" else 0.9},
                 "roc_auc_score": auc, "compute_metrics": compute_metrics,
                 "np": SimpleNamespace(mean=lambda values: sum(values) / len(values)),
                 "Dict": dict}
        function = archived_function("external/DCMI/MIA.py", "run_real_dcmi", scope)
        result = function(1)
        self.assertEqual(result["metrics"]["auc"], 0.0)
        self.assertEqual(result["score_orientation"], "predeclared_original")

    def test_rag_cannot_replace_missing_answers_with_retrieval_hits(self):
        path = self.root / "rag.json"
        path.write_text(json.dumps({"doc": {"mem": "yes", "retrieved_doc_ids": [["doc"]],
                                           "llm_responses": ["Yes"], "accuracy": 100}}))
        with self.assertRaises(ValueError):
            parse_json_file("rag_mia", str(path))

    def test_rag_uses_reference_answer_agreement(self):
        path = self.root / "rag.json"
        path.write_text(json.dumps({
            "member": {"mem": "yes", "retrieved_doc_ids": [[]], "answers": ["Yes"], "llm_responses": ["Yes"]},
            "nonmember": {"mem": "no", "retrieved_doc_ids": [["nonmember"]], "answers": ["Yes"], "llm_responses": ["No"]}}))
        result = parse_json_file("rag_mia", str(path))
        self.assertEqual(result["scores"], [1.0, 0.0])
        self.assertEqual(result["metrics"]["auc"], 1.0)
        self.assertEqual(result["score_source"], "reference_answer_agreement")

    def test_rag_source_scorer_requires_reference_answers(self):
        import re
        from types import SimpleNamespace
        path = REPO_ROOT / "external/RAG_MIA/mia_utils/mia.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MIA_Attacker")
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_score")
        scope = {"re": re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        current = SimpleNamespace(target_docs={"doc": {"retrieved_doc_ids": [["doc"]], "llm_responses": ["Yes"]}},
                                  save_target_docs=lambda: self.fail("Invalid scoring must not save a successful result"))
        with self.assertRaises(ValueError):
            scope["calculate_score"](current)


if __name__ == "__main__":
    unittest.main()
