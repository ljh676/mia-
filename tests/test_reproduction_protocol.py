"""Numerical/protocol checks runnable without downloading any model weights."""
import ast
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extracted_function(module, name, scope):
    path = ROOT / 'reproductions' / module
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class ReproductionProtocolTests(unittest.TestCase):
    def test_padding_does_not_change_causal_likelihood(self):
        try:
            import torch
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError:
            self.skipTest('Torch/Transformers are checked on the GPU server')
        from reproductions.lm import causal_losses
        torch.manual_seed(7)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=16, n_layer=1, n_head=1, n_embd=8)).eval()
        with torch.no_grad():
            original = causal_losses(model, torch.tensor([[4, 5, 6]]), torch.ones(1, 3, dtype=torch.long))
            left = causal_losses(model, torch.tensor([[0, 0, 4, 5, 6]]), torch.tensor([[0, 0, 1, 1, 1]]))
            right = causal_losses(model, torch.tensor([[4, 5, 6, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        self.assertAlmostEqual(original.item(), left.item(), places=6)
        self.assertAlmostEqual(original.item(), right.item(), places=6)

    def test_ia_unknown_penalty_and_no_gold_fabrication(self):
        score = extracted_function('rag.py', 'ia_score', {})
        self.assertEqual(score(['yes', 'no', 'yes'], ['yes', 'yes', 'UNK'], 5), -4 / 3)
        with self.assertRaises(ValueError):
            score(['UNK'], ['UNK'], 5)
        with self.assertRaises(ValueError):
            score(['yes'], [], 5)

    def test_spv_calibration_cancels_identical_models(self):
        score = extracted_function('spv.py', 'calibrated_score', {'math': math})
        self.assertEqual(score(1., [1.2, .9], 1., [1.2, .9]), 0.)
        # A target's larger probability drop under perturbation indicates membership.
        self.assertGreater(score(1., [2., 2.], 1., [1., 1.]), 0.)

    def test_antonym_edits_use_exact_word_positions_and_record_budget(self):
        import re
        apply = extracted_function('rag.py', 'apply_edits', {'re': re})
        document = 'High levels and high quality were reported.'
        record = {'edits': [{'position': 3, 'original': 'high', 'replacement': 'low'}]}
        changed, meta = apply(document, record, .5)
        self.assertEqual(changed, 'High levels and low quality were reported.')
        self.assertFalse(meta['requested_budget_met'])
        self.assertEqual(meta['words_replaced'], 1)

    def test_sparse_reconstruction_rejects_model_deletions_and_inflections(self):
        import re
        import difflib
        extract = extracted_function('rag.py', 'sparse_replacements', {'re': re, 'difflib': difflib})
        apply = extracted_function('rag.py', 'apply_edits', {'re': re})
        original = 'The acute condition identifies common symptoms. Further tests are needed.'
        proposal = 'The chronic condition identify rare symptoms.'
        edits = extract(original, proposal)
        self.assertEqual([(e['original'], e['replacement']) for e in edits], [('acute', 'chronic'), ('common', 'rare')])
        changed, _ = apply(original, {'edits': edits}, .5)
        self.assertEqual(changed, 'The chronic condition identifies rare symptoms. Further tests are needed.')

    def test_held_out_labels_cannot_select_threshold(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest('scikit-learn is installed on the experiment server')
        from reproductions.common import evaluate
        calibration = [{'id': 'a', 'split': 'calibration', 'label': 0, 'score': .2},
                       {'id': 'b', 'split': 'calibration', 'label': 1, 'score': .8}]
        test = [{'id': 'c', 'split': 'test', 'label': 0, 'score': .3},
                {'id': 'd', 'split': 'test', 'label': 1, 'score': .7}]
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = evaluate(calibration + test, first, 'check')
            b = evaluate(calibration + [{**r, 'label': 1 - r['label']} for r in test], second, 'check')
        self.assertEqual(a['threshold'], b['threshold'])
        self.assertNotEqual(a['auc'], b['auc'])


if __name__ == '__main__':
    unittest.main()
