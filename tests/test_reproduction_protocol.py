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

    def test_ia_rejects_open_questions_and_incomplete_format(self):
        import re
        validate = extracted_function('query_generation.py', 'valid_binary_question', {'re': re})
        self.assertTrue(validate('Did the study enroll 127 adolescents?'))
        self.assertFalse(validate('What was the average age of the participants?'))
        self.assertFalse(validate('Was the study randomized'))

    def test_antonym_selection_cannot_invent_pairs_or_duplicate_positions(self):
        select = extracted_function('query_generation.py', 'select_lexical_edits', {})
        candidates = [{'position': 1, 'original': 'high', 'replacement': 'low'},
                      {'position': 1, 'original': 'high', 'replacement': 'bad'},
                      {'position': 4, 'original': 'positive', 'replacement': 'negative'}]
        selected = select(candidates, [True, -1, 80, '0', 0, 1, 2, 2])
        self.assertEqual([(e['position'], e['replacement']) for e in selected], [(1, 'low'), (4, 'negative')])

    def test_context_pos_filter_rejects_noun_with_adjective_dictionary_sense(self):
        import re
        from types import SimpleNamespace
        from unittest.mock import patch
        calls = []
        opposite = SimpleNamespace(name=lambda: 'rough', key=lambda: 'rough-key')
        lemma = SimpleNamespace(name=lambda: 'smooth', antonyms=lambda: [opposite], key=lambda: 'smooth-key',
                                synset=lambda: SimpleNamespace(definition=lambda: 'not rough'))
        def lemmas(word, pos):
            calls.append((word, pos))
            return [lemma]
        wordnet = SimpleNamespace(ensure_loaded=lambda: None, ADJ='a', ADV='r', lemmas=lemmas)
        corpus = SimpleNamespace(wordnet=wordnet)
        nltk = SimpleNamespace(corpus=corpus, pos_tag=lambda words: list(zip(words, ['DT', 'JJ', 'NN'])))
        extract = extracted_function('query_generation.py', 'lexical_candidates', {'re': re})
        with patch.dict('sys.modules', {'nltk': nltk, 'nltk.corpus': corpus}):
            candidates = extract('The smooth surface.')
        self.assertEqual(calls, [('smooth', 'a')])
        self.assertEqual([(e['original'], e['replacement']) for e in candidates], [('smooth', 'rough')])

    def test_edit_reconstruction_rejects_stale_document_offsets(self):
        import re
        apply = extracted_function('rag.py', 'apply_edits', {'re': re})
        with self.assertRaises(ValueError):
            apply('The high value.', {'edits': [{'position': 1, 'original': 'low', 'replacement': 'high'}]}, .1)

    def test_ia_batched_generation_resumes_and_keeps_required_question_count(self):
        import json
        import re
        from reproductions.common import read_json, write_json, text_hash
        class UnscorableSample(RuntimeError):
            pass
        generate = extracted_function('query_generation.py', 'generate_ia_queries', {
            'Path': Path, 'read_json': read_json, 'write_json': write_json, 'text_hash': text_hash,
            'json': json, 're': re, 'json_object': json.loads, 'UnscorableSample': UnscorableSample,
            'valid_binary_question': extracted_function('query_generation.py', 'valid_binary_question', {'re': re})})
        class Generator:
            calls = 0
            def generate(self, prompt, maximum, temperature=0.):
                self.calls += 1
                return json.dumps({'summary': 'My inquiry concerns trials.', 'questions': [
                    f'Did cohort {self.calls * 10 + i} receive the treatment?' for i in range(8)]})
            def answer(self, document, question):
                return 'yes', {'raw': 'Yes'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'questions.json'
            generator = Generator()
            result = generate(generator, 'A document about several cohorts.', 30, path)
            self.assertEqual(len(result['queries']), 30)
            self.assertEqual(len({q['probe'] for q in result['queries']}), 30)
            self.assertEqual(generator.calls, 4)
            self.assertEqual(generate(generator, 'A document about several cohorts.', 30, path), result)
            # The previous 24-attempt cap used to make these partials impossible
            # to resume. Preserve the accepted references and append missing ones.
            resumed_path = Path(folder) / 'resumed.json'
            partial = {**result, 'queries': result['queries'][:27], 'attempts': 24}
            write_json(resumed_path.with_suffix('.v3.partial.json'), partial)
            resumed = generate(generator, 'A document about several cohorts.', 30, resumed_path)
            self.assertEqual(len(resumed['queries']), 30)
            self.assertEqual(resumed['queries'][:27], result['queries'][:27])
            self.assertEqual(resumed['attempts'], 25)

    def test_dcmi_exclusions_preserve_coverage_and_freeze_calibration_before_test(self):
        import re
        from reproductions import common
        class UnscorableSample(RuntimeError):
            pass
        samples = [{'id': f'{split}_{label}_{i}', 'split': split, 'label': label}
                   for split in ('calibration', 'test') for label in (0, 1) for i in range(2)]
        excluded_ids = {'calibration_1_1', 'test_0_0'}
        corpus = {r['id']: {'text': r['id'] + ' high levels'} for r in samples}
        evaluated = []
        class Retriever:
            def __init__(self, *args):
                pass
            def search(self, *args):
                return []
        def edits(generator, document, output):
            if document.split()[0] in excluded_ids:
                raise UnscorableSample('No contextual antonym')
            return {'edits': [{'position': 1, 'original': 'high', 'replacement': 'low'}]}
        def evaluate(records, output, method, *args, **kwargs):
            evaluated.append((method, records))
            return {'status': 'completed'}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            testcase = self
            class Target:
                def __init__(self, *args, **kwargs):
                    pass
                def query(self, question, contexts):
                    if 'test_' in question:
                        testcase.assertTrue((output / 'selected_calibration.json').exists())
                    return {'probability_yes': .9 if ' high ' in question else .1, 'answer': 'yes'}
            scope = {k: getattr(common, k) for k in ('read_json', 'write_json', 'read_records', 'append_record', 'text_hash')}
            scope.update(Path=Path, time=__import__('time'), seed_all=lambda s: None,
                         sample_seed=lambda seed, identifier: seed, model_fingerprint=lambda path: {},
                         prepare=lambda config, path: (corpus, {'member_ids': [], 'records': samples}),
                         Retriever=Retriever, LocalGenerator=lambda path: None, TargetRAGGenerator=Target,
                         antonym_edits=edits, UnscorableSample=UnscorableSample, evaluate=evaluate,
                         apply_edits=extracted_function('rag.py', 'apply_edits', {'re': re}))
            run = extracted_function('rag.py', 'run_rag', scope)
            result = run({'method': 'dcmi', 'seed': 42, 'auxiliary_model': 'aux', 'target_model': 'target',
                          'retriever': 'bm25', 'max_input_tokens': 2048, 'retrieve_k': 4,
                          'calibration_per_class': 2, 'theta_grid': [.04], 'gamma_grid': [.1, .5],
                          'blackbox_theta': .03}, output)
            self.assertEqual(result['status'], 'completed_with_exclusions')
            self.assertEqual(result['coverage']['configured'], 8)
            self.assertEqual(result['coverage']['scored'], 6)
            self.assertEqual(result['coverage']['unscorable'], 2)
            self.assertEqual({r['id'] for r in common.read_records(output / 'unavailable_samples.jsonl')}, excluded_ids)
            self.assertTrue(all(r['id'] not in excluded_ids for _, records in evaluated for r in records))


if __name__ == '__main__':
    unittest.main()
