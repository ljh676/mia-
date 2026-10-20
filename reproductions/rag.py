"""End-to-end IA and DCMI with real retrieval, target generation and saved evidence.

API-generating models are explicitly substituted with a local instruction model.
The target RAG signal is always obtained from the generator, never retrieval hits.
"""
from pathlib import Path
import difflib
import re
import time
from .common import (read_json, write_json, read_records, append_record, text_hash,
                     seed_all, sample_seed, evaluate, model_fingerprint)
from .generation import LocalGenerator, TargetRAGGenerator
from .rag_data import prepare, Retriever
from .query_generation import generate_ia_queries, antonym_edits, UnscorableSample


def ia_score(expected, responses, penalty=5):
    if not expected or len(expected) != len(responses) or any(g not in {'yes', 'no'} for g in expected):
        raise ValueError('IA requires aligned non-unknown reference answers')
    return sum(int(answer == gold) - penalty * int(answer == 'UNK')
               for gold, answer in zip(expected, responses)) / len(expected)


def sparse_replacements(document, proposal):
    pattern = r"\b\w+(?:[-']\w+)*\b"
    original = [m.group() for m in re.finditer(pattern, document)]
    changed = [m.group() for m in re.finditer(pattern, proposal)]
    stop = set('a an the of to in on at for by with from as and or but if then is are was were be been '
               'being has have had do does did this that these those it its they their we our you your '
               'he she his her which who what how when where not no yes'.split())
    edits = []
    for tag, start, end, other_start, other_end in difflib.SequenceMatcher(a=original, b=changed, autojunk=False).get_opcodes():
        if tag != 'replace' or end - start != other_end - other_start:
            continue  # Never apply deletions, insertions or paragraph-level paraphrases.
        for offset, (old, new) in enumerate(zip(original[start:end], changed[other_start:other_end])):
            left, right = old.lower(), new.lower()
            stem = lambda word: word[:-3] + 'y' if word.endswith('ies') else word.rstrip('s')
            if (left == right or left in stop or right in stop or old.isupper()
                    or not old.replace('-', '').isalpha() or not new.replace('-', '').isalpha()
                    or stem(left) == stem(right)):
                continue
            edits.append({'position': start + offset, 'original': old, 'replacement': new})
    return edits


def apply_edits(document, edit_record, theta):
    words = list(re.finditer(r"\b\w+(?:[-']\w+)*\b", document))
    desired = max(1, int(theta * len(words)))
    selected = edit_record['edits'][:desired]
    result = document
    for item in sorted(selected, key=lambda r: r['position'], reverse=True):
        if not 0 <= item['position'] < len(words):
            raise ValueError('Perturbation position outside document')
        span = words[item['position']]
        if span.group() != item['original']:
            raise ValueError('Perturbation original word does not match document')
        result = result[:span.start()] + item['replacement'] + result[span.end():]
    if result == document:
        raise ValueError('DCMI perturbation did not change the sample')
    return result, {'theta_requested': theta, 'words_requested': desired, 'words_replaced': len(selected),
                    'fraction_replaced': len(selected) / len(words), 'available_edits': len(edit_record['edits']),
                    'requested_budget_met': len(selected) == desired}


def run_rag(config, output):
    output = Path(output)
    seed_all(config['seed'])
    corpus, datastore = prepare(config, output / 'data')
    retriever = Retriever(config, corpus, datastore['member_ids'], output / 'index')
    generator = LocalGenerator(config['auxiliary_model'])
    target = TargetRAGGenerator(config['target_model'], config['max_input_tokens'], style=config['method'])
    write_json(output / 'models.json', {'target': model_fingerprint(config['target_model']),
                'auxiliary': model_fingerprint(config['auxiliary_model']),
                'retrieval': model_fingerprint(config['retrieval_model']) if config['retriever'] == 'dense' else None,
                'api_substitution': True, 'user_authorized_local_generation': True})
    path = output / 'records.jsonl'
    records = read_records(path)
    completed = {r['id'] for r in records}
    unavailable_path = output / 'unavailable_samples.jsonl'
    unavailable = read_records(unavailable_path)
    unavailable_ids = {r['id'] for r in unavailable}
    query_path = output / 'query_records.jsonl'
    query_records = read_records(query_path)
    queried = {(r['sample_id'], r['query_id']): r for r in query_records}

    def query(identifier, query_id, question):
        key = (identifier, query_id)
        if key in queried:
            if queried[key]['question'] != question:
                raise ValueError('Query identity changed on resume')
            return queried[key]
        rewritten = question
        rewrite_evidence = None
        if config.get('query_rewrite'):
            prompt = 'Paraphrase this user question without changing its meaning or requested answer format. Return only the rewritten question:\n' + question
            rewritten = generator.generate(prompt, 256)
            rewrite_evidence = {'prompt': prompt, 'raw': rewritten}
        retrieved = retriever.search(rewritten, config['retrieve_k'])
        signal = target.query(rewritten, [r['text'] for r in retrieved])
        result = {'sample_id': identifier, 'query_id': query_id, 'question': question, 'rewritten_question': rewritten,
                  'rewrite_evidence': rewrite_evidence, 'retrieved': retrieved, **signal}
        append_record(query_path, result)
        queried[key] = result
        return result

    def select_calibration(calibration):
        options = []
        for theta in config['theta_grid']:
            for gamma in config['gamma_grid']:
                correct = sum((r['variants'][f'{theta:.2f}']['score'] > gamma) == r['label'] for r in calibration)
                options.append((correct, -theta, gamma))
        correct, negative_theta, gamma = max(options)
        choice = {'theta': -negative_theta, 'threshold': gamma,
                  'calibration_accuracy': correct / len(calibration), 'test_labels_used': False}
        write_json(output / 'selected_calibration.json', choice)
        return choice

    selected = read_json(output / 'selected_calibration.json') if (output / 'selected_calibration.json').exists() else None
    calibration = [r for r in records if r['split'] == 'calibration']
    if config['method'] == 'dcmi' and selected is None and len(calibration) == 2 * config['calibration_per_class']:
        selected = select_calibration(calibration)
    all_samples = sorted(datastore['records'], key=lambda r: 0 if r['split'] == 'calibration' else 1)
    # Every configured document is accounted for, including unscorable inputs.
    # Exclusions are never assigned artificial scores or silently resampled.
    for index, sample in enumerate(all_samples):
        if sample['id'] in completed or sample['id'] in unavailable_ids:
            continue
        if config['method'] == 'dcmi' and sample['split'] == 'test' and selected is None:
            calibration = [r for r in records if r['split'] == 'calibration']
            if {r['label'] for r in calibration} != {0, 1}:
                raise RuntimeError('Insufficient scorable calibration classes; test queries not started')
            selected = select_calibration(calibration)
        identifier = sample['id']
        seed_all(sample_seed(config['seed'], identifier))
        document = corpus[identifier]['text']
        cache_name = text_hash(identifier)
        try:
            if config['method'] == 'ia':
                generated = generate_ia_queries(generator, document, config['questions_per_document'],
                                                 output / 'generated_queries' / f'{cache_name}.json',
                                                 max_attempts=config.get('max_question_generation_attempts', 72))
            else:
                edits = antonym_edits(generator, document, output / 'perturbations' / f'{cache_name}.json')
        except UnscorableSample as error:
            item = {**sample, 'reason': str(error), 'document_sha256': text_hash(document),
                    'status': 'unscorable_no_synthetic_score', 'time': time.time()}
            append_record(unavailable_path, item)
            unavailable.append(item)
            unavailable_ids.add(identifier)
            write_json(output / 'progress.json', {'stage': config['method'],
                'done': len(records), 'unscorable': len(unavailable),
                'processed': len(completed) + len(unavailable_ids), 'total': len(all_samples),
                'updated_at': time.time()})
            print(f'{config["method"]} unscorable {identifier}: {error}', flush=True)
            continue
        if config['method'] == 'ia':
            responses = [query(identifier, str(i), item['query']) for i, item in enumerate(generated['queries'])]
            gold = [item['reference_answer'] for item in generated['queries']]
            answers = [r['answer'] for r in responses]
            result = {**sample, 'reference_answers': gold, 'target_answers': answers,
                      'score': ia_score(gold, answers, config['unknown_penalty']),
                      'questions': len(gold), 'unknown_responses': answers.count('UNK'),
                      'target_document_retrieved_count': sum(any(d['id'] == identifier for d in r['retrieved']) for r in responses)}
        else:
            original_query = f'Is this: "{document}" right? Answer with Yes or No.'
            original = query(identifier, 'original', original_query)
            thetas = config['theta_grid'] if sample['split'] == 'calibration' else [selected['theta']]
            thetas = sorted(set(thetas + [config['blackbox_theta']]))
            variants = {}
            for theta in thetas:
                perturbed, perturb_meta = apply_edits(document, edits, theta)
                perturbed_query = f'Is this: "{perturbed}" right? Answer with Yes or No.'
                response = query(identifier, f'theta_{theta:.2f}', perturbed_query)
                variants[f'{theta:.2f}'] = {'score': original['probability_yes'] - response['probability_yes'],
                        'original_probability': original['probability_yes'], 'perturbed_probability': response['probability_yes'],
                        'original_answer': original['answer'], 'perturbed_answer': response['answer'],
                        'blackbox_score': (int(original['answer'] == 'yes') - int(response['answer'] == 'yes')
                                          if original['answer'] != 'UNK' and response['answer'] != 'UNK' else None),
                        'perturbation': perturb_meta}
            result = {**sample, 'variants': variants,
                      'score': variants[f'{selected["theta"]:.2f}']['score'] if selected else None,
                      'uncalibrated_probability_score': original['probability_yes']}
        append_record(path, result)
        records.append(result)
        completed.add(identifier)
        # Select DCMI parameters only after all reference samples, before any test queries.
        if config['method'] == 'dcmi' and selected is None:
            calibration = [r for r in records if r['split'] == 'calibration']
            if len(calibration) == 2 * config['calibration_per_class']:
                selected = select_calibration(calibration)
        write_json(output / 'progress.json', {'stage': config['method'], 'done': len(records),
                    'unscorable': len(unavailable), 'processed': len(completed) + len(unavailable_ids),
                    'total': len(all_samples), 'updated_at': time.time(), 'selected_calibration': selected})
        print(f'{config["method"]} {len(records)}/{len(all_samples)}', flush=True)
    coverage = {'configured': len(all_samples), 'scored': len(records), 'unscorable': len(unavailable),
        'by_split_and_label': {f'{split}_{label}': {
            'configured': sum(r['split'] == split and r['label'] == label for r in all_samples),
            'scored': sum(r['split'] == split and r['label'] == label for r in records),
            'unscorable': sum(r['split'] == split and r['label'] == label for r in unavailable)}
            for split in ('calibration', 'test') for label in (0, 1)},
        'scope': 'conditional_on_scorable_documents' if unavailable else 'all_configured_documents'}
    write_json(output / 'coverage.json', coverage)
    absent = [split for split in ('calibration', 'test')
              if {r['label'] for r in records if r['split'] == split} != {0, 1}]
    if absent:
        raise RuntimeError(f'{config["method"]}: only {len(records)}/{len(all_samples)} documents scorable; '
                           f'missing membership class in {absent}. See coverage.json and unavailable_samples.jsonl; '
                           'unscorable documents were not assigned scores.')
    if config['method'] == 'ia':
        summary = evaluate(records, output / 'evaluation', 'ia_local_generator_substitution')
    else:
        primary = [{**r, 'score': r['variants'][f'{selected["theta"]:.2f}']['score']} for r in records]
        write_json(output / 'scored_records.json', primary)
        summary = evaluate(primary, output / 'evaluation', 'dcmi_gray_local_perturbation', selected['threshold'])
        # Adversary 2 uses only nonmember reference scores and their third quartile.
        import numpy as np
        fixed_key = f'{config["blackbox_theta"]:.2f}'
        reference_scores = [r['variants'][fixed_key]['score'] for r in records if r['split'] == 'calibration' and r['label'] == 0]
        q3 = float(np.quantile(reference_scores, .75))
        evaluate([{**r, 'score': r['variants'][fixed_key]['score']} for r in records],
                 output / 'adversary_2', 'dcmi_distribution_aware_q3', q3,
                 threshold_source='third_quartile_of_nonmember_reference_only')
        black = [{**r, 'score': r['variants'][fixed_key]['blackbox_score']} for r in records]
        if all(r['score'] is not None for r in black):
            evaluate(black, output / 'adversary_3', 'dcmi_blackbox_decision_reversal', 0.0,
                     threshold_source='paper_decision_reversal_rule_fixed_without_calibration')
        else:
            write_json(output / 'adversary_3' / 'unavailable.json', {'reason': 'some target responses were not Yes/No',
                        'invalid_count': sum(r['score'] is None for r in black), 'total': len(black)})
        evaluate([{**r, 'score': r['uncalibrated_probability_score']} for r in records],
                 output / 'uncalibrated_baseline', 'rag_probability_yes')
    summary.update({'api_generation_replaced_with_local_model': True,
                    'original_paper_models_fully_matched': False, 'actual_target_rag_outputs': True,
                    'datastore_size': len(datastore['member_ids']), 'query_count': len(queried),
                    'protocol_differences': config.get('protocol_differences', [])})
    summary['coverage'] = coverage
    if unavailable:
        summary['status'] = 'completed_with_exclusions'
        summary['evaluation_scope'] = 'conditional_on_scorable_documents_not_full_paper_protocol'
    if config['method'] == 'dcmi':
        perturbations = [variant['perturbation'] for r in records for variant in r['variants'].values()]
        summary['perturbation_budget_checks'] = {
            'queries': len(perturbations),
            'requested_budget_met': sum(p['requested_budget_met'] for p in perturbations),
            'unmet_budget_count': sum(not p['requested_budget_met'] for p in perturbations),
            'mean_actual_fraction': sum(p['fraction_replaced'] for p in perturbations) / len(perturbations)}
    # Keep the commonly read metric artifact as explicit as the run marker.
    write_json(output / 'evaluation' / 'result.json', summary)
    for name in ('adversary_2', 'adversary_3', 'uncalibrated_baseline'):
        metric_path = output / name / 'result.json'
        if metric_path.exists():
            metric = read_json(metric_path)
            metric['coverage'] = summary['coverage']
            if unavailable:
                metric.update(status='completed_with_exclusions',
                              evaluation_scope='conditional_on_scorable_documents_not_full_paper_protocol')
            write_json(metric_path, metric)
    write_json(output / 'result.json', summary)
    return summary
