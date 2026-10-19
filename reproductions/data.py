"""Deterministic public AG News splits, with explicit fine-tuning membership."""
from pathlib import Path
import random
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from .common import text_hash, read_json, write_json, append_record


def prepare(config, output):
    output = Path(output)
    if (output / 'data_manifest.json').exists():
        return read_json(output / 'data_manifest.json')
    files = sorted((Path(config['assets']) / 'ag_news' / 'data').glob('train*.parquet'))
    tests = sorted((Path(config['assets']) / 'ag_news' / 'data').glob('test*.parquet'))
    if not files or not tests:
        raise ValueError('Pinned AG News parquet assets not prepared')
    train_texts = [t for f in files for t in pq.read_table(f, columns=['text']).column('text').to_pylist()]
    valid_texts = [t for f in tests for t in pq.read_table(f, columns=['text']).column('text').to_pylist()]
    rng = random.Random(config['seed'])
    kind = config['method']
    if kind == 'neighbour':
        order = list(range(len(train_texts)))
        rng.shuffle(order)
        members = [train_texts[i] for i in order[:config['train_samples']]]
        member_hashes = {text_hash(t) for t in members}
        nonmembers = [train_texts[i] for i in order[config['train_samples']:] if text_hash(train_texts[i]) not in member_hashes]
        valid = [t for t in valid_texts if text_hash(t) not in member_hashes][:1000]
        train = members
        prompts = []
    elif kind == 'spv':
        tokenizer = AutoTokenizer.from_pretrained(config['base_model'], local_files_only=True)

        def pack(texts):
            blocks = []
            # Match the upstream Dataset.map default batch size and tail dropping.
            for start in range(0, len(texts), 1000):
                parts = tokenizer(texts[start:start + 1000], truncation=False)['input_ids']
                ids = [token for part in parts for token in part]
                for i in range(0, len(ids) - config['block_size'] + 1, config['block_size']):
                    block = ids[i:i + config['block_size']]
                    text = tokenizer.decode(block)
                    if len(tokenizer.encode(text)) == config['block_size']:
                        blocks.append(text)
            return blocks
        boundary = int(len(train_texts) * .9)
        packed = pack(train_texts[:boundary])
        heldout = pack(train_texts[boundary:])
        train = packed[:config['train_samples']]
        members = train
        hashes = {text_hash(t) for t in train}
        available_nonmembers = [t for t in heldout if text_hash(t) not in hashes]
        valid = available_nonmembers[:config.get('validation_samples', 1000)]
        validation_hashes = {text_hash(t) for t in valid}
        nonmembers = [t for t in available_nonmembers if text_hash(t) not in validation_hashes]
        prompts = packed[config['train_samples']:config['train_samples'] + config['reference_samples']]
    else:
        raise ValueError(kind)
    required = config['calibration_per_class'] + config['test_per_class']

    def unique(texts):
        seen = set()
        result = []
        for text in texts:
            key = text_hash(text)
            if key not in seen:
                seen.add(key)
                result.append(text)
        return result
    mem_eval, non_eval = unique(members), unique(nonmembers)
    rng.shuffle(mem_eval)
    rng.shuffle(non_eval)
    if len(train) != config['train_samples'] or min(len(mem_eval), len(non_eval)) < required:
        raise ValueError('Insufficient unique train/evaluation records; no silent downsizing')
    if kind == 'spv' and len(prompts) != config['reference_samples']:
        raise ValueError('Insufficient non-target self-prompt blocks')
    output.mkdir(parents=True, exist_ok=True)
    for name, texts in (('train', train), ('validation', valid), ('prompts', prompts)):
        write_json(output / f'{name}.json', texts)
    eval_path = output / 'evaluation.jsonl'
    if eval_path.exists():
        raise ValueError('Incomplete prior data preparation; use a fresh output directory')
    for label, texts in ((1, mem_eval), (0, non_eval)):
        for i, text in enumerate(texts[:required]):
            append_record(eval_path, {'id': text_hash(text), 'text': text, 'label': label,
                                     'split': 'calibration' if i < config['calibration_per_class'] else 'test'})
    manifest = {'method': kind, 'dataset': 'fancyzhx/ag_news', 'seed': config['seed'],
                'membership_definition': 'included_in_this_target_finetuning_not_claimed_pretraining_membership',
                'train_count': len(train), 'train_text_hashes': [text_hash(t) for t in train],
                'nonmember_count_available': len(non_eval), 'evaluation_per_class': required,
                'validation_count': len(valid), 'prompt_count': len(prompts),
                'train_nonmember_exact_overlap': 0,
                'packing': '128-token blocks; batch=1000; drop tails' if kind == 'spv' else None,
                'self_prompt_source': 'held-out same-domain AG News blocks as upstream script' if kind == 'spv' else None}
    write_json(output / 'data_manifest.json', manifest)
    return manifest
