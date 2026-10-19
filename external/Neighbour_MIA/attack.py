"""Neighbour_MIA ?????????

???????????????????????????????????
1. ?? GPT-2 ????????????? log probability?
2. ?? BERT MLM ? one-word replacement??? neighbourhood?
3. ? original_logprob - mean(neighbour_logprob) ?????????
4. ?? pickle?? unified_mia_framework ?????

?????? attack.py ?? <path_to_attack_model_*> ?????cuda:0/cuda:1 ???
one_word_neighbours ????????????????????????????????
"""

import argparse
import json
import os
import pickle
import random
import re
import time
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer


def parse_args() -> argparse.Namespace:
    """???????????????? --proc-id/--model/--dataset ???"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc-id', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'], default='bert')
    parser.add_argument('--dataset', type=str, choices=['news', 'twitter', 'wiki', 'sample'], default='wiki')
    parser.add_argument('--attack-model-path', type=str, default='')
    parser.add_argument('--search-model-path', type=str, default='/root/autodl-tmp/models/bert-base-uncased')
    parser.add_argument('--members-file', type=str, default='')
    parser.add_argument('--nonmembers-file', type=str, default='')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--max-neighbours', type=int, default=5)
    parser.add_argument('--mlm-top-k', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-file', type=str, default='')
    parser.add_argument('--analysis-file', type=str, default='')
    parser.add_argument('--perturbation-modes', type=str, default='mlm,synonym,phrase,deletion,sentence')
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    """?????????????? CPU ???GPU ??????? GPU?"""
    if name and name != 'auto':
        return torch.device(name)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_jsonl_texts(path: str, label: int, max_count: int) -> List[Dict[str, object]]:
    """?? jsonl ? txt ?????? member/non-member ???"""
    items: List[Dict[str, object]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if path.endswith('.jsonl'):
                obj = json.loads(line)
                text = obj.get('text') if isinstance(obj, dict) else str(obj)
            else:
                text = line
            text = str(text).replace('\\', ' ').strip()
            if text:
                items.append({'text': text, 'label': int(label)})
            if len(items) >= max_count:
                break
    return items


def clean_decoded(text: str) -> str:
    """?? MLM ??????????????????? GPT-2?"""
    return (
        text.replace('[CLS]', '')
        .replace('[SEP]', '')
        .replace('[PAD]', '')
        .replace('  ', ' ')
        .strip()
    )


def one_word_neighbours(
    text: str,
    search_tokenizer: BertTokenizer,
    search_model: BertForMaskedLM,
    device: torch.device,
    max_neighbours: int,
    mlm_top_k: int,
    max_length: int,
) -> List[Tuple[str, float]]:
    """?? BERT MLM ? one-word replacement ??????

    ??????????????????????????????????????
    ????????? token???? MLM ?????????????????
    """
    encoded = search_tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    special_ids = set(search_tokenizer.all_special_ids)
    neighbours: List[Tuple[str, float]] = []
    seen = set()

    with torch.no_grad():
        for token_index in range(1, input_ids.shape[1] - 1):
            original_id = int(input_ids[0, token_index].item())
            original_token = search_tokenizer.convert_ids_to_tokens(original_id)
            # ???? token ? BERT ?????????? one-word replacement ????
            if original_id in special_ids or str(original_token).startswith('##'):
                continue

            masked_ids = input_ids.clone()
            masked_ids[0, token_index] = search_tokenizer.mask_token_id
            outputs = search_model(input_ids=masked_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits[0, token_index], dim=-1)
            top_probs, top_ids = torch.topk(probs, k=mlm_top_k)

            for prob, candidate_id in zip(top_probs.tolist(), top_ids.tolist()):
                candidate_id = int(candidate_id)
                candidate_token = search_tokenizer.convert_ids_to_tokens(candidate_id)
                if candidate_id == original_id or candidate_id in special_ids:
                    continue
                if str(candidate_token).startswith('##'):
                    continue

                alt_ids = input_ids.clone()
                alt_ids[0, token_index] = candidate_id
                alt_text = clean_decoded(search_tokenizer.decode(alt_ids[0], skip_special_tokens=True))
                if alt_text and alt_text != text and alt_text not in seen:
                    seen.add(alt_text)
                    neighbours.append((alt_text, float(prob)))
                if len(neighbours) >= max_neighbours:
                    return neighbours
    return neighbours



SYNONYM_MAP = {
    'study': ['research', 'analysis'],
    'patient': ['individual', 'person'],
    'patients': ['individuals', 'people'],
    'disease': ['condition', 'illness'],
    'treatment': ['therapy', 'care'],
    'risk': ['likelihood', 'chance'],
    'benefit': ['advantage', 'positive effect'],
    'health': ['wellbeing', 'medical health'],
    'vitamin': ['nutrient', 'supplement'],
    'therapy': ['treatment', 'care approach'],
}

PHRASE_MAP = {
    'cardiovascular disease': 'heart-related condition',
    'vitamin a deficiency': 'nutritional deficiency',
    'reduced risk': 'lower likelihood',
    'regular consumption': 'consistent use',
    'clinical trial': 'medical study',
    'public health': 'population health',
}


def _clean_variant(text: str) -> str:
    # ????????????? GPT-2 loss ???
    return ' '.join(str(text).replace('  ', ' ').split()).strip()


def _append_neighbour(neighbours: List[Tuple[str, float]], seen: set, text: str, weight: float) -> None:
    # ??????? neighbour?
    text = _clean_variant(text)
    key = text.lower()
    if text and key not in seen:
        seen.add(key)
        neighbours.append((text, float(weight)))


def augment_neighbours(text: str, base_neighbours: List[Tuple[str, float]], max_neighbours: int, modes: str) -> List[Tuple[str, float]]:
    # ?? neighbour generation?MLM + synonym + phrase + deletion + sentence perturbation?
    selected_modes = {m.strip().lower() for m in modes.split(',') if m.strip()}
    neighbours: List[Tuple[str, float]] = []
    seen = {text.lower()}
    for candidate, weight in base_neighbours:
        _append_neighbour(neighbours, seen, candidate, weight)
        if len(neighbours) >= max_neighbours:
            return neighbours[:max_neighbours]

    words = text.split()
    lowered = [w.strip('.,;:!?()[]{}"\'').lower() for w in words]

    if 'synonym' in selected_modes:
        for i, token in enumerate(lowered):
            if token in SYNONYM_MAP:
                for syn in SYNONYM_MAP[token]:
                    out = words[:]
                    out[i] = syn
                    _append_neighbour(neighbours, seen, ' '.join(out), 0.55)
                    if len(neighbours) >= max_neighbours:
                        return neighbours[:max_neighbours]

    if 'phrase' in selected_modes:
        lower_text = text.lower()
        for phrase, repl in PHRASE_MAP.items():
            if phrase in lower_text:
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                _append_neighbour(neighbours, seen, pattern.sub(repl, text, count=1), 0.50)
                if len(neighbours) >= max_neighbours:
                    return neighbours[:max_neighbours]

    if 'deletion' in selected_modes and len(words) > 8:
        step = max(1, len(words) // 8)
        for i in range(1, len(words) - 1, step):
            if len(words[i]) <= 3:
                continue
            out = words[:i] + words[i + 1:]
            _append_neighbour(neighbours, seen, ' '.join(out), 0.35)
            if len(neighbours) >= max_neighbours:
                return neighbours[:max_neighbours]

    if 'sentence' in selected_modes:
        templates = [
            'In summary, ' + text,
            text + ' This statement is described in general terms.',
            re.sub(r'\b(is|are|was|were)\b', 'may be', text, count=1, flags=re.IGNORECASE),
        ]
        for variant in templates:
            _append_neighbour(neighbours, seen, variant, 0.30)
            if len(neighbours) >= max_neighbours:
                return neighbours[:max_neighbours]

    return neighbours[:max_neighbours]


def build_detailed_analysis(details: List[Dict[str, object]], scores: List[float], labels: List[int]) -> Dict[str, object]:
    # ?????? Neighbour_MIA ?????
    member_details = [d for d in details if int(d.get('label', 0)) == 1]
    nonmember_details = [d for d in details if int(d.get('label', 0)) == 0]
    member_scores = [float(d['membership_score']) for d in member_details]
    nonmember_scores = [float(d['membership_score']) for d in nonmember_details]
    member_losses = [float(d['original_loss']) for d in member_details]
    nonmember_losses = [float(d['original_loss']) for d in nonmember_details]
    variances = []
    for d in details:
        vals = [float(v) for v in d.get('neighbour_losses', [])]
        if vals:
            mean = sum(vals) / len(vals)
            variances.append(sum((v - mean) ** 2 for v in vals) / len(vals))
    thresholds = sorted(set(float(s) for s in scores))
    sweep = []
    roc_points = []
    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        acc = (tp + tn) / max(1, len(labels))
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        sweep.append({'threshold': thr, 'accuracy': acc, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})
        roc_points.append({'threshold': thr, 'tpr': tpr, 'fpr': fpr})
    def mean(vals): return float(sum(vals) / len(vals)) if vals else 0.0
    return {
        'num_members': len(member_details),
        'num_nonmembers': len(nonmember_details),
        'member_mean_loss': mean(member_losses),
        'nonmember_mean_loss': mean(nonmember_losses),
        'member_mean_score': mean(member_scores),
        'nonmember_mean_score': mean(nonmember_scores),
        'neighbour_variance_mean': mean(variances),
        'score_distribution': {'member': member_scores, 'nonmember': nonmember_scores},
        'threshold_sweep': sweep,
        'roc_points': roc_points,
    }

def gpt2_logprob(text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, device: torch.device, max_length: int) -> float:
    """?? GPT-2 ?????? log probability????????????????"""
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
    return float(-loss.item())


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.dataset == 'sample':
        args.attack_model_path = args.attack_model_path or '/root/autodl-tmp/Neighbour_MIA/finetuned_gpt2'
        args.members_file = args.members_file or '/root/autodl-tmp/Neighbour_MIA/data/train_members.txt'
        args.nonmembers_file = args.nonmembers_file or '/root/autodl-tmp/Neighbour_MIA/data/test_nonmembers.txt'
    else:
        args.attack_model_path = args.attack_model_path or '/root/autodl-tmp/models/gpt2'
        args.members_file = args.members_file or '/root/autodl-tmp/Neighbour_MIA/data/sample_members.jsonl'
        args.nonmembers_file = args.nonmembers_file or '/root/autodl-tmp/Neighbour_MIA/data/sample_nonmembers.jsonl'

    print(f'[Neighbour_MIA] device={device}')
    print(f'[Neighbour_MIA] target_model={args.attack_model_path}')
    print(f'[Neighbour_MIA] search_model={args.search_model_path}')

    attack_tokenizer = GPT2Tokenizer.from_pretrained(args.attack_model_path)
    attack_tokenizer.pad_token = attack_tokenizer.eos_token
    attack_model = GPT2LMHeadModel.from_pretrained(args.attack_model_path).to(device)
    attack_model.eval()

    # ??????????? bert??? --model ???????????????
    if args.model != 'bert':
        print(f'[Neighbour_MIA] ?????????? BERT MLM??? model={args.model}??? bert ???')
    search_tokenizer = BertTokenizer.from_pretrained(args.search_model_path)
    search_model = BertForMaskedLM.from_pretrained(args.search_model_path).to(device)
    search_model.eval()

    per_class = max(1, args.max_samples // 2)
    samples = load_jsonl_texts(args.members_file, label=1, max_count=per_class)
    samples += load_jsonl_texts(args.nonmembers_file, label=0, max_count=args.max_samples - len(samples))
    print(f'[Neighbour_MIA] loaded_samples={len(samples)} members={sum(s["label"] for s in samples)}')

    scores: List[float] = []
    labels: List[int] = []
    predictions: List[int] = []
    details: List[Dict[str, object]] = []
    raw_score_dicts: List[Dict[str, object]] = []

    for idx, sample in enumerate(tqdm(samples, desc='neighbour-mia')):
        text = str(sample['text'])
        label = int(sample['label'])
        start = time.time()
        original_logprob = gpt2_logprob(text, attack_tokenizer, attack_model, device, args.max_length)
        neighbours = one_word_neighbours(
            text,
            search_tokenizer,
            search_model,
            device,
            args.max_neighbours,
            args.mlm_top_k,
            args.max_length,
        )
        neighbours = augment_neighbours(text, neighbours, args.max_neighbours, args.perturbation_modes)
        neighbour_logprobs = [
            gpt2_logprob(neighbour_text, attack_tokenizer, attack_model, device, args.max_length)
            for neighbour_text, _ in neighbours
        ]
        if neighbour_logprobs:
            attack_score = original_logprob - (sum(neighbour_logprobs) / len(neighbour_logprobs))
        else:
            attack_score = original_logprob

        scores.append(float(attack_score))
        labels.append(label)
        detail = {
            'index': idx,
            'label': label,
            'original_text': text,
            'generated_neighbours': neighbours,
            'original_logprob': float(original_logprob),
            'original_loss': float(-original_logprob),
            'neighbour_logprobs': [float(v) for v in neighbour_logprobs],
            'neighbour_losses': [float(-v) for v in neighbour_logprobs],
            'membership_score': float(attack_score),
            'elapsed_seconds': float(time.time() - start),
        }
        details.append(detail)
        raw_score_dicts.append({
            f'<original_text>: {text}': float(original_logprob),
            **{neighbour_text: float(score) for (neighbour_text, _), score in zip(neighbours, neighbour_logprobs)},
        })

        print('\n[Sample {}] label={}'.format(idx, label))
        print('original text:', text)
        print('generated neighbours:', neighbours)
        print('original loss:', -original_logprob)
        print('neighbour losses:', [-v for v in neighbour_logprobs])
        print('membership score:', attack_score)

    if scores:
        sorted_scores = sorted(scores)
        threshold = sorted_scores[len(sorted_scores) // 2]
        predictions = [1 if score >= threshold else 0 for score in scores]
    print('[Neighbour_MIA] predictions:', predictions)
    print('[Neighbour_MIA] labels:', labels)

    detailed_analysis = build_detailed_analysis(details, scores, labels)
    if args.analysis_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.analysis_file)), exist_ok=True)
        with open(args.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analysis, f, ensure_ascii=False, indent=2)
        print('[Neighbour_MIA] wrote detailed analysis:', args.analysis_file)

    output_file = args.output_file or os.path.join(os.getcwd(), f'all_scores_{args.dataset}_{args.model}_{args.proc_id}.pkl')
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    payload = {
        'scores': scores,
        'labels': labels,
        'predictions': predictions,
        'details': details,
        'raw_score_dicts': raw_score_dicts,
        'target_model_path': args.attack_model_path,
        'search_model_path': args.search_model_path,
        'detailed_analysis': detailed_analysis,
        'perturbation_modes': args.perturbation_modes,
    }
    with open(output_file, 'wb') as f:
        pickle.dump(payload, f)
    # ???????????????????? all_scores*.pkl?
    default_output = os.path.join(os.getcwd(), f'all_scores_{args.dataset}_{args.model}_{args.proc_id}.pkl')
    if os.path.abspath(default_output) != os.path.abspath(output_file):
        with open(default_output, 'wb') as f:
            pickle.dump(payload, f)
    print('[Neighbour_MIA] wrote pickle:', output_file)
    print('[Neighbour_MIA] wrote compatibility pickle:', default_output)


if __name__ == '__main__':
    main()
