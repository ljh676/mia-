"""SPV-MIA ?? membership scoring ???

?? SPV-MIA ?? self-prompt reference model ? probabilistic variation?
???????????????? privacy evaluation??????????
1. ???? GPT-2 target model?
2. ?? member/non-member JSONL?
3. ?????? deterministic paraphrase / perturbation?
4. ?? original loss ? perturbed loss?
5. ?? perturbed_loss - original_loss ?? probabilistic variation membership score?

??????????????????????? self-prompt reference pipeline?
"""

import argparse
import json
import os
import re
from typing import Dict, List

import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_yaml(path: str) -> Dict[str, object]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def light_paraphrase(text: str) -> str:
    """??????????? probabilistic variation scoring?"""
    replacements = {
        r"\bstudy\b": "analysis",
        r"\bresearch\b": "investigation",
        r"\bshows\b": "indicates",
        r"\bsuggests\b": "indicates",
        r"\bimportant\b": "notable",
        r"\bcommon\b": "frequent",
        r"\bpatients\b": "individuals",
        r"\bdisease\b": "condition",
        r"\brisk\b": "likelihood",
        r"\beffect\b": "impact",
    }
    value = text
    for pattern, repl in replacements.items():
        value = re.sub(pattern, repl, value, flags=re.IGNORECASE)
    words = value.split()
    if len(words) > 18:
        # ???????????????? paraphrase-like ???
        i = min(6, len(words) - 3)
        words[i], words[i + 1] = words[i + 1], words[i]
        value = " ".join(words)
    return value


def sequence_loss(text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, device: torch.device, max_length: int) -> float:
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    encoded = {k: v.to(device) for k, v in encoded.items()}
    labels = encoded['input_ids']
    with torch.no_grad():
        loss = model(**encoded, labels=labels).loss
    return float(loss.item())


def compute_metrics(labels: List[int], predictions: List[int], scores: List[float]) -> Dict[str, float]:
    return {
        'accuracy': float(accuracy_score(labels, predictions)) if labels else 0.0,
        'auc': float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0,
        'precision': float(precision_score(labels, predictions, zero_division=0)) if labels else 0.0,
        'recall': float(recall_score(labels, predictions, zero_division=0)) if labels else 0.0,
        'f1': float(f1_score(labels, predictions, zero_division=0)) if labels else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Minimal SPV-MIA privacy evaluation benchmark')
    parser.add_argument('--config', default='/root/autodl-tmp/SPV_MIA/configs/spv_minimal.yml')
    parser.add_argument('--method', default=None, choices=['probabilistic_variation'])
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--members-file', default=None)
    parser.add_argument('--nonmembers-file', default=None)
    parser.add_argument('--output-path', default=None)
    parser.add_argument('--max-length', type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_path = args.model_path or cfg.get('model_path', '/root/autodl-tmp/models/gpt2')
    members_file = args.members_file or cfg.get('members_file', '/root/autodl-tmp/SPV_MIA/data/sample_members.jsonl')
    nonmembers_file = args.nonmembers_file or cfg.get('nonmembers_file', '/root/autodl-tmp/SPV_MIA/data/sample_nonmembers.jsonl')
    output_path = args.output_path or cfg.get('output_path', '/root/autodl-tmp/SPV_MIA/outputs/spv_mia_result.json')
    max_length = int(args.max_length or cfg.get('max_length', 128))
    method = args.method or cfg.get('method', 'probabilistic_variation')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    rows = load_jsonl(members_file) + load_jsonl(nonmembers_file)
    scores: List[float] = []
    labels: List[int] = []
    details: List[Dict[str, object]] = []

    for idx, row in enumerate(rows):
        text = str(row['text'])
        label = int(row.get('label', 0))
        perturbed = light_paraphrase(text)
        original_loss = sequence_loss(text, tokenizer, model, device, max_length)
        perturbed_loss = sequence_loss(perturbed, tokenizer, model, device, max_length)
        score = perturbed_loss - original_loss
        labels.append(label)
        scores.append(float(score))
        details.append({
            'index': idx,
            'label': label,
            'text': text,
            'perturbed_text': perturbed,
            'original_loss': original_loss,
            'perturbed_loss': perturbed_loss,
            'probabilistic_variation_score': float(score),
        })
        print(f'[SPV-MIA minimal] index={idx} label={label} original_loss={original_loss:.4f} perturbed_loss={perturbed_loss:.4f} score={score:.4f}')

    threshold = sorted(scores)[len(scores)//2] if scores else 0.0
    predictions = [1 if score >= threshold else 0 for score in scores]
    metrics = compute_metrics(labels, predictions, scores)
    result = {
        'attack_name': 'spv_mia',
        'benchmark_name': 'spv_mia',
        'method': method,
        'scores': scores,
        'predictions': predictions,
        'labels': labels,
        'metrics': metrics,
        'threshold': float(threshold),
        'details': details,
        'model_path': model_path,
        'returncode': 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[SPV-MIA minimal] wrote result: {output_path}')
    print(json.dumps({'metrics': metrics, 'threshold': threshold, 'num_samples': len(labels)}, indent=2))


if __name__ == '__main__':
    main()
