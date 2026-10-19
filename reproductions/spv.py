"""SPV-MIA, paper Algorithm 1: symmetrical embedding noise and reference calibration."""
import math
from pathlib import Path
import time
import torch
from .common import (read_records, append_record, seed_all, sample_seed, write_json,
                     evaluate, model_fingerprint)
from .lm import load_causal, batch_tokens, causal_losses


def calibrated_score(target_loss, target_pair_losses, reference_loss, reference_pair_losses):
    target_variation = sum(math.exp(-v) for v in target_pair_losses) / len(target_pair_losses) - math.exp(-target_loss)
    reference_variation = sum(math.exp(-v) for v in reference_pair_losses) / len(reference_pair_losses) - math.exp(-reference_loss)
    # Paper Eq. 11 is target variation minus reference variation; more negative
    # indicates membership. Negate once to make larger scores mean member.
    return -(target_variation - reference_variation)


@torch.inference_mode()
def embedding_pairs(model, ids, mask, noises):
    embeddings = model.get_input_embeddings()(ids)
    original = causal_losses(model, ids, mask).item()
    result = []
    for noise in noises:
        paired = torch.cat([embeddings + noise, embeddings - noise], 0)
        losses = causal_losses(model, ids.expand(2, -1), mask.expand(2, -1), inputs_embeds=paired)
        result.extend(losses.cpu().tolist())
    return original, result


def attack(config, data, target, reference, output):
    output = Path(output)
    path = output / 'records.jsonl'
    existing = read_records(path)
    completed = {r['id'] for r in existing}
    model, tokenizer = load_causal(target)
    ref_model, _ = load_causal(reference)
    write_json(output / 'models.json', {'target': model_fingerprint(target),
                                      'reference': model_fingerprint(reference)})
    if model.config.hidden_size != ref_model.config.hidden_size:
        raise ValueError('Target and reference embedding dimensions must agree')
    records = read_records(Path(data) / 'evaluation.jsonl')
    for index, record in enumerate(records):
        if record['id'] in completed:
            continue
        current_seed = sample_seed(config['seed'], record['id'])
        seed_all(current_seed)
        tokens = batch_tokens(tokenizer, [record['text']], config['block_size'])
        shape = (1, tokens.input_ids.shape[1], model.config.hidden_size)
        noises = [torch.randn(shape, device='cuda', dtype=torch.float32).to(model.dtype) * config['noise_sigma']
                  for _ in range(config['perturbation_pairs'])]
        target_loss, target_pairs = embedding_pairs(model, tokens.input_ids, tokens.attention_mask, noises)
        ref_loss, ref_pairs = embedding_pairs(ref_model, tokens.input_ids, tokens.attention_mask, noises)
        score = calibrated_score(target_loss, target_pairs, ref_loss, ref_pairs)
        result = {**record, 'seed': current_seed, 'target_loss': target_loss, 'target_pair_losses': target_pairs,
                  'reference_loss': ref_loss, 'reference_pair_losses': ref_pairs, 'score': score,
                  'loss_baseline_score': -target_loss, 'paraphrasing': 'paper_algorithm_1_symmetrical_embedding',
                  'noise_sigma': config['noise_sigma'], 'pairs': config['perturbation_pairs']}
        append_record(path, result)
        existing.append(result)
        write_json(output / 'progress.json', {'stage': 'spv_attack', 'done': index + 1,
                                              'total': len(records), 'updated_at': time.time()})
        print(f'SPV {index + 1}/{len(records)} score={score:.8f}', flush=True)
    result = evaluate(existing, output, 'spv_self_prompt_symmetrical_embedding')
    evaluate([{**r, 'score': r['loss_baseline_score']} for r in existing], output / 'loss_baseline', 'loss')
    del model, ref_model
    torch.cuda.empty_cache()
    return result
