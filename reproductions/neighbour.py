"""ACL 2023 neighbourhood comparison: embedding dropout, ranked one-token neighbours."""
from pathlib import Path
import time
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from .common import seed_all, sample_seed, append_record, read_records, write_json, evaluate, model_fingerprint, file_hash
from .lm import load_causal, text_losses


@torch.inference_mode()
def generate_neighbours(text, model, tokenizer, count=100, max_length=512, batch_size=8,
                        dropout=.7, embedding_mode='paper'):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length,
                       return_offsets_mapping=True)
    offsets = inputs.pop('offset_mapping')[0].tolist()
    inputs = inputs.to('cuda')
    ids = inputs.input_ids
    special = set(tokenizer.all_special_ids)
    positions = [i for i, token in enumerate(ids[0].tolist()) if token not in special]
    if not positions:
        raise ValueError('No replaceable tokens')
    # The released script passes the full BERT embedding output back as inputs_embeds,
    # adding position/type embeddings a second time. The paper formulation perturbs
    # token embeddings only. Both paths are explicit; special-token replacements are excluded.
    base = (model.bert.embeddings(ids) if embedding_mode == 'upstream' else model.get_input_embeddings()(ids))
    candidates = {}
    for offset in range(0, len(positions), batch_size):
        selected = positions[offset:offset + batch_size]
        embeddings = base.expand(len(selected), -1, -1).clone()
        for row, position in enumerate(selected):
            embeddings[row, position] = torch.nn.functional.dropout(embeddings[row, position], dropout, training=True)
        mask = inputs.attention_mask.expand(len(selected), -1)
        logits = model(inputs_embeds=embeddings, attention_mask=mask).logits.float()
        for row, position in enumerate(selected):
            probs = logits[row, position].softmax(-1)
            original = ids[0, position].item()
            denominator = max(1 - probs[original].item(), 1e-12)
            values, tokens = probs.topk(10)
            for prob, token in zip(values.tolist(), tokens.tolist()):
                if token == original or token in special:
                    continue
                replaced = ids[0].clone()
                replaced[position] = token
                if embedding_mode == 'upstream':
                    neighbour = tokenizer.decode(replaced, skip_special_tokens=True)
                else:
                    # Uncased BERT decoding would lowercase and detokenize the entire
                    # sample. Replace only the selected character span in paper mode.
                    start, end = offsets[position]
                    fragment = tokenizer.convert_ids_to_tokens(token)
                    fragment = fragment[2:] if fragment.startswith('##') else fragment
                    neighbour = text[:start] + fragment + text[end:]
                rank = prob / denominator
                item = {'text': neighbour, 'position': position, 'original_token': original,
                        'replacement_token': token, 'character_span': offsets[position], 'rank': rank}
                if neighbour not in candidates or rank > candidates[neighbour]['rank']:
                    candidates[neighbour] = item
    ranked = sorted(candidates.values(), key=lambda r: (-r['rank'], r['text']))[:count]
    if len(ranked) != count:
        raise ValueError(f'Only {len(ranked)} distinct neighbours; required {count}')
    return ranked


def attack(config, data, target, output):
    output = Path(output)
    path = output / 'records.jsonl'
    existing = read_records(path)
    completed = {r['id'] for r in existing}
    model, tokenizer = load_causal(target, config['base_model'])
    tokenizer.padding_side = 'right'
    search_tokenizer = AutoTokenizer.from_pretrained(config['search_model'], local_files_only=True)
    search_model = AutoModelForMaskedLM.from_pretrained(config['search_model'], local_files_only=True).to('cuda').eval()
    write_json(output / 'models.json', {'target': model_fingerprint(target),
                                      'search': model_fingerprint(config['search_model']),
                                      'neighbour_implementation_sha256': file_hash(__file__)})
    (output / 'neighbour_implementation.py').write_bytes(Path(__file__).read_bytes())
    records = read_records(Path(data) / 'evaluation.jsonl')
    for index, record in enumerate(records):
        if record['id'] in completed:
            continue
        seed_all(sample_seed(config['seed'], record['id']))
        neighbours = generate_neighbours(record['text'], search_model, search_tokenizer,
                                          count=config['neighbours'], max_length=config['block_size'],
                                          embedding_mode=config.get('embedding_mode', 'paper'))
        losses = text_losses(model, tokenizer, [record['text']] + [n['text'] for n in neighbours],
                             batch_size=config['attack_batch_size'], max_length=config['block_size'])
        for neighbour, loss in zip(neighbours, losses[1:]):
            neighbour['target_loss'] = loss
        result = {**record, 'original_loss': losses[0], 'neighbours': neighbours,
                  'score': sum(losses[1:]) / len(neighbours) - losses[0], 'loss_baseline_score': -losses[0]}
        append_record(path, result)
        existing.append(result)
        write_json(output / 'progress.json', {'stage': 'neighbour_attack', 'done': index + 1,
                                              'total': len(records), 'updated_at': time.time()})
        print(f'Neighbour {index + 1}/{len(records)} score={result["score"]:.6f}', flush=True)
    result = evaluate(existing, output, 'neighbour_embedding_dropout')
    baseline = [{**r, 'score': r['loss_baseline_score']} for r in existing]
    evaluate(baseline, output / 'loss_baseline', 'loss')
    del model, search_model
    torch.cuda.empty_cache()
    return result
