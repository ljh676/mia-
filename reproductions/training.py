"""Auditable full-parameter causal-LM fine-tuning and self-prompt generation."""
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from .common import (read_json, write_json, seed_all, model_fingerprint, file_hash,
                     read_records, append_record, digest)
from .lm import load_causal, causal_losses, text_losses


def save_inference_checkpoint(model, tokenizer, output):
    state, converted = {}, {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        if key not in converted:
            converted[key] = tensor.detach().to(device='cpu', dtype=torch.bfloat16 if tensor.is_floating_point() else tensor.dtype)
        state[name] = converted[key]
    model.save_pretrained(output, state_dict=state, safe_serialization=True)
    tokenizer.save_pretrained(output)


def train(config, data, output, role='target'):
    output, data = Path(output), Path(data)
    marker = output / 'training_complete.json'
    source = data / 'train.json'
    texts = read_json(source)
    token_source = data / 'train_token_ids.json'
    validation = read_json(data / 'validation.json')
    identity = {'base_model': config['base_model'], 'data_sha256': file_hash(source),
                'config_sha256': digest(config), 'role': role}
    if token_source.exists():
        identity['token_ids_sha256'] = file_hash(token_source)
    if marker.exists():
        if read_json(marker)['identity'] != identity:
            raise ValueError('Training checkpoint identity mismatch')
        return output
    seed_all(config['seed'])
    torch.set_num_threads(config.get('cpu_threads', 8))
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_causal(config['base_model'], training=True)
    model.config.use_cache = False
    if config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
    maximum = config['block_size']
    encoded = (read_json(token_source) if token_source.exists() else
               tokenizer(texts, truncation=True, max_length=maximum, add_special_tokens=False)['input_ids'])
    if len(encoded) != len(texts) or any(len(ids) > maximum for ids in encoded):
        raise ValueError('Authoritative training token IDs do not match the dataset')
    if any(len(ids) < 2 for ids in encoded):
        raise ValueError('Invalid training sequence')

    def collate(items):
        tokens = tokenizer.pad({'input_ids': items}, padding=True, return_tensors='pt')
        return tokens.to('cuda')

    micro = config['micro_batch_size']
    accumulation = config['effective_batch_size'] // micro
    if micro * accumulation != config['effective_batch_size']:
        raise ValueError('Effective batch size must be divisible by micro batch size')
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(encoded, batch_size=micro, shuffle=True, collate_fn=collate, generator=generator)
    epochs = config['target_epochs'] if role == 'target' else config['reference_epochs']
    learning_rate = config['learning_rate']
    gpu_parameters = list(model.parameters())
    cpu_optimizer = config.get('optimizer_device', 'cuda') == 'cpu'
    master_parameters = ([torch.nn.Parameter(parameter.detach().cpu().clone()) for parameter in gpu_parameters]
                          if cpu_optimizer else gpu_parameters)
    optimizer = torch.optim.AdamW(master_parameters, lr=learning_rate,
                                  weight_decay=config.get('weight_decay', 0), foreach=False)
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch * epochs)
    best_validation = math.inf
    history = []
    step = 0
    started = time.time()
    write_json(output / 'training_started.json', {'identity': identity, 'started_at': started,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'precision': 'float32_parameters_with_bfloat16_autocast',
               'optimizer': 'AdamW', 'optimizer_state_precision': 'float32',
               'optimizer_device': 'cpu' if cpu_optimizer else 'cuda',
               'cpu_threads': torch.get_num_threads(),
               'inference_checkpoint_dtype': 'bfloat16',
               'micro_batch_size': micro, 'gradient_accumulation': accumulation,
               'effective_batch_size': config['effective_batch_size'], 'epochs_requested': epochs})
    for epoch in range(epochs):
        model.train()
        total, batches = 0., 0
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        for index, tokens in enumerate(loader):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = causal_losses(model, tokens.input_ids, tokens.attention_mask)
            token_counts = (tokens.attention_mask[:, 1:] * tokens.attention_mask[:, :-1]).sum(1)
            loss = (losses * token_counts).sum() / token_counts.sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; refusing to continue')
            # The final partial accumulation group has its own true denominator.
            group_start = (index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            total += loss.item()
            batches += 1
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradient norm')
                if cpu_optimizer:
                    for parameter, master in zip(gpu_parameters, master_parameters):
                        master.grad = parameter.grad.detach().cpu() if parameter.grad is not None else None
                optimizer.step()
                if cpu_optimizer:
                    with torch.no_grad():
                        for parameter, master in zip(gpu_parameters, master_parameters):
                            parameter.copy_(master)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    progress = {'stage': f'{role}_training', 'epoch': epoch + 1, 'epochs': epochs,
                                'step': step, 'total_steps': steps_per_epoch * epochs,
                                'mean_train_loss': total / batches, 'elapsed_seconds': time.time() - started}
                    write_json(output / 'progress.json', progress)
                    print(progress, flush=True)
        model.eval()
        validation_losses = text_losses(model, tokenizer, validation,
                                        batch_size=config.get('validation_batch_size', micro), max_length=maximum)
        validation_loss = sum(validation_losses) / len(validation_losses)
        record = {'epoch': epoch + 1, 'train_loss': total / batches, 'validation_loss': validation_loss,
                  'validation_perplexity': math.exp(min(validation_loss, 80)), 'steps': step}
        history.append(record)
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_inference_checkpoint(model, tokenizer, output)
            write_json(output / 'best_epoch.json', record)
        write_json(output / 'history.json', history)
        print(f'{role} epoch completed: {record}', flush=True)
        if config.get('stop_at_validation_perplexity') and math.exp(validation_loss) <= config['stop_at_validation_perplexity']:
            break
    write_json(marker, {'identity': identity, 'status': 'completed', 'epochs_run': len(history),
                        'elapsed_seconds': time.time() - started, 'checkpoint': model_fingerprint(output),
                        'best_epoch': read_json(output / 'best_epoch.json'),
                        'validation_perplexity_target_met': (math.exp(best_validation) <= config['stop_at_validation_perplexity']
                           if config.get('stop_at_validation_perplexity') else None)})
    del model, optimizer, scheduler, gpu_parameters, master_parameters
    torch.cuda.empty_cache()
    return output


def self_prompt(config, target, data, output):
    output, data = Path(output), Path(data)
    marker = output / 'self_prompt_complete.json'
    if marker.exists():
        return output
    output.mkdir(parents=True, exist_ok=True)
    prompts = read_json(data / 'prompts.json')
    expected = config['reference_samples']
    if len(prompts) != expected:
        raise ValueError('Self-prompt source has unexpected sample count')
    path = output / 'generated.jsonl'
    generated = read_records(path)
    if any(r['index'] != i for i, r in enumerate(generated)):
        raise ValueError('Self-prompt records out of order')
    model, tokenizer = load_causal(target)
    batch_size = config['generation_batch_size']
    for start in range(len(generated), expected, batch_size):
        seed_all(config['seed'] + start)
        batch = prompts[start:start + batch_size]
        prompt_ids = [tokenizer.encode(t, add_special_tokens=False)[:16] for t in batch]
        if any(len(ids) != 16 for ids in prompt_ids):
            raise ValueError('Each self-prompt must have at least 16 tokens')
        ids = torch.tensor(prompt_ids, device='cuda')
        with torch.inference_mode():
            sequences = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                        temperature=1., top_k=50, top_p=1.,
                                        min_new_tokens=config['block_size'] - 16,
                                        max_new_tokens=config['block_size'] - 16,
                                        pad_token_id=tokenizer.eos_token_id, use_cache=True)
        for offset, tokens in enumerate(sequences.cpu().tolist()):
            text = tokenizer.decode(tokens)
            if len(tokens) != config['block_size']:
                raise ValueError('Generated self-prompt sample has an unexpected token length')
            record = {'index': start + offset, 'prompt_ids': prompt_ids[offset],
                      'input_ids': tokens, 'text': text, 'seed': config['seed'] + start,
                      'text_round_trip_preserves_ids': tokenizer.encode(text, add_special_tokens=False) == tokens}
            append_record(path, record)
            generated.append(record)
        write_json(output / 'progress.json', {'stage': 'self_prompt', 'done': len(generated), 'total': expected})
        print(f'Self-prompt {len(generated)}/{expected}', flush=True)
    write_json(output / 'train.json', [r['text'] for r in generated])
    # Byte-level GPT token decoding can be lossy at invalid UTF-8 boundaries.
    # Preserve generated IDs as the authoritative reference training input.
    write_json(output / 'train_token_ids.json', [r['input_ids'] for r in generated])
    # Model selection uses held-out real nonmembers, never target MIA test labels.
    write_json(output / 'validation.json', read_json(data / 'validation.json'))
    write_json(marker, {'status': 'completed', 'count': len(generated), 'length': config['block_size'],
                        'source_prompt_tokens': 16, 'generated_sha256': file_hash(path),
                        'target': model_fingerprint(target)})
    del model
    torch.cuda.empty_cache()
    return output
