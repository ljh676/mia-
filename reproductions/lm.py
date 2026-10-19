import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal(path, tokenizer_path=None, training=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                torch_dtype=torch.float32 if training else torch.bfloat16,
                                                attn_implementation='sdpa')
    model.to('cuda')
    model.train(training)
    return model, tokenizer


def batch_tokens(tokenizer, texts, max_length):
    return tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                     max_length=max_length).to('cuda')


def causal_losses(model, input_ids, attention_mask, inputs_embeds=None):
    kwargs = {'input_ids': input_ids} if inputs_embeds is None else {'inputs_embeds': inputs_embeds}
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    logits = model(**kwargs, attention_mask=attention_mask, position_ids=position_ids,
                   use_cache=False).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float() * attention_mask[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
    if (mask.sum(1) == 0).any():
        raise ValueError('Sequence has fewer than two non-padding tokens')
    return (loss * mask).sum(1) / mask.sum(1)


@torch.inference_mode()
def text_losses(model, tokenizer, texts, batch_size=16, max_length=512):
    result = []
    for start in range(0, len(texts), batch_size):
        tokens = batch_tokens(tokenizer, texts[start:start + batch_size], max_length)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=next(model.parameters()).is_cuda):
            losses = causal_losses(model, tokens.input_ids, tokens.attention_mask)
        result.extend(losses.cpu().tolist())
    return result
