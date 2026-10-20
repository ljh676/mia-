"""Local-model substitution for paper API calls; raw prompts and outputs are retained."""
import json
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .lm import load_causal


def binary_answer(text):
    value = str(text).replace('\u2019', "'").strip()
    if re.search(r"\b(?:unknown|unsure|not enough information)\b|\bi (?:do not|don't) know\b", value, re.I):
        return 'UNK'
    match = re.match(r'^[\s\"\x27]*(?:answer\s*:\s*)?(yes|no)\b', value, re.I)
    return match.group(1).lower() if match else 'UNK'


def json_object(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('Model did not return a JSON object')
    return json.loads(text[start:end + 1])


class LocalGenerator:
    def __init__(self, path):
        self.model, self.tokenizer = load_causal(path)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=1024, thinking=False, temperature=0.):
        messages = [{'role': 'user', 'content': prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=thinking)
        tokens = self.tokenizer(rendered, return_tensors='pt', truncation=False).to('cuda')
        if tokens.input_ids.shape[1] + max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError('Generation request exceeds local-model context window')
        sampling = {'do_sample': temperature > 0}
        if temperature > 0:
            sampling.update(temperature=temperature, top_p=.9, top_k=40)
        result = self.model.generate(**tokens, max_new_tokens=max_new_tokens, **sampling,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(result[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        if thinking and '</think>' not in text:
            raise RuntimeError('Local model exceeded the reasoning budget without a final answer')
        # Qwen hybrid models can emit a closing think tag even with thinking disabled.
        return text.split('</think>')[-1].strip()

    def answer(self, document, query):
        prompt = ('Based solely on the context below, answer the question with Yes, No, or I don\'t know. '
                  'Do not use outside knowledge. Return only the answer.\nContext:\n' + document + '\nQuestion: ' + query)
        raw = self.generate(prompt, 12)
        return binary_answer(raw), {'prompt': prompt, 'raw': raw}


class TargetRAGGenerator:
    def __init__(self, path, max_input_tokens=2048, style='ia'):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True,
                            torch_dtype=torch.bfloat16).to('cuda').eval()
        self.maximum = max_input_tokens
        self.style = style
        yes = self.tokenizer.encode('Yes', add_special_tokens=False)
        if len(yes) != 1:
            raise ValueError('This implementation requires a single-token Yes response')
        self.yes_id = yes[0]

    @torch.inference_mode()
    def query(self, question, contexts):
        context = '\n\n'.join(contexts)
        if self.style == 'dcmi':
            # DCMI Appendix E and the released FlashRAG base prompt.
            prompt = f'context:{context}\nAnswer the question based on the provided context.\n{question}'
        else:
            prompt = ('You are a helpful assistant, below is a query from a user and some relevant contexts. '
                      'Answer the question given the information in those contexts. Your answer should be short and concise. '
                      'If you cannot find the answer to the question, just say "I don\'t know". '
                      f'\n\nContexts: {context}\n\nQuery: {question}\n\nAnswer:')
        full = self.tokenizer(prompt, add_special_tokens=True).input_ids
        # FlashRAG preserves both ends of long prompts; this retains the question
        # at the end rather than silently dropping it during prefix truncation.
        used = full if len(full) <= self.maximum else full[:self.maximum // 2] + full[-(self.maximum - self.maximum // 2):]
        input_ids = torch.tensor([used], device='cuda')
        tokens = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids)}
        # Teacher-force only decoder start: the exact next-token P(Yes), not lexical or retrieval scores.
        decoder = torch.tensor([[self.model.config.decoder_start_token_id]], device='cuda')
        logits = self.model(**tokens, decoder_input_ids=decoder).logits[0, 0].float()
        probability = logits.softmax(-1)[self.yes_id].item()
        response = self.model.generate(**tokens, max_new_tokens=8, do_sample=False)
        text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        return {'probability_yes': probability, 'raw_response': text, 'answer': binary_answer(text),
                'prompt': prompt, 'input_tokens_before_truncation': len(full),
                'input_tokens_used': len(used), 'input_truncated': len(full) > self.maximum,
                'truncation_policy': 'flashrag_head_tail', 'prompt_style': self.style}
