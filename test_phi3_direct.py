"""直接测试 Phi3 原始 generate，排除 RAG_MIA wrap_prompt 干扰。"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "/root/autodl-tmp/models/phi3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
cfg.rope_scaling = None
cfg.use_cache = True
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=cfg,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
).eval()
model.config.use_cache = True

def gen(messages):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
    return prompt, tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

user = "Context: The paper discusses obesity and cardiovascular disease.\nQuestion: Does the paper mention obesity?\nAnswer only Yes, No, or I don't know.\nAnswer:"
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer only with Yes, No, or I don't know."},
    {"role": "user", "content": user},
]
prompt, output = gen(messages)
print("== CHAT PROMPT ==")
print(prompt)
print("== EXPECTED ==")
print("Yes")
print("== CHAT OUTPUT ==")
print(output)
print("== ENV ==")
print("dtype=bf16 use_cache=True attn=eager eos", tok.eos_token_id, "pad", tok.pad_token_id)
