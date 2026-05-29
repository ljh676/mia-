import torch, transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
model_name = "/root/autodl-tmp/models/phi3-mini-4k-instruct"
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
for trust in [False, True]:
    for cache in [True, False]:
        print(f"\n== trust_remote_code={trust} use_cache={cache} ==")
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust, local_files_only=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust, local_files_only=True)
            cfg.rope_scaling = None
            cfg.use_cache = cache
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=cfg,
                trust_remote_code=trust,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            ).eval()
            model.config.use_cache = cache
            messages = [
                {"role":"system","content":"You are a helpful assistant. Answer only with Yes, No, or I don't know."},
                {"role":"user","content":"Context: The paper discusses obesity and cardiovascular disease.\nQuestion: Does the paper mention obesity?\nAnswer only Yes, No, or I don't know."},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt")
            device = getattr(model, "device", None) or next(model.parameters()).device
            enc = {k:v.to(device) for k,v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, use_cache=cache)
            print("class", model.__class__.__name__)
            print("output", tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print("ERROR", type(e).__name__, str(e)[:1000])
