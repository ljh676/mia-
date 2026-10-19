from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import re
import torch

from .Model import Model


class Phi3(Model):
    """Phi-3 mini 本地模型适配器。

    该类遵循 RAG_MIA 现有 Model 接口，只负责把本地 Phi-3 权重
    包装成 query(msg, max_output_tokens) 调用；不改写原论文攻击流程。
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = min(int(config["params"].get("max_output_tokens", 5)), 5)
        self.device = config["params"].get("device", "cuda")

        # 使用配置中的本地路径加载，避免再次联网访问 HuggingFace。
        model_name = config["model_info"].get("name", "/root/autodl-tmp/models/phi3-mini-4k-instruct")
        trust_remote_code = bool(config["params"].get("trust_remote_code", True))
        device_map = config["params"].get("device_map", "auto")
        dtype_name = str(config["params"].get("torch_dtype", "bfloat16")).lower()
        if dtype_name in ["bfloat16", "bf16"]:
            torch_dtype = torch.bfloat16
        elif dtype_name in ["auto"]:
            torch_dtype = "auto"
        elif dtype_name in ["float32", "fp32"]:
            torch_dtype = torch.float32
        else:
            # 不再优先使用 float16；只有配置显式指定时才退到 fp16。
            torch_dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = self.tokenizer.eos_token or "<|end|>"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        # Phi-3 mini 4k 的原始 config.json 中 rope_scaling 为 null。
        model_config.rope_scaling = None
        model_config.use_cache = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        self.model.config.use_cache = True
        self.model.eval()

    def to(self, device):
        # device_map="auto" 时 accelerate 会管理模型分布；这里保持接口兼容。
        return None

    def _normalize_answer(self, text: str) -> str:
        """把 Phi-3 原始输出收敛到 RAG_MIA 需要的三类短答案。"""
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        lowered = normalized.lower()
        if "i don't know" in lowered or "i do not know" in lowered or "unknown" in lowered:
            return "I don't know"
        if re.search(r"\byes\b", lowered):
            return "Yes"
        if re.search(r"\bno\b", lowered):
            return "No"
        return "I don't know"

    def _move_inputs_to_model(self, inputs):
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device
        return {key: value.to(model_device) for key, value in inputs.items()}

    def query(self, msg: str, max_output_tokens=None):
        """按 RAG_MIA 约定返回 Yes/No/I don't know。"""
        max_new_tokens = min(int(max_output_tokens or self.max_output_tokens), 5)
        strict_instruction = (
            "Answer ONLY with one of:\n"
            "- Yes\n"
            "- No\n"
            "- I don't know\n\n"
            "Do not explain.\n"
            "Do not generate any other text.\n\n"
            f"Question:\n{msg}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer only with Yes, No, or I don't know.",
            },
            {"role": "user", "content": strict_instruction},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {
                "input_ids": chat_input_ids,
                "attention_mask": torch.ones_like(chat_input_ids),
            }
        except Exception:
            prompt = f"<|system|>\nAnswer only with Yes, No, or I don't know.<|end|>\n<|user|>\n{strict_instruction}<|end|>\n<|assistant|>\n"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)

        print("==================================================")
        print("[FINAL RAG PROMPT BEGIN]")
        print("==================================================")
        print(prompt)
        print("==================================================")
        print("[FINAL RAG PROMPT END]")
        print("==================================================")

        tokenized_no_trunc = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        raw_token_length = int(tokenized_no_trunc["input_ids"].shape[1])
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        input_len = int(inputs["input_ids"].shape[1])
        print(f"[Phi3 Token Length] {raw_token_length}")
        print(f"[Phi3 Chat Input Length] {input_len}")
        print(f"[Phi3 Tokenizer model_max_length] {model_max_length}")
        print(f"[Phi3 Truncation Enabled] False for chat_template tokenize path")
        print(f"[Phi3 Truncated] {raw_token_length > 4096}")
        print(f"[Phi3 eos_token_id] {self.tokenizer.eos_token_id}")
        print(f"[Phi3 pad_token_id] {self.tokenizer.pad_token_id}")
        print(f"[Phi3 model.use_cache] {self.model.config.use_cache}")

        inputs = self._move_inputs_to_model(inputs)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=1.0,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        output_tokens = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        normalized_output = self._normalize_answer(raw_output)
        print(f"[Phi3 Raw Output] {raw_output}")
        print(f"[Phi3 Normalized Output] {normalized_output}")
        return normalized_output
