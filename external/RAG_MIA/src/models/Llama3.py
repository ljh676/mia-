import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from .Model import Model

class Llama3(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
        self.model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct',
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16)

    def query(self, msg: str, max_output_tokens=None):
        chat_template_extra_tokens = 4 # Chat template will add 4 tokens

        # Phi-4 expects chat-template
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg}
        ]
        wrapped_msg = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_msg, return_tensors="pt", max_length=12296, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        num_input_tokens = inputs['input_ids'].shape[1]

        max_new_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        max_new_tokens += chat_template_extra_tokens

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        output_tokens = outputs[0][num_input_tokens:]
        result = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return result
