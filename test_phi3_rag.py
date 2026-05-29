"""Phi3 + 手工 RAG context 最小测试。"""
import os
import sys

RAG_MIA_PATH = "/root/autodl-tmp/RAG_MIA"
if RAG_MIA_PATH not in sys.path:
    sys.path.append(RAG_MIA_PATH)
os.chdir(RAG_MIA_PATH)

from src.models import create_model
from src.prompts import wrap_prompt

llm = create_model("model_configs/phi3_config.json")

context = ["The paper discusses obesity and cardiovascular disease."]
question = "Does the paper mention obesity?"
rag_prompt = wrap_prompt(question, context, prompt_id=4, context_free_response=False)

print("== Manual RAG Prompt ==")
print(rag_prompt)
print("== Expected ==")
print("Yes")
print("== Phi3 Answer ==")
print(llm.query(rag_prompt, max_output_tokens=5))
