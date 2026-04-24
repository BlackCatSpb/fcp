"""Quick FCP Test."""
import sys
import time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

print("=" * 60)
print("FCP Quick Test")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
model.eval()
print(f"Model loaded: {len(model.model.layers)} layers")

queries = [
    "Привет!",
    "Что такое ИИ?",
    "Как работает компьютер?",
]

for i, q in enumerate(queries):
    print(f"\n{'='*60}")
    print(f"Query {i+1}: {q}")
    print(f"{'='*60}")
    
    messages = [{"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    generated = []
    for step in range(32):
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits[0, -1]
        next_token = logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]])], dim=1)
    
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    print(f"Ответ: {response}")
    print(f"Tokens: {len(generated)}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)