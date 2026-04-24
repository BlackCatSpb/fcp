"""Quick FCP with Cache Hit Test."""
import sys
import os
import time
import torch

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src/memory")

from hybrid_cache import HybridTokenCache
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("FCP Cache Hit Test")
print("=" * 60)

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
model.eval()
print("Model ready")

cache = HybridTokenCache(max_memory_tokens=50, disk_cache_dir="C:/Users/black/OneDrive/Desktop/FCP/cache")
print("Cache ready")

def generate(query, max_tokens=12):
    cached = cache.get(query)
    if cached:
        return cached, True
    
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    
    generated = []
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits[0, -1]
        next_token = logits.argmax().item()
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        ids = torch.cat([ids, torch.tensor([[next_token]])], dim=1)
    
    response = tokenizer.decode(generated, skip_special_tokens=True)
    cache.put(query, response)
    return response, False

# Run test with repeat
query = "Что такое нейросеть?"

print(f"\n[1] First: {query}")
r1, cached1 = generate(query)
print(f"    Response: {r1}")
print(f"    Cached: {cached1}")

print(f"\n[2] Repeat: {query}")
r2, cached2 = generate(query)
print(f"    Response: {r2}")
print(f"    Cached: {cached2}")

print(f"\nCache stats: {cache.get_stats()}")
print("\nTEST COMPLETE")