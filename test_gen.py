"""Quick FCP Generation Test."""
import sys
import os
import time
import torch

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src/memory")

from hybrid_cache import HybridTokenCache
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("FCP Generation Test")
print("=" * 60)

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

# Load model
print("\n[1] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
model.eval()
print(f"    Model ready: {len(model.model.layers)} layers")

# Load cache
print("\n[2] Loading cache...")
cache = HybridTokenCache(max_memory_tokens=50, disk_cache_dir="C:/Users/black/OneDrive/Desktop/FCP/cache")
print(f"    Cache ready")

# Generate function
def generate(query, max_tokens=16):
    # Check cache
    cached = cache.get(query)
    if cached:
        return cached, True
    
    # Generate
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
    
    # Cache it
    cache.put(query, response)
    
    return response, False

# Run tests
print("\n" + "=" * 60)
print("Running queries...")
print("=" * 60)

queries = [
    "Привет!",
    "Что такое ИИ?",
    "Как работает компьютер?",
]

for i, q in enumerate(queries):
    print(f"\nQuery {i+1}: {q}")
    print("-" * 40)
    
    start = time.time()
    response, from_cache = generate(q, max_tokens=16)
    elapsed = time.time() - start
    
    print(f"Ответ: {response}")
    print(f"From cache: {from_cache}")
    print(f"Time: {elapsed:.1f}s")

print("\n" + "=" * 60)
print(f"Cache stats: {cache.get_stats()}")
print("=" * 60)
print("\nTEST COMPLETE")