"""FCP Cache Hit Test v2."""
import sys
import time
import torch

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src/memory")

from hybrid_cache import HybridTokenCache
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("FCP Cache Hit Test v2")
print("=" * 60)

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
model.eval()

cache = HybridTokenCache(max_memory_tokens=100, disk_cache_dir="C:/Users/black/OneDrive/Desktop/FCP/cache")

def generate(query, max_tokens=8):
    start = time.time()
    
    # Check cache FIRST
    cached = cache.get(query)
    if cached:
        return cached, True, time.time() - start
    
    # Generate NEW
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
    
    # Save to cache
    cache.put(query, response)
    
    return response, False, time.time() - start

# Test 1: Query 1
q1 = "Сколько будет 2+2?"
print(f"\n[1] Query: {q1}")
r1, c1, t1 = generate(q1)
print(f"    Response: {r1}")
print(f"    Cached: {c1}, Time: {t1:.1f}s")

# Test 2: Same query - should CACHE HIT
q2 = "Сколько будет 2+2?"
print(f"\n[2] Query: {q2}")
r2, c2, t2 = generate(q2)
print(f"    Response: {r2}")
print(f"    Cached: {c2}, Time: {t2:.1f}s")

print(f"\n\nCache stats: {cache.get_stats()}")
print("\nDONE")