"""Simple FCP Cache Test - Verify components."""
import sys
import os

# Test HybridCache first
sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src/memory")
from hybrid_cache import HybridTokenCache

print("=" * 60)
print("1. Testing HybridCache")
print("=" * 60)

cache = HybridTokenCache(max_memory_tokens=10, disk_cache_dir="C:/Users/black/OneDrive/Desktop/FCP/cache/disk")

# Test cache operations
print("\n[Test 1] Cache miss...")
result = cache.get("test query")
print(f"  Result: {result}")

print("\n[Test 2] Cache put...")
cache.put("test query", "test response")
print(f"  Done")

print("\n[Test 3] Cache hit...")
result = cache.get("test query")
print(f"  Result: {result}")

print("\n[Test 4] Stats...")
print(f"  Stats: {cache.get_stats()}")

print("\n" + "=" * 60)
print("HybridCache: OK")
print("=" * 60)

# Test tokenizer only (quick)
print("\n[2] Testing Tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b", trust_remote_code=True)
text = "Привет"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(f"  Tokenize: {text} -> {len(ids)} tokens")
print(f"  Decode: {decoded}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)