"""FCP with Hybrid Cache - Full Integration Test."""
import sys
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set paths
MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
CACHE_DIR = "C:/Users/black/OneDrive/Desktop/FCP/cache"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src/memory")

from graph_search import create_graph_search
from temporal_context import TemporalContextMemory
from hybrid_cache import HybridTokenCache


class FCPWithCache:
    """FCP with Hybrid Cache integration."""
    
    def __init__(self):
        print("=" * 60)
        print("FCP with Hybrid Cache")
        print("=" * 60)
        
        # Model
        print("[1] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cpu", torch_dtype=torch.float32)
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        print(f"    Model: {self.num_layers} layers")
        
        # HNSW
        print("[2] Loading HNSW...")
        self.graph = create_graph_search(db_path=GRAPH_PATH)
        self.graph.build_index()
        print(f"    Graph: {len(self.graph.node_ids)} nodes")
        
        # TCM
        print("[3] Loading TCM...")
        self.tcm = TemporalContextMemory(max_segments=50, embedding_dim=2048)
        print(f"    TCM: ready")
        
        # Hybrid Cache
        print("[4] Loading Hybrid Cache...")
        self.cache = HybridTokenCache(
            max_memory_tokens=100,
            disk_cache_dir=os.path.join(CACHE_DIR, "disk")
        )
        print(f"    Cache: ready")
        
        print("[5] Ready!")
        print("=" * 60)
    
    def generate(self, query, max_tokens=48):
        start = time.time()
        
        # 1. Check cache
        cached = self.cache.get(query)
        if cached:
            print(f"    [CACHE] Using cached response")
            self.tcm.write(query, np.zeros(2048))
            self.tcm.write(cached, np.zeros(2048))
            return {
                "response": cached,
                "from_cache": True,
                "tokens": len(cached.split()),
                "tcm_segments": len(self.tcm._segments),
                "cache_stats": self.cache.get_stats(),
                "time": 0.01
            }
        
        # 2. Get graph context
        results = self.graph.search(query, k=3)
        context = ""
        if results:
            context = " | ".join([r.content[:40] for r in results])
        
        # 3. Get TCM context
        tcm_context = ""
        if self.tcm._segments:
            recent = self.tcm._segments[-2:]
            tcm_context = " | ".join([s.text[:30] for s in recent])
        
        # 4. Build prompt
        full_prompt = query
        if context:
            full_prompt = f"[Контекст: {context}] {full_prompt}"
        if tcm_context:
            full_prompt = f"[Память: {tcm_context}] {full_prompt}"
        
        # 5. Generate
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        generated = []
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1]
            next_token = logits.argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # 6. Save to cache
        self.cache.put(query, response)
        
        # 7. Save to TCM
        self.tcm.write(query, np.zeros(2048))
        self.tcm.write(response, np.zeros(2048))
        
        elapsed = time.time() - start
        
        return {
            "response": response,
            "from_cache": False,
            "tokens": len(generated),
            "tcm_segments": len(self.tcm._segments),
            "cache_stats": self.cache.get_stats(),
            "time": round(elapsed, 2)
        }


def main():
    fcp = FCPWithCache()
    
    # Test queries
    queries = [
        ("Привет!", "simple"),
        ("Что такое нейросеть?", "facts"),
        ("Как работает компьютер?", "medium"),
    ] + [
        ("Привет!", "simple"),
    ]  # Same as first - should use cache
    
    print("\n" + "=" * 60)
    print("Running queries...")
    print("=" * 60)
    
    for i, (q, category) in enumerate(queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}/{len(queries)} [{category}]: {q}")
        print(f"{'='*60}")
        
        r = fcp.generate(q)
        
        print(f"\nОтвет: {r['response']}")
        print(f"From cache: {r['from_cache']}")
        print(f"Tokens: {r['tokens']}, TCM: {r['tcm_segments']}, Time: {r['time']}s")
        print(f"Cache: {r['cache_stats']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()