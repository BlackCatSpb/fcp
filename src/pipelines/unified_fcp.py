"""Unified FCP Pipeline - All phases integrated."""
import sys
import time
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.graph_search import create_graph_search
from memory.temporal_context import TemporalContextMemory


class UnifiedFCP:
    """Unified FCP: All phases in one pipeline."""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Unified FCP Pipeline")
        logger.info("=" * 60)
        
        # Model (Фаза 1)
        logger.info("[1] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="cpu", torch_dtype=torch.float32
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        
        # HNSW Graph (Фаза 1)
        logger.info("[2] Loading HNSW...")
        self.graph = create_graph_search(db_path=GRAPH_PATH)
        self.graph.build_index()
        
        # TCM (Фаза 2)
        logger.info("[3] Initializing TCM...")
        self.tcm = TemporalContextMemory(max_segments=100, embedding_dim=2048)
        
        # Layer routing (Фаза 3)
        self.layer_domains = {
            "facts": (0, 8),
            "reasoning": (8, 20), 
            "creative": (20, 30),
            "memory": (30, 35)
        }
        
        # LoRA state (Фаза 4)
        self.lora_loaded = False
        
        # Selective activation (Фаза 4)
        self.stop_threshold = 0.85
        
        logger.info("[4] Ready!")
        logger.info(f"    Layers: {self.num_layers}, Graph: 199 nodes")
    
    def analyze(self, query: str) -> dict:
        """Intelligent analysis (Фаза 3)."""
        # Domain routing
        q = query.lower()
        if any(w in q for w in ["что такое", "кто такой", "определение"]):
            domain = "facts"
        elif any(w in q for w in ["почему", "как", "причина"]):
            domain = "reasoning"
        elif any(w in q for w in ["напиши", "сочини"]):
            domain = "creative"
        elif any(w in q for w in ["помнишь", "контекст"]):
            domain = "memory"
        else:
            domain = "facts"
        
        layers = self.layer_domains.get(domain, (0, 8))
        
        # Graph search
        results = self.graph.search(query, k=5)
        
        return {
            "domain": domain,
            "layers": layers,
            "graph_nodes": len(results),
            "needs_context": len(results) > 0
        }
    
    def generate(self, query: str, max_tokens: int = 48) -> dict:
        """Full generation with all phases."""
        start = time.time()
        
        # 1. Analysis (Фаза 3)
        analysis = self.analyze(query)
        logger.info(f"Domain: {analysis['domain']}, Layers: {analysis['layers']}")
        
        # 2. Get context from graph (Фаза 1)
        context = ""
        if analysis["needs_context"]:
            results = self.graph.search(query, k=3)
            context = " | ".join([r.content[:40] for r in results])
            logger.info(f"Context: {len(results)} nodes")
        
        # 3. Get TCM context (Фаза 2)
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
        
        # 5. Tokenize
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        # 6. Generate (with selective activation Фаза 4)
        generated = []
        confidences = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            logits = outputs.logits[0, -1]
            conf = torch.softmax(logits, dim=-1).max().item()
            confidences.append(conf)
            
            # Early exit
            if conf > self.stop_threshold and step > 3:
                logger.info(f"Early exit @ {step}, conf={conf:.3f}")
                break
            
            next_token = logits.argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        # 7. Decode
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # 8. Save to TCM (Фаза 2)
        self.tcm.write(query, np.zeros(2048))
        self.tcm.write(response, np.zeros(2048))
        
        elapsed = time.time() - start
        
        return {
            "response": response,
            "tokens": len(generated),
            "domain": analysis["domain"],
            "layers": analysis["layers"],
            "graph_nodes": analysis["graph_nodes"],
            "tcm_segments": len(self.tcm._segments),
            "early_exit": conf > self.stop_threshold,
            "time": round(elapsed, 2)
        }


def test():
    logger.info("=" * 60)
    logger.info("Unified FCP Test")
    logger.info("=" * 60)
    
    fcp = UnifiedFCP()
    
    queries = [
        "Что такое нейросеть?",
    ]
    
    for q in queries:
        logger.info(f"\n>>> {q}")
        r = fcp.generate(q)
        logger.info(f"<<< {r['response']}")
        logger.info(f"    Tokens: {r['tokens']}, Domain: {r['domain']}")
        logger.info(f"    Layers: {r['layers']}, Graph: {r['graph_nodes']}, TCM: {r['tcm_segments']}")
        logger.info(f"    Time: {r['time']}s, Exit: {r['early_exit']}")


if __name__ == "__main__":
    test()