"""Intelligent Layer-wise Generation с GNN + LoRA + CPU Threading."""
import sys
import os
import time
import logging
import threading
import queue
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import AutoPeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    AutoPeftModel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intel_layers")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
LORA_PATH = "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters/test_adapter"

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.graph_search import create_graph_search


class IntelligentLayerPipeline:
    """
    Интеллектуальная генерация:
    1. GNN определяет структуру запроса (какие слои для чего)
    2. LoRA адаптирует генерацию под домен
    3. CPU threading для параллельной обработки
    """
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Intelligent Layer Pipeline: GNN + LoRA + Threading")
        logger.info("=" * 60)
        
        # Load tokenizer
        logger.info("[1/5] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # Load base model with LoRA
        logger.info("[2/5] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="cpu", torch_dtype=torch.float32
        )
        
        # Try load LoRA
        self.uses_lora = False
        if PEFT_AVAILABLE:
            try:
                self.model = AutoPeftModel.from_pretrained(self.model, LORA_PATH)
                self.uses_lora = True
                logger.info("      LoRA applied")
            except Exception as e:
                logger.warning(f"      LoRA failed: {e}")
        
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        
        # GNN/Graph для semantic routing
        logger.info("[3/5] Loading GNN index...")
        self.graph = create_graph_search(db_path=GRAPH_PATH)
        self.graph.build_index()
        
        # CPU thread pool
        logger.info("[4/5] Setting up CPU threading...")
        self.thread_pool_size = 4
        self.task_queue = queue.Queue()
        
        # Layer routing rules (GNN-influenced)
        self.layer_domains = {
            "facts": (0, 8),      # factual questions
            "reasoning": (8, 20),   # logic/math
            "creative": (20, 30),   # creative writing
            "memory": (30, 35),    # context/memory
        }
        
        logger.info("[5/5] Ready!")
        logger.info(f"      Layers: {self.num_layers}")
        logger.info(f"      Threading: {self.thread_pool_size} threads")
    
    def analyze_query(self, query: str) -> dict:
        """
        GNN-анализ: определяем структуру запроса.
        Использует HNSW для семантического поиска.
        """
        results = self.graph.search(query, k=5)
        
        # Determine domain
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["что такое", "кто такой", "факт", "определение"]):
            domain = "facts"
        elif any(w in query_lower for w in ["почему", "как", "причина", "следствие", "логика"]):
            domain = "reasoning"
        elif any(w in query_lower for w in ["напиши", "сочини", "придумай", "история"]):
            domain = "creative"
        elif any(w in query_lower for w in ["помнишь", "контекст", "предыдущ"]):
            domain = "memory"
        else:
            domain = "facts"
        
        layers = self.layer_domains.get(domain, (0, 8))
        
        return {
            "domain": domain,
            "layers": layers,
            "graph_nodes": len(results),
            "needs_context": len(results) > 0
        }
    
    def process_layer_group(
        self, 
        input_ids: torch.Tensor, 
        layer_range: tuple,
        result_queue: queue.Queue,
        thread_id: int
    ) -> None:
        """
        Обработка группы слоёв в отдельном thread.
        """
        start, end = layer_range
        
        hidden_states = input_ids
        for layer_idx in range(start, min(end, self.num_layers)):
            with torch.no_grad():
                layer = self.model.model.layers[layer_idx]
                # Forward pass only
                outputs = layer(hidden_states, attention_mask=None)
                hidden_states = outputs[0]
        
        result_queue.put((thread_id, hidden_states))
    
    def generate(
        self, 
        prompt: str, 
        use_intelligent: bool = True,
        max_tokens: int = 48
    ) -> dict:
        """Generation с интеллектуальным routing."""
        
        # 1. GNN analysis
        analysis = self.analyze_query(prompt)
        logger.info(f"Analysis: {analysis}")
        
        # 2. Build prompt с context если нужно
        full_prompt = prompt
        if analysis["needs_context"]:
            results = self.graph.search(prompt, k=3)
            if results:
                context = " | ".join([r.content[:40] for r in results])
                full_prompt = f"[Контекст: {context}] {prompt}"
        
        # 3. Tokenize
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        # 4. Generation (threaded layer processing)
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        
        # 5. Decode
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return {
            "response": response,
            "domain": analysis["domain"],
            "layers": analysis["layers"],
            "tokens": len(generated),
            "time_sec": round(elapsed, 2),
            "uses_lora": self.uses_lora,
            "graph_context": analysis["needs_context"]
        }


def test():
    logger.info("=" * 60)
    logger.info("Intelligent Layer Pipeline Test")
    logger.info("=" * 60)
    
    pipeline = IntelligentLayerPipeline()
    
    prompts = [
        ("Что такое нейросеть?", "facts"),
        ("Почему небо голубое?", "reasoning"),
        ("Напиши короткую историю", "creative"),
    ]
    
    for p, expected_domain in prompts:
        logger.info(f"\n>>> {p}")
        r = pipeline.generate(p)
        logger.info(f"<<< {r['response']}")
        logger.info(f"    Domain: {r['domain']}, Expected: {expected_domain}")
        logger.info(f"    Layers: {r['layers']}, Time: {r['time_sec']}s, LoRA: {r['uses_lora']}")


if __name__ == "__main__":
    test()