"""Unified FCP - Save output to file for viewing."""
import sys
import time
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# File output handler
class FileLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
    
    def write(self, msg):
        self.file.write(msg + '\n')
        self.file.flush()
        print(msg)
    
    def close(self):
        self.file.close()

# Setup file logging
log_file = "C:/Users/black/OneDrive/Desktop/FCP/test_output.txt"
file_logger = FileLogger(log_file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified")

# Override to write to file
class DualHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        file_logger.write(msg)

dh = DualHandler()
logging.getLogger().addHandler(dh)

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.graph_search import create_graph_search
from memory.temporal_context import TemporalContextMemory


class UnifiedFCP:
    def __init__(self):
        file_logger.write("=" * 60)
        file_logger.write("Unified FCP Pipeline Test")
        file_logger.write("=" * 60)
        
        file_logger.write("[1] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="cpu", torch_dtype=torch.float32
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        
        file_logger.write("[2] Loading HNSW...")
        self.graph = create_graph_search(db_path=GRAPH_PATH)
        self.graph.build_index()
        
        file_logger.write("[3] Initializing TCM...")
        self.tcm = TemporalContextMemory(max_segments=100, embedding_dim=2048)
        
        self.layer_domains = {
            "facts": (0, 8),
            "reasoning": (8, 20), 
            "creative": (20, 30),
            "memory": (30, 35)
        }
        
        self.stop_threshold = 0.85
        
        file_logger.write(f"[4] Ready! Layers: {self.num_layers}, Graph: 199 nodes\n")
    
    def analyze(self, query):
        q = query.lower()
        if any(w in q for w in ["что такое", "кто такой", "определение"]):
            domain = "facts"
        elif any(w in q for w in ["почему", "как", "причина", "решить"]):
            domain = "reasoning"
        elif any(w in q for w in ["напиши", "сочини"]):
            domain = "creative"
        elif any(w in q for w in ["помнишь", "контекст", "говорили"]):
            domain = "memory"
        else:
            domain = "facts"
        
        layers = self.layer_domains.get(domain, (0, 8))
        results = self.graph.search(query, k=5)
        
        return {
            "domain": domain,
            "layers": layers,
            "graph_nodes": len(results),
            "needs_context": len(results) > 0
        }
    
    def generate(self, query, max_tokens=48):
        start = time.time()
        
        analysis = self.analyze(query)
        
        context = ""
        if analysis["needs_context"]:
            results = self.graph.search(query, k=3)
            context = " | ".join([r.content[:40] for r in results])
        
        tcm_context = ""
        if self.tcm._segments:
            recent = self.tcm._segments[-2:]
            tcm_context = " | ".join([s.text[:30] for s in recent])
        
        full_prompt = query
        if context:
            full_prompt = f"[Контекст: {context}] {full_prompt}"
        if tcm_context:
            full_prompt = f"[Память: {tcm_context}] {full_prompt}"
        
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        generated = []
        confidences = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            logits = outputs.logits[0, -1]
            conf = torch.softmax(logits, dim=-1).max().item()
            confidences.append(conf)
            
            if conf > self.stop_threshold and step > 3:
                break
            
            next_token = logits.argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
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
    fcp = UnifiedFCP()
    
    queries = [
        ("Привет!", "simple"),
        ("Что такое ИИ?", "simple"),
        ("Как работает компьютер?", "medium"),
        ("Почему небо голубое?", "medium"),
        ("Как решить уравнение x+2=5?", "medium"),
        ("Напиши короткое стихотворение про нейросеть", "creative"),
        ("О чем мы говорили?", "memory"),
    ]
    
    for i, (q, category) in enumerate(queries):
        file_logger.write("")
        file_logger.write("=" * 60)
        file_logger.write(f"Query {i+1}/{len(queries)} [{category}]: {q}")
        file_logger.write("=" * 60)
        
        r = fcp.generate(q, max_tokens=64)
        
        file_logger.write("")
        file_logger.write(f"ОТВЕТ: {r['response']}")
        file_logger.write("")
        file_logger.write(f"Tokens: {r['tokens']} | Domain: {r['domain']} | Layers: {r['layers']}")
        file_logger.write(f"Graph: {r['graph_nodes']} nodes | TCM: {r['tcm_segments']} segments | Time: {r['time']}s")
        file_logger.write(f"Early Exit: {r['early_exit']}")
    
    file_logger.write("")
    file_logger.write("=" * 60)
    file_logger.write("TEST COMPLETE")
    file_logger.write("=" * 60)
    file_logger.close()


if __name__ == "__main__":
    test()