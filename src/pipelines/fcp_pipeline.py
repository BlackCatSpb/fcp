"""Full FCP Pipeline: Real Split + HNSW + ruadapt."""
import sys
import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fcp")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.graph_search import create_graph_search


class FCPPipeline:
    """Full FCP Pipeline с Real Split и Graph Injection."""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("FCP Pipeline: Real Split + HNSW")
        logger.info("=" * 60)
        
        # 1. Load Model
        logger.info("[1/4] Loading ruadapt model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        self.split_layer = 8
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"      Model: {len(self.model.model.layers)} layers, hidden={self.hidden_size}")
        
        # 2. Register hooks
        logger.info("[2/4] Registering split hooks...")
        self.hidden_states = {}
        self._register_hooks()
        
        # 3. Load HNSW
        logger.info("[3/4] Loading HNSW index...")
        self.graph_search = create_graph_search(db_path=GRAPH_PATH)
        self.graph_search.build_index()
        logger.info(f"      HNSW: {len(self.graph_search.node_ids)} nodes")
        
        logger.info("[4/4] Ready!")
        logger.info("")
    
    def _register_hooks(self):
        """Hook на split слое."""
        layer = self.model.model.layers[self.split_layer]
        layer.register_forward_hook(
            lambda m, i, o: self.hidden_states.update({self.split_layer: o[0].detach()})
        )
    
    def get_graph_context(self, query: str, k: int = 5) -> str:
        """Получить контекст из графа."""
        results = self.graph_search.search(query, k=k)
        if not results:
            return ""
        
        context = " | ".join([f"{r.content[:50]}..." for r in results])
        logger.info(f"Graph: found {len(results)} nodes")
        return context
    
    def generate(self, prompt: str, use_graph: bool = True, max_tokens: int = 48) -> str:
        """Generation с graph injection."""
        
        # Get graph context
        graph_ctx = ""
        if use_graph:
            graph_ctx = self.get_graph_context(prompt)
        
        # Build prompt
        full_prompt = prompt
        if graph_ctx:
            full_prompt = f"{prompt}\n\nКонтекст: {graph_ctx}\n\nОтветь используя контекст."
        
        # Format messages
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Extract hidden states
        self.hidden_states.clear()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Log hidden state
        if self.split_layer in self.hidden_states:
            hs = self.hidden_states[self.split_layer]
            logger.info(f"Hidden state: {hs.shape}")
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response


def test():
    """Test pipeline."""
    pipeline = FCPPipeline()
    
    prompts = [
        "Что такое нейросеть?",
    ]
    
    for p in prompts:
        logger.info(f">>> {p}")
        r = pipeline.generate(p, use_graph=True)
        logger.info(f"<<< {r}")
        logger.info("")


if __name__ == "__main__":
    test()