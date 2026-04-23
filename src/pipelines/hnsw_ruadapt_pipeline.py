"""Full Pipeline: HNSW Graph Search + ruadapt Generation."""
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.graph_search import create_graph_search, GraphVectorExtractor


class FullPipeline:
    """HNSW Graph + ruadapt Generation."""
    
    def __init__(self):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        logger.info(f"Tokenizer: vocab={len(self.tokenizer)}")
        
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        self.model.eval()
        logger.info("Model ready")
        
        logger.info("Loading HNSW index...")
        self.graph_search = create_graph_search(db_path=GRAPH_PATH)
        self.graph_search.build_index()
        self.extractor = GraphVectorExtractor(self.graph_search, output_dim=2048)
        logger.info("HNSW ready: 199 nodes")
    
    def generate(self, prompt: str, use_graph: bool = True, max_tokens: int = 64) -> str:
        """Generate with graph context."""
        
        # Get graph context
        graph_context = ""
        if use_graph:
            results = self.graph_search.search(prompt, k=5)
            if results:
                graph_context = " kontext: " + " | ".join([r.content[:100] for r in results])
                logger.info(f"Graph context: {len(results)} nodes")
        
        # Build full prompt
        full_prompt = prompt
        if graph_context:
            full_prompt = f"{prompt}\n{graph_context}\nОтветь на вопрос используя контекст."
        
        # Format messages
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response


def test():
    logger.info("=" * 60)
    logger.info("FULL PIPELINE: HNSW + ruadapt")
    logger.info("=" * 60)
    
    pipeline = FullPipeline()
    
    prompts = [
        "Что такое нейросеть?",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        r = pipeline.generate(p, use_graph=True, max_tokens=64)
        logger.info(f"<<< {r}")


if __name__ == "__main__":
    test()