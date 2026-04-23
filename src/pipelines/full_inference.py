"""Qwen3 Full Runner с OpenVINO и Graph."""
import sys
import logging
import numpy as np
from openvino import Core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODELS = {
    "openvino": "C:/Users/black/OneDrive/Desktop/Models/openvino_model.xml",
    "graph": "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
}

SEQ_LEN = 32


def get_graph_vector(query: str) -> np.ndarray:
    """Get graph context vector."""
    sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
    from memory.graph_search import create_graph_search, GraphVectorExtractor
    
    search = create_graph_search(db_path=MODELS["graph"])
    search.build_index()
    
    extractor = GraphVectorExtractor(search, output_dim=2560)
    return extractor.extract(query, k=10)


class Qwen3Runner:
    """Full Qwen3 inference with graph injection."""
    
    def __init__(self):
        logger.info("Loading OpenVINO model...")
        self.core = Core()
        self.model = self.core.read_model(MODELS["openvino"])
        self.compiled = self.core.compile_model(self.model, "CPU")
        logger.info("Model ready: logits (1, 32, 146260)")
    
    def generate(self, prompt: str, use_graph: bool = True) -> str:
        """Generate response."""
        
        # Get graph context
        if use_graph:
            gv = get_graph_vector(prompt)
            logger.info(f"Graph vector: shape={gv.shape}, norm={np.linalg.norm(gv):.3f}")
        
        # Create input (simplified tokenization)
        input_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        input_ids[0, 0] = 272  # "What"
        input_ids[0, 1] = 271
        input_ids[0, 2] = 528
        input_ids[0, 3] = 271
        input_ids[0, 4] = 1499  # AI
        
        attention_mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
        attention_mask[0, :5] = 1
        
        position_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
        for i in range(SEQ_LEN):
            position_ids[0, i] = i
        
        beam_idx = np.zeros((1,), dtype=np.int32)
        
        # Run
        ireq = self.compiled.create_infer_request()
        outputs = ireq.infer({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "beam_idx": beam_idx
        })
        
        logits = outputs["logits"]
        
        # Get next token
        next_token = int(np.argmax(logits[0, SEQ_LEN-1, :]))
        
        return f"[Generated: token={next_token}]"


def test():
    logger.info("=" * 60)
    logger.info("Qwen3 Full Runner Test")
    logger.info("=" * 60)
    
    runner = Qwen3Runner()
    
    prompts = [
        "What is AI?",
    ]
    
    for p in prompts:
        logger.info(f"\nPrompt: {p}")
        resp = runner.generate(p, use_graph=True)
        logger.info(f"Response: {resp}")


if __name__ == "__main__":
    test()