"""Qwen3 Full Runner с OpenVINO и Graph."""
import sys
import logging
import numpy as np
from openvino import Core
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODELS = {
    "openvino": "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_openvino_ModelB/openvino_model.xml",
    "tokenizer": "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_openvino_ModelB",
    "graph": "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
}

MAX_NEW_TOKENS = 64
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
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS["tokenizer"], trust_remote_code=True)
        logger.info(f"Tokenizer ready: vocab={len(self.tokenizer.get_vocab())}")
        
        logger.info("Loading OpenVINO model...")
        self.core = Core()
        self.model = self.core.read_model(MODELS["openvino"])
        self.compiled = self.core.compile_model(self.model, "CPU")
        logger.info("Model ready: logits (1, 32, 146260)")
    
    def generate(self, prompt: str, use_graph: bool = True, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate full response with autoregressive decoding."""
        
        # Get graph context
        graph_vec = None
        if use_graph:
            graph_vec = get_graph_vector(prompt)
            logger.info(f"Graph vector: shape={graph_vec.shape}, norm={np.linalg.norm(graph_vec):.3f}")
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = input_ids[0]  # (seq_len,)
        
        # Start with prompt tokens
        generated_ids = list(input_ids)
        past_key_values = None
        
        logger.info(f"Generating {max_new_tokens} new tokens...")
        
        for step in range(max_new_tokens):
            # Prepare input
            seq_len = min(len(generated_ids), SEQ_LEN)
            current_input = generated_ids[-seq_len:]
            
            # Pad to SEQ_LEN
            input_ids_padded = np.zeros((1, SEQ_LEN), dtype=np.int64)
            input_ids_padded[0, :seq_len] = current_input
            
            # Attention mask
            attention_mask = np.zeros((1, SEQ_LEN), dtype=np.int64)
            attention_mask[0, :seq_len] = 1
            
            # Position ids
            position_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)
            for i in range(SEQ_LEN):
                position_ids[0, i] = i
            
            beam_idx = np.zeros((1,), dtype=np.int32)
            
            # Run inference
            ireq = self.compiled.create_infer_request()
            outputs = ireq.infer({
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "beam_idx": beam_idx
            })
            
            logits = outputs["logits"]  # (1, SEQ_LEN, vocab_size)
            
            # Get next token (last position)
            next_token_logits = logits[0, seq_len-1, :]
            next_token = int(np.argmax(next_token_logits))
            
            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                logger.info(f"EOS token at step {step}")
                break
            
            generated_ids.append(next_token)
            
            if step % 8 == 0:
                logger.info(f"Step {step}: next_token={next_token}")
        
        # Decode
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Remove prompt from response
        prompt_len = len(self.tokenizer.decode(input_ids, skip_special_tokens=True))
        if response.startswith(prompt[:len(prompt)]):
            response = response[len(prompt):].strip()
        
        return response


def test():
    logger.info("=" * 60)
    logger.info("Qwen3 Full Runner Test")
    logger.info("=" * 60)
    
    runner = Qwen3Runner()
    
    prompts = [
        "Что такое искусственный интеллект?",
    ]
    
    for p in prompts:
        logger.info(f"\nPrompt: {p}")
        resp = runner.generate(p, use_graph=True, max_new_tokens=32)
        logger.info(f"Response: {resp}")


if __name__ == "__main__":
    test()