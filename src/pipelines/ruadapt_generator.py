"""Qwen3-ruadapt PyTorch Generator."""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

MAX_NEW_TOKENS = 64


class Qwen3AdaptGenerator:
    """Qwen3-ruadapt generation."""
    
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
    
    def generate(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate response."""
        
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
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
    logger.info("Qwen3-ruadapt Generation Test")
    logger.info("=" * 60)
    
    gen = Qwen3AdaptGenerator()
    
    prompts = [
        "Что такое искусственный интеллект?",
        "Как работает блокчейн?",
        "Объясни квантовую запутанность простыми словами",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        r = gen.generate(p, max_tokens=48)
        logger.info(f"<<< {r}")


if __name__ == "__main__":
    test()