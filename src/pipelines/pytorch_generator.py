"""Qwen3-4B PyTorch Generator."""
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B-PyTorch"
TOKENIZER_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_openvino_ModelB"

MAX_NEW_TOKENS = 64


class Qwen3Generator:
    """Qwen3-4B generation using PyTorch."""
    
    def __init__(self):
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
        logger.info(f"Tokenizer ready: vocab={len(self.tokenizer)}")
        
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        self.model.eval()
        logger.info("Model ready")
    
    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """Generate response."""
        
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        logger.info(f"Input tokens: {input_ids.shape[1]}")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response


def test():
    logger.info("=" * 60)
    logger.info("Qwen3-4B PyTorch Generation Test")
    logger.info("=" * 60)
    
    generator = Qwen3Generator()
    
    prompts = [
        "Что такое искусственный интеллект?",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        response = generator.generate(p, max_new_tokens=48)
        logger.info(f"<<< {response}")


if __name__ == "__main__":
    test()