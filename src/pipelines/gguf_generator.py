"""Qwen3 GGUF Generator."""
import logging
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runner")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf"


class Qwen3GGUFGenerator:
    """Qwen3 generation using GGUF model."""
    
    def __init__(self):
        logger.info("Loading GGUF model...")
        self.model = Llama(MODEL_PATH, n_ctx=512, n_threads=4, verbose=False)
        logger.info("Model ready")
    
    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        """Generate response."""
        
        # Format prompt for instruction
        formatted = f"<|im_start|>system\nТы - интеллектуальный ассистент.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        output = self.model(
            formatted,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>"]
        )
        
        response = output["choices"][0]["text"].strip()
        return response


def test():
    logger.info("=" * 60)
    logger.info("Qwen3 GGUF Generation Test")
    logger.info("=" * 60)
    
    generator = Qwen3GGUFGenerator()
    
    prompts = [
        "Что такое искусственный интеллект?",
        "Как работает квантовый компьютер?",
        "Объясни теорию относительности простыми словами",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        response = generator.generate(p, max_tokens=48)
        logger.info(f"<<< {response}")


if __name__ == "__main__":
    test()