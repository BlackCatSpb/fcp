"""
FCP Pipeline - Этап 1: MVP Базовый Pipeline
"""
import sys
import os
import time
import logging
import io

# Fix encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.pipeline")

# Paths
MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf"
EVA_PATH = "C:/Users/black/OneDrive/Desktop/EVA-Ai"

if EVA_PATH not in sys.path:
    sys.path.insert(0, EVA_PATH)


class FCPV1:
    """
    FCP v1 - MVP Pipeline.
    
    Компоненты:
    - openvino_genai.Tokenizer (GGUF)
    - LLMPipeline (OpenVINO GenAI)
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = None
        self.pipeline = None
        
        self._loaded = False
        self._vocab_size = 0
        self._num_layers = 0
        self._hidden_dim = 0
    
    def load(self) -> bool:
        """Load tokenizer and pipeline."""
        try:
            import openvino_genai as ov_genai
            
            # Tokenizer
            logger.info("[FCP] Loading OpenVINO Tokenizer...")
            self.tokenizer = ov_genai.Tokenizer(self.model_path)
            vocab = self.tokenizer.get_vocab()
            self._vocab_size = len(vocab)
            logger.info(f"[FCP] Vocab size: {self._vocab_size}")
            
            # Pipeline
            logger.info("[FCP] Loading LLMPipeline...")
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path,
                self.tokenizer,
                "CPU",
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": 8,
                    "NUM_STREAMS": 1,
                }
            )
            
            # Model info defaults (Qwen3 4B)
            self._num_layers = 36
            self._hidden_dim = 2560
            
            self._loaded = True
            logger.info("[FCP] Pipeline loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"[FCP] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40
    ) -> str:
        """Generate response."""
        if not self._loaded:
            return "[FCP] Not loaded"
        
        try:
            response = self.pipeline.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True
            )
            return response
            
        except Exception as e:
            logger.error(f"[FCP] Generate error: {e}")
            return ""
    
    def generate_with_context(
        self,
        prompt: str,
        context: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        """Generate with context enrichment."""
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt
        
        return self.generate(full_prompt, max_new_tokens, temperature)
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def num_layers(self) -> int:
        return self._num_layers
    
    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_config_summary(self) -> str:
        return f"FCP: {self._num_layers} layers, {self._hidden_dim}d, vocab={self._vocab_size}"


def test_mvp():
    """Test MVP Pipeline."""
    print("=" * 50)
    print("FCP v1 - MVP Test")
    print("=" * 50)
    
    fcp = FCPV1()
    
    if not fcp.load():
        print("[ERROR] Failed to load")
        return 1
    
    print(f"[Config] {fcp.get_config_summary()}")
    
    # Test prompts
    test_prompts = [
        ("Hello!", 256),
        ("What is quantum mechanics?", 384),
    ]
    
    for prompt, max_tokens in test_prompts:
        print(f"\n[Prompt] {prompt}")
        
        start = time.time()
        response = fcp.generate(prompt, max_new_tokens=max_tokens)
        latency = time.time() - start
        
        if response:
            print(f"[Response] {response[:300]}...")
            print(f"[Latency] {latency:.1f}s | ~{len(response.split())} words")
    
    print("\n" + "=" * 50)
    print("FCP v1 - Ready!")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(test_mvp())