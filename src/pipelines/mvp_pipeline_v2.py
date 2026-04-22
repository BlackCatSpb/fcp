"""
FCP Pipeline v2 - Simple version without complex graph loading
"""
import sys
import os
import time
import logging
import codecs
import numpy as np

# Fix encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.pipeline")

# Paths
MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf"
EVA_PATH = "C:/Users/black/OneDrive/Desktop/EVA-Ai"

if EVA_PATH not in sys.path:
    sys.path.insert(0, EVA_PATH)


class FCPTemporalContext:
    """TCM - история диалога."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
    
    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content[:2000]})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, max_chars: int = 1500) -> str:
        if not self.history:
            return ""
        
        parts = []
        total = 0
        for msg in self.history[-self.max_history:]:
            text = f"{msg['role'].capitalize()}: {msg['content']}"
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        
        return "\n".join(parts)


class FCPV2:
    """FCP v2 - Pipeline с TCM."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.tokenizer = None
        self.pipeline = None
        self.tcm = FCPTemporalContext(max_history=10)
        
        self._loaded = False
        self._vocab_size = 0
        self._num_layers = 36
        self._hidden_dim = 2560
    
    def load(self) -> bool:
        try:
            import openvino_genai as ov_genai
            
            logger.info("[FCP] Loading...")
            self.tokenizer = ov_genai.Tokenizer(self.model_path)
            self._vocab_size = len(self.tokenizer.get_vocab())
            logger.info(f"[FCP] Vocab: {self._vocab_size}")
            
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
            
            self._loaded = True
            logger.info("[FCP] Loaded")
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
        temperature: float = 0.2
    ) -> str:
        if not self._loaded:
            return "[FCP] Not loaded"
        
        try:
            return self.pipeline.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        except Exception as e:
            logger.error(f"[FCP] Error: {e}")
            return ""
    
    def generate_with_context(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True
    ) -> str:
        """Generation с контекстом из TCM."""
        if not self._loaded:
            return "[FCP] Not loaded"
        
        # TCM context
        context = ""
        if use_tcm:
            context = self.tcm.get_context(max_chars=1000)
        
        # Build prompt
        if context:
            full_prompt = f"History:\n{context}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        try:
            response = self.pipeline.generate(
                full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            # Save to TCM
            self.tcm.add("user", prompt)
            self.tcm.add("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"[FCP] Error: {e}")
            return ""
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


def test():
    print("=" * 50)
    print("FCP v2 - Test")
    print("=" * 50)
    
    fcp = FCPV2()
    
    if not fcp.load():
        print("[ERROR] Failed to load")
        return 1
    
# Test 1
    print("\n[1] Simple")
    start = time.time()
    r1 = fcp.generate("Hi!", max_new_tokens=128)
    t1 = time.time() - start
    print(f"[{t1:.1f}s] {r1[:100]}...")
    
    # Test 2: With context
    print("\n[2] With TCM")
    start = time.time()
    r2 = fcp.generate_with_context("Tell me about quantum", max_new_tokens=256)
    t2 = time.time() - start
    print(f"[{t2:.1f}s] {r2[:200]}...")
    
    # Test 3: Continue 
    print("\n[3] Continue")
    start = time.time()
    r3 = fcp.generate_with_context("Tell me more", max_new_tokens=256)
    t3 = time.time() - start
    print(f"[{t3:.1f}s] {r3[:200]}...")
    
    print("\n" + "=" * 50)
    print("FCP v2 - Ready!")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(test())