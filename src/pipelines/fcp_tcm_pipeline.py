"""Full FCP Pipeline с TCM + Selective Activation."""
import sys
import time
import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fcp_tcm")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
from memory.temporal_context import TemporalContextMemory


class FCPWithTCM:
    """FCP Pipeline с Temporal Context Memory."""
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("FCP + TCM Pipeline")
        logger.info("=" * 60)
        
        # Model
        logger.info("[1/3] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="cpu", torch_dtype=torch.float32
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        self.stop_threshold = 0.99  # Almost never exit
        logger.info(f"      Model: {self.num_layers} layers")
        
        # TCM
        logger.info("[2/3] Initializing TCM...")
        self.tcm = TemporalContextMemory(
            max_segments=100,
            embedding_dim=2048,
            time_scales=4
        )
        logger.info(f"      TCM: max_segments=100, scales=4")
        
        logger.info("[3/3] Ready!")
        logger.info("")
    
    def compute_confidence(self, logits: torch.Tensor) -> float:
        """Уверенность по logits."""
        probs = torch.softmax(logits, dim=-1)
        return probs.max().item()
    
    def generate(
        self, 
        prompt: str, 
        use_tcm: bool = True,
        max_tokens: int = 48
    ) -> dict:
        """Generation с TCM и Early Exit."""
        
        # Retrieve from TCM
        tcm_context = ""
        if use_tcm and self.tcm._segments:
            recent = self.tcm._segments[-3:]
            if recent:
                tcm_context = " | ".join([s.text[:50] for s in recent])
                logger.info(f"TCM context: {len(recent)} segments")
        
        # Build prompt
        full_prompt = prompt
        if tcm_context:
            full_prompt = f"Контекст из предыдущих разговоров: {tcm_context}\n\n{prompt}"
        
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        generated = []
        confidences = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            logits = outputs.logits[0, -1]
            next_token = logits.argmax().item()
            
            conf = self.compute_confidence(logits)
            confidences.append(conf)
            
            if conf > self.stop_threshold and step > 2:
                logger.info(f"Early exit @ step {step}, conf={conf:.3f}")
                break
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Save to TCM
        if use_tcm:
            self.tcm.write(prompt, np.zeros(2048), {"role": "user"})
            self.tcm.write(response, np.zeros(2048), {"role": "assistant"})
        
        return {
            "response": response,
            "tokens": len(generated),
            "early_exit": conf > self.stop_threshold,
            "tcm_segments": len(self.tcm._segments),
            "avg_conf": sum(confidences) / len(confidences) if confidences else 0
        }


def test():
    logger.info("=" * 60)
    logger.info("FCP + TCM Test")
    logger.info("=" * 60)
    
    fcp = FCPWithTCM()
    
    prompts = [
        "Привет! Как дела?",
        "Что такое нейросеть?",
        "Расскажи о квантовых компьютерах",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        r = fcp.generate(p, use_tcm=True, max_tokens=32)
        logger.info(f"<<< {r['response']}")
        logger.info(f"    Tokens: {r['tokens']}, TCM: {r['tcm_segments']}, Exit: {r['early_exit']}")


if __name__ == "__main__":
    test()