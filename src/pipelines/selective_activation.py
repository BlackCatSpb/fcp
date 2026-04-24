"""Selective Activation Pipeline с Early Exit."""
import sys
import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("selective")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"


class SelectiveActivation:
    """Early Exit: останавливает генерацию когда уверенность высока."""
    
    def __init__(self):
        logger.info("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size
        
        self.hidden_states = {}
        self._register_hooks()
        
        self.stop_threshold = 0.7
        logger.info(f"Model: {self.num_layers} layers, Early Exit @ {self.stop_threshold}")
    
    def _register_hooks(self):
        """Hook на каждом слое для early exit."""
        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            layer.register_forward_hook(
                lambda m, inp, out, idx=i: self._capture(idx, out)
            )
    
    def _capture(self, idx, output):
        self.hidden_states[idx] = output[0].detach()
    
    def compute_confidence(self, logits: torch.Tensor) -> float:
        """Вычисляет уверенность по logits (softmax probability)."""
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max().item()
        return max_prob
    
    def generate(self, prompt: str, max_tokens: int = 64) -> dict:
        """Generation с early exit."""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        
        generated = []
        confidences = []
        
        logger.info(f"Generating (max={max_tokens})...")
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            
            logits = outputs.logits[0, -1]
            next_token = logits.argmax().item()
            
            # Early exit check using logits probability
            conf = self.compute_confidence(logits)
            confidences.append(conf)
            
            if conf > self.stop_threshold and step > 2:
                logger.info(f"Early exit @ step {step}, confidence={conf:.3f}")
                break
            
            if next_token == self.tokenizer.eos_token_id:
                logger.info(f"EOS @ step {step}")
                break
            
            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)
        
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return {
            "response": response,
            "tokens": len(generated),
            "early_exit": conf > self.stop_threshold,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0
        }


def test():
    logger.info("=" * 60)
    logger.info("Selective Activation Test")
    logger.info("=" * 60)
    
    sa = SelectiveActivation()
    
    prompts = [
        "Что такое ИИ?",
        "Сколько будет 2+2?",
    ]
    
    for p in prompts:
        logger.info(f"\n>>> {p}")
        result = sa.generate(p, max_tokens=32)
        logger.info(f"<<< {result['response']}")
        logger.info(f"    Tokens: {result['tokens']}, Early Exit: {result['early_exit']}")


if __name__ == "__main__":
    test()