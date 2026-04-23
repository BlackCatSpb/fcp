"""Real Split Exporter for ruadapt_qwen3_4b."""
import sys
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("split")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b"


class RealSplitExporter:
    """Real Split: экспорт part1/part2 для graph injection."""
    
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
        logger.info(f"Model: {self.num_layers} layers, hidden={self.model.config.hidden_size}")
        
        self.hidden_states = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Регистрация хуков для извлечения hidden states."""
        split_layers = [4, 8, 16, 24, 32]
        
        for layer_idx in split_layers:
            if layer_idx < self.num_layers:
                layer = self.model.model.layers[layer_idx]
                
                def hook(module, input, output, idx=layer_idx):
                    self.hidden_states[idx] = output[0].detach().clone()
                
                layer.register_forward_hook(hook)
                logger.info(f"Registered hook on layer {layer_idx}")
    
    def extract_hidden_states(self, prompt: str, split_layer: int = 8) -> dict:
        """Извлечение hidden states на указанном слое."""
        
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        self.hidden_states.clear()
        
        with torch.no_grad():
            _ = self.model(**input_ids)
        
        if split_layer in self.hidden_states:
            return {
                'hidden': self.hidden_states[split_layer],
                'shape': self.hidden_states[split_layer].shape
            }
        return None
    
    def generate(self, prompt: str, max_tokens: int = 32) -> str:
        """Generation для сравнения."""
        
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
        
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def test():
    logger.info("=" * 60)
    logger.info("Real Split Exporter Test")
    logger.info("=" * 60)
    
    exporter = RealSplitExporter()
    
    # Test hidden state extraction
    logger.info("\n[1] Hidden states extraction:")
    result = exporter.extract_hidden_states("Что такое ИИ?", split_layer=8)
    if result:
        logger.info(f"  Layer 8: {result['shape']}")
    
    # Test generation
    logger.info("\n[2] Generation:")
    prompt = "Что такое нейросеть?"
    response = exporter.generate(prompt, max_tokens=32)
    logger.info(f"  Prompt: {prompt}")
    logger.info(f"  Response: {response}")


if __name__ == "__main__":
    test()