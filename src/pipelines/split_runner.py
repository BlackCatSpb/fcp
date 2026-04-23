"""
Qwen3 Split Runner - Runtime с инъекцией графа
"""
import sys
import os
import logging
import json
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("runner")

GRAPH_DB = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
MODEL_DIR = "C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B-PyTorch"
SPLIT_CONFIG = "C:/Users/black/OneDrive/Desktop/FCP/models/split/split_config.json"
INJECTION_LAYERS = [4, 8, 16, 24]
INJECTION_ALPHA = 0.3


@dataclass
class SplitConfig:
    model_type: str
    hidden_size: int
    num_layers: int
    split_layers: list
    vocab_size: int


class GraphInjector:
    """Graph vector injection into model."""
    
    def __init__(self, db_path: str):
        sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
        from memory.graph_search import create_graph_search, GraphVectorExtractor
        
        self.search = create_graph_search(db_path=db_path)
        self.extractor = GraphVectorExtractor(self.search, output_dim=2560)
        logger.info(f"GraphInjector ready: {db_path}")
    
    def get_graph_vector(self, query: str) -> np.ndarray:
        """Get graph context vector for query."""
        return self.extractor.extract(query, k=10)


class Qwen3SplitRunner:
    """
    Split runner с инъекцией графа.
    
    Flow:
    1. Токенизация запроса
    2. Получение graph vector
    3. Full model inference (Placeholder - требует transformers)
    4. Инъекция graph vector в hidden states
    5. Генерация токенов
    """
    
    def __init__(self, model_dir: str, config_path: str, graph_db: str):
        self.model_dir = model_dir
        self.config = self._load_config(config_path)
        self.graph_injector = GraphInjector(graph_db)
        
        # Placeholder for actual model
        self.model = None
        logger.info("Qwen3SplitRunner ready (placeholder)")
    
    def _load_config(self, path: str) -> SplitConfig:
        with open(path) as f:
            data = json.load(f)
        return SplitConfig(**data)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        use_graph: bool = True,
        temperature: float = 0.7
    ) -> str:
        """Generate response с опциональной инъекцией графа."""
        
        logger.info(f"Generating: {prompt[:50]}...")
        
        # Get graph context if enabled
        graph_vec = None
        if use_graph:
            graph_vec = self.graph_injector.get_graph_vector(prompt)
            logger.info(f"Graph vector: {graph_vec.shape}, norm={np.linalg.norm(graph_vec):.2f}")
        
        # Placeholder: return enhanced prompt
        # Real implementation would use transformers inference
        enhanced_prompt = prompt
        if graph_vec is not None and np.linalg.norm(graph_vec) > 0.1:
            graph_context = f"[Context: {np.sum(graph_vec[:10]):.2f}] "
            enhanced_prompt = graph_context + prompt
            logger.info(f"Enhanced with graph context")
        
        # Placeholder response
        response = f"[Generated from {len(prompt)} chars]"
        
        return response
    
    def generate_stream(self, prompt: str, use_graph: bool = True):
        """Streaming generation (placeholder)."""
        result = self.generate(prompt, use_graph=use_graph)
        yield result


def run_test():
    """Test the split runner."""
    logger.info("=" * 60)
    logger.info("Qwen3 Split Runner Test")
    logger.info("=" * 60)
    
    runner = Qwen3SplitRunner(MODEL_DIR, SPLIT_CONFIG, GRAPH_DB)
    
    test_prompts = [
        "What is artificial intelligence?",
        "How does neural network work?"
    ]
    
    for prompt in test_prompts:
        logger.info(f"\n--- Prompt: {prompt} ---")
        response = runner.generate(prompt, use_graph=True)
        logger.info(f"Response: {response}")
    
    logger.info("=" * 60)
    logger.info("Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_test()