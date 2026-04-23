"""Qwen3 Split Exporter."""
import sys
import os
import logging
import json
from typing import Dict

import torch
import torch.nn as nn
from safetensors import safe_open
import onnx
from onnx import helper, TensorProto

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("split")

MODEL_DIR = "C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B-PyTorch"
OUTPUT_DIR = "C:/Users/black/OneDrive/Desktop/FCP/models/split"
SPLIT_LAYERS = [4, 8, 16, 24]


def load_model():
    """Load model weights."""
    logger.info(f"Loading from {MODEL_DIR}")
    
    shards = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('model-') and f.endswith('.safetensors')])
    logger.info(f"Shards: {len(shards)}")
    
    state_dict = {}
    for shard in shards:
        with safe_open(os.path.join(MODEL_DIR, shard), framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    
    logger.info(f"Tensors: {len(state_dict)}")
    
    emb = state_dict['model.embed_tokens.weight']
    logger.info(f"Embed: {emb.shape}, {emb.dtype}")
    
    return state_dict


def export():
    """Export split config."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    state_dict = load_model()
    
    # Count layers
    layers = set()
    for k in state_dict:
        if k.startswith('model.layers.'):
            parts = k.split('.')
            if len(parts) > 2:
                layers.add(int(parts[2]))
    num_layers = max(layers) + 1
    logger.info(f"Layers: {num_layers}")
    
    # Config
    config = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_layers": num_layers,
        "split_layers": SPLIT_LAYERS,
        "vocab_size": 151936
    }
    
    with open(f"{OUTPUT_DIR}/split_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config: {OUTPUT_DIR}/split_config.json")
    
    # ONNX placeholders
    inp = helper.make_tensor_value_info('input', TensorProto.INT64, [1, 128])
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 128, 2560])
    graph = helper.make_graph([helper.make_node('Identity', ['input'], ['output'])], 'model', [inp], [out])
    onnx.save(helper.make_model(graph), f"{OUTPUT_DIR}/part1.onnx")
    onnx.save(helper.make_model(graph), f"{OUTPUT_DIR}/part2.onnx")
    
    logger.info("Done!")


if __name__ == "__main__":
    export()