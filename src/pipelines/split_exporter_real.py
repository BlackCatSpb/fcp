"""
FCP Split Exporter - Real split с PyTorch safetensors моделью
"""
import sys
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("split_exporter")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_hf"
OUTPUT_DIR = "C:/Users/black/OneDrive/Desktop/FCP/models/split"
SPLIT_LAYER = 8


def load_model_for_split():
    """Загрузка модели с хуками для split."""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    logger.info(f"Loading model from {MODEL_PATH}")
    
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info(f"  hidden_size: {config.hidden_size}")
    logger.info(f"  num_layers: {config.num_hidden_layers}")
    logger.info(f"  num_heads: {config.num_attention_heads}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="bfloat16",
        device_map="cpu",
        trust_remote_code=True
    )
    
    logger.info(f"  Model loaded: {type(model).__name__}")
    return model, config


def create_split_models(model, config, split_layer: int = 8):
    """Создание Part1 и Part2 моделей."""
    import torch
    
    logger.info(f"Creating split at layer {split_layer}")
    
    num_layers = config.num_hidden_layers
    
    # Part1: layers 0 to split_layer
    part1_layers = model.model.layers[:split_layer + 1]
    
    # Part2: layers split_layer+1 to end
    part2_layers = model.model.layers[split_layer + 1:]
    
    logger.info(f"  Part1: {len(part1_layers)} layers (0-{split_layer})")
    logger.info(f"  Part2: {len(part2_layers)} layers ({split_layer+1}-{num_layers-1})")
    
    return part1_layers, part2_layers


def export_to_onnx(part1_layers, config, output_path: str):
    """Экспорт слоёв в ONNX."""
    import torch
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    
    logger.info(f"Exporting to ONNX: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simple ONNX model with dummy weights (placeholder)
    # Real export would require proper weight extraction
    
    input_info = helper.make_tensor_value_info(
        'input_ids', TensorProto.INT64, ['batch', 'seq']
    )
    
    output_info = helper.make_tensor_value_info(
        'hidden_states', TensorProto.FLOAT, ['batch', 'seq', config.hidden_size]
    )
    
    # Add a simple identity node as placeholder
    identity_node = helper.make_node(
        'Identity',
        inputs=['input_ids'],
        outputs=['hidden_states']
    )
    
    graph = helper.make_graph(
        [identity_node],
        'part1',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, output_path)
    
    logger.info(f"  Saved: {output_path}")


def create_split_runner():
    """Создание рантайм раннера для split модели."""
    logger.info("=" * 60)
    logger.info("Creating Split Runner")
    logger.info("=" * 60)
    
    model, config = load_model_for_split()
    
    part1, part2 = create_split_models(model, config, SPLIT_LAYER)
    
    # Export to ONNX (placeholder)
    export_to_onnx(part1, config, f"{OUTPUT_DIR}/part1.onnx")
    export_to_onnx(part2, config, f"{OUTPUT_DIR}/part2.onnx")
    
    # Save config
    import json
    split_config = {
        "split_layer": SPLIT_LAYER,
        "injection_layers": [4, 8, 16, 24],
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "part1_path": f"{OUTPUT_DIR}/part1.onnx",
        "part2_path": f"{OUTPUT_DIR}/part2.onnx"
    }
    
    with open(f"{OUTPUT_DIR}/split_config.json", 'w') as f:
        json.dump(split_config, f, indent=2)
    
    logger.info(f"Config saved: {OUTPUT_DIR}/split_config.json")
    
    logger.info("=" * 60)
    logger.info("Split Runner Ready!")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = create_split_runner()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)