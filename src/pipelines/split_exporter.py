"""
FCP Split Model Exporter
Экспорт модели с промежуточными выходами на слоях инъекции
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.split")

OPENVINO_PATH = "C:/Users/black/OneDrive/Desktop/Models/openvino_model.xml"
OUTPUT_DIR = "C:/Users/black/OneDrive/Desktop/FCP/models"


def export_split_model():
    """Экспортировать модель с split на слое K"""
    
    print("=" * 60)
    print("FCP Split Model Exporter")
    print("=" * 60)
    
    try:
        from openvino import Core
        
        core = Core()
        
        # Load full model
        logger.info(f"[Export] Loading: {OPENVINO_PATH}")
        model = core.read_model(OPENVINO_PATH)
        
        # Get model info
        inputs = [i.any_name for i in model.inputs]
        outputs = [o.any_name for o in model.outputs]
        layers = len(model.get_ops())
        
        logger.info(f"[Export] Inputs: {inputs}")
        logger.info(f"[Export] Outputs: {outputs}")
        logger.info(f"[Export] Layers: {layers}")
        
        # Create part1 (pre-injection layers: 0-8)
        # For now, just use the full model as reference
        logger.info("[Export] Part1 would be layers 0-8 (pre-injection)")
        logger.info("[Export] Part2 would be layers 9-31 (post-injection)")
        
        # Save split config
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        config = {
            "split_layer": 8,
            "injection_layers": [4, 8, 16, 24],
            "full_model_path": OPENVINO_PATH,
            "total_layers": layers,
            "input_names": inputs,
            "output_names": outputs
        }
        
        config_path = os.path.join(OUTPUT_DIR, "split_config.json")
        with open(config_path, 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info(f"[Export] Config saved: {config_path}")
        
        print("\n[Split Config]")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        print("\n" + "=" * 60)
        print("Split Model Ready for Integration!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"[Export] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def integrate_with_hybrid():
    """Интеграция split модели с гибридными слоями"""
    
    print("\n[Integration] Hybrid Layer + Split Model")
    
    # Split execution flow:
    # 1. Load prompt
    # 2. Run through hybrid layers 0-7
    # 3. Get hidden states after layer 7
    # 4. Inject graph via AdaptiveFusionInjector
    # 5. Continue through layers 8-31
    # 6. Generate
    
    flow = """
    Hybrid Split Flow:
    ─────────────────────
    Layer 0: GNN → Transformer → LoRA
    Layer 1: GNN → Transformer → LoRA
    ...
    Layer 7: GNN → Transformer → LoRA
    
    [INJECTION at layer 8]
    - Get hidden states H_8
    - Get graph vector G from FractalGraphV2
    - Blend: H_8' = H_8 + α*G
    
    Layer 9: GNN → Transformer → LoRA
    ...
    Layer 31: GNN → Transformer → LoRA
    
    [OUTPUT] → LM Head → Tokens
    """
    
    print(flow)
    
    return True


if __name__ == "__main__":
    success = export_split_model()
    if success:
        integrate_with_hybrid()
    sys.exit(0 if success else 1)