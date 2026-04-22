"""
FCP LoRA Training - Simple CPU version.
For machines without strong GPU.
"""
import sys
import os
import json

def main():
    print("=" * 60)
    print("FCP LoRA Setup")
    print("=" * 60)
    
    # Config
    output_dir = "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters/fcp_adapter"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset check
    dataset_path = "C:/Users/black/OneDrive/Desktop/FCP/data/lora_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Dataset: {len(dataset)} pairs")
    print(f"Output: {output_dir}")
    print()
    
    # Show config
    print("LoRA Config:")
    print("  rank: 8")
    print("  alpha: 8")
    print("  modules: q_proj, k_proj, v_proj")
    print()
    
    print("Training commands:")
    print("=" * 60)
    print()
    print("# Option 1: Google Colab (T4 GPU)")
    print("!pip install peft transformers trl")
    print("python src/train_lora.py")
    print()
    print("# Option 2: Local with strong GPU")
    print("pip install peft transformers bitsandbytes trl")
    print("python src/train_lora.py")
    print()
    print("=" * 60)
    
    # Create dummy adapter to show structure
    config = {
        "r": 8,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    config_path = os.path.join(output_dir, "adapter_config.json")
    import json as j
    with open(config_path, 'w') as f:
        j.dump(config, f, indent=2)
    
    print(f"Adapter config saved to: {config_path}")
    print()
    print("Ready for training!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())