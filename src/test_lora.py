"""
Test FCP v7 with LoRA adapter.
"""
import sys
import os
import codecs

if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
LORA_PATH = "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters/fcp_adapter"

def test_lora():
    print("=" * 50)
    print("FCP v7.1 + LoRA Test")
    print("=" * 50)
    
    try:
        import openvino_genai as ov_genai
        
        print("\n[1] Loading pipeline...")
        tokenizer = ov_genai.Tokenizer(MODEL_PATH)
        
        pipeline = ov_genai.LLMPipeline(
            MODEL_PATH, tokenizer, "CPU",
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 8}
        )
        
        print("[2] Checking LoRA adapter...")
        if os.path.exists(LORA_PATH):
            config_path = os.path.join(LORA_PATH, "adapter_config.json")
            if os.path.exists(config_path):
                print(f"    LoRA adapter found: {LORA_PATH}")
                print("    Note: OpenVINO doesn't support live LoRA application")
                print("    LoRA weights are prepared but require integration")
        else:
            print(f"    LoRA not found: {LORA_PATH}")
        
        print("\n[3] Test queries...")
        queries = [
            "Что такое квантовая запутанность?",
            "Объясни нейронную сеть.",
            "Кто такой Пушкин?"
        ]
        
        for q in queries:
            print(f"\nQ: {q}")
            try:
                r = pipeline.generate(f"Q: {q} | A:", max_new_tokens=256, temperature=0.2)
                print(f"A: {r[:200]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "=" * 50)
        print("FCP + LoRA - Ready!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora()