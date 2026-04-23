# Real Split Implementation Guide

## Status

**BLOCKED**: Safetensors file corrupted
```
safetensors_rust.SafetensorError: Error while deserializing header: header too large
```

**Available Models:**
- GGUF: `C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf` (2.4GB, Q4 quantization)
- Safetensors: `C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_hf/model.safetensors` (corrupted)

## Solution: Convert GGUF to PyTorch

### Option 1: llama.cpp (recommended)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=OFF
cmake --build . --config Release

# Convert GGUF to PyTorch
./build/bin/llama-ggufv2-to-pt C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf \
    C:/Users/black/OneDrive/Desktop/Models/qwen3_4b_pt/model.pt
```

### Option 2: re-sell with other tools

```bash
pip install gguf
python -c "
from gguf import GGUFReader
reader = GGUFReader('C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf')
# Export to numpy arrays
"
```

### Option 3: Re-download model

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B
```

## Implementation When Model Available

```python
# src/pipelines/split_exporter_real.py (updated)

import torch
from transformers import AutoModelForCausalLM

def create_hooked_model(model_path: str, split_layers: list[int]):
    """Создание модели с хуками на-split слоях."""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    hidden_states_cache = {}
    
    for layer_idx in split_layers:
        layer = model.model.layers[layer_idx]
        
        def hook(module, input, output, idx=layer_idx):
            hidden_states_cache[idx] = output[0].detach().clone()
        
        layer.register_forward_hook(hook)
    
    return model, hidden_states_cache


def extract_layer(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Извлечение состояний на конкретном слое."""
    # This would use intermediate model outputs
    pass


def export_split():
    """Экспорт разделенной модели."""
    
    model, cache = create_hooked_model(
        "path/to/model",
        split_layers=[4, 8, 16, 24]
    )
    
    # Export to OpenVINO
    from optimum.exporters.openvino import main_export
    
    main_export(
        model=model,
        output="C:/Users/black/OneDrive/Desktop/FCP/models/split",
        task="text-generation",
        split_layers=[4, 8, 16, 24]
    )
```

## Current Workaround

Until model is available, use full model with **Graph Injection** as fallback:

```python
def run_with_graph_injection(prompt: str, graph_search):
    """Full model inference with graph context injection."""
    
    # Get graph vector
    graph_vec = graph_search.extract(prompt, k=10)
    
    # Concatenate with prompt as context
    enhanced_prompt = f"Context: {graph_vec}\n\nQuestion: {prompt}"
    
    # Run full model
    response = full_model(enhanced_prompt)
    
    return response
```

## Next Action

1. Download new PyTorch model OR convert GGUF
2. Run split_exporter_real.py
3. Integrate with FractalGraphSearch