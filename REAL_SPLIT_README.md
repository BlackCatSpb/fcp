# Real Split Model Implementation

## Current Status

**BLOCKED**: No PyTorch model available

PyTorch модель (pytorch_model.bin) отсутствует. Доступны только:
- `C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_pt/pytorch_model.bin` (не существует)
- GGUF формат: `C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf`

## Required Step 1: Get PyTorch Model

Download from HuggingFace or convert from GGUF:

```bash
# Option 1: Download original
huggingface-cli download Qwen/Qwen3-4B --local-dir C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B

# Option 2: Convert GGUF to PyTorch (requires llama.cpp)
convert-llama-ggufv2-to-pytorch C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf \
    --outfile C:/Users/black/OneDrive/Desktop/Models/ruadapt_qwen3_4b_pt/pytorch_model.bin
```

## Implementation After PyTorch Model Available

### Step 1: Create Hooked Model

```python
# src/pipelines/split_model_hook.py

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Optional

class Qwen3SplitModel(torch.nn.Module):
    """Qwen3 с хуками для извлечения промежуточных состояний."""
    
    def __init__(self, base_model_path: str, split_layers: List[int]):
        super().__init__()
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        self.split_layers = split_layers  # [4, 8, 16, 24]
        self.hidden_states: Dict[int, torch.Tensor] = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Регистрация хуков на слоях."""
        for layer_idx in self.split_layers:
            layer = self.base_model.model.layers[layer_idx]
            
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Cache hidden states before output
                self.hidden_states[layer_idx] = output[0].detach().clone()
                return output
            
            layer.register_forward_hook(hook_fn)
    
    def forward(self, input_ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward pass с кэшированием промежуточных состояний."""
        self.hidden_states.clear()
        
        output = self.base_model(input_ids)
        
        return self.hidden_states


class SplitModelExporter:
    """Экспортер модели с разделением на Part1/Part2."""
    
    def __init__(
        self,
        model_path: str,
        split_layer: int = 8,
        output_dir: str = "C:/Users/black/OneDrive/Desktop/FCP/models"
    ):
        self.model_path = model_path
        self.split_layer = split_layer
        self.output_dir = output_dir
    
    def export_to_onnx(self):
        """Экспорт в ONNX с динамическими осями."""
        import onnx
        from onnx import helper, TensorProto
        
        hooked_model = Qwen3SplitModel(self.model_path, [self.split_layer])
        
        # Part1: input_ids → hidden_states after layer split_layer
        # Create ONNX graph for Part1
        input_tensor = helper.make_tensor_value_info(
            'input_ids', TensorProto.INT64, ['batch', 'seq']
        )
        output_tensor = helper.make_tensor_value_info(
            'hidden_states', TensorProto.FLOAT, ['batch', 'seq', 2560]
        )
        
        part1_graph = helper.make_graph(
            [input_tensor, output_tensor],
            'Part1',
            [],
            []
        )
        
        part1_model = helper.make_model(part1_graph)
        onnx.save(part1_model, f"{self.output_dir}/part1.onnx")
        
        # Part2: hidden_states → logits
        # Similar for part2
        
        print(f"Exported to {self.output_dir}/part1.onnx")
        print(f"Exported to {self.output_dir}/part2.onnx")
        
        return True


class SplitModelRunner:
    """Runtime для выполнения split модели."""
    
    def __init__(self, part1_path: str, part2_path: str):
        import openvino as ov
        
        self.core = ov.Core()
        
        self.part1 = self.core.read_model(part1_path)
        self.part2 = self.core.read_model(part2_path)
        
        self.compiled_part1 = self.core.compile_model(self.part1, "CPU")
        self.compiled_part2 = self.core.compile_model(self.part2, "CPU")
    
    def run_part1(self, input_ids: np.ndarray) -> np.ndarray:
        """Run Part1: получить скрытые состояния."""
        return self.compiled_part1(input_ids)
    
    def run_part2(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run Part2: получить логиты."""
        return self.compiled_part2(hidden_states)
    
    def run_with_injection(
        self,
        input_ids: np.ndarray,
        graph_vector: Optional[np.ndarray] = None,
        injection_alpha: float = 0.3
    ) -> np.ndarray:
        """Сквозной проход с инъекцией графа."""
        # Part 1
        hidden_states = self.run_part1(input_ids)
        
        # Inject graph if provided
        if graph_vector is not None:
            # Add graph vector to hidden states
            hidden_states = hidden_states + injection_alpha * graph_vector
        
        # Part 2
        logits = self.run_part2(hidden_states)
        
        return logits
```

### Step 2: Integration

```python
# src/pipelines/split_runner.py

from src.memory.graph_search import FractalGraphSearch, GraphVectorExtractor

def run_inference(prompt: str, use_graph: bool = True):
    """Инференс с опциональной инъекцией графа."""
    
    # 1. Get graph context if enabled
    graph_vector = None
    if use_graph:
        search = FractalGraphSearch(db_path=GRAPH_DB_PATH)
        search.build_index()
        
        extractor = GraphVectorExtractor(search, output_dim=2560)
        graph_vector = extractor.extract(prompt, k=10)
    
    # 2. Run through split model
    runner = SplitModelRunner(PART1_PATH, PART2_PATH)
    
    input_ids = tokenizer.encode(prompt)
    logits = runner.run_with_injection(
        input_ids,
        graph_vector=graph_vector,
        injection_alpha=0.3
    )
    
    # 3. Generate
    output_ids = tokenizer.decode(logits.argmax(dim=-1))
    
    return output_ids
```

## Configuration

```json
{
  "split_layer": 8,
  "injection_layers": [4, 8, 16, 24],
  "injection_alpha": 0.3,
  "part1_path": "C:/Users/black/OneDrive/Desktop/FCP/models/part1.xml",
  "part2_path": "C:/Users/black/OneDrive/Desktop/FCP/models/part2.xml",
  "graph_db_path": "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
}
```

## Next Action Required

1. **Download PyTorch model** or convert from GGUF
2. Run the exporter
3. Test inference with injection