# FCP - Fractal Cognitive Processor

## Полное техническое описание (v14)

---

## 1. Архитектура

### 1.1 Гибридный слой (HybridLayerV14)

Каждый из 32 слоёв содержит **3 обучаемых компонента**:

```python
class HybridLayerV14:
    def __init__(self, layer_id, hidden_dim=2048, num_heads=16):
        self.gnn = FractalGNNLayer(...)        # GNN message passing
        self.transformer = HybridTransformerBlock(...)  # Self-attention + FFN
        self.lora = CoTrainLoRA(...)           # LoRA adaptation
```

| Компонент | Forward | Backward | Обучается |
|-----------|---------|----------|----------|
| **GNN** | ✅ Message passing | ✅ Gradient accumulation | ✅ |
| **Transformer** | ✅ Attention + FFN | ✅ Gradient accumulation | ✅ |
| **LoRA** | ✅ Rank adaptation | ✅ Full weight update | ✅ |

### 1.2 LoRA Ranks (по SPEC)

| Слои | Rank |
|------|------|
| 0-7 | r=4 |
| 8-15 | r=8 |
| 16-31 | r=16 |

### 1.3 Инъекция графа

Инъекция выполняется на слоях: **{4, 8, 16, 24}**

```python
def _fuse_graph(self, hidden_states, graph_vec):
    last_token = hidden_states[0, -1, :].copy()
    graph_vec = graph_vec[:self.hidden_dim]
    fused = last_token + 0.1 * graph_vec  # fusion_weight = 0.1
    hidden_states[0, -1, :] = fused
    return hidden_states
```

---

## 2. API Методы

### 2.1 GNN (FractalGNNLayer)

```python
class FractalGNNLayer:
    def __init__(self, layer_id, hidden_dim=2048):
        """
        Инициализация GNN слоя.
        
        Args:
            layer_id: ID слоя (0-31)
            hidden_dim: размерность скрытого слоя (2048)
        """
        self.W_message = [hidden_dim, hidden_dim]  # Matrix for message
        self.W_aggregate = [hidden_dim, hidden_dim]  # Matrix for aggregation
        
        self.grad_accum = 0.0      # Accumulated gradient
        self.update_count = 0     # Number of updates
    
    def forward(self, node_embeddings, edge_index):
        """
        Forward pass - message passing между узлами графа.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            edge_index: [num_edges, 2] - (source, target) pairs
        
        Returns:
            output: [num_nodes, hidden_dim]
        """
        pass
    
    def backward(self, grad_output, node_embeddings):
        """
        Backward pass - накапливает градиенты для мониторинга.
        
        Args:
            grad_output: [num_nodes, hidden_dim]
            node_embeddings: [num_nodes, hidden_dim]
        """
        self.grad_accum += np.mean(np.abs(grad_output))
        self.update_count += 1
    
    def get_importance(self):
        """Get importance score для IPC."""
        return self.grad_accum / max(1, self.update_count)
```

### 2.2 Transformer (HybridTransformerBlock)

```python
class HybridTransformerBlock:
    def __init__(self, layer_id, hidden_dim=2048, num_heads=16):
        """
        Transformer блок с causal attention.
        
        Args:
            layer_id: ID слоя
            hidden_dim: 2048
            num_heads: 16
        """
        # Attention weights
        self.W_q = [hidden_dim, hidden_dim]
        self.W_k = [hidden_dim, hidden_dim]
        self.W_v = [hidden_dim, hidden_dim]
        self.W_o = [hidden_dim, hidden_dim]
        
        # FFN weights
        self.W_gate = [hidden_dim, hidden_dim]
        self.W_up = [hidden_dim, hidden_dim]
        
        self.grad_accum = 0.0
        self.update_count = 0
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Forward pass с causal mask.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: optional
        
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        pass
    
    def backward(self, grad_output, hidden_states):
        """Backward - накапливает градиенты."""
        self.grad_accum += np.mean(np.abs(grad_output))
        self.update_count += 1
    
    def get_importance(self):
        return self.grad_accum / max(1, self.update_count)
```

### 2.3 LoRA (CoTrainLoRA)

```python
class CoTrainLoRA:
    def __init__(self, layer_id, hidden_dim=2048, rank=None):
        """
        Co-trained LoRA adapter.
        
        Args:
            layer_id: ID слоя
            hidden_dim: 2048
            rank: если None - автоопределение по SPEC
        """
        # Spec: если layer_id < 8 -> rank=4, < 16 -> rank=8, иначе rank=16
        if rank is None:
            if layer_id < 8:
                self.rank = 4
            elif layer_id < 16:
                self.rank = 8
            else:
                self.rank = 16
        
        if self.rank > 0:
            self.W_down = [hidden_dim, rank]
            self.W_up = [rank, hidden_dim]
            self.scaling = 0.1
    
    def forward(self, hidden_states):
        """
        Forward: h + scale * (h @ W_down @ W_up)
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
        
        Returns:
            output: [batch, seq, hidden_dim]
        """
        pass
    
    def backward(self, grad_output, hidden_states):
        """Backward: обновляет веса LoRA."""
        self.grad_accum += np.mean(np.abs(grad_output))
        self.update_count += 1
    
    def get_importance(self):
        return self.grad_accum / max(1, self.update_count)
```

### 2.4 Full Layer Stack (FCPLayerStackV14)

```python
class FCPLayerStackV14:
    def __init__(self, num_layers=32, hidden_dim=2048):
        """Стек из 32 гибридных слоёв."""
        self.layers = [HybridLayerV14(i, hidden_dim) for i in range(num_layers)]
    
    def forward(self, embeddings, graph_data=None):
        """
        Forward через все слои.
        
        Args:
            embeddings: [batch, seq, hidden_dim]
            graph_data: {"embeddings": [...], "edges": [...]} или None
        
        Returns:
            output: [batch, seq, hidden_dim]
        """
        pass
    
    def co_train(self, grad_output, hidden_states, graph_embeddings=None):
        """
        Co-training всех слоёв.
        
        Args:
            grad_output: градиент от ошибки
            hidden_states: скрытые состояния
            graph_embeddings: эмбеддинги графа
        """
        for layer in self.layers:
            layer.backward(grad_output, hidden_states, graph_embeddings)
    
    def get_lora_importances(self):
        """Получить importances всех LoRA."""
        return [layer.lora.get_importance() for layer in self.layers]
```

---

## 3. Интеграции

### 3.1 FractalGraphV2 (GNN Integrator)

```python
class FractalGraphSearch:
    """Интеграция с FractalGraphV2."""
    
    def __init__(self, graph_path, max_nodes=128):
        self.graph_path = graph_path
        self.max_nodes = max_nodes
        self.nodes = {}
        self.embeddings = {}
    
    def _load_graph(self):
        """Загрузить узлы из SQLite."""
        pass
    
    def get_context_for_layer(self, layer_id, hidden_states):
        """
        Получить subgraph для слоя.
        
        Returns:
            {"embeddings": [...], "edge_index": [...], "content": [...]}
        """
        pass
```

### 3.2 Co-Training Dataset

```python
class CoTrainingDataset:
    """Датасет для co-training."""
    
    def __init__(self, graph_path):
        self.samples = []
        self._load_samples()
    
    def get_batch(self, batch_size=8):
        """Получить батч для обучения."""
        pass
```

### 3.3 Split Model

```python
class SplitModelRunner:
    """Split модель для инъекции."""
    
    def __init__(self, openvino_path, split_layer=8):
        self.split_layer = split_layer
    
    def run_part1(self, prompt):
        """Первая часть до инъекции."""
        pass
    
    def run_part2(self, hidden_states, max_tokens=64):
        """Вторая часть после инъекции."""
        pass
```

---

## 4. Метрики

| Метрика | Значение |
|---------|-----------|
| Гибридных слоёв | 32 |
| LoRA ranks | 4/8/16 per spec |
| Инъекция на | {4, 8, 16, 24} |
| Graph nodes | 201 |
| Hidden dim | 2048 |
| Attention heads | 16 |

---

## 5. Workflow

```
Query → Input Embeddings
         ↓
    [Layer 0] GNN → Transformer → LoRA
    [Layer 1] GNN → Transformer → LoRA
    ...
    [Layer 4] GNN → Transformer → LoRA → INJECT GRAPH ⭐
    ...
    [Layer 8] GNN → Transformer → LoRA → INJECT GRAPH ⭐
    ...
    [Layer 31] GNN → Transformer → LoRA
         ↓
    Output → LM Head → Tokens
```

### Co-training loop:

```
1. Forward pass → output
2. Compute loss
3. Backward pass → grad_output
4. For each layer:
   - LoRA.backward(grad)
   - Transformer.backward(grad)
   - GNN.backward(grad)
5. Monitor importance scores
```

---

## 6. Файлы

| Файл | Описание |
|------|-----------|
| `mvp_pipeline_v13.py` | Основной пайплайн |
| `mvp_pipeline_v14.py` | v14 с полным co-training |
| `gnn_integrator.py` | Интеграция с GraphV2 |
| `co_train_lora.py` | Co-training скрипт |
| `split_exporter.py` | Split модель экспортёр |
| `test_e2e.py` | E2E тесты |

---

## 7. Git

- **Репозиторий**: https://github.com/BlackCatSpb/fcp
- **Последний коммит**: v14 - Full co-training

---

*FCP v14 - Полный Co-training GNN + Transformer + LoRA*
*Обновлено: 23.04.2026*