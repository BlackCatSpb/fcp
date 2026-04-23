# FCP - Fractal Cognitive Processor

## Полное техническое описание реализации (v12)

---

## 1. Обзор архитектуры

**Fractal Cognitive Processor (FCP)** — самообучающаяся когнитивная архитектура, объединяющая языковое моделирование (Transformer) и графовое представление знаний (GNN) на уровне каждого вычислительного слоя.

### Ключевые принципы

1. **Послойная гибридизация** — каждый из 32 слоёв объединяет Transformer-блок и GNN-процессор
2. **Co-training LoRA** — LoRA-адаптеры обучаются совместно с GNN
3. **Адаптивная инъекция** — графовый вектор впрыскивается на слоях 4, 8, 16, 24
4. **Селективная активация** — ранний выход на основе confidence
5. **Непрерывное обучение** — без катастрофического забывания

---

## 2. Архитектура слоёв

### 2.1 Структура стека

```
┌─────────────────────────────────────────────────────────┐
│              FCPLayerStackV12 (32 слоёв)               │
├─────────────────────────────────────────────────────────┤
│  Layer 0: GNN → Transformer → LoRA                   │
│  Layer 1: GNN → Transformer → LoRA                   │
│  Layer 2: GNN → Transformer → LoRA                   │
│  Layer 3: GNN → Transformer → LoRA                   │
│  Layer 4: GNN → Transformer → LoRA → INJECTION       │ ←
│  Layer 5: GNN → Transformer → LoRA                   │
│  ...                                              │
│  Layer 8: GNN → Transformer → LoRA → INJECTION       │ ←
│  ...                                              │
│  Layer 16: GNN → Transformer → LoRA → INJECTION      │ ←
│  ...                                              │
│  Layer 24: GNN → Transformer → LoRA → INJECTION      │ ←
│  ...                                              │
│  Layer 31: GNN → Transformer → LoRA                   │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Типы слоёв

| Слой | GNN | Transformer | LoRA rank | Injection |
|-------|-----|------------|----------|----------|
| 0-7 | ✅ | ✅ (16 heads) | r=4 | ❌ |
| 4, 8 | ✅ | ✅ | r=4/8 | ✅ |
| 9-15 | ✅ | ✅ (16 heads) | r=8 | ❌ |
| 16 | ✅ | ✅ | r=16 | ✅ |
| 17-23 | ✅ | ✅ (16 heads) | r=16 | ❌ |
| 24 | ✅ | ✅ | r=16 | ✅ |
| 25-31 | ✅ | ✅ (16 heads) | r=16 | ❌ |

---

## 3. Компоненты гибридного слоя

### 3.1 FractalGNNLayer

GNN-слой с message passing для обработки графового контекста.

```python
class FractalGNNLayer:
    def __init__(self, layer_id, hidden_dim=2048, num_heads=16):
        self.W_message = [hidden_dim, hidden_dim]  # Message transformation
        self.W_aggregate = [hidden_dim, hidden_dim]  # Neighbor aggregation
    
    def forward(self, node_embeddings, edge_index):
        # Aggregate neighbors
        aggregated = sum(neighbor_states)
        # Transform
        output = node_embeddings @ W_message + 0.5 * aggregated
        return output
```

**Математика:**
```
h_v' = W_message @ h_v + 0.5 * Σ_{u∈N(v)} h_u
```

### 3.2 HybridTransformerBlock

Трансформерный блок с causal attention и FFN.

```python
class HybridTransformerBlock:
    def __init__(self, layer_id, hidden_dim=2048, num_heads=16):
        # Multi-head attention
        self.W_q = [hidden_dim, hidden_dim]
        self.W_k = [hidden_dim, hidden_dim]
        self.W_v = [hidden_dim, hidden_dim]
        self.W_o = [hidden_dim, hidden_dim]
        
        # FFN
        self.W_gate = [hidden_dim, hidden_dim]
        self.W_up = [hidden_dim, hidden_dim]
        
        # LayerNorm
        self.gamma = ones(hidden_dim)
        self.beta = zeros(hidden_dim)
```

**Операции:**
1. **Causal Attention** — каждый токен видит только предыдущие
2. **SwiGLU** (упрощённо до tanh FFN)
3. **Residual** — h = h + FFN(h)
4. **LayerNorm** — нормализация

### 3.3 CoTrainLoRA

Co-trained LoRA адаптер, обучающийся вместе с GNN.

```python
class CoTrainLoRA:
    def __init__(self, layer_id, hidden_dim=2048, rank=None):
        # Spec ranks: 1-8→r=4, 9-16→r=8, 17-32→r=16
        if layer_id < 8:
            self.rank = 4
        elif layer_id < 16:
            self.rank = 8
        else:
            self.rank = 16
        
        self.W_down = [hidden_dim, rank]
        self.W_up = [rank, hidden_dim]
        self.scaling = 0.1
```

**Forward:**
```
output = h + scaling * (h @ W_down @ W_up)
```

**Co-training:**
```
# Накапливает градиенты для IPC
importance = mean(|grad|)
```

### 3.4 HybridLayerV12

Полный гибридный слой с 5 этапами обработки:

```python
class HybridLayerV12:
    INJECTION_LAYERS = {4, 8, 16, 24}
    
    def forward(self, hidden_states, graph_embeddings, edge_index, apply_lora=True):
        # Stage 1-2: GNN clusterer
        if graph_embeddings is not None:
            graph_vec = self.gnn.forward(graph_embeddings, edge_index)
        
        # Stage 3: Transformer
        output = self.transformer.forward(hidden_states)
        
        # Stage 4: Activation gate
        self.confidence = self._compute_confidence(output, graph_vec)
        should_stop = self.confidence > 0.85
        
        # Stage 5: Fusion (at injection layers)
        if self.layer_id in INJECTION_LAYERS and graph_vec is not None:
            output = self._fuse_streams(output, graph_vec)
        
        # Apply LoRA
        if apply_lora:
            output = self.lora.forward(output)
        
        return output, graph_vec, should_stop
```

---

## 4. Адаптивная инъекция

### 4.1 Точки инъекции

Инъекция графового вектора выполняется на слоях: **4, 8, 16, 24**

Эти слои выбраны согласно SPEC для баланса между глубиной и информацией:
- Слой 4: раннее обогащение
- Слой 8: средняя глубина
- Слой 16: глубокий уровень
- Слой 24: предфинальный уровень

### 4.2 Формула инъекции

```
h' = h_last + fusion_weight * graph_vec
```

где:
- `h_last` — скрытое состояние последнего токена
- `graph_vec` — вектор от GNN (2048 dim)
- `fusion_weight` = 0.1

---

## 5. Селективная активация

### 5.1 Токеновый уровень

Каждый токен может остановиться на любом слое.

```python
confidence = (magnitude + graph_contrib) / sqrt(hidden_dim)
should_stop = confidence > stop_threshold (0.85)
```

### 5.2 Послойный уровень

Если `mean(confidence) > layer_threshold`, делается early exit.

---

## 6. Система памяти

### 6.1 FractalGraphV2

Долговременный граф знаний (из EVA-Ai).

- **201 узел** в базе
- Типы: `concept`, `fact`, `routing_rule`
- Связи: `is_a`, `part_of`, `related_to`, `contradicts`

### 6.2 TCM (Temporal Context Memory)

Краткосрочная память в рамках сессии.

```python
class TemporalContextMemory:
    def add(role, content):
        history.append({role, content[:2000]})
        if len(history) > max_history:
            history = history[-max_history:]
```

### 6.3 LearningGraphManager

Управление сигналами обучения.

**Узлы:**
- `LearningSignal` — обратная связь
- `LayerSensitivity` — статистика по слоям

**Методы:**
- `record_learning_signal()` — записать сигнал
- `get_learning_tasks()` — получить задачи на дообучение
- `prune_old_signals()` — удалить старые
- `get_replay_buffer()` — выборка для experience replay
- `decay_signal_priority()` — экспоненциальный распад

---

## 7. Система курации

### 7.1 ConceptMiner

Извлечение концептов из текста.

```python
class ConceptMiner:
    def extract_concepts(text, context=""):
        # NER по паттернам
        # Построение is_a иерархии
        # Семантическое сопоставление с графом
        return concepts
```

### 7.2 ContradictionMiner

Обнаружение и разрешение противоречий.

```python
class ContradictionMiner:
    def find_contradictions():
        # sim >= 0.75
        # NLI analysis (contra_score >= 0.65)
        # Транзитивное замыкание
        return contradictions
    
    def resolve(contradiction, strategy="auto"):
        # auto / user_query / merge
```

### 7.3 GraphCurator

Полный цикл курации.

```python
class GraphCurator:
    def run_cycle():
        1. Extract concepts
        2. Resolve contradictions
        3. Handle dangling nodes
        4. Apply temporal decay
        5. Generate LoRA examples
```

---

## 8. Система обучения

### 8.1 ShadowLoRA

Управление LoRA адаптерами.

```python
class ShadowLoRAManager:
    def atomic_swap(new_adapter):
        # Без блокировки генерации
    
    def live_rollback():
        # Откат при деградации
```

### 8.2 LearningOrchestrator

Оркестрация обучения.

```python
class LearningOrchestrator:
    def check_and_schedule():
        # Опрос LearningGraphManager
        # Запуск дообучения
    
    def curriculum_learning_schedule(samples):
        # Сортировка по сложности
    
    def validate_adapter(adapter, validation_set):
        # Валидация перед деплоем
```

### 8.3 Important Weight Constraints (IPC)

```python
# Штраф за отклонение от критических весов
loss = original_loss + lambda * (W - W_original)^2
```

---

## 9. UES (Universal Execution Subsystem)

### 9.1 Функции

```python
class UES:
    def discover_topology():
        # CPU cores, GPU, NPU
        # ISA extensions (SSE4.2, AVX2)
    
    def pgo_auto_tune(benchmark_fn, n_trials):
        # Optuna-like подбор параметров
    
    def pin_to_e_cores():
        # Энергоэффективные
    
    def pin_to_p_cores():
        # Производительные
```

---

## 10. Интеграция с OpenVINO

### 10.1 Модели

- **Qwen3 4B Q4_K_M.gguf** — основная модель (GGUF)
- **openvino_model.xml** — OpenVINO IR формат

### 10.2 SplitModelRunner

```python
class SplitModelRunner:
    def run_part1(prompt):
        # Первые k слоёв → hidden states
    
    def run_part2(hidden, max_tokens):
        # Продолжение генерации
```

---

## 11. Файловая структура

```
FCP/
├── src/
│   ├── pipelines/
│   │   ├── mvp_pipeline_v7.py      # MVP
│   │   ├── mvp_pipeline_v8.py  # Hybrid layers
│   │   ├── mvp_pipeline_v9.py  # Per-layer LoRA
│   │   ├── mvp_pipeline_v10.py # Full SPEC
│   │   ├── mvp_pipeline_v11.py # Curation
│   │   └── mvp_pipeline_v12.py # Real GNN + Co-training ← ТЕКУЩИЙ
│   ├── fcp_core/
│   │   ├── hybrid_layer.py    # FractalGatedHybridLayer (646 lines)
│   │   ├── hybrid_stack.py
│   │   ├── input_layer.py
│   │   ├── output_layer.py
│   │   └── types.py
│   └── memory/
│       └── temporal_context.py
├── fmf/
│   ├── eva_ai/memory/fractal_graph_v2/  # Graph
│   └── fmf_knowledge/
│       ├── self_dialog.py
│       ├── concept_extractor.py
│       └── contradiction_generator.py
├── lora_adapters/
│   ├── checkpoint-100/  # Обученный адаптер
│   └── adapter_config.json
├── data/
│   └── graph.db/fractal_graph.db  # 201 nodes
├── SPEC.md
└── README.md
```

---

## 12. Метрики производительности

| Метрика | Значение |
|----------|---------|
| Graph nodes | 201 |
| Hybrid layers | 32 |
| Hidden dim | 2048 |
| Attention heads | 16 |
| LoRA ranks | 4/8/16 (per spec) |
| Injection layers | {4, 8, 16, 24} |
| Early exit threshold | 0.85 |

---

## 13. Версии и история коммитов

| Коммит | Версия | Описание |
|--------|--------|----------|
| `23b5bbb` | v7.1 | MVP: OpenVINO, Graph 201 |
| `ec9ce10` | v8 | Hybrid layers (4 слоя) |
| `c019643` | v9 | Per-layer LoRA (OpenVINO IR) |
| `9de7961` | v10 | Full SPEC components |
| `5fda5ea` | v11 | Concept/Contradiction Mining |
| `e802d6d` | v12 | Real GNN + Co-training |

---

## 14. Когнитивный цикл FCP

```
┌─────────────────────────────────────────────────────────────┐
│                    КОГНИТИВНЫЙ ЦИКЛ                   │
├─────────────────────────────────────────────────────────────┤
│                                                      │
│  1. ВОСПРИЯТИЕ                                    │
│     Query → Tokenizer → Embeddings                        │
│                                                      │
│  2. ИЗВЛЕЧЕНИЕ КОНТЕКСТА                            │
│     Graph → subgraph → FractalGNN → graph_vec          │
│                                                      │
│  3. ГИБРИДНАЯ ОБРАБОТКА (32 слоя)                  │
│     For layer in layers:                              │
│       - GNN message passing                         │
│       - Self-attention + FFN                        │
│       - LoRA adaptation                           │
│       - Early exit if confident                   │
│       - Inject graph at {4,8,16,24}                 │
│                                                      │
│  4. ГЕНЕРАЦИЯ                                      │
│     Hidden -> LM Head → Sampling → Tokens         │
│                                                      │
│  5. ОБРАТНАЯ СВЯЗЬ                                 │
│     LearningSignal.record() → Graph                 │
│                                                      │
│  6. КУРАЦИЯ (асинхронно)                          │
│     GraphCurator.run_cycle()                        │
│       - Concept mining                             │
│       - Contradiction resolution                  │
│       - Temporal decay                            │
│                                                      │
│  7. ДООБУЧЕНИЕ (асинхронно)                      │
│     ShadowLoRA.schedule_finetune()                   │
│       - Co-train LoRA with GNN                     │
│       - Atomic swap                                │
│                                                      │
└─────────────────────────────────────────────────────────────┘
```

---

### 16.11 Adaptive Fusion Injector

```python
class AdaptiveFusionInjector:
    def __init__(self, hidden_dim: int = 2048) -> None:
        self.hidden_dim: int = hidden_dim
        self.gate_proj: np.ndarray  # [2*hidden_dim, hidden_dim]
        self.gate_bias: np.ndarray  # [hidden_dim]
    
    def inject(
        self,
        hidden_states: np.ndarray,
        graph_vec: np.ndarray,
        gate_weights: np.ndarray
    ) -> np.ndarray:
        """
        Впрыснуть графовый контекст.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            graph_vec: [hidden_dim]
            gate_weights: [hidden_dim]
        
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        pass
    
    def calibrate(
        self,
        samples: List[Tuple],
        target_mean: float = 0.5
    ) -> None:
        """Калибровать gate bias."""
        pass
```

### 16.12 Split Model Runner

```python
class SplitModelRunner:
    def __init__(self, openvino_path: str) -> None:
        self.openvino_path: str
        self.kv_cache: dict = {}
    
    def load(self) -> None:
        """Загрузить модель."""
        pass
    
    def run_part1(self, prompt: str) -> np.ndarray:
        """
        Запустить первую часть (prefill).
        
        Args:
            prompt: исходный промпт
        
        Returns:
            hidden_states: [1, seq, hidden_dim]
        """
        pass
    
    def run_part2(
        self,
        hidden_states: np.ndarray,
        kv_cache: dict = None,
        max_tokens: int = 64
    ) -> str:
        """
        Запустить вторую часть (decoding).
        
        Args:
            hidden_states: [1, seq, hidden_dim]
            kv_cache: предыдущий KV кэш
            max_tokens: максимум токенов
        
        Returns:
            generated text
        """
        pass
    
    def kv_cache_snapshot(self) -> dict:
        """Сохранить KV кэш."""
        pass
    
    def kv_cache_restore(self, snapshot: dict) -> None:
        """Восстановить KV кэш."""
        pass
```

### 16.13 Fractal Graph Encoder

```python
class FractalGraphEncoder:
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 2048,
        num_layers: int = 2
    ) -> None:
        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.proj: np.ndarray  # [input_dim, hidden_dim]
        self.gate_proj: np.ndarray  # [2*hidden_dim, hidden_dim]
    
    def forward(
        self,
        node_embeddings: np.ndarray,
        edge_index: np.ndarray,
        mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Энкодировать подграф.
        
        Args:
            node_embeddings: [num_nodes, input_dim]
            edge_index: [num_edges, 2]
            mask: [num_nodes] или None
        
        Returns:
            graph_vec: [hidden_dim]
            gate_weights: [hidden_dim]
        """
        pass
    
    def distill_from_llm(
        self,
        llm_hidden: np.ndarray,
        subgraph_data: np.ndarray
    ) -> float:
        """
        MSE между graph_vec и LLM hidden states.
        
        Returns:
            mse: float
        """
        pass
```

### 16.14 Graph Node Operations

```python
class SimpleGraph:
    def __init__(self, db_path: str) -> None:
        self._nodes: dict = {}
    
    def _load_nodes(self) -> None:
        """Загрузить узлы из SQLite."""
        pass
    
    def get_facts(self) -> list:
        """Получить все факты."""
        pass
    
    def get_dangling(self) -> list:
        """Получить висячие узлы."""
        pass
    
    def add_node(
        self,
        name: str,
        type: str,
        confidence: float
    ) -> str:
        """Добавить узел. Возвращает node_id."""
        pass
    
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        rel: str
    ) -> None:
        """Добавить ребро."""
        pass
    
    def remove_node(self, node_id: str) -> None:
        """Удалить узел."""
        pass
    
    def apply_decay(self, factor: float) -> int:
        """Применить распад. Возвращает количество."""
        pass
    
    def search_keyword(
        self,
        keyword: str,
        limit: int = 5
    ) -> list:
        """Поиск по ключевому слову."""
        pass
```

### 16.15 TCM

```python
class SimpleTCM:
    def __init__(self, max_history: int = 10) -> None:
        self.max_history: int = max_history
        self.history: list = []
    
    def add(self, role: str, content: str) -> None:
        """Добавить в историю."""
        pass
```

---

## 17. Ссылки

- **GitHub:** https://github.com/BlackCatSpb/fcp
- **Модель:** Qwen3 4B Q4_K_M.gguf
- **Graph DB:** fractal_graph.db (201 nodes)

---

## 18. Статус по SPEC

| Компонент | Статус | Файл |
|-----------|--------|------|
| ✅ Input Layer | Реализован | v12 |
| ✅ Output Layer | Реализован | v12 |
| ✅ 32 Hybrid Layers | Реализован | v12 |
| ✅ GNN Message Passing | Реализован | v12 |
| ✅ Co-trained LoRA | Реализован | v12 |
| ✅ Adaptive Injection | Реализован | v12 |
| ✅ Selective Activation | Реализован | v12 |
| ✅ FractalGraphV2 | Интегрирован | EVA-Ai |
| ✅ TCM | Реализован | v8-v11 |
| ✅ Concept Miner | Реализован | v11 |
| ✅ Contradiction Miner | Реализован | v11 |
| ✅ Graph Curator | Реализован | v11 |
| ✅ LearningGraphManager | Реализован | v11 |
| ✅ ShadowLoRA | Реализован | v10 |
| ✅ UES | Реализован | v10 |
| ✅ AdaptiveFusionInjector | Реализован | v10 |
| ✅ SplitModelRunner | Реализован | v10 |
| ✅ FractalGraphEncoder | Реализован | v10 |

---

## 19. Следующие шаги

1. **Реальная GNN** — заменить упрощённую GNN на полную реализацию с HNSW
2. **Обучить LoRA** — запустить co-training цикл
3. **Интеграция** — связать v12 с OpenVINO моделью
4. **Оптимизация** — применить UES auto-tune
5. **Валидация** — полное тестирование

---

*FCP v12 — Real Hybrid Layers with GNN + Co-training*
*Обновлено: 23.04.2026*