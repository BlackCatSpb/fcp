# FCP Development Plan - Based on "Последовательные решения.txt"

**Generated:** 25.04.2026  
**Source:** `C:\Users\black\OneDrive\Desktop\Последовательные решения.txt`

---

## Overview

FCP v15 требует реализацию ~25 компонентов в 4 фазах:

| Phase | Components | Status |
|-------|-----------|--------|
| 1: Core | Data Loaders, AdaLoRA, GraphEncoder, Co-training | [P] In Progress |
| 2: OpenVINO | GNN→OV, Injector, LoRA Manager, Quantization | [ ] |
| 3: Advanced | Cache Evictor, Clarification, TCM, Multi-Agent | [ ] |
| 4: Integration | Toolformer, Thinking, Attribution, MVP Pipeline | [ ] |

---

## PHASE 1: Core Components

### 1.1 Data Loaders
**File:** `fcp/data/ru_data_loaders.py`

| Class | Description | Priority |
|-------|-------------|----------|
| ConceptNetLoader | Загрузка русского ConceptNet | HIGH |
| RuBQLoader | Загрузка RuBQ 2.0 | HIGH |
| SaigaLoader | Загрузка Saiga2_70b dataset | HIGH |
| NERELLoader | Загрузка NEREL | MEDIUM |

### 1.2 AdaLoRA
**File:** `fcp_core/adaptive_lora.py`

| Method | Description | Status |
|--------|-------------|--------|
| AdaLoRALayer | Адаптивный LoRA слой с dynamic rank | [P] |
| adapt_rank() | Изменение rank во время инференса | [P] |

### 1.3 FractalGraphEncoder
**File:** `fcp/gnn/graph_encoder.py`

| Component | Description | Status |
|---------|-------------|--------|
| SAGEConv layers | Graph convolution (input→hidden→output) | [P] |
| HNSW index | Semantic search | [P] |
| retrieve_subgraph() | Получение подграфа по запросу | [P] |
| build_hnsw_index() | Построение индекса | [ ] |

### 1.4 Co-Training Pipeline
**File:** `colab/train_full_pipeline.py`

| Step | Description | Status |
|------|-------------|--------|
| Data loading | ConceptNet + RuBQ + Saiga | [P] |
| GNN training | Обучение GraphEncoder | [ ] |
| LoRA training | AdaLoRA на Saiga | [ ] |

---

## PHASE 2: OpenVINO Integration

### 2.1 GNN Conversion
**File:** `fcp/gnn/convert_gnn_to_ov.py`

| Method | Description | Status |
|--------|-------------|--------|
| convert_gnn_to_ov() | PyTorch → ONNX → OpenVINO | [ ] |

### 2.2 GNN Runtime
**File:** `fcp/gnn/gnn_runtime_ov.py`

| Class | Description | Status |
|-------|-------------|--------|
| GNNEncoderOV | OpenVINO wrapper | [ ] |
| encode() | Инференс | [ ] |

### 2.3 AdaptiveFusionInjector
**File:** `fcp/gnn/injector.py`

| Method | Description | Status |
|--------|-------------|--------|
| inject() | Инъекция graph_vec в hidden_states | [ ] |

### 2.4 LoRA Manager
**File:** `fcp/lora/shadow_lora_ov.py`

| Class | Description | Status |
|-------|-------------|--------|
| ShadowLoRAManagerOV | Управление адаптерами | [ ] |
| atomic_swap() | Атомарная смена адаптера | [ ] |

### 2.5 Quantization
**File:** `fcp/quantization/quantize_activations.py`

| Method | Description | Status |
|--------|-------------|--------|
| quantize_activations() | NNCF квантование | [ ] |
| KV-cache precision | U8 сжатие | [ ] |

### 2.6 Speculative Decoding
**File:** `fcp/pipelines/speculative.py`

| Component | Description | Status |
|---------|-------------|--------|
| Draft model | 8-слойная модель | [ ] |
| SpeculativeController | Управление | [ ] |

---

## PHASE 3: Advanced Features

### 3.1 Semantic Cache Evictor
**File:** `fcp/semantic_cache/evictor.py`

| Method | Description | Status |
|--------|-------------|--------|
| _token_importance() | Оценка важности токена | [ ] |
| select_blocks_to_evict() | Выбор блоков для вытеснения | [ ] |

### 3.2 Clarification Generator
**File:** `fcp/active_learning/clarification.py`

| Class | Description | Status |
|-------|-------------|--------|
| ClarificationGenerator | Генерация уточняющих вопросов | [ ] |

### 3.3 ScenarioTCM
**File:** `fcp/memory/scenario_tcm.py`

| Class | Description | Status |
|-------|-------------|--------|
| ScenarioTCM | Эпизодическая память | [ ] |
| add_turn() | Добавление хода диалога | [ ] |
| _save_chain() | Сохранение сценария | [ ] |

### 3.4 Expert System
**File:** `fcp/multi_agent/expert_system.py`

| Class | Description | Status |
|-------|-------------|--------|
| ExpertSystem | Мультиагентное обсуждение | [ ] |
| discuss() | Параллельная генерация экспертами | [ ] |

---

## PHASE 4: Integration

### 4.1 Toolformer
**File:** `fcp/tools/orchestrator.py`

| Class | Description | Status |
|-------|-------------|--------|
| ToolOrchestrator | Оркестрация инструментов | [ ] |
| CalculatorTool | Калькулятор | [ ] |
| WebSearchTool | Веб-поиск | [ ] |

### 4.2 Thinking Controller
**File:** `fcp/thinking_controller.py`

| Class | Description | Status |
|-------|-------------|--------|
| ThinkingController | Управление режимом мышления | [ ] |
| should_enable_thinking() | Решение о включении | [ ] |
| build_chat_prompt() | Построение промпта | [ ] |

### 4.3 Attribution
**File:** `fcp/explain/attribution.py`

| Class | Description | Status |
|-------|-------------|--------|
| AttributionReport | Отслеживание атрибуции | [ ] |
| track() | Запись активаций | [ ] |
| explain() | Генерация объяснения | [ ] |

### 4.4 MVP Pipeline
**File:** `fcp/pipelines/mvp_pipeline_v15.py`

| Class | Description | Status |
|-------|-------------|--------|
| FCPPipelineV15 | Полный пайплайн | [ ] |
| generate() | Генерация с всеми компонентами | [ ] |

---

## File Structure

```
FCP/
├── fcp/
│   ├── data/
│   │   └── ru_data_loaders.py      # [P] Data loaders
│   ├── gnn/
│   │   ├── graph_encoder.py       # [P] FractalGraphEncoder
│   │   ├── convert_gnn_to_ov.py  # [ ] GNN→OV
│   │   ├── gnn_runtime_ov.py     # [ ] GNN Runtime
│   │   └── injector.py          # [ ] AdaptiveFusionInjector
│   ├── lora/
│   │   └── shadow_lora_ov.py    # [ ] LoRA Manager
│   ├── quantization/
│   │   └── quantize_activations.py  # [ ] Quantization
│   ├── semantic_cache/
│   │   └── evictor.py          # [ ] Cache Evictor
│   ├── active_learning/
│   │   └── clarification.py   # [ ] Clarification
│   ├── memory/
│   │   └── scenario_tcm.py    # [ ] Episodic Memory
│   ├── multi_agent/
│   │   └── expert_system.py     # [ ] Expert System
│   ├── tools/
│   │   └── orchestrator.py      # [ ] Toolformer
│   ├── explain/
│   │   └── attribution.py      # [ ] Attribution
│   └── pipelines/
│       ├── speculative.py       # [ ] Speculative Decoding
│       └── mvp_pipeline_v15.py # [ ] Full Pipeline
├── fcp_core/
│   └── adaptive_lora.py       # [P] AdaLoRA
├── colab/
│   └── train_full_pipeline.py  # [P] Co-training
└── data/
    └── lora_dataset.json      # [X] Dataset
```

---

## Dependencies

```bash
# Core
pip install torch-geometric sentence-transformers hnswlib

# OpenVINO
pip install openvino openvino-genai nncf

# Data
pip install datasets ruconceptnet gdown

# Training
pip install transformers peft accelerate bitsandbytes
```

---

## Current Status

| Component | Status | Notes |
|----------|--------|-------|
| ru_data_loaders.py | [P] Partial | Нужны реальные загрузчики |
| adaptive_lora.py | [P] Draft | AdaLoRALayer готов |
| graph_encoder.py | [P] Draft | SAGEConv готов |
| lora_dataset.json | [X] Complete | 146 примеров |
| train_lora_v2.ipynb | [X] Complete | Для Colab |

---

## Next Steps (Priority Order)

1. [P] **LoRA Training в Colab** - Запустить обучение
2. [P] **GraphEncoder implementation** - SAGEConv + HNSW
3. [ ] **AdaLoRA layer** - Финализировать
4. [ ] **Data loaders** - Реализовать загрузчики
5. [ ] **GNN → OpenVINO** - Конвертация
6. [ ] **MVP Pipeline** - Полная сборка

---

## Model Sources

| Model | Source | Usage |
|-------|--------|-------|
| RefalMachine/RuadaptQwen3-4B-Instruct | HuggingFace | Base model |
| intfloat/multilingual-e5-small | HuggingFace | Embeddings |
| IlyaGusev/saiga2_70b_lora | HuggingFace | Training data |