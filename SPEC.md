# FCP - Fractal Cognitive Processor

## Полная спецификация реализации

---

## 1. Обзор

**Цель:** Реализовать архитектуру FCP согласно спецификации из `Fractal Cognitive Processor (FCP).txt`

**Основные принципы:**
- Гибридное моделирование: трансформер + граф знаний на уровне каждого слоя
- Непрерывное обучение без катастрофического забывания
- Аппаратная независимость через UES
- Селективная активация (токеновая + послойная)

---

## 2. Архитектура системы

```
FCP/
├── src/
│   ├── fcp_core/              # Вычислительное ядро
│   │   ├── __init__.py
│   │   ├── input_layer.py     # Входной слой (BPE, RoPE)
│   │   ├── hybrid_layer.py   # Гибридный слой
│   │   ├── output_layer.py    # Выходной слой
│   │   └── stack.py         # Стек слоёв
│   │
│   ├── memory/               # Система памяти
│   │   ├── __init__.py
│   │   ├── temporal_context.py  # TCM
│   │   └── fractal_graph.py # FractalGraphV2
│   │
│   ├── curation/             # Анализ и курация
│   │   ├── __init__.py
│   │   ├── concept_miner.py
│   │   ├── contradiction.py
│   │   └── graph_curator.py
│   │
│   ├── adapters/            # Адаптеры
│   │   ├── __init__.py
│   │   ├── openvino_adapter.py
│   │   ├── llm_adapter.py
│   │   └── ues.py        # Universal Execution Subsystem
│   │
│   ├── training/          # Обучение
│   │   ├── __init__.py
│   │   ├── lora_adapters.py
│   │   └── continual.py
│   │
│   ├── config.py            # Конфигурация
│   └── main.py             # Точка входа
│
├── models/                 # Модели
│   └── .gitkeep
│
├── data/                  # Данные
│   └── .gitkeep
│
├── tests/
│   └── .gitkeep
│
├── README.md
├── SPEC.md               # Этот файл
└── requirements.txt
```

---

## 3. Компоненты

### 3.1 Input Layer (Входной слой)

**Функции:**
- Токенизация через BPE
- Создание эмбеддингов (embeddings table)
- Rotary Positional Encoding (RoPE)

**Интерфейс:**
```python
class InputLayer:
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int)
    def forward(self, token_ids: np.ndarray) -> np.ndarray  # (batch, seq_len, embedding_dim)
    def get_positional_embedding(position: int, dim: int) -> np.ndarray
```

### 3.2 Hybrid Layer (Гибридный слой)

**5 этапов обработки:**

1. **Контекстуальный токенизатор:**
   - Вычисляет структурный запрос для каждого токена
   - HNSW поиск в FractalGraphV2
   - Извлечение подграфа
   - Узловая маршрутизация (graph-only vs full LM)

2. **Графовый кластеризатор:**
   - Message passing между узлами
   - Soft кластеризация → мастер-токены
   - Метрика структурной полноты
   - **Сохранение состояния** между слоями

3. **Трансформерный блок:**
   - Multi-head causal attention
   - SwiGLU FFN
   - RMSNorm + Residual connections
   - Early exit для "остановленных" токенов

4. **Активационный гейт:**
   - Вероятность остановки токена
   - Послойная уверенность
   - Early exit Decision

5. **Слияние потоков:**
   - Cross-attention или additive gate
   - Объединение transformer + graph мастер-токенов

**Интерфейс:**
```python
class HybridLayer:
    def __init__(self, config: LayerConfig)
    def forward(
        self,
        hidden_states,           # (batch, seq_len, dim)
        graph_context,        # Извлечённый подграф
        layer_state          # Состояние кластеризации
    ) -> tuple:
        # Returns: (output, new_layer_state, stop_mask, layer_confidence)
```

### 3.3 Temporal Context Memory (TCM)

**Функции:**
- Энергонезависимое хранилище в рамках сессии
- Иерархическое временное кодирование (час, день, неделя, месяц)
- Soft addressing при извлечении (семантика + время + релевантность)

**Режимы:**
- Синхронный: доступ на чтение during generation
- Асинхронный: обновление параметров, контрастивный лосс, консолидация в граф

**Интерфейс:**
```python
class TemporalContextMemory:
    def __init__(self, max_segments: int, embedding_dim: int)
    def add(self, text: str, embedding: np.ndarray, metadata: dict)
    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> list[Segment]
    def consolidate_stable(self, graph: FractalGraphV2)  # Асинхронно
    def update_parameters(self, contrastive_loss)     # Асинхронно
```

### 3.4 FractalGraphV2

**Использовать из EVA-Ai:**
- `eva_ai/memory/fractal_graph_v2/` полностью

**Интеграция:**
- Индекс HNSW для семантического поиска
- Concept mining из диалогов
- Contradiction detection and resolution

### 3.5 Concept Miner

**Функции:**
- Извлечение именованных сущностей
- Разрешение кореференций
- Фильтрация по частоте/специфичности
- Семантическое сопоставление с графом
- Построение таксономии

**Интеграция:**
- Асинхронный запуск через DCS
- Сигнал: `curator.knowledge_extracted`

### 3.6 Contradiction Miner

**Функции:**
- Сканирование графовых пар (sim >= 0.75)
- NLI анализ противоречий
- Транзитивное замыкание --> кластеры
- Стратегии разрешения

**Интеграция:**
- Асинхронный запуск через DCS
- Сигнал: `contradiction.detected`

### 3.7 Graph Curator

**Функции:**
- Периодический запуск mining + contradiction
- Обработка "висячих" узлов
- Временной распад весов
- Корректирующие примеры для LoRA

**Интеграция:**
- Запуск через DCS на `system.idle`

### 3.8 UES (Universal Execution Subsystem)

**Функции:**
- Зондирование платформы (CPU, GPU, NPU)
- Построение графа вычислительных ресурсов
- MLIR трансляция (будущая реализация)
- Адаптивный планировщик задач
- Двойная буферизация дл�� ас��нхронного обучения

**Текущий приоритет:** LOW (базовая поддержка через OpenVINO)

### 3.9 LoRA Adapters

**Архитектура:**
- Иерархические адаптеры: low-rank (base), mid-rank (domain), high-rank (reasoning)
- Локальные обучающие сигналы на каждом слое

**Интеграция:**
- Асинхронные микро-обновления
- Experience replay buffer
- Important weight constraints

**Текущий приоритет:** MEDIUM (после базовой архитектуры)

### 3.10 Селективная активация

**Токеновый уровень:**
- Вероятность остановки на каждом слое
- Адаптивный порог
- Freeze остановленных токенов

**Послойный уровень:**
- Послойная уверенность
- Early exit Decision
- Динамическая калибровка порогов

**Интеграция:**
- Интегрировано в HybridLayer (этап 4)

---

## 4. Главный цикл (Main Loop)

```python
def process_query(query: str, model: "FCPModel") -> str:
    # 1. Токенизация
    token_ids = model.tokenizer.encode(query)
    embeddings = model.input_layer(token_ids)
    
    # 2. Извлечение контекста
    context_from_tcm = model.tcm.retrieve(query_embedding)
    initial_subgraph = model.graph.retrieve(query_embedding)
    
    # 3. Подача на стек
    hidden = embeddings
    layer_state = None
    layer_confidences = []
    stop_masks = []
    
    for layer in model.layers:
        # Контекстуальный токенизатор
        graph_context = model.graph.retrieve_subgraph(
            hidden, 
            layer_state
        )
        
        # Forward pass
        hidden, layer_state, stop_mask, conf = layer.forward(
            hidden, 
            graph_context,
            layer_state
        )
        
        layer_confidences.append(conf)
        stop_masks.append(stop_mask)
        
        # Early exit check
        if np.mean(conf) > model.early_exit_threshold:
            break
    
    # 4. Выходной слой
    logits = model.output_layer(hidden)
    tokens = model.sample(logits)
    
    # 5. Сохранение в TCM (асинхронно)
    model.tcm.add(query, response, metadata)
    
    # 6. Запуск курации (асинхронно)
    model.schedule_curation()
    
    return response
```

---

## 5. Зависимости

```
openvino
openvino.genai
numpy
scipy
hnswlib
transformers (для токенизации)
peft (для LoRA, опционально)
```

---

## 6. Приоритеты реализации

### Фаза 1: Базовая архитектура (MVP)
- [ ] Input Layer + Output Layer
- [ ] FractalGraphV2 из EVA-Ai
- [ ] Basic Hybrid Layer (только transformer + graph retrieval)
- [ ] Интеграция с OpenVINO Generator
- [ ] Basic селективная активация

### Фаза 2: Память
- [ ] Temporal Context Memory
- [ ] Интеграция TCM --> Graph

### Фаза 3: КУРАЦИЯ
- [ ] Concept Miner
- [ ] Contradiction Miner
- [ ] Graph Curator

### Фаза 4: Продвинутые функции
- [ ] LoRA адаптеры
- [ ] Селективная активация (полная)
- [ ] UES базовая реализация

---

## 7. Тестирование

### Unit тесты
- Input Layer: эмбеддинги, RoPE
- Hybrid Layer: все 5 этапов
- TCM: add/retrieve/consolidate

### Интеграционные тесты
- Full pipeline: query --> response
- Graph retrieval: скорость, точн��ст��

### Бенчмарки
- Latency на разных конфигурациях
- Memory usage
- Graph size vs retrieval time

---

## 8. Критерии готовности MVP

- [ ] Input --> Output pipeline работает
- [ ] FractalGraphV2 интегрирован
- [ ] Graph retrieval в hybrid слое работает
- [ ] Early exit работает
- [ ] Latency < 30 сек на CPU (1B params)
- [ ] Concept extraction --> Graph работает
- [ ] Contradiction generation --> Graph работает
- [ ] Self-dialog (из FMF_EVA) интегрирован