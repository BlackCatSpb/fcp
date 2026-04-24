# FCP v15 План доработок

## Анализ: Уже реализовано ✅

| Компонент | Статус |
|----------|--------|
| Tokenizer + Model (ruadapt) | ✅ 36 слоёв |
| HNSW Graph Search | ✅ 199 nodes |
| Real Split hooks | ✅ layers 8/16/24 |
| Selective Activation (Early Exit) | ✅ confidence-based |
| Temporal Context Memory (TCM) | ✅ basic |
| Intelligent routing | ✅ domain-based |
| HybridCache | ✅ RAM + Disk |
| Generation with cache hit | ✅ 39s→0s |

## Требует доработки ❌

### Приоритет 1: Внешние зависимости

1. **Конвертация GNN в OpenVINO** - нет модели
2. **ShadowLoRA Manager** - нет API адаптеров
3. **Speculative Decoding** - нужна драфт-модель
4. **NNCF Quantization** - требует deep learning libs

### Приоритет 2: Расширенные модули

5. **AdaLoRA Layer** - адаптивный ранг
6. **FractalGraphEncoder** - SAGEConv из torch_geometric
7. **AdaptiveFusionInjector** - инъекция graph vector
8. **SemanticCacheEvictor** - семантическое вытеснение

### Приоритет 3: Продвинутые функции

9. **ClarificationGenerator** - активное обучение
10. **ScenarioTCM** - эпизодическая память
11. **ExpertSystem** - мультиагентность
12. **ToolOrchestrator** - Toolformer интеграция
13. **ThinkingController** - управление мышлением
14. **AttributionReport** - объяснимость

### Приоритет 4: Интеграция

15. **FCPPipelineV15** - финальная сборка

---

## Roadmap реализации

### Этап 1: GNN + LoRA (в Colab)
- Обучение FractalGraphEncoder с SageConv
- Обучение AdaLoRA с разными rank (4/8/16)
- Сохранение весов

### Этап 2: OpenVINO конвертация
- Конвертация GNN → ONNX → OpenVINO
- Интеграция инъектора

### Этап 3: LoRA менеджмент
- ShadowLoRA loading/unloading
- Адаптер switching

### Этап 4: Спекулятивное декодирование
- Драфт-модель
- Верификация

### Этап 5: Квантование
- NNCFquantization
- KV-cache compression

### Этап 6: Расширенная память
- SemanticCacheEvictor
- ScenarioTCM

### Этап 7: Активное обучение
- ClarificationGenerator

### Этап 8: Мультиагентность
- ExpertSystem

### Этап 9: Toolformer
- ToolOrchestrator

### Этап 10: Интеграция V15
- FCPPipelineV15

---

## Текущие ограничения

1. **Нет данных для обучения GNN** - нужен Russian dataset
2. **Нетtorch_geometric** - требует установки
3. **Нет OpenVINO GenAI** - для спекулятивного декодирования
4. **LoRAadapter incompatibility** - нужна конвертация

---

## Ближайшие шаги

1. Установить torch_geometric
2. Создать AdaLoRA layer
3. Интегрировать AdaptiveFusionInjector в pipeline
4. Расширить HybridCache с семантическим вытеснением