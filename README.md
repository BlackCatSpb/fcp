# FCP - Fractal Cognitive Processor

## Overview

FCP/FMF - полная когнитивная система, объединяющая FCP и FMF.

## Current Architecture (v15)

```
┌─────────────────────────────────────────────────────────────────┐
│                      FCP Pipeline v15                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Input Layer (Tokenizer + Embeddings)                        │
│     └── ruadapt_qwen3_4b: 36 слоёв, hidden=2048, vocab=151665   │
├─────────────────────────────────────────────────────────────────┤
│  2. Memory System                                              │
│     ├── HNSW Graph Search: 199 semantic nodes                  │
│     ├── Temporal Context Memory (TCM): 50 сегментов          │
│     └── HybridCache: RAM + Disk (50% hit rate)                 │
├─────────────────────────────────────────────────────────────────┤
│  3. Processing Layers                                           │
│     ├── Real Split hooks: layers 8/16/24                       │
│     ├── Intelligent Routing: facts/reasoning/creative/memory  │
│     └── Selective Activation: Early Exit (confidence≥0.85)    │
├─────────────────────────────────────────────────────────────────┤
│  4. Generation                                                 │
│     └── Generation with caching: 39s→0s on cache hit           │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### FCP Core (src/)
- `pipelines/` - Генеративные пайплайны
- `fcp_core/` - Гибридные слои
- `memory/` - TCM, HNSW, HybridCache

### FMF (fmf/)
- `eva_ai/` - Основная система
- `fmf_knowledge/` - Knowledge системы

## Features

| Feature | Status | Implementation |
|----------|--------|----------------|
| Tokenizer | ✅ | ruadapt tokenizer (151665 vocab) |
| Model | ✅ | ruadapt_qwen3_4b (36 layers) |
| Knowledge Graph | ✅ | HNSW: 199 semantic nodes |
| Graph Search | ✅ | HNSW cosine similarity |
| Real Split | ✅ | Hooks layers 8/16/24 |
| Intelligent Routing | ✅ | Domain-based layer selection |
| Selective Activation | ✅ | Early Exit confidence≥0.85 |
| Temporal Memory | ✅ | TCM 50 segments max |
| HybridCache | ✅ | RAM + Disk, 50% hit rate |
| Generation + Cache | ✅ | 39s → 0s on hit |

## Upcoming Features (Plan)

### Phase 1: GNN + LoRA (in Colab)
- [ ] FractalGraphEncoder with SAGEConv
- [ ] AdaLoRA with adaptive rank (4/8/16)
- [ ] Co-training pipeline

### Phase 2: OpenVINO Integration
- [ ] GNN → ONNX → OpenVINO converter
- [ ] AdaptiveFusionInjector
- [ ] Graph vector injection

### Phase 3: LoRA Management
- [ ] ShadowLoRA Manager
- [ ] Atomic adapter switching
- [ ] Multiple adapters (r4/r8/r16)

### Phase 4: Advanced Decoding
- [ ] Speculative Decoding
- [ ] Draft model integration
- [ ] NNCF Quantization

### Phase 5: Semantic Memory
- [ ] SemanticCacheEvictor
- [ ] ScenarioTCM (episodic memory)
- [ ] ClarificationGenerator

### Phase 6: Multi-Agent
- [ ] ExpertSystem
- [ ] ContradictionDetector
- [ ] Toolformer integration

### Phase 7: Integration
- [ ] FCPPipelineV15 (full pipeline)
- [ ] AttributionReport
- [ ] ThinkingController

## Quick Start

```bash
# Test generation with cache
python test_hit2.py

# Run unified pipeline
python src/pipelines/unified_fcp.py

# Validate all phases
python src/pipelines/fcp_validate.py
```

## Documentation

- `SPEC.md` - Full specification
- `FCP_IMPLEMENTATION.md` - Technical details
- `PLAN_V15.md` - Dvelopment roadmap from "Последовательные решения.txt"

## GitHub

https://github.com/BlackCatSpb/fcp

## License

MIT