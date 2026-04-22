# FCP - Fractal Cognitive Processor

## Overview

FCP/FMF - полная когнитивная система, объединяющая FCP и FMF.

## Components

### FCP Core (src/)
- `pipelines/` - Генеративные пайплайны
- `fcp_core/` - Гибридные слои
- `memory/` - Temporal Context Memory
- `adapters/` - OpenVINO адаптеры
- `training/` - LoRA обучение

### FMF (fmf/)
- `eva_ai/` - Основная система
- `fmf_knowledge/` - Knowledge системы
- `fmf_config.py` - Конфигурация
- `fmf_interactive.py` - Интерактивный режим

## Features

| Feature | Status |
|---------|--------|
| OpenVINO Generation | ✅ |
| Knowledge Graph | ✅ 201 nodes |
| Early Exit | ✅ ~10s |
| LoRA Adapter | ✅ Trained |
| Self-Dialog | ✅ |
| Concept Extraction | ✅ |
| Contradiction Detection | ✅ |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run FCP
cd src/pipelines
python mvp_pipeline_v7.py
```

## Documentation

См. `SPEC.md` и файлы `.txt` на рабочем столе.

## GitHub

https://github.com/BlackCatSpb/fcp

## License

MIT