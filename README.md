# FCP - Fractal Cognitive Processor

## Overview

FCP - полностью функциональная когнитивная система с:
- OpenVINO генерация (Qwen3 4B)
- Knowledge Graph (201 узлов)
- Early Exit (~10s)
- LoRA адаптер

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
cd src/pipelines
python mvp_pipeline_v7.py
```

## Components

| Component | Status |
|-----------|--------|
| OpenVINO Pipeline | ✅ |
| Knowledge Graph | ✅ 201 nodes |
| Early Exit | ✅ |
| LoRA | ✅ Trained |

## Documentation

Подробнее в `SPEC.md` и файлах `*.txt` на рабочем столе.

## License

MIT