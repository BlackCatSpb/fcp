# MemBind: Multi-Head Covariance Memory Language Model

**MemBind** — архитектура языка следующего поколения, заменяющая механизм внимания
(Transformer) на **multi-head covariance memory + bilinear bind + спектральный оператор**.
Без softmax, без sigmoid-гейтов, без attention.

Контекст — **∞** на 2GB GPU. Сложность — **O(D²)**, не зависит от длины последовательности.

---

## Документация

- **[LAMBDA_ARCHITECTURE.md](LAMBDA_ARCHITECTURE.md)** — полное техническое описание архитектуры
  (12 разделов: философия, все компоненты, математика, сравнение, масштабирование)

---

## Быстрый старт

```bash
# Тренировка (MemBind, fib_seq spectrum)
train_fibseq.bat

# Тренировка с произвольными параметрами
train.bat --spectrum linear --n_modes 16

# Мониторинг лога
Get-Content training_fibseq.log -Tail 10 -Wait
```

## Архитектурные нововведения (Jul 2026)

Все включены по умолчанию, отключаются через `--no-*` флаги:

| Фича | Флаг отключения | Суть |
|------|-----------------|------|
| **Multi-timescale heads** | `--no-multi-tau` | 4 головы с τ=[3, 12, 49, 200] frozen. Каждая специализируется на своём горизонте. |
| **Random Features** | `--no-rf` | D→64→16 через frozen R_fixed. Ёмкость k в 4× больше в тех же 16×16. |
| **First Moment µ** | `--no-first-moment` | µ[t]=d·µ[t-1]+i·k. Знак/смещение, который теряет Σ = k·kᵀ. Ёмкость 16×16 ×2. |
| **Cognitive Mirror** | `--no-mirror` | std(head_out) → bind-коррекция. Модель смотрит на собственные multi-head выходы. |

```powershell
# Полная конфигурация (все фичи включены)
python train_large.py --dct --slide

# Без зеркала и первого момента (отладка)
python train_large.py --dct --slide --no-mirror --no-first-moment
```

---

## Ключевые характеристики

| Параметр | Значение |
|----------|----------|
| Параметры | 89.1M trainable (108.4M total) |
| D | 896 |
| Слои | 24 |
| Головы памяти | 4, каждая r=16 |
| Bind rank | 16 |
| Спектр | fib_seq, 8 мод, λ∈[0.8, 1.8] |
| Блоки | [10, 21, 31, 51, 82, 134, 216, 351] |
| Длина контекста (inference) | Неограничена |
| VRAM (inference, fp32) | ~356MB + 0 за контекст |
| VRAM (training, B=2) | ~2.0GB |

---

## Структура репозитория

```
ld_model/
  core.py          — MemBindBlock, MemBindStack, parallel_prefix_scan,
                     LDBlock, LDStack, CausalConv1d, BottleneckMLP
  readout.py       — Zeckendorf readout (экспериментальный)

train_phase2.py   — Тренировка (--arch membind|ld)
analyze_scaling.py  — Анализ масштабирования и FLOPs
analyze_context.py  — Анализ контекста при 2GB

*.bat             — Ярлыки для запуска тренировки
LAMBDA_ARCHITECTURE.md — Полное описание архитектуры
```

---

## Сравнение: MemBind vs Transformer

| Характеристика | Transformer | MemBind |
|---------------|------------|---------|
| Сложность на токен | O(L·D) | O(D²) — константа |
| Память контекста | O(L·D) KV cache | 96KB covariance |
| Контекст на 2GB (D=896) | ~18K tok | ∞ |
| Attention | QK^T + softmax | Covariance scan + bind |
| Нелинейность | Softmax | Bilinear (u*v) |

---

## Требования

- Python 3.12
- PyTorch 2.x
- GPU 2GB+ (MX550)
- 16GB RAM
