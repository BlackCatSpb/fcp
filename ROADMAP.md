# λ_d: Roadmap обучения

## Фазы, задачи, ресурсы, milestones

> **Примечание:** После внешнего аудита внесены корректировки.
> - **Phase 3.0:** FCF (VSA bind/unbind) признан избыточным — заменён на HierarchyLD (стек LDBlock + Linear compression). Код FCF в `fcf_cell.py`/`fcf_hierarchy4.py` заменён на `hierarchy_ld.py`. Раздел 3 оставлен для истории.
> - **Phase 4:** Оценка FLOPs скорректирована (8B на 1T токенов требует ~555 дней на 8×A100). Цель изменена на 2.5B / 160B tok (Chinchilla-optimal).

---

**Цель:** построить языковую модель на архитектуре λ_d, конкурентоспособную с transformer-based моделями (Qwen, LLaMA), с принципиальным преимуществом: O(1) память контекста, линейная сложность, edge-ready.

---

## Содержание

1. [Сводка статуса](#1-сводка-статуса)
2. [Phase 2.5: Consolidation](#2-phase-25-consolidation)
   - [2.7 36-Layer Scale-up & Tokenizer](#27-36-layer-scale-up--tokenizer)
3. [Phase 3.0: FCF Cell Prototype (с кодом)](#3-phase-30-fcf-cell-prototype)
4. [Phase 3.5: Hierarchical λ_d medium (с кодом)](#4-phase-35-hierarchical-λ_d-medium)
5. [Phase 4: Production λ_d](#5-phase-4-production-λ_d)
6. [Дорожная карта: график](#6-дорожная-карта-график)
7. [Приложение: Ресурсы по фазам](#приложение-а-сводка-ресурсов-по-фазам)
8. [Приложение: Риски](#приложение-б-риски-и-mitigation)

---

## 1. Сводка статуса

### 1.1. Что готово

| Компонент | Статус | Детали |
|-----------|--------|--------|
| λ_d архитектура (ядро) | ✅ **Phase 2 complete** | LDBlock, CausalConv1d, BottleneckMLP, LDStack |
| Обучение на wikitext | ✅ **3 эпохи, best PPL 131.5** | 105M params, 0 NaN, стабильная кривая |
| Обучение на русском | ✅ **1+ эпоха, PPL ~125** | Colab T4, batch=64, mmap dataset, auto-resume |
| BPE tokenizer | ✅ **Vocab=50K, русский** | ByteLevel, обучен на Wikipedia RU |
| Датасет (русский) | ✅ **6.6M chunks, 844M токенов** | russian_chunks.npy, 3.4 GB |
| Colab pipeline | ✅ **Работает** | mmap, fp16 ckpt, batch rescale, auto-resume |
| Гейты (learned) | ✅ **Работают, entropy 0.15** | Доказано: bootstrapping V-анизотропии |
| Learnable V (Cayley) | ✅ **Реализован** | Явный solve R = (I-S)^{-1}(I+S), machine epsilon |
| Adaptive depth | ✅ **Реализован** | Per-token routing по spread гейтов, soft/hard |
| ZeckendorfReadout | ❌ **Закрыт** | Co-training failed, lm_head оставлен (0.04% модели) |
| Анализатор модели | ✅ **analyze_model.py** | HTML-отчёт: гейты, Cayley, нормы, adaptive depth |
| Документация | ✅ **README.md (единая)** | Архитектура, математика, результаты, весь проект |

### 1.2. Ключевые метрики Phase 2

| Метрика | Значение |
|---------|----------|
| Параметры | 105.5M (95.1M trainable) |
| Train PPL (wikitext, эпоха 3) | **49.78** |
| Eval PPL (wikitext, эпоха 3) | **131.53** |
| Eval PPL (русский, step 20k) | **~125** |
| NaN за всё обучение | **0** |
| Память контекста | **3.5 KB** (O(D), не O(L)) |
| Энтропия гейтов | 0.15 (max 1.386) |
| Длина контекста (надёжная) | 128 токенов |
| Длина контекста (экстраполяция) | 256-896 токенов |

### 1.3. Ограничения Phase 2

- **D=896** — ёмкость контекста ~896 токенов (информационный предел)
- **K=4** — только 4 временны́х масштаба
- **V frozen** — не учится под данные (random orthogonal)
- **Плоский стек** — все 12 слоёв одинаковы
- **Только русский** — нет мультиязычности
- **3B токенов** — недостаточно для competitive PPL

---

## 2. Phase 2.5: Consolidation

**Цель:** выжать максимум из текущей архитектуры, подготовить инфраструктуру для Phase 3.

**Оборудование:** T4 16GB (Colab), RTX 3090 (local — перспектива)

### 2.1. Завершить Phase 2 Russian (3 эпохи)

| Задача | Описание | Приоритет | Зависимости |
|--------|----------|-----------|-------------|
| 2.1.1 | Дождаться 3 эпох русского на Colab | **High** | Текущая Colab сессия |
| 2.1.2 | Замерить eval PPL на каждой эпохе | **High** | 2.1.1 |
| 2.1.3 | Сравнить с wikitext baseline | Medium | 2.1.2 |
| 2.1.4 | Обновить README.md с русскими метриками | Medium | 2.1.2 |

**Milestone M2.1:** Phase 2 Russian completed, known eval PPL. ✅/❌

### 2.2. Scaling law: зависимость PPL от размера

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.2.1 | Обучить модель D=512, L=8 (~40M) на 10% данных | **High** |
| 2.2.2 | Обучить D=768, L=10 (~80M) на 10% данных | **High** |
| 2.2.3 | Обучить D=1024, L=14 (~140M) на 10% данных | **High** |
| 2.2.4 | Построить график PPL(params) — Chinchilla-style | **High** |
| 2.2.5 | Определить оптимальное params/tokens ratio | Medium |

**Зачем:** Chinchilla scaling law говорит, что большинство моделей недотренированы. Нам нужно знать, сколько данных нужно для данной архитектуры при заданном размере. Это критично для запроса ресурсов.

**Milestone M2.2:** Scaling law curves for λ_d. Known optimal data/params ratio. ✅/❌

### 2.3. Benchmark: λ_d vs transformer равного размера

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.3.1 | Выбрать transformer baseline (GPT-2 124M, OPT-125M) | **High** |
| 2.3.2 | Обучить baseline на тех же данных (русский wikitext) | **High** |
| 2.3.3 | Сравнить PPL, скорость, память на 128K контекста | **High** |
| 2.3.4 | Публикация сравнения в таблице WHITEPAPER | Medium |

**Зачем:** Аргумент для ресурсов. Нужно показать, что λ_d не уступает transformer того же размера, но использует в 3000× меньше памяти на длинных контекстах.

**Milestone M2.3:** Direct comparison λ_d vs transformer on same data. ✅/❌

### 2.4. ZeckendorfReadout distillation (❌ ЗАКРЫТ)

Zeckendorf (86K) побеждал lm_head (44.8M) на замороженном stack (PPL 1,435 vs 32,652),
но разваливался при co-training (PPL 6e18). lm_head — 0.04% модели — оставлен.
Код и чекпоинты удалены. Подробности: SUMMARY.md → ZeckendorfReadout experiment.

### 2.5. Инфраструктура для Phase 3

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.5.1 | Настроить FSDP/DeepSpeed на RTX 3090 | **High** |
| 2.5.2 | Протестировать Multi-GPU training (2×3090) | **High** |
| 2.5.3 | Подготовить скрипты для A100 кластера | Medium |
| 2.5.4 | Настроить W&B / MLflow для логов | Medium |

**Milestone M2.5:** Infra ready for 1B+ param training. ✅/❌

### 2.6. Ресурсы Phase 2.5

| Ресурс | Для чего | Оценка |
|--------|----------|--------|
| T4 16GB (Colab) | 2.1 — завершение русского | ~неделя сессий |
| RTX 3090 24GB | 2.2 — scaling law sweep | ~2 недели |
| RTX 3090 24GB | 2.3 — transformer baseline | ~1 неделя |
| RTX 3090 24GB | 2.5 — инфраструктура | ~1 неделя |
| **Итого** | | **~5 недель** |

### 2.7. Новые фичи (реализовано)

| Фича | Статус | Описание |
|------|--------|----------|
| **Learnable V (Cayley)** | ✅ | V_eff = V_frozen @ R, R = solve(I-S, I+S), S = A·B^T - B·A^T. Orth_err < 1e-9. |
| **Adaptive depth** | ✅ | Gate spread → per-token routing. Soft на обучении, hard на инференсе. 0 доп. параметров. |
| **Анализатор модели** | ✅ | `analyze_model.py` — ASCII + `--html` с цветными таблицами и визуализацией гейтов. |

### 2.8. 36-Layer Scale-up & Tokenizer

**Цель:** масштабировать модель до 36 слоёв, доработать токенизатор, подготовить инфраструктуру для multi-agent reasoning.

**Оборудование:** T4 16GB (Colab) — 36L модель (106M params) влезает (875MB VRAM).

#### 2.7.1. 36 layers (warm-start)

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.7.1.1 | Скопировать LDBlock веса 12→36 (identity для слоёв 1-12, random для 13-36) | **High** |
| 2.7.1.2 | Freeze embed/lm_head на первые 10K шагов (экономия VRAM) | **High** |
| 2.7.1.3 | Обучение 3 эпохи на русском (D=896, 106M, 36L) | **High** |
| 2.7.1.4 | Замерить eval PPL, сравнить с 12L | **High** |

**Ожидание:** 36L даст -15..-25 PPL относительно 12L на том же D.

#### 2.7.2. Tokenizer (спецтокены + ретокенизация)

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.7.2.1 | Добавить 20-50 служебных токенов: `<reason>`, `<alt>`, `<synth>`, `<agent_a>`, `<agent_b>` и т.д. | **High** |
| 2.7.2.2 | Init embed новых токенов = avg(embed субтокенов) — проверено | **High** |
| 2.7.2.3 | Ретокенизировать `russian_chunks.npy` с новым vocab (однократно, ~30 мин CPU) | **High** |
| 2.7.2.4 | Resume обучение с 36L чекпойнта | Medium |

#### 2.7.3. Distillation (Qwen 1.5B → λ_d)

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 2.7.3.1 | Sequence-level: Qwen 1.5B генерирует русский текст → наш 50K BPE → fine-tune 36L | **High** |
| 2.7.3.2 | KL(NLL + β·KL) догонялка | Medium |
| 2.7.3.3 | Сравнить PPL до/после | Medium |

#### 2.7.4. Multi-agent reasoning (концепт)

> Внутренний диалог двух агентов A и B. Вместо одного `h` — два параллельных `h_A`, `h_B` с cross-attention между ними. Обучение: contrastive loss — штраф B за согласие с A, поощрение за альтернативу. Детали — после 36L + дистилляции.

**Milestone M2.7:** 36L обучен, токенизатор расширен, baseline дистилляции известен. ✅/❌

---

## 3. Phase 3.0: FCF Cell Prototype

**Цель:** реализовать и проверить FCF-клетку (иерархический λ_d с binding) на маленьких данных. Доказать, что иерархия работает и даёт больший контекст, чем плоский λ_d.

**Оборудование:** RTX 3090 24GB (local)

### 3.1. Теория: binding/unbinding для λ_d

**Ключевая операция FCF:**

```
state[t] = λ · state[t-1] + bind(input[t], position[t])
output[t] = unbind(state[t], position[t])
```

**Bind (VSA binding):** для векторов a, b ∈ ℝ^D:
```
bind(a, b) = a ⊙ b  (покоординатное умножение)
```
или через FFT:
```
bind(a, b) = FFT^{-1}(FFT(a) ⊙ FFT(b))
```

**Unbind (VSA unbinding):** обратная операция:
```
unbind(c, b) = c ⊙ b^{-1} ≈ c ⊙ b (поскольку b ⊙ b = 1 для биполярных векторов)
```

**Почему это даёт иерархию:** bind «связывает» информацию с контекстом (позицией). Unbind извлекает. Это позволяет:

1. **Ретрив:** из state[t] извлечь input в любой позиции через unbind
2. **Композиция:** state содержит сумму всех прошлых input'ов, каждый «помечен» своей позицией
3. **Разные масштабы:** Level 0 bind с token-position, Level 1 bind с segment-position, etc.

> **Примечание (после аудита):** Bind через Hadamard product (a⊙b) требует биполярных векторов, иначе unbind искажает сигнал. Корректное решение для непрерывных векторов — **HRR (Holographic Reduced Representations)** через FFT: `bind(a,b)=FFT⁻¹(FFT(a)⊙FFT(b))`, `unbind(c,key)=FFT⁻¹(FFT(c)⊙conj(FFT(key)))`. Однако FCF заменён на HierarchyLD (без VSA), поэтому HRR не внедрялся. Оставлено для справки.

### 3.2. Задачи реализации

| Задача | Описание | Приоритет | Ожидаемый код |
|--------|----------|-----------|---------------|
| 3.1.1 | Реализовать FCFCell (bind/unbind, λ_d recur) | **Critical** | `ld_model/fcf_cell.py` |
| 3.1.2 | Реализовать позиционное кодирование (cos) | **Critical** | В составе FCFCell |
| 3.1.3 | Реализовать V биполярный (нормированный) | **Critical** | `V = sign(randn(D))` |
| 3.1.4 | Реализовать 3-level иерархию | **High** | `ld_model/fcf_stack.py` |
| 3.1.5 | Реализовать сжатие между уровнями | **High** | unbind + average pooling |
| 3.1.6 | Прототип: предсказание следующего токена | **High** | lm_head поверх иерархии |
| 3.1.7 | Тест: PPL на wikitext (L=128) | **High** | Сравнение с Phase 2 |

### 3.3. Эксперименты

| Эксперимент | Описание | Что измеряем |
|-------------|----------|--------------|
| E3.1 | FCFCell 1 level, D=256, vs LDBlock D=896 | PPL, кол-во параметров |
| E3.2 | FCFCell 3 level, D=256 each | PPL, контекст (extrapolation) |
| E3.3 | FCFCell 3 level + learned gates (гибрид) | PPL vs чисто FCF |
| E3.4 | FCFCell с V обучаемым (Procrustes) | Растёт ли качество |

**Ключевой вопрос E3.2:** даёт ли 3-level иерархия лучший PPL на L=256-512, чем плоский λ_d? Если да — доказательство концепции.

### 3.4. Полный код прототипа

```python
"""
fcf_cell.py — FCF Cell (VSA binding + λ_d recurrence).

Запуск:
    python fcf_cell.py                     # тест на малых данных
    python fcf_cell.py --train             # обучение на wikitext
    python fcf_cell.py --train --epochs 5  # 5 эпох

Зависимости: torch, numpy, tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, sys, json
import numpy as np

# ─── Позиционное кодирование (cos) ──────────────────────────────────────

def position_encoding(positions: torch.Tensor, D: int) -> torch.Tensor:
    """
    Аргументы:
        positions: (B, L) — целочисленные позиции
        D: размерность кодирования (должна быть чётной)
    Возвращает:
        (B, L, D) — позиционные векторы с ∥pos_enc∥ = 1
    """
    B, L = positions.shape
    freqs = 10.0 ** torch.linspace(0, 2, D // 2)  # (D/2,)
    freqs = freqs.to(positions.device)
    angles = positions.unsqueeze(-1) * freqs  # (B, L, D/2)
    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, L, D)
    pe = pe / (pe.norm(dim=-1, keepdim=True) + 1e-8)
    return pe


# ─── VSA операции ───────────────────────────────────────────────────────

def bipolar(D: int) -> torch.Tensor:
    """Случайный биполярный вектор: sign(randn). Норма = sqrt(D)."""
    return torch.sign(torch.randn(D))


def random_orthogonal_bipolar(D: int) -> torch.Tensor:
    """Ортогональная биполярная матрица: V · V^T = I, V_ij ∈ {-1, +1}.
    Строится как sign(произведение Householder отражений случайных bipolar векторов)."""
    V = torch.eye(D)
    for _ in range(min(32, D)):
        u = bipolar(D).float()
        u = u / u.norm()
        V = V - 2 * torch.outer(V @ u, u)
    return V.sign()  # биполярная аппроксимация


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """VSA binding: a ⊙ b (покоординатное умножение)."""
    return a * b


def unbind(c: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """VSA unbinding: c ⊙ key^{-1} ≈ c ⊙ key (для биполярных key)."""
    return c * key


# ─── FCF Cell ───────────────────────────────────────────────────────────

class FCFCell(nn.Module):
    """
    FCF Cell: state[t] = λ · state[t-1] + bind(V · input[t], pos_enc[t])
    
    Параметры:
        D: размерность пространства
        lambda_k: корень Фибоначчи (λ)
        use_gate: если True, добавляет learned gate α = σ(W_gate · input)
    """
    def __init__(self, D: int, lambda_k: float, use_gate: bool = False):
        super().__init__()
        self.D = D
        self.lambda_k = lambda_k
        self.use_gate = use_gate
        
        # V — фиксированная биполярная ортогональная матрица
        V = random_orthogonal_bipolar(D).float()
        self.register_buffer('V', V)
        
        # Опциональный learned gate
        if use_gate:
            self.W_gate = nn.Parameter(torch.randn(D, 1) * 0.01)
            self.b_gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_seq: torch.Tensor,
                pos_seq: torch.Tensor,
                state: torch.Tensor = None) -> tuple:
        """
        Аргументы:
            input_seq: (B, L, D) — входные векторы (уже спроецированные)
            pos_seq: (B, L, D) — позиционные кодирования
            state: (B, D) — начальное состояние (None = нули)
        
        Возвращает:
            outputs: (B, L, D) — unbind(state, pos) на каждом шаге
            state: (B, D) — финальное состояние
        """
        B, L, D = input_seq.shape
        if state is None:
            state = torch.zeros(B, D, device=input_seq.device)
        
        # Проецируем V · input
        V_proj = input_seq @ self.V.T  # (B, L, D)
        
        outputs = []
        for t in range(L):
            inp = V_proj[:, t, :]  # (B, D)
            pos = pos_seq[:, t, :]  # (B, D)
            
            # Gate (если есть)
            if self.use_gate:
                gate = torch.sigmoid(inp @ self.W_gate + self.b_gate)  # (B, 1)
                inp = inp * gate
            
            # λ_d обновление
            state = self.lambda_k * state + bind(inp, pos)
            
            # Читаем состояние
            out = unbind(state, pos)
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)  # (B, L, D)
        return outputs, state


# ─── 3-Level Hierarchy ──────────────────────────────────────────────────

class FCF3Level(nn.Module):
    """
    3-level иерархическая λ_d модель.

    Level 0 (token, λ₁): обрабатывает токены, сжатие каждые N0 шагов
    Level 1 (segment, λ₂): обрабатывает сегменты, сжатие каждые N1 шагов
    Level 2 (block, λ₃): обрабатывает блоки

    Архитектура:
        h = Embed(tokens) → V₀ · h [gate]
        state_0[t] = λ₁·state_0[t-1] + bind(V₀·h[t], pos_enc[t])
        каждые N0 шагов: compress(state_0) → z₁
    
        state_1[s] = λ₂·state_1[s-1] + bind(V₁·z₁[s], pos_enc[s])
        каждые N1 шагов: compress(state_1) → z₂
    
        state_2[b] = λ₃·state_2[b-1] + bind(V₂·z₂[b], pos_enc[b])
    
        output = Combine(read(state_0), read(state_1), read(state_2))
    """
    def __init__(self, vocab: int, D: int,
                 lambda_roots: list[float],
                 compress_steps: list[int],
                 use_gate_level0: bool = True):
        super().__init__()
        self.D = D
        self.vocab = vocab
        self.compress_steps = compress_steps  # [N0, N1]
        
        # Embedding + LM head (shared)
        self.embed = nn.Embedding(vocab, D)
        self.lm_head = nn.Linear(D, vocab, bias=False)
        
        # Level cells
        self.cells = nn.ModuleList([
            FCFCell(D, lambda_roots[0], use_gate=use_gate_level0),
            FCFCell(D, lambda_roots[1], use_gate=False),
            FCFCell(D, lambda_roots[2], use_gate=False),
        ])
        
        # Выучиваемые веса для комбинации уровней
        self.combine_w = nn.Parameter(torch.ones(3) / 3)
    
    def _compress(self, outputs: torch.Tensor,
                  pos: torch.Tensor) -> torch.Tensor:
        """Сжатие: unbind всех выходов + mean pool."""
        # outputs: (B, L, D), pos: (B, L, D)
        unbound = unbind(outputs, pos)  # (B, L, D)
        return unbound.mean(dim=1)  # (B, D)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            tokens: (B, L) — токены
        Возвращает:
            logits: (B, L, V) — логиты предсказания
        """
        B, L = tokens.shape
        device = tokens.device
        
        # Позиционные кодирования для Level 0 (токены)
        pos_0 = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pe_0 = position_encoding(pos_0, self.D)  # (B, L, D)
        
        # Embedding
        h = self.embed(tokens)  # (B, L, D)
        
        # Level 0 (token-level)
        out_0, state_0 = self.cells[0](h, pe_0)
        
        # Сжатие: каждые N0 токенов
        N0 = self.compress_steps[0]
        n_segments = (L + N0 - 1) // N0
        seg_inputs = []
        seg_pos = []
        for s in range(n_segments):
            start = s * N0
            end = min(start + N0, L)
            # Сжимаем выходы Level 0 за этот сегмент
            z = self._compress(out_0[:, start:end, :],
                               pe_0[:, start:end, :])
            seg_inputs.append(z)
            seg_pos.append(s)
        
        if n_segments == 0:
            seg_inputs = [torch.zeros(B, self.D, device=device)]
            seg_pos = [0]
        
        z_1 = torch.stack(seg_inputs, dim=1)  # (B, S, D)
        pos_1 = torch.tensor(seg_pos, device=device).unsqueeze(0).expand(B, -1)
        pe_1 = position_encoding(pos_1, self.D)  # (B, S, D)
        
        # Level 1 (segment-level)
        out_1, state_1 = self.cells[1](z_1, pe_1)
        
        # Сжатие для Level 2
        N1 = self.compress_steps[1]
        n_blocks = (n_segments + N1 - 1) // N1
        block_inputs = []
        block_pos = []
        for b in range(n_blocks):
            start = b * N1
            end = min(start + N1, n_segments)
            z = self._compress(out_1[:, start:end, :],
                               pe_1[:, start:end, :])
            block_inputs.append(z)
            block_pos.append(b)
        
        if n_blocks == 0:
            block_inputs = [torch.zeros(B, self.D, device=device)]
            block_pos = [0]
        
        z_2 = torch.stack(block_inputs, dim=1)  # (B, T, D)
        pos_2 = torch.tensor(block_pos, device=device).unsqueeze(0).expand(B, -1)
        pe_2 = position_encoding(pos_2, self.D)
        
        # Level 2 (block-level)
        out_2, state_2 = self.cells[2](z_2, pe_2)
        
        # Комбинируем выходы всех уровней
        # Для каждого токена: weighted sum of current readout from each level
        w = torch.softmax(self.combine_w, dim=0)
        combined = (w[0] * out_0 + w[1] * unbind(state_1.unsqueeze(1), pe_0)
                    + w[2] * unbind(state_2.unsqueeze(1), pe_0))
        
        # LM head
        logits = self.lm_head(combined)  # (B, L, V)
        return logits
    
    def generate(self, prompt: torch.Tensor, max_len: int = 100,
                 temperature: float = 1.0) -> torch.Tensor:
        """Авторегрессивная генерация."""
        self.eval()
        generated = prompt.clone()
        B = prompt.shape[0]
        
        with torch.no_grad():
            for _ in range(max_len):
                logits = self.forward(generated[:, -128:])  # последние 128
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# ─── Обучение ───────────────────────────────────────────────────────────

def train_model(model, data_file, vocab_size=50000, seq_len=128,
                batch_size=16, epochs=3, lr=1e-3):
    """Обучение модели на .npy данных (формат: (N_chunks, seq_len+1))."""
    arr = np.load(data_file)
    n_total = arr.shape[0]
    n_train = n_total - 500
    
    train_x = torch.from_numpy(arr[:n_train, :-1].copy()).long()
    train_y = torch.from_numpy(arr[:n_train, 1:].copy()).long()
    eval_x = torch.from_numpy(arr[n_train:, :-1].copy()).long()
    eval_y = torch.from_numpy(arr[n_train:, 1:].copy()).long()
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * epochs)
    
    DEVICE = next(model.parameters()).device
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), by.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                ppl = math.exp(total_loss / n_batches)
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  [{epoch+1}/{epochs}] step {n_batches}: loss={loss.item():.4f} ppl={ppl:.1f} lr={lr_now:.2e}')
        
        # Eval
        model.eval()
        with torch.no_grad():
            eval_logits = model(eval_x[:100].to(DEVICE))
            eval_loss = F.cross_entropy(eval_logits.reshape(-1, vocab_size),
                                        eval_y[:100].reshape(-1).to(DEVICE))
            eval_ppl = math.exp(eval_loss.item())
        
        train_ppl = math.exp(total_loss / n_batches)
        print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}  eval_ppl={eval_ppl:.1f}')
    
    return model


# ─── Main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Обучить модель')
    parser.add_argument('--D', type=int, default=256, help='Размерность')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--data', default='russian_chunks.npy')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {DEVICE}')
    
    lambda_roots = [1.6180, 1.8393, 1.9276]  # λ₁, λ₂, λ₃
    compress_steps = [32, 8]  # N0=32 токенов на сегмент, N1=8 сегментов на блок
    
    model = FCF3Level(
        vocab=50000, D=args.D,
        lambda_roots=lambda_roots,
        compress_steps=compress_steps,
        use_gate_level0=True,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {n_params/1e6:.1f}M params')
    
    if args.train:
        train_model(model, args.data, seq_len=128,
                    batch_size=args.batch, epochs=args.epochs)
        
        # Тест экстраполяции: длина > 128
        print()
        print('=== Extrapolation test (L=256) ===')
        data = np.load(args.data)
        test_seq = torch.from_numpy(data[-1:, :].copy()).long().to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(test_seq[:, :-1])
            loss_128 = F.cross_entropy(logits.reshape(-1, 50000),
                                       test_seq[:, 1:].reshape(-1))
            print(f'  PPL (L=128): {math.exp(loss_128.item()):.1f}')
        
        torch.save(model.state_dict(), 'fcf_prototype.pt')
        print('Saved: fcf_prototype.pt')
    else:
        # Быстрый тест: forward pass
        B, L = 2, 64
        tokens = torch.randint(0, 50000, (B, L), device=DEVICE)
        logits = model(tokens)
        print(f'Forward OK: logits shape = {logits.shape}')
        print(f'  min={logits.min().item():.2f}  max={logits.max().item():.2f}')
        
        # Проверка на NaN
        assert not torch.isnan(logits).any(), 'NaN в logits!'
        print('  NaN check: OK')

        # Тест генерации
        prompt = torch.randint(0, 50000, (1, 8), device=DEVICE)
        gen = model.generate(prompt, max_len=20)
        print(f'  Generation OK: {gen.shape}')
```

**Параметры прототипа:**

| Компонент | Параметры |
|-----------|-----------|
| Embedding | 50K × 256 = 12.8M |
| V₀, V₁, V₂ | 3 × 256 = 768 (биполярные, frozen) |
| Level 0 gate | 256 + 1 = 257 |
| Combine weights | 3 |
| lm_head | 256 × 50K = 12.8M |
| **Итого** | **~25.6M** |

**Запуск:**
```bash
# Тест (forward pass + NaN check)
python fcf_cell.py

# Обучение на русском датасете
python fcf_cell.py --train --data russian_chunks.npy --epochs 3
```

### 3.5. Milestones и ресурсы

| Milestone | Описание | Время |
|-----------|----------|-------|
| M3.1 | FCFCell реализован, тест на wikitext | 1 неделя |
| M3.2 | 3-level иерархия работает, PPL известен | 2 недели |
| M3.3 | Extrapolation на L=512 замерен | 2.5 недели |
| M3.4 | Вывод: работает/не работает, path decision | 3 недели |

**Ресурсы:** RTX 3090 24GB (или T4), ~3 недели.

**Gate decision:** Если FCF иерархия показывает PPL хуже Phase 2 плоского λ_d → возвращаем learned gates в иерархию (гибрид). Если лучше → pure FCF.

---

## 4. Phase 3.5: Hierarchical λ_d (medium)

**Цель:** масштабировать иерархическую λ_d до 1B параметров, получить competitive PPL, продемонстрировать 4K+ контекст.

**Оборудование:** 4× A100 80GB (облако)

### 4.1. Архитектура (гибрид: своя λ_d глубина на каждом уровне)

**Идея:** каждый уровень — полноценная λ_d модель (стек LDBlock + BottleneckMLP), но на своём временно́м разрешении. Между уровнями — простая линейная проекция (compression), без VSA binding.

**Преимущества над FCF:**
- Никакой новой теории — те же LDBlock, что уже работают (PPL 131)
- Глубина на каждом уровне (8+6+4 = 18 слоёв) — сложный вывод
- Compression = Linear(Dₖ → Dₖ₊₁), не bind/unbind — предсказуемо и стабильно
- Каждый уровень — самодостаточная λ_d модель, может тестироваться отдельно

```
Vocab: 150K (мультиязычный)
Embed: 150K × 2048 = 307M

Level 0 (token-level, D₀=2048, λ=[1.62, 1.84], 8 слоёв LDBlock + MLP):
  ← каждый токен, stride=1 →
  Выход: h₀ ∈ ℝ^{B×L×2048}
  ↓ Compression каждые 64 токена: mean_pool(L=64) → Linear(2048→2048)

Level 1 (segment-level, D₁=2048, λ=[1.93, 1.97], 6 слоёв LDBlock + MLP):
  ← каждый 64-й токен, stride=64 →
  Выход: h₁ ∈ ℝ^{B×S×2048}, S = L/64
  ↓ Compression каждые 32 сегмента: mean_pool(S=32) → Linear(2048→2048)

Level 2 (block-level, D₂=2048, λ=[1.98, 1.99], 4 слоя LDBlock + MLP):
  ← каждый 2048-й токен, stride=2048 →
  Выход: h₂ ∈ ℝ^{B×B×2048}, B = L/2048

Выходная комбинация:
  h_out = Linear(2048×3 → 2048) · [h₀, unpool(h₁), unpool(h₂)]
  → RMSNorm → lm_head

Total params: ~1.2B (Embed 307M + 18× LDBlock (~30M) + BottleneckMLPs + lm_head)
```

**Контекст:**

| Уровень | Слоёв | Шагов | Размер шага | Контекст | Память |
|---------|-------|-------|-------------|----------|--------|
| Level 0 | 8 | каждый | 1 токен | 128 токенов (deep) | 2048 floats |
| Level 1 | 6 | 1/64 | 64 токена | 128×64 = 8K токенов | 2048 floats |
| Level 2 | 4 | 1/2048 | 2048 токенов | ∞ (λ→2, unbounded) | 2048 floats |
| **Итого** | **18** | | | **∞** | **~24 KB** |

### 4.2. Код: HierarchyLD (многослойная иерархия)

```python
"""
fcf_hierarchy4.py — 4-level иерархический λ_d с гибридными гейтами и Procrustes-V.

Запуск:
    python fcf_hierarchy4.py --train --data /path/to/data --vocab 150000 --D 2048

Зависимости: torch, numpy, tqdm, wandb (опционально)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np


# ═══════════════════════════════════════════════════════════════════════
# VSA операции
# ═══════════════════════════════════════════════════════════════════════

def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b

def unbind(c: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return c * key

def bipolar_orthogonal(D: int) -> torch.Tensor:
    """Случайная биполярная ортогональная матрица V ∈ {-1, +1}^{D×D}."""
    V = torch.eye(D)
    for _ in range(min(32, D)):
        u = torch.sign(torch.randn(D)).float()
        u = u / u.norm()
        V = V - 2 * torch.outer(V @ u, u)
    return V.sign()

def position_encoding(positions: torch.Tensor, D: int) -> torch.Tensor:
    B, L = positions.shape
    freqs = 10.0 ** torch.linspace(0, 2, D // 2, device=positions.device)
    angles = positions.unsqueeze(-1) * freqs  # (B, L, D/2)
    pe = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return pe / (pe.norm(dim=-1, keepdim=True) + 1e-8)


# ═══════════════════════════════════════════════════════════════════════
# Procrustes — поддержание ортогональности V
# ═══════════════════════════════════════════════════════════════════════

def procrustes(V: torch.Tensor) -> torch.Tensor:
    """
    Procrustes projection: V ← argmin_{Q ∈ O(D)} ||Q - V||_F
    Решение: V ← V · (V^T · V)^{-1/2}
    Используется после каждого шага SGD для восстановления ортогональности.
    """
    U, _, Vt = torch.linalg.svd(V, full_matrices=False)
    return U @ Vt  # ближайшая ортогональная матрица


# ═══════════════════════════════════════════════════════════════════════
# Гибридный FCFCell (learned gates + VSA binding)
# ═══════════════════════════════════════════════════════════════════════

class HybridFCFCell(nn.Module):
    """
    Гибридная FCF клетка:
    state[t] = λ · state[t-1] + α · bind(V · input[t], pos[t])

    α = softmax(4.0 · (W_gate · RMSNorm(input) + b_gate))  — learned routing

    V — обучаемый параметр (Procrustes каждые N шагов для ортогональности).
    """
    def __init__(self, D: int, lambda_k: float, n_gates: int = 8,
                 trainable_V: bool = False):
        super().__init__()
        self.D = D
        self.lambda_k = lambda_k
        self.n_gates = n_gates  # K — количество подпространств в V
        self.block_size = D // n_gates

        # V — ортогональная, опционально обучаемая
        V_init = bipolar_orthogonal(D).float()
        if trainable_V:
            self.V = nn.Parameter(V_init)
        else:
            self.register_buffer('V', V_init)
        self.trainable_V = trainable_V

        # Learned gates
        self.W_gate = nn.Parameter(torch.randn(D, n_gates) * 0.01)
        self.b_gate = nn.Parameter(torch.randn(n_gates) * 0.01)

        # RMS norm weight
        self.norm_w = nn.Parameter(torch.ones(D))

    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(dim=-1, keepdim=True) / (self.D ** 0.5)
        return x / (rms + 1e-6) * self.norm_w

    def forward(self, input_seq: torch.Tensor,
                pos_seq: torch.Tensor,
                state: torch.Tensor = None) -> tuple:
        B, L, D = input_seq.shape
        if state is None:
            state = torch.zeros(B, D, device=input_seq.device)

        outputs = []
        V_T = self.V.T.contiguous()  # (D, D)

        for t in range(L):
            inp = input_seq[:, t, :]  # (B, D)
            pos = pos_seq[:, t, :]    # (B, D)

            # Learned gate (block-wise, как в Phase 2)
            h_norm = self.rms_norm(inp)
            gate_logits = (h_norm @ self.W_gate + self.b_gate) * 4.0
            alpha = F.softmax(gate_logits, dim=-1)  # (B, K)
            lambda_alpha = alpha * self.lambda_k  # (B, K) — λ·α для анализа

            # V-проекция + binding
            V_proj = (inp.view(B, 1, D) @ V_T).squeeze(1)  # (B, D)
            bound = bind(V_proj, pos)

            # Gate: block-wise масштабирование через α
            alpha_eff = alpha.repeat_interleave(self.block_size, dim=-1)  # (B, D)
            gated_input = bound * alpha_eff

            # λ_d обновление
            state = state + gated_input  # λ = 1 (residual, как в Phase 2)
            out = unbind(state, pos)
            outputs.append(out)

        return torch.stack(outputs, dim=1), state

    def apply_procrustes(self):
        """Восстановить ортогональность V, если обучаемый."""
        if self.trainable_V:
            with torch.no_grad():
                self.V.data.copy_(procrustes(self.V.data))


# ═══════════════════════════════════════════════════════════════════════
# Pure FCFCell (без learned gates)
# ═══════════════════════════════════════════════════════════════════════

class PureFCFCell(nn.Module):
    """
    Pure FCF: state[t] = λ · state[t-1] + bind(V · input[t], pos[t])
    0 learnable params (кроме V, опционально).
    """
    def __init__(self, D: int, lambda_k: float, trainable_V: bool = False):
        super().__init__()
        self.D = D
        self.lambda_k = lambda_k
        V_init = bipolar_orthogonal(D).float()
        if trainable_V:
            self.V = nn.Parameter(V_init)
        else:
            self.register_buffer('V', V_init)
        self.trainable_V = trainable_V

    def forward(self, input_seq: torch.Tensor,
                pos_seq: torch.Tensor,
                state: torch.Tensor = None) -> tuple:
        B, L, D = input_seq.shape
        if state is None:
            state = torch.zeros(B, D, device=input_seq.device)

        V_T = self.V.T.contiguous()
        outputs = []
        for t in range(L):
            inp = input_seq[:, t, :]
            pos = pos_seq[:, t, :]
            V_proj = (inp.view(B, 1, D) @ V_T).squeeze(1)
            state = self.lambda_k * state + bind(V_proj, pos)
            out = unbind(state, pos)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state

    def apply_procrustes(self):
        if self.trainable_V:
            with torch.no_grad():
                self.V.data.copy_(procrustes(self.V.data))


# ═══════════════════════════════════════════════════════════════════════
# Компрессия между уровнями
# ═══════════════════════════════════════════════════════════════════════

def compress_segment(outputs: torch.Tensor,
                     pos_seq: torch.Tensor) -> torch.Tensor:
    """Сжатие сегмента: unbind всех выходов → mean pool."""
    return unbind(outputs, pos_seq).mean(dim=1)  # (B, D)


# ═══════════════════════════════════════════════════════════════════════
# 4-Level Hierarchy
# ═══════════════════════════════════════════════════════════════════════

class FCFHierarchy4(nn.Module):
    """
    4-level иерархия:
      Level 0 (token, λ₁): gates + bind, сжатие каждые N0 токенов
      Level 1 (segment, λ₂): pure bind, сжатие каждые N1 сегментов
      Level 2 (block, λ₃): pure bind, сжатие каждые N2 блоков
      Level 3 (document, λ₄): pure bind

    Контекст: N0 × N1 × N2 × ... × длина шага на верхнем уровне
    Память: 4 × D × 4 байта (не зависит от длины)
    """
    def __init__(self, vocab: int, D: int,
                 lambda_roots: list[float],
                 compress_steps: list[int],  # [N0, N1, N2]
                 trainable_V: bool = False):
        super().__init__()
        self.D = D
        self.vocab = vocab
        self.compress_steps = compress_steps

        self.embed = nn.Embedding(vocab, D)
        self.lm_head = nn.Linear(D, vocab, bias=False)
        self.final_norm_w = nn.Parameter(torch.ones(D))

        # Level 0: гибрид (gates + bind)
        self.cell_0 = HybridFCFCell(
            D, lambda_roots[0], n_gates=8, trainable_V=trainable_V)

        # Levels 1-3: pure FCF
        self.cell_1 = PureFCFCell(D, lambda_roots[1], trainable_V=trainable_V)
        self.cell_2 = PureFCFCell(D, lambda_roots[2], trainable_V=trainable_V)
        self.cell_3 = PureFCFCell(D, lambda_roots[3], trainable_V=trainable_V)

        # Выучиваемые веса комбинации уровней
        self.combine_w = nn.Parameter(torch.ones(4) / 4)

    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(dim=-1, keepdim=True) / (self.D ** 0.5)
        return x / (rms + 1e-6) * self.final_norm_w

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        device = tokens.device

        # ─── Level 0: token-level ────────────────────────────────────
        pos_0 = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pe_0 = position_encoding(pos_0, self.D)

        h = self.embed(tokens)
        out_0, state_0 = self.cell_0(h, pe_0)

        # Compress → Level 1
        N0 = self.compress_steps[0]
        n_segments = max(1, (L + N0 - 1) // N0)
        seg_in, seg_pos = [], []
        for s in range(n_segments):
            start, end = s * N0, min((s + 1) * N0, L)
            z = compress_segment(out_0[:, start:end], pe_0[:, start:end])
            seg_in.append(z); seg_pos.append(s)

        z_1 = torch.stack(seg_in, dim=1)  # (B, S, D)
        pos_1 = torch.tensor(seg_pos, device=device).unsqueeze(0).expand(B, -1)
        pe_1 = position_encoding(pos_1, self.D)

        # ─── Level 1: segment-level ──────────────────────────────────
        out_1, state_1 = self.cell_1(z_1, pe_1)

        # Compress → Level 2
        N1 = self.compress_steps[1]
        n_blocks = max(1, (n_segments + N1 - 1) // N1)
        blk_in, blk_pos = [], []
        for b in range(n_blocks):
            start, end = b * N1, min((b + 1) * N1, n_segments)
            z = compress_segment(out_1[:, start:end], pe_1[:, start:end])
            blk_in.append(z); blk_pos.append(b)

        z_2 = torch.stack(blk_in, dim=1)  # (B, T, D)
        pos_2 = torch.tensor(blk_pos, device=device).unsqueeze(0).expand(B, -1)
        pe_2 = position_encoding(pos_2, self.D)

        # ─── Level 2: block-level ────────────────────────────────────
        out_2, state_2 = self.cell_2(z_2, pe_2)

        # Compress → Level 3
        N2 = self.compress_steps[2]
        n_docs = max(1, (n_blocks + N2 - 1) // N2)
        doc_in, doc_pos = [], []
        for d in range(n_docs):
            start, end = d * N2, min((d + 1) * N2, n_blocks)
            z = compress_segment(out_2[:, start:end], pe_2[:, start:end])
            doc_in.append(z); doc_pos.append(d)

        z_3 = torch.stack(doc_in, dim=1)  # (B, D_docs, D)
        pos_3 = torch.tensor(doc_pos, device=device).unsqueeze(0).expand(B, -1)
        pe_3 = position_encoding(pos_3, self.D)

        # ─── Level 3: document-level ─────────────────────────────────
        out_3, state_3 = self.cell_3(z_3, pe_3)

        # ─── Комбинация уровней ──────────────────────────────────────
        # Каждый токен получает взвешенную сумму readout со всех уровней
        w = torch.softmax(self.combine_w, dim=0)
        # Level 1-3 unbind с позициями ТЕКУЩЕГО токена (pe_0)
        read_1 = unbind(state_1.unsqueeze(1), pe_0)  # (B, L, D)
        read_2 = unbind(state_2.unsqueeze(1), pe_0)
        read_3 = unbind(state_3.unsqueeze(1), pe_0)

        combined = (w[0] * out_0 + w[1] * read_1 +
                    w[2] * read_2 + w[3] * read_3)
        combined = self.rms_norm(combined)

        return self.lm_head(combined)  # (B, L, V)

    def apply_procrustes_all(self):
        """Procrustes на все обучаемые V."""
        for cell in [self.cell_0, self.cell_1, self.cell_2, self.cell_3]:
            if hasattr(cell, 'apply_procrustes'):
                cell.apply_procrustes()


# ═══════════════════════════════════════════════════════════════════════
# Обучение с Procrustes
# ═══════════════════════════════════════════════════════════════════════

def train_hierarchy(model: FCFHierarchy4, data_path: str,
                    vocab: int, batch_size: int, epochs: int,
                    lr: float = 1e-3, procrustes_every: int = 100):
    import numpy as np
    data = np.load(data_path)
    n_train = len(data) - 500

    train_x = torch.from_numpy(data[:n_train, :-1].copy()).long()
    train_y = torch.from_numpy(data[:n_train, 1:].copy()).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * epochs)
    DEVICE = next(model.parameters()).device

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (bx, by) in enumerate(loader):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, vocab), by.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Procrustes: восстанавливаем ортогональность V
            if step % procrustes_every == 0:
                model.apply_procrustes_all()

            total_loss += loss.item()
            if step % 50 == 0:
                ppl = math.exp(total_loss / (step + 1))
                print(f'  [{epoch+1}/{epochs}] step {step}: ppl={ppl:.1f}')

        train_ppl = math.exp(total_loss / len(loader))
        print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}')

    torch.save(model.state_dict(), 'fcf_hierarchy4.pt')
    print('Saved: fcf_hierarchy4.pt')


# ═══════════════════════════════════════════════════════════════════════
# Extrapolation test
# ═══════════════════════════════════════════════════════════════════════

def test_extrapolation(model: FCFHierarchy4, data_path: str,
                       lengths: list[int] = [128, 256, 512, 1024, 4096]):
    """Измеряет PPL на разных длинах последовательности."""
    import numpy as np
    data = np.load(data_path)
    DEVICE = next(model.parameters()).device
    model.eval()

    # Ищем непрерывный кусок данных
    for L in lengths:
        seq = torch.from_numpy(data[0, :L].copy()).long().to(DEVICE)
        if seq.shape[0] < L + 1:
            continue
        with torch.no_grad():
            logits = model(seq[:-1].unsqueeze(0))
            loss = F.cross_entropy(logits.reshape(-1, model.vocab),
                                   seq[1:].reshape(-1))
            ppl = math.exp(loss.item())
        print(f'  L={L:5d}: PPL = {ppl:.1f}')
```

### 4.3. Контекст

| Уровень | Шагов | Размер шага | Полный контекст |
|---------|-------|-------------|-----------------|
| Level 0 | 64 | 1 токен | 64 токена |
| Level 1 | 32 | 64 токена | 2.048 токена |
| Level 2 | 32 | 2048 токенов | 65.536 токенов |
| Level 3 | 64 | 65K токенов | **4.1M токенов** |

**Память:** 4 × 2048 × 4 байта = **32 KB**.

### 4.4. Задачи

| Задача | Описание | Приоритет |
|--------|----------|-----------|
| 4.1.1 | Имплементировать HierarchyLevel (стек LDBlock + MLP) | **Critical** |
| 4.1.2 | Имплементировать Compression (pool + Linear) между уровнями | **Critical** |
| 4.1.3 | Настроить Distributed training (FSDP) | **High** |
| 4.1.4 | Подготовить датасет 100B+ токенов | **High** |
| 4.1.5 | Обучить baseline (L=128) для сравнения | **High** |
| 4.1.6 | Extrapolation тест: PPL на L=256, 512, 1024, 4096 | **High** |
| 4.1.7 | Сравнение PPL на длинных контекстах vs transformer | Medium |
| 4.1.8 | ZeckendorfReadout для edge deployment | Medium |

### 4.5. Эксперименты

| Эксперимент | Описание |
|-------------|----------|
| E4.1 | HierarchyLD 3-level (8+6+4) vs flat λ_d 1.2B |
| E4.2 | 3 уровня vs 4 уровня (стоит ли Level 3?) |
| E4.3 | D=2048 vs D=1024 (trade-off качество/скорость) |
| E4.4 | mean_pool vs λ_d-based pooling vs Cross-Attention compression |

### 4.6. Milestones и ресурсы

| Milestone | Описание | Время |
|-----------|----------|-------|
| M4.1 | HierarchyLD реализован, тест на разных D ✅ | 1 неделя |
| M4.2 | 3-level стек обучен, PPL baseline известен | 6 недель |
| M4.3 | Extrapolation на 4K+ токенов замерен | 7 недель |
| M4.4 | Решение: Phase 4 (2.5B) или pivot | 8 недель |

**Ресурсы:**

| Ресурс | Количество | Стоимость |
|--------|-----------|-----------|
| GPU | 4 × A100 80GB | ~$5K (Lambda/Colab) |
| RAM | 256GB+ | — |
| Storage | 2TB+ | — |
| Время | 6-8 недель | — |
| Данные | 100B токенов | — |

### 4.7. Возможные результаты

| Сценарий | PPL | Контекст | Действие |
|----------|-----|----------|----------|
| **Лучший** | < 25 | > 4K | → Phase 4 (2.5B) |
| **Средний** | 25-35 | 1K-4K | Оптимизация compression |
| **Худший** | > 35 | < 1K | Пересмотр архитектуры |

### 4.8. Post-Phase 3.5: дистилляция + warm-start (заметка)

> **Варианты ускорения λ_d через ruadapt / Qwen:**

1. **Logit-level distillation**: teacher (ruadapt/Qwen) → logits → KL(NLL + β·KL) для λ_d. Двухфазно: Phase A — экспорт logits (Colab, один раз), Phase B — тренировка λ_d на них.
2. **Embed + MLP transfer**: SVD-перенос весов Qwen embed (V×4096 → V×2048) и MLP (FFN → bottleneck 256) в λ_d. Остальное (95%) — случайная инициализация. Даёт ~10-15% ускорение сходимости.
3. **Sequence-level distillation**: генерация текста teacher'ом → дообучение λ_d на нём.

**Формат teacher'а не важен** — OpenVINO/HF одинаковы для экспорта logits (frozen, только forward).
INT8 шум умеренный — может даже регуляризовать.

**Оптимально:** Qwen 2.5-1.5B как teacher (влезает на T4 16GB вместе с λ_d 95M).
Варианты 1+2 не конфликтуют и могут использоваться одновременно.

---

## 5. Phase 4: Production λ_d

**Цель:** 2.5B параметров, 1M+ контекст, конкуренция с Qwen 3 ruadapt.

> **Корректировка после аудита:** Исходная цель 8B / 1T токенов → ~555 дней на 8×A100. Chinchilla-optimal для 2.5B — 160B токенов (~65 дней на 8×A100). Стратегия: доказать превосходство на малом размере, затем масштабировать.

**Оборудование:** 8 × A100 80GB / H100

### 5.1. Архитектура (HierarchyLD — 3 уровня)

```
Vocab: 150K
Embed+LM Head (shared): 150K × 4096 = 614M

Level 0 (token, D₀=4096, 12 слоёв LDBlock + BottleneckMLP):
  λ=[1.618, 1.839, 1.928, 1.966] × 3 повтора = 12 мод
  stride=1, каждый токен
  ↓ Compression: mean_pool(128) → Linear(4096→8192)

Level 1 (segment, D₁=8192, 6 слоёв LDBlock + BottleneckMLP):
  λ=[1.984, 1.991, 1.996] × 2 повтора = 6 мод
  stride=128 (каждый 128-й токен)
  ↓ Compression: mean_pool(64) → Linear(8192→8192)

Level 2 (block, D₂=8192, 4 слоя LDBlock + BottleneckMLP):
  λ=[1.998, 1.999] × 2 повтора = 4 моды
  stride=8192 (каждый 8192-й токен)

Выход: Linear(3·8192 → 8192) · [h₀_proj, h₁_proj, h₂_proj] → RMSNorm → lm_head

Total: ~8B params (Embed 614M + 22 слоя LDBlock (~200M) + BottleneckMLPs + output)

⚠️ **8B — верхняя граница.** Chinchilla-optimal для Phase 4 — **2.5B / D=4096**: embed 150K×4096=614M, 22 слоя LDBlock ~1.9B, total ~2.5B. 160B токенов, ~65 дней на 8×A100. 8B потребует ~555 дней или 64×H100.
```

**Контекст и скорость:**

| Уровень | D | Слоёв | Stride | Контекст | FLOPs/токен (forward) |
|---------|---|-------|--------|----------|----------------------|
| Level 0 | 4096 | 12 | 1 | 128 (глубокая) | 12 × 33M = 400M |
| Level 1 | 8192 | 6 | 128 | 128×128 = 16K | 6 × 134M / 128 = 6.3M |
| Level 2 | 8192 | 4 | 8192 | ∞ (λ→2) | 4 × 134M / 8192 = 0.065M |
| **Итого** | | **22** | | **∞** | **~406M** |

**Память контекста:** 4096 + 8192 + 8192 = ~20K floats = **~80 KB**.

### 5.2. Ключевое преимущество: скорость инференса на длинном контексте

Главный аргумент λ_d hierarchy — не экономия памяти (на сервере память дешёвая), а **принципиально более быстрый инференс на длинных контекстах**.

**Почему трансформер медленный на 1M токенов:**
- Prefill: attention O(L²) = 10¹² операций → **30-60 сек** на A100
- Per-token decoding: attention O(L) = 10⁶ → замедляется линейно с L
- Решение: дробление контекста, sliding window, RAG — всё lossy, всё компромисс

**Почему λ_d hierarchy быстрый:**
- Prefill: λ_d O(L·D) = 10⁹ операций → **0.3-1 сек** на A100 (в 60× быстрее)
- Per-token decoding: λ_d O(D²) = 10⁷ → **константа**, не зависит от L
- Иерархия: Level 0 (D=896, каждый токен) + Level 1 (D=2048, 1/64) + Level 2 (D=4096, 1/2048) = **~17.5M FLOPs/токен** против **9.6B FLOPs/токен** у трансформера

**Полный контекст × быстрый инференс × без KV-cache — одновременно:**

| Сценарий | Qwen 2.5-7B | λ_d 8B hierarchy |
|----------|-------------|-------------------|
| 128K контекст, prefill | ~10 сек | **~0.3 сек** |
| 128K контекст, 1 tok decode | ~8 ms (зависит от L) | **~0.3 ms** (константа) |
| 1M контекст, prefill | ~60 сек, OOM риск | **~1 сек** |
| Скорость генерации (1M) | падает с L | **константа ~3000 tok/s** |

**Итог:** λ_d hierarchy даёт то, чего не может ни один трансформер — **масштабируемый до миллионов токенов инференс без потери скорости**. Не за счёт чуда, а за счёт O(L) вместо O(L²) и O(D²) вместо O(L·D).

### 5.3. Ожидаемые метрики

| Метрика | Цель (2.5B) | Цель (8B, future) |
|---------|-------------|-------------------|
| PPL (Russian eval) | < 25 | < 18 |
| Контекст (effective) | > 1M токенов | > 1M токенов |
| Inference speed (token/s) | > 3000 tok/s (A100) | > 2000 tok/s (A100) |
| Prefill 1M токенов | < 1 сек | < 2 сек |
| Сравнение с Qwen 3 ruadapt | ≤ 2× PPL gap | ≤ 1.2× PPL gap |
| Training tokens | 160B (Chinchilla-optimal) | 160B-1T |
| GPU hours | ~18K A100-hours | ~60K A100-hours |

### 5.4. Ресурсы

> **Корректировка FLOPs (после аудита):** λ_d имеет ~2.4B FLOPs/token на обучение (forward+backward) на D=4096.
> Для 2.5B модели на 160B токенов: ~3.7e21 FLOPs. 8×A100 (40% MFU ≈ 1 PFLOPS) → ~43 дней.
> Для 8B модели на 1T токенов: ~4.8e22 FLOPs → ~555 дней.
> **Рекомендация:** начинать с 2.5B/160B, масштабировать при доказанном превосходстве.

| Ресурс | Количество | Вариант 2.5B | Вариант 8B |
|--------|-----------|--------------|------------|
| GPU | 8 × A100 80GB | ~$15K (6-8 нед аренды) | ~$50K (3 мес) |
| GPU (альтернатива) | 8 × H100 | ~$25K (4 нед) | ~$80K (2 мес) |
| RAM | 512GB+ | — | 1TB+ |
| Storage | 5TB+ (2.5B) / 10TB+ (8B) | — | — |
| Время | — | 6-8 недель | 3-5 месяцев |
| Данные | 160B токенов | 160B (русский + English + код) | 1T+ токенов |

**Примечание по компрессии между уровнями:** mean_pool — простейший вариант. При падении качества на длинных контекстах — замена на **адаптивную компрессию** (λ_d-based pooling, zero extra params) или **Cross-Attention compression** (Perceiver-style, добавляет O(L·D) FLOPs).

---

## 6. Дорожная карта: график

```
Месяц 1-2:    Phase 2.5 (Consolidation)
  │  ├─ Завершить русский Phase 2
  │  ├─ Scaling law sweep (3-4 модели)
  │  ├─ Transformer baseline
  │  └─ Инфраструктура (FSDP)
  │
Месяц 2-3:    Phase 3.0 (HierarchyLD Prototype)
  │  ├─ HierarchyLD: 3 уровня LDBlock + compression
  │  ├─ Экстраполяция контекста
  │  ├─ Тест: D=512,768,896,1024,1536,2048 ✅
  │  └─ Gate decision: HierarchyLD vs flat λ_d
  │
Месяц 3-5:    Phase 3.5 (Medium Scale)
  │  ├─ HierarchyLD 3-level, D=2048, 8+6+4 слоёв
  │  ├─ 1.2B params, 300B token training
  │  ├─ Extrapolation до 4K+
  │  └─ Сравнение с Qwen 2.5-1.5B
  │
Месяц 5-7:    Phase 4 (Production, 2.5B)
  │  ├─ 2.5B params, D=4096, 12+6+4 слоёв
  │  ├─ 160B token training (Chinchilla-optimal)
  │  ├─ 1M+ контекст
  │  └─ PPL < 25, скорость > 3000 tok/s
  │
Месяц 7+:     Phase 4+ (Scale to 8B, опционально)
  │  ├─ 8B params, D=8192, при доказанном превосходстве
  │  ├─ 555+ days на 8×A100, либо 64×H100 для ускорения
  │  └─ Требует отдельного финансирования (~$80-400K)
```

### 6.1. Gate decisions

На каждой фазе принимается решение о продолжении:

```
Phase 2.5 → PPL < 200 на русском? → Да → Phase 3.0
                                    → Нет → Пересмотр архитектуры

Phase 3.0 → Extrapolation лучше Phase 2? → Да (HierarchyLD) → Phase 3.5
                                           → Нет → Оптимизация compression

Phase 3.5 → PPL < 30, контекст > 4K? → Да → Phase 4 (2.5B)
                                        → Нет → Оптимизация / pivot

Phase 4   → PPL < 25, контекст > 1M? → Готово к релизу (2.5B)
           PPL < 18, контекст > 1M?  → Scale to 8B (требует финансирования)
                                       → Нет → Итерация
```

### 6.2. Что можно параллелить

```
Phase 2.5:
  ┌─ 2.1 Русский (Colab) ────────────────────────┐
  ├─ 2.2 Scaling law (RTX 3090)                   ├── Все параллельно
  ├─ 2.3 Transformer baseline (RTX 3090)          │
  └─ 2.5 Инфраструктура (RTX 3090) ──────────────┘
                     │
                     ▼
Phase 3.0:
  HierarchyLD реализация + тест D-свапа (RTX 3090)
                     │
                     ▼
Phase 3.5:
  HierarchyLD 1.2B (4×A100) ── единственный поток
                     │
                     ▼
Phase 4:
  2.5B Training (8×A100/H100, ~65 дней)
```

---

## Приложение А: Сводка ресурсов по фазам

| Фаза | Параметры | Данные | GPU | Время | Стоимость |
|------|-----------|--------|-----|-------|-----------|
| Phase 2 | 105M | 3B tok | T4 16GB | 2 мес ✅ | $0 (Colab) |
| Phase 2.5 | 40-140M ×4 | 3B tok | RTX 3090 | 5 нед | ~$200 (аренда 3090) |
| Phase 3.0 | 25M-217M | 1B tok | RTX 3090 | 3 нед | ~$100 |
| Phase 3.5 | 1.2B | 100B tok | 4×A100 | 8 нед | ~$5K |
| **Phase 4 (2.5B)** | **2.5B** | **160B tok** | **8×A100** | **6-8 нед** | **~$15K** |
| Phase 4+ (8B) | 8B+ | 160B-1T tok | 64×H100 | 2-5 мес | ~$80-400K |

> **Корректировка:** Phase 4 переориентирована с 8B/1T на 2.5B/160B (Chinchilla-optimal).
> 8B требует отдельного раунда финансирования ($80-400K в зависимости от кластера).

## Приложение Б: Риски и mitigation

| Риск | Вероятность | Влияние | Mitigation |
|------|------------|---------|------------|
| VSA bind/unbind (FCF) не работает | Снят | — | FCF заменён на HierarchyLD (без VSA) |
| mean_pool теряет информацию | Средняя | Среднее | λ_d-based pooling / Cross-Attention compression |
| Экстраполяция контекста не растёт | Средняя | Высокое | Ring-buffer state, увеличение D |
| Colab T4 не тянет 1B+ | Высокая | Среднее | RTX 3090 / rent A100 |
| Overfitting на малых данных | Низкая | Среднее | Регуляризация, dropout |
| FLOPs недооценены | Учтено | — | Phase 4 перепланирована на 2.5B/160B tokens |
| Недостаточно данных для русского | Низкая | Высокое | English + код + multilingua |
