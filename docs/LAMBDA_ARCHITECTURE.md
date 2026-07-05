# MemBind: Multi-Head Covariance Memory Language Model

> Архитектура следующего поколения: без softmax, без sigmoid-гейтов, без attention.
> Замена — multi-head covariance memory + bilinear bind + спектральный оператор.

---

## Содержание

1. [Философия: почему не attention](#1-философия-почему-не-attention)
2. [Общая схема модели](#2-общая-схема-модели)
3. [Компонент 1: Causal Convolution (локальное смешивание)](#3-компонент-1-causal-convolution-локальное-смешивание)
4. [Компонент 2: Bilinear Bind (адаптация без гейтов)](#4-компонент-2-bilinear-bind-адаптация-без-гейтов)
5. [Компонент 3: Multi-Head Covariance Memory (глобальная память)](#5-компонент-3-multi-head-covariance-memory-глобальная-память)
   - [5.1 Проекции Key / Query / Gate](#51-проекции-key--query--gate)
   - [5.2 Covariance delta: k^T @ k](#52-covariance-delta-kt--k)
   - [5.3 Linear recurrence: M[t] = d·M[t-1] + i·Δ](#53-linear-recurrence-mt--dmt-1--iδ)
   - [5.4 Parallel prefix scan (Hillis-Steele)](#54-parallel-prefix-scan-hillis-steele)
   - [5.5 Memory read: q @ M @ W_read](#55-memory-read-q--m--w_read)
   - [5.6 Stateful inference: контекст = ∞](#56-stateful-inference-контекст--∞)
6. [Компонент 4: Bind Enhancement (Memory → Bind feedback)](#6-компонент-4-bind-enhancement-memory--bind-feedback)
7. [Компонент 5: γ-λ Spectral Operator](#7-компонент-5-γ-λ-spectral-operator)
   - [7.1 V: случайная ортогональная матрица](#71-v-случайная-ортогональная-матрица)
   - [7.2 λ: спектральный алфавит](#72-λ-спектральный-алфавит)
   - [7.3 Block-wise scaling](#73-block-wise-scaling)
   - [7.4 Типы спектров](#74-типы-спектров)
8. [Компонент 6: Bottleneck MLP](#8-компонент-6-bottleneck-mlp)
9. [Полный forward pass MemBindBlock](#9-полный-forward-pass-membindblock)
10. [Сравнение с Transformer](#10-сравнение-с-transformer)
11. [Масштабирование и ёмкость](#11-масштабирование-и-ёмкость)
12. [Приложение: математические детали](#12-приложение-математические-детали)

---

## 1. Философия: почему не attention

**Проблема attention.** Attention-механизм в трансформерах имеет вычислительную сложность O(L²·D) и память O(L·D) для KV cache. Это делает длинный контекст (L > 8K) непропорционально дорогим.

**Идея MemBind.** Вместо того чтобы хранить все прошлые Key/Value пары и сканировать их линейным поиском (QK^T), MemBind **сжимает историю в ковариационную матрицу** — второй момент распределения скрытых состояний. Это даёт O(1) памяти на контекст и O(D²) вычислений на токен — независимо от длины последовательности.

**Ключевые принципы:**

| Принцип | Attention | MemBind |
|---------|-----------|---------|
| Представление контекста | Все K/V пары (L × D) | Ковариация r×r (r ≪ D) |
| Сложность на токен | O(L·D) | O(D²) — константа |
| Память на контекст | O(L·D) | O(r²) — константа |
| Нелинейность | Softmax | Нет (билинейное взаимодействие) |
| Межтокенное взаимодействие | QK^T (все пары) | Covariance scan (r²) |

---

## 2. Общая схема модели

```
                      ┌──────────────────────────────┐
                      │         lm_head (Linear)      │
                      └────────────┬─────────────────┘
                                   │
                      ┌────────────▼─────────────────┐
                      │        RMSNorm (final)        │
                      └────────────┬─────────────────┘
                                   │
                      ┌────────────▼─────────────────┐
                      │       MemBindStack × L        │
                      │  ┌─────────────────────────┐  │
                      │  │     MemBindBlock × 1     │  │
                      │  │  ┌───────────────────┐  │  │
                      │  │  │  CausalConv1d      │  │  │
                      │  │  │  ↓                 │  │  │
                      │  │  │  +/RMSNorm         │  │  │
                      │  │  │  ↓                 │  │  │
                      │  │  │  Bilinear Bind     │  │  │
                      │  │  │  ↓                 │  │  │
                      │  │  │  Cov Memory (scan) │  │  │
                      │  │  │  ↓                 │  │  │
                      │  │  │  Bind Enhancement  │  │  │
                      │  │  │  ↓                 │  │  │
                      │  │  │  γ-λ Spectral Op   │  │  │
                      │  │  └───────────────────┘  │  │
                      │  │  + BottleneckMLP        │  │
                      │  └─────────────────────────┘  │
                      └────────────┬─────────────────┘
                                   │
                      ┌────────────▼─────────────────┐
                      │         Embedding (V × D)     │
                      └────────────┬─────────────────┘
                                   │
                                input_ids
```

**Параметры (current config):**

| Компонент | Параметры | Доля |
|-----------|-----------|------|
| Embedding (weight-tied) | V × D = 44.8M | 50.3% |
| MLP (24 слоя) | 24 × 2 × D × bottleneck = 38.6M | 43.3% |
| Covariance (24 слоя) | 24 × (H·2·D·r + H·r·D + D·br) = 4.5M | 5.0% |
| Bind (24 слоя) | 24 × 3·D·br = 1.0M | 1.1% |
| Frozen V (24 слоя) | 24 × D² = 19.3M | (не обучается) |
| **Trainable total** | **89.1M** | |

---

## 3. Компонент 1: Causal Convolution (локальное смешивание)

**Назначение:** обеспечить локальный n-граммный контекст до K токенов.

**Устройство:** Depthwise CausalConv1d — каждый канал обрабатывается независимо.

```python
# kernel_size = 48, padding = kernel_size - 1 (только слева)
h_conv = causal_conv1d(h)   # (B, L, D)
h_norm = rms_norm(h + h_conv)
```

**Receptive field:** каждый токен "видит" K-1 предыдущих токенов через свёртку. RF слоя = K, полный RF стека = 1 + K·L ≈ 1153 токена для L=24, K=48.

**Почему свертка, а не attention:** свёртка — линейный оператор с весом, не зависящим от контента. Это даёт детерминированное локальное смешивание без квадратичной стоимости. Глобальное смешивание — задача Covariance Memory.

**Stateful inference:** при инкрементальном инференсе сохраняется буфер последних K-1 входов (B × D × (K-1) = 896 × 47 × 4 ≈ 168KB). Буфер конкатенируется с новым токеном перед свёрткой, обеспечивая идентичность полному скану.

---

## 4. Компонент 2: Bilinear Bind (адаптация без гейтов)

**Назначение:** замена FCF (Fast Weight) — контент-зависимая адаптация скрытого состояния через билинейное взаимодействие, без sigmoid/softmax.

**Устройство:**

```python
# Две проекции на low-rank пространство
u = h_norm @ W_u      # (B, L, D) @ (D, bind_r) → (B, L, bind_r)
v = h_norm @ W_v      # (B, L, D) @ (D, bind_r) → (B, L, bind_r)

# Покомпонентное произведение (binding)
h_bind = (u * v_enh) @ W_out   # (B, L, bind_r) @ (bind_r, D) → (B, L, D)
h_adapt = h_norm + h_bind
```

**Принцип:** `u * v` — это bilinear binding в терминах VSA (Vector Symbolic Architectures). В отличие от attention, где взаимодействие идёт через QK^T (dot product всех пар), bind работает в пределах одного токена: два линейных преобразования одного входа перемножаются покомпонентно. Это даёт нелинейность без softmax.

**Bind_r = 16 при D = 896:** сжатие в 56×. Параметры bind: 3·D·bind_r = 43K — менее 2.3% слоя.

---

## 5. Компонент 3: Multi-Head Covariance Memory (глобальная память)

**Это ключевое нововведение MemBind.** Замена KV-cache attention на сжатую ковариационную матрицу, обновляемую через линейную рекуррентность.

### 5.1 Проекции Key / Query / Gate

```python
k_h = h_norm @ W_k_h     # (B, L, r)
q_h = h_norm @ W_q_h     # (B, L, r)
i_h = exp(h_norm @ W_i_h + b_i_h)     # скалярный импульс (0, ∞)
d_h = sigmoid(h_norm @ W_decay_h + b_decay_h)  # decay в (0, 1)
```

**H = 4 головы**, каждая с размерностью r = 16.

- **Key k_h:** кодирует текущий токен в r-мерное пространство для ковариации
- **Query q_h:** запрос к памяти — извлекает информацию из ковариационной матрицы
- **Impulse i_h = exp(⋅):** определяет, сколько новой информации записывается. В отличие от sigmoid (насыщение в 0/1), экспонента не насыщается — может пропустить любое количество информации.
- **Decay d_h = sigmoid(⋅):** скорость забывания. b_i = 1.0 (i_gate ≈ 2.7).

**Multi-timescale heads.** Каждая из H=4 голов имеет свой frozen decay, задающий горизонт памяти τ:

| Голова | τ | d = σ(b_decay) | Назначение |
|--------|---|-----------------|------------|
| 0 | 3 | 0.667 | Локальные n-gram (3-5 токенов) |
| 1 | 12 | 0.920 | Фразы и короткие паттерны |
| 2 | 49 | 0.980 | Предложения |
| 3 | 200 | 0.995 | Абзацы и глобальный контекст |

Значения τ заморожены (register_buffer). Градиент не может их изменить — каждая голова обязана специализироваться на своём горизонте.

**Random Features (cov_rf).** Проекция h_norm в r-мерное пространство k_h делается не напрямую, а через случайное замороженное бутылочное горлышко:

```python
h_rf = h_norm @ R_frozen   # D → p=64, frozen random
k_h = einsum('blp,hpr->bhlr', h_rf, W_k_rf)  # 64 → r, learnable
q_h = einsum('blp,hpr->bhlr', h_rf, W_q_rf)  # 64 → r, learnable
```

Это даёт каждой голове 64-мерное промежуточное представление, сжатое до r=16. Емкость ковариации 16×16 — та же, но информация в k_h богаче в 4 раза. R_frozen (D×64) фиксирован — градиент не может "схлопнуть" базис.

### 5.2 Covariance delta: k^T @ k

```python
K_e = k_h.unsqueeze(-1)              # (B, H, L, r, 1)
delta_h = K_e @ K_e.transpose(-2, -1)  # (B, H, L, r, r) — outer product
delta_scaled = delta_h * i_h.unsqueeze(-1)  # gate-масштабирование
```

**k_h^T @ k_h** — внешнее произведение, дающее r×r матрицу ранга 1. Это ковариация проекций k_h: она кодирует корреляции между r компонентами в ответ на текущий токен. Умножение на импульс i_h определяет силу записи.

### 5.3 Linear recurrence: M[t] = d·M[t-1] + i·Δ

**Сердце памяти.** Каждая голова поддерживает r×r матрицу M_h[t], которая обновляется как взвешенное скользящее среднее ковариаций:

```
M_h[t] = d_h[t] · M_h[t-1] + i_h[t] · (k_h[t]^T @ k_h[t])
```

Это **линейная рекуррентность первого порядка** — та же форма, что у S4/Mamba, но без структуры HiPPO. M_h ∈ ℝ^{r×r} растёт не с длиной контекста, а с квадратом ранга.

**Что хранит M_h?** Сумму затухающих ковариаций. Каждый новый токен добавляет rank-1 матрицу k^T@k, а decay амортизирует старые. После τ = 1/(1-d) токенов вклад убывает в e раз. При d = 0.99 горизонт τ ≈ 100 токенов.

### 5.4 Parallel prefix scan (Hillis-Steele)

Для тренировки (обработка полной последовательности длины L) рекуррентность вычисляется параллельно через ассоциативный сканер:

```python
def parallel_prefix_scan(a, b, state=None):
    """
    M[t] = a[t]·M[t-1] + b[t], M[-1] = state (или 0)
    a: (B, L, H) decay
    b: (B, L, H, r, r) delta
    Returns: M_all, final_state
    """
    A = a.unsqueeze(-1).unsqueeze(-1)  # (B, L, H, 1, 1)
    M = b.clone()
    stride = 1
    while stride < L:
        # Комбинируем блоки размером stride
        A = combine(A)
        M = combine(M)
        stride *= 2
    # После scan: M[t] = рекуррентность от M[-1]=0
    if state is not None:
        M = M + A * state.unsqueeze(1)  # добавляем влияние начального состояния
    return M, M[:, -1]
```

**Алгоритм Hillis-Steele (O(L log L)):**

На каждой итерации удваивается расстояние, на которое распространяется информация:
- Шаг 1: M[t] = a[t]·M[t-1] + b[t] (дистанция 1)
- Шаг 2: M[t] = a[t]·a[t-1]·M[t-2] + ... (дистанция 2)
- Шаг 4: дистанция 4
- ... до L

Количество итераций: log₂(L). Для L=128: 7 итераций.

### 5.5 Memory read: q @ M @ W_read

```python
mem_r = q_h[t] @ M_h[t]            # (B, L, r) — чтение из памяти
mem_d = mem_r @ W_read_h            # (B, L, D) — проекция обратно в D
mem_sum = Σ_h mem_d                 # (B, L, D) — сумма по головам
```

Ковариационная матрица M_h применяется к запросу q_h: `q_h @ M @ W_read`. Это линейное преобразование контекстно-зависимо (через q_h) и исторически-зависимо (через M_h). Результат суммируется по всем головам.

**Аналогия с attention:**

| Attention | Cov Memory |
|-----------|------------|
| QK^T: все пары (L²) | k^T@k: один токен (r²) |
| softmax: competition | i_h: independent scaling |
| AV: weighted sum | q@M: linear read |
| KV cache: O(L·D) | M: O(r²) |

### 5.6 Stateful inference: контекст = ∞

При инференсе (1 токен за раз) рекуррентность вычисляется инкрементально:

```python
M[t] = a[t] * M[t-1] + b[t]    # O(r²), не O(L·log(L))
```

Состояние M (96KB для 24 слоёв × 4 головы × 16×16) передаётся между шагами. Контекст неограничен — каждый новый токен просто обновляет M с весовым коэффициентом. **Никакого роста памяти или вычислений с длиной контекста.**

### 5.7 First Moment (cov_first_moment)

Наряду со вторым моментом Σ[t] = d·Σ[t-1] + i·(k@kᵀ), модель отслеживает **первый момент**:

```
µ[t] = d·µ[t-1] + i·k
```

где µ ∈ ℝʳ — скользящее среднее проекций k. Чтение: `µ_read = q_µ @ µ`, где q_µ ∈ ℝʳ⁻¹ — обучаемый параметр (64 числа на слой). Результат добавляется к mem_sum перед bind.

**Зачем?** Второй момент (ковариация) теряет знак — k·kᵀ одинаков для k и -k. Первый момент сохраняет направление: положительные и отрицательные компоненты k разделяются. Это удваивает ёмкость 16×16 без увеличения r.

### 5.8 Parallel prefix scan для векторов

Для первого момента используется `parallel_prefix_scan_1d` — тот же Hillis-Steele, но для (H, r)-векторов вместо (H, r, r)-матриц:

```python
def parallel_prefix_scan_1d(a, b, state=None):
    # v[t] = a[t]·v[t-1] + b[t]
    # a: (B, L, H), b: (B, L, H, r), state: (B, H, r)
```

FLOPs: O(L·H·r) против O(L·H·r²) для ковариации — накладные расходы < 1%.

---

## 6. Компонент 4: Bind Enhancement (Memory → Bind feedback + Cognitive Mirror)

**Назначение:** feedback от ковариационной памяти к bind-адаптации.

### 6.1 Consensus read (bind enhancement)

```python
v_enh = v + (mem_sum @ W_mem2v)     # (B, L, bind_r)
```

Память модулирует bind: `v` усиливается прочитанным из памяти содержимым.

**W_mem2v** ∈ ℝ^{D × bind_r} — проекция mem_sum в пространство bind.

### 6.2 Cognitive Mirror

**Назначение:** модель "смотрит" на собственные multi-head выходы и корректирует себя, если головы расходятся.

После чтения каждой головы из Σ получается H разных D-мерных векторов. Если все головы согласны — std(head_out) ≈ 0, зеркало молчит. Если головы расходятся — std(head_out) > 0, зеркало активируется:

```python
head_out = einsum('blhr,hro->blho', mem_r, W_read)  # (B, L, H, D)

# Consensus
mem_sum = head_out.sum(dim=2)  # (B, L, D)

# Mirror: disagreement → correction via bind
disagreement = head_out.std(dim=2)           # (B, L, D) — std across heads
u_m = disagreement @ W_u_m                   # (B, L, bind_r)
v_m = h_norm @ W_v_m                         # (B, L, bind_r)
mirror_delta = (u_m * v_m) @ W_out_m         # (B, L, D)
mem_sum = mem_sum + mirror_delta * mirror_scale
```

**Ключевое:** зеркало использует **тот же bind-механизм** (u * v @ W_out), что и основная модель. Единственное отличие — вход для u_m — disagreement, а не h_norm. Никаких новых операций, никаких softmax.

**Параметры:** W_u_m, W_v_m ∈ ℝ^{D × bind_r}, W_out_m ∈ ℝ^{bind_r × D}, mirror_scale — скаляр. Всего 3·D·bind_r + 1 ≈ 43K — менее 2% слоя при D=896.

**Когнитивный эффект:** если heads с разными τ (3, 12, 49, 200) дают разные ответы, модель видит конфликт между коротким и длинным контекстом. Зеркало превращает этот спор в коррекцию скрытого состояния — без внешнего модуля, без softmax arbitration.

---

## 7. Компонент 5: γ-λ Spectral Operator

### 7.1 V: случайная ортогональная матрица

V ∈ O(D) — случайная ортогональная матрица, замороженная после инициализации:

```python
def random_orthogonal(D):
    V = torch.eye(D)
    for _ in range(32):
        u = torch.randn(D)
        u = u / u.norm()
        V = V - 2 * torch.outer(V @ u, u)   # Householder reflection
    return V
```

Случайное ортогональное V = случайный базис в ℝ^D. Spectral operator использует V как словарь частот: проекция h_adapt на V^T раскладывает сигнал по этому базису; применение λ масштабирует каждую компоненту; обратная проекция V восстанавливает сигнал с изменённым спектром.

**Математика:** Δ = V · diag(λ) · V^T · h_adapt — это разложение по сингулярным числам (SVD-like), где λ — эквивалент сингулярных чисел, а V — ортогональный базис.

### 7.2 λ: спектральный алфавит

λ_k — спектральные коэффициенты, определяющие резонанс/затухание каждой частотной компоненты:

| λ < 1 | λ = 1 | λ > 1 |
|-------|-------|-------|
| Демпфирование (информация затухает) | Сохранение (нейтрально) | Усиление (информация резонирует) |

Аналогия с пластиной Хлади: каждый λ_k — это частота собственного колебания пластины. λ_k < 1 — затухающая мода, λ_k > 1 — резонирующая.

**Биполярный спектр [0.8, 1.8]:** первые моды λ < 1 (демпфирование, локальная информация), последние λ > 1 (усиление, глобальные паттерны). Тот же диапазон, что температура в LLM.

### 7.3 Block-wise scaling

Спектральные коэффициенты применяются не ко всем D размерностям одинаково — D разбивается на K блоков, каждый со своим λ_k:

```python
h_proj = h_adapt @ V_T               # (B, L, D) — проекция на базис V
# Разбиение на K блоков
h_blocks = split(h_proj, block_sizes, dim=-1)
# Масштабирование каждого блока своим λ_k
h_scaled = cat([b * lam for b, lam in zip(h_blocks, λ)])
# Обратная проекция
Δ = h_scaled @ V                     # (B, L, D)
h_out = h + Δ
```

**Block sizes** зависят от типа спектра:
- **fib_root, linear, hybrid**: равные блоки (D/K)
- **fib_seq**: блоки ∝ числам Фибоначчи: [10, 21, 31, 51, 82, 134, 216, 351] при D=896, K=8

### 7.4 Типы спектров

| Тип | λ распределение | Block sizes | Описание |
|-----|----------------|-------------|----------|
| `fib_root` | Корни Fibonacci: 1.618, 1.839, 1.928,... | Равные | Оригинальный, λ сходятся к 2 |
| `fib_seq` | F₂..F₉ нормализованные в [lo, hi] | По Фибоначчи | Увеличение блоков с частотой |
| `fib_ratio` | F_{k+1}/F_k нормализованные | Равные | Ритмы золотого сечения |
| `linear` | Равномерно в [lo, hi] | Равные | Полный спектральный охват |
| `hybrid` | Среднее fib_ratio + linear | Равные | Компромисс |

---

## 8. Компонент 6: Bottleneck MLP

```python
h_mlp = self.down(silu(self.up(h_layer)))   # D → bottleneck → D
```

Стандартный SwiGLU-style MLP без расширения (bottleneck = D). 2·D·bottleneck параметров. Для D=896: 1.6M на слой, 42.6% слоя.

---

## 9. Полный forward pass MemBindBlock

```
Вход: h (B, L, D), state = (M_prev, µ_prev, conv_buf) or None
──────────────────────────────────────────────────────────────
1. h_conv, conv_buf ← CausalConv1d(h, conv_buf)
2. h_norm ← rms_norm(h + h_conv)
3. u, v ← h_norm @ W_u, h_norm @ W_v
4. Для каждой головы h = 1..H:
   h_rf ← h_norm @ R_frozen               (Random Features, D→p=64)
   k ← einsum('blp,hpr→bhlr', h_rf, W_k_rf)   (Key, 64→r)
   q ← einsum('blp,hpr→bhlr', h_rf, W_q_rf)   (Query, 64→r)
   i ← exp(einsum(·) + b_i)                (Impulse gate, b_i=1.0)
   d ← sigmoid(einsum(·) + b_decay_h)      (Decay gate, τ frozen per head)
   Δ_Σ ← i · (k^T @ k)                     (Second moment delta, r×r)
   Δ_µ ← i · k                              (First moment delta, r)
   M_all, M_new ← parallel_prefix_scan(d, Δ_Σ, M_prev_h)   (Cov scan)
   µ_all, µ_new ← parallel_prefix_scan_1d(d, Δ_µ, µ_prev_h) (1st moment)
   mem ← q @ M_all @ W_read_h               (Memory read: Σ)
   mem_mu ← q_µ @ µ_all                     (Memory read: µ)
5. head_out = Σ_h [mem @ W_read_h + mem_mu]
   mem_sum = head_out.sum(dim=2)             (Consensus over heads)
6. Cognitive Mirror (если heads disagree):
   disagreement ← head_out.std(dim=2)        (std across H heads)
   u_m ← disagreement @ W_u_m                (Mirror u)
   v_m ← h_norm @ W_v_m                      (Mirror v)
   mirror_delta ← (u_m * v_m) @ W_out_m      (Mirror correction)
   mem_sum += mirror_delta * mirror_scale
7. v_enh ← v + mem_sum @ W_mem2v             (Bind enhancement)
8. h_adapt ← h_norm + (u * v_enh) @ W_out
9. Δ_spec ← V · diag(λ) · V^T · h_adapt     (Spectral transform)
10. h_out ← h + Δ_spec                       (Residual)
──────────────────────────────────────────────────────────────
Выход: h_out (B, L, D), state = (M_new, µ_new, conv_buf)
```

---

## 10. Сравнение с Transformer

| Характеристика | Transformer | MemBind |
|---------------|------------|---------|
| Вычислительная сложность на токен | O(L·D) attention + O(D²) MLP | O(D²) spectral + O(D²) MLP |
| Память контекста (inference) | O(L·D) KV cache | O(r²) covariance (константа) |
| Max контекст на 2GB (D=896) | ~18K токенов (fp32) | **Неограничен** |
| FLOPs на токен (L=128) | 9.9M | 3.45M |
| FLOPs на токен (L=8K) | 24.3M | 3.45M (та же!) |
| FLOPs на токен (L=128K) | 229M | 3.45M (та же!) |
| Параметры | V·D + 12·L·D² | V·D + 2.38·L·D² |
| Нелинейность | Softmax + GeLU | Bilinear bind (u*v) + SiLU + exp |
| Gradient flow | O(L²) через softmax | O(L) через линейную рекуррентность |

---

## 11. Масштабирование и ёмкость

### Параметры

```python
N_trainable = V·D + L·(3·D·bind_r + H·(4·D·r) + 2·D·bottleneck + 2·D)

# При D = bottleneck, bind_r ≈ D/50, r ≈ D/50, H = 4:
N_trainable ≈ V·D + L·(2.38·D²)
N_frozen = L·D²                               # V-матрицы (заморожены)
```

### FLOPs на токен (на слой)

| Компонент | FLOPs | Доля при D=896 |
|-----------|-------|-----------------|
| Spectral | 2·D² | 46.7% |
| MLP | 2·D·bottleneck | 46.7% |
| Covariance | H·(3·D·r + 2·r²) | 5.3% |
| Bind | 3·D·bind_r | 1.3% |

### Информационная ёмкость памяти

Covariance memory M ∈ ℝ^{H × r × r} = 4 × 16 × 16 = 1024 чисел (4KB fp32).
Степени свободы: при τ = 1/(1-d) эффективный ранг min(r, τ/2).
При τ = 8 (начальный decay): ≈ 67 DoF/head → 8.5K бит всего.
При τ = 100 (обученный decay): 128 DoF/head → 16K бит.

### Контекст на 2GB VRAM (MX550)

| | Transformer (D=768) | **MemBind (D=896)** |
|---|-------------------|---------------------|
| fp32 inference | 18K токенов | **∞** |
| fp16 inference | 43K токенов | **∞** |
| fp32 training (L=128, B=2) | — | **2.0GB** |

---

## 12. Приложение: математические детали

### RMSNorm

```python
def rms_norm(x, weight):
    rms = ||x|| / √D
    return x / rms * weight
```

Нормализация по среднеквадратичному отклонению, без центрирования (среднее не вычитается). Быстрее LayerNorm.

### CausalConv1d

```python
# Depthwise: groups = D, kernel = K
weight: (D, 1, K)   # отдельный фильтр на канал
pad: (K-1, 0)       # padding только слева (каузальность)
```

### Parallel prefix scan

Рекуррентность M[t] = a[t]·M[t-1] + b[t] — **линейное ассоциативное отображение**:

```
(M[t], A[t]) = (a[t]·M[t-1] + b[t], a[t]·A[t-1])
```

Комбинатор: `(M_i, A_i) ∘ (M_j, A_j) = (A_j·M_i + M_j, A_j·A_i)`.

Hillis-Steele использует этот комбинатор для параллельного вычисления за O(log L) шагов.

### Почему λ ∈ [0.8, 1.8]?

Диапазон совпадает с температурой в LLM:

- **λ = 0.8**: сильное демпфирование — эквивалент низкой температуры (детерминизм, локальная точность)
- **λ = 1.0**: нейтрально — информация сохраняется без изменений
- **λ = 1.8**: усиление — эквивалент высокой температуры (креативность, глобальные паттерны)

Биполярный спектр означает, что часть спектральных компонент гасит информацию, часть усиливает. Это создаёт wavelet-like multiresolution: низкие λ работают на локальных деталях, высокие — на глобальной структуре.

---

> **MemBind**: Language modeling through covariance memory, not attention.
> 94% качества CovGate при 3.7× скорости.
> Контекст = ∞ на 2GB GPU.
