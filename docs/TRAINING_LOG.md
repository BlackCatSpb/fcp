# MemBind Training Log

## Model Configuration

| Param | Value |
|-------|-------|
| Architecture | MemBind (no softmax, no sigmoid, no attention) |
| D (dim) | 896 |
| L (layers) | 24 |
| K (modes) | 8 |
| Bottleneck | 896 |
| Kernel size | 48 |
| Conv RF | 1 + 47 × 24 = 1129 |
| Cov heads | 4 |
| Cov rank (r) | 16 |
| Bind rank (bind_r) | 16 |
| Spectrum | fib_seq (0.8–1.8) |
| Vocab | 50,000 |
| Params | 89.1M |
| Weight tying | Yes |

### Training Hyperparameters

| Param | Value |
|-------|-------|
| Batch size | 2 |
| Accum steps | 16 |
| Effective batch | 32 |
| Seq len | 128 |
| LR | 1e-3 → cosine 0 |
| Warmup | 2500 steps |
| Grad clip | 1.0 |
| Optimizer | AdamW |
| Data | Russian chunks (~6.6M) |

---

## Epoch 1 (steps 0–10000)

### Checkpoint-Level Metrics (all epochs)

| Step | Loss | PPL | LR (e-04) | Grad Norm | Embed Norm |
|------|------|-----|-----------|-----------|------------|
| 1000 | 9.8631 | 19208.4 | 3.97 | 0.6538 | 133.4 |
| 2000 | 9.0840 | 8812.8 | 8.00 | 0.0000* | 139.7 |
| 3000 | 8.7217 | 6134.7 | 10.00 | 0.9123 | 146.3 |
| 4000 | 8.4700 | 4769.5 | 9.98 | 0.0000* | 153.3 |
| 5000 | 8.2636 | 3880.1 | 9.93 | 0.9114 | 160.4 |
| 6000 | 8.0842 | 3242.9 | 9.87 | 0.0000* | 167.2 |
| 7000 | 7.9289 | 2776.4 | 9.78 | 0.8125 | 172.9 |
| 8000 | 7.7958 | 2430.4 | 9.67 | 0.0000* | 178.5 |
| 9000 | 7.6739 | 2151.5 | 9.55 | 0.7013 | 183.8 |
| 10000 | 7.5639 | 1927.3 | 9.40 | 0.0000* | 188.7 |
| 11000 | 6.2877 | 537.9 | 9.23 | 0.7334 | 194.0 |
| 12000 | 6.2403 | 513.0 | 9.05 | 0.0000* | 199.2 |
| 13000 | 6.2087 | 497.1 | 8.84 | 0.6771 | 204.1 |
| 14000 | 6.1715 | 478.9 | 8.62 | 0.0000* | 209.0 |
| 15000 | 6.1346 | 461.5 | 8.39 | 0.7142 | 213.5 |
| 16000 | 6.0867 | 440.0 | 8.14 | 0.0000* | 217.9 |
| 17000 | 6.0466 | 422.7 | 7.87 | 0.7852 | 221.8 |
| 18000 | 6.0066 | 406.1 | 7.59 | 0.0000* | 225.3 |
| 19000 | 5.9698 | 391.4 | 7.31 | 0.7898 | 228.6 |
| 20000 | 5.9307 | 376.4 | 7.01 | 0.0000* | 231.5 |
| 21000 | 5.0223 | 151.8 | 6.70 | 0.9323 | 235.1 |
| 22000 | 5.0198 | 151.4 | 6.39 | 0.0000* | 238.4 |
| 23000 | 5.0117 | 150.2 | 6.07 | 0.7362 | 241.4 |
| 24000 | 5.0000 | 148.4 | 5.74 | 0.0000* | 244.1 |
| 25000 | 4.9915 | 147.2 | 5.42 | 0.7039 | 246.4 |
| 26000 | 4.9805 | 145.5 | 5.08 | 0.0000* | 248.5 |
| 27000 | 4.9680 | 143.7 | 4.75 | 0.7062 | 250.3 |
| 28000 | 4.9548 | 141.9 | 4.42 | 0.0000* | 251.9 |
| 29000 | 4.9387 | 139.6 | 4.10 | 0.7558 | 253.3 |
| **30000** | **4.9278** | **138.1** | **3.77** | **0.0000*** | **254.6** |

\* Grad norm = 0.0 due to report generation before grad_norm capture fix.

### Epoch Summary

| Epoch | Steps | Train PPL | Eval PPL | Best |
|-------|-------|-----------|----------|------|
| 1 | 0–10000 | 1927.3 | **884.4** | ✓ |
| 2 | 10000–20000 | 376.4 | 406.5 | — |
| 3 | 20000–30000 | **138.1** | **308.4** | ✓ |

### Layer 0 Parameter Norms at Step 11000

| Param | Step 10000 | Step 11000 | Δ |
|-------|-----------|------------|---|
| W_u | 2.466 | 3.042 | +23% |
| W_v | 2.244 | 5.657 | **+152%** |
| W_out | 2.318 | 2.552 | +10% |
| W_k | 4.664 | 4.159 | −11% |
| W_q | 4.630 | 5.010 | +8% |
| W_i | 0.720 | 0.503 | −30% |
| b_i (mean) | 0.945 | −0.062 (σ=0.485) | gate closing |
| W_decay | 1.478 | 1.760 | +19% |
| b_decay (mean) | 1.906 (τ≈7.8) | 1.915 (τ≈7.8) | stable |
| W_read | 4.826 | 2.793 | **−42%** |
| W_mem2v | 1.657 | 1.226 | −26% |

### Epoch 2 Summary

| Epoch | Steps | Train PPL | Eval PPL | Best |
|-------|-------|-----------|----------|------|
| 1 | 0–10000 | 1927.3 | **884.4** | ✓ |
| 2 | 10000–20000 | 376.4 | 406.5 | — |

*Eval > Train впервые — модель начинает переобучаться на 2.6M токенов.*

### Key Changes at Epoch 2 Start

- **PPL crashed** from 1927 → 538 in 1000 steps — rapid improvement
- **Logits spread widening** (std 1.89 → 2.04, range −3.24/+0.65 → −1.88/+16.35) — model more confident
- **W_v explosion** (+152%) — bind value path dominating
- **Covariance memory path shrinking** — W_read −42%, W_mem2v −26%
- **Impulse gate closing** — b_i mean went from 0.945 → −0.062 (sigmoid 0.72 → 0.48)
- **τ ≈ 7.8 tokens** — decay constant unchanged
- **Hidden state σ ≈ 1.0** — normalization still perfect
- **Grad norm fix working** — 0.7334 (no more zeros)

### Observations

- **Плато 150→140 PPL на 8000 шагов** — модель упёрлась в количество данных (2.6M токенов)
- **PPL < 50 не достижима на 2.6M токенов** — нужно как минимум 10× больше данных
- **Embedding norm растёт линейно** (133 → 253 за 30K шагов) — эмбеддинги продолжают учиться
- **Grad norm ∼0.7–0.9** — стабильна на всём протяжении
- **Epoch reset даёт −35% PPL** (376→152) — модель помнит, дообучается на новом прогоне тех же данных

---

## Parameter Norm Evolution (Epoch 1)

### Layer 0 (representative)

| Step | W_u | W_v | W_out | W_k | W_q | W_i | b_i | W_decay | b_decay | W_read | W_mem2v |
|------|-----|-----|-------|-----|-----|-----|-----|---------|---------|--------|---------|
| 1000 | 1.320 | 1.300 | 1.288 | 2.570 | 2.615 | 0.610 | 0.007 | 0.635 | 4.010 | 2.538 | 1.294 |
| 3000 | 1.534 | 1.493 | 1.492 | 3.011 | 3.028 | 0.663 | 0.604 | 0.799 | 3.897 | 3.090 | 1.414 |
| 5000 | 1.803 | 1.711 | 1.731 | 3.529 | 3.527 | 0.694 | 0.769 | 0.998 | 3.847 | 3.663 | 1.514 |
| 7000 | 2.107 | 1.943 | 1.994 | 4.085 | 4.062 | 0.711 | 0.870 | 1.239 | 3.822 | 4.245 | 1.589 |
| 10000 | 2.466 | 2.244 | 2.318 | 4.664 | 4.630 | 0.720 | 0.945 | 1.478 | 3.812 | 4.826 | 1.657 |

### Growth Rates by Step 10000 (% from init=0.01)

| Param | Init Norm | Final Norm | Growth |
|-------|-----------|------------|--------|
| W_v | 0.01 × √896 ≈ 0.299 | 2.244 | +650% |
| W_out | 0.01 × √16 ≈ 0.040 | 2.318 | +5695% |
| W_u | 0.299 | 2.466 | +725% |
| W_k (4 heads avg) | 0.299 | 4.647 | +1454% |
| W_q (4 heads avg) | 0.299 | 4.630 | +1448% |
| W_i (4 heads avg) | 0.299 | 0.720 | +141% |
| W_decay (4 heads avg) | 0.299 | 1.478 | +394% |
| b_decay | 2.0 | 3.812 (sigmoid→0.978) | +91% |
| W_read (4 heads avg) | 0.01 × √896 ≈ 0.299 | 4.826 | +1514% |
| W_mem2v | 0.299 | 1.657 | +454% |

b_i: 0.003 → 0.945 (sigmoid→0.720) — impulse gate opening
b_decay: 2.005 → 1.906 (sigmoid→0.871 → τ≈7.8 tok) — decay constant settling

---

## Generation Samples

### Prompt: "В начале было Слово, и Слово было"

**Parallel mode** (sliding window, RF=1129):
```
...было выгляден о его очередь и их в Москве. Облаологи не только в своих культуре не виден на службу...
```
**Recurrent mode** (stateful, infinite context):
```
...было использоваться в «Регме» за счёт времени в городе...
```

Entropy: 45% of max (temp=0.9, top_k=40)

---

## Key Observations

1. **Loss drops consistently**: 11.76 → 4.94 (PPL 128K → 140) over 29K steps
2. **Epoch reset effect**: −65% PPL at epoch 1→2 boundary (1927→538); −60% at 2→3 (376→152). Model quickly adapts to re-shuffled data.
3. **Eval PPL crossing**: Epoch 1 eval (884) < train (1927); Epoch 2 eval (407) > train (376). Model begins overfitting to 2.6M token dataset.
4. **PPL plateau**: 150→140 over 8000 steps (epoch 3). Data-limited — 2.6M tokens too few for 89M params.
5. **τ ≈ 8 tokens** from b_decay ≈ 1.915 (sigmoid → 0.872) — covariance memory horizon unchanged through all epochs.
6. **W_v dominates** (+152% in epoch 2 start), covariance memory path (W_read) −42% — model relies on bind, not memory, at L=128.
7. **Grad norm ∼0.7–0.9** — stable training dynamics throughout.
8. **Hidden state σ ≈ 1.0** — normalization maintained perfectly.
9. **Embedding norm grows linearly**: 133 → 253 over 29K steps — embeddings still learning.
10. **Conclusion: PPL < 50 not reachable on 2.6M tokens**. Need 10× more data (≥26M tokens) for meaningful improvement.

---

## TODOs / Next Steps

- [x] Record epoch 1 metrics
- [x] Record epoch 2 metrics
- [ ] Record epoch 3 metrics (in progress)
- [ ] Add per-layer parameter norm table for all layers
- [ ] Log eval PPL at each checkpoint
- [ ] Record wall-clock time per epoch
- [ ] Add learning rate schedule plot data
