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

### Step-Level Metrics

| Step | Loss | PPL | LR | Grad Norm | Embed Norm | Logits Mean | Logits Std | Hidden Mean | Hidden Std |
|------|------|-----|----|-----------|------------|-------------|------------|-------------|------------|
| 1000 | 9.8631 | 19208.4 | 0.000400 | 0.6538 | 133.38 | −2.2306 | 1.5975 | 0.0762 | 0.9971 |
| 2000 | 9.0840 | 8812.8 | 0.000800 | 0.0000* | 139.68 | −2.2462 | 1.6179 | 0.0228 | 0.9926 |
| 3000 | 8.7217 | 6134.7 | 0.001000 | 0.9123 | 146.27 | −2.4804 | 1.6885 | 0.0198 | 0.9924 |
| 4000 | 8.4700 | 4769.5 | 0.000998 | 0.0000* | 153.32 | −2.5857 | 1.7260 | 0.0185 | 0.9918 |
| 5000 | 8.2636 | 3880.1 | 0.000993 | 0.9114 | 160.35 | −2.7270 | 1.7681 | 0.0188 | 0.9920 |
| 6000 | 8.0842 | 3242.9 | 0.000987 | 0.0000* | 167.18 | −2.8605 | 1.8005 | 0.0205 | 0.9931 |
| 7000 | 7.9289 | 2776.4 | 0.000978 | 0.8125 | 172.92 | −2.9481 | 1.8246 | 0.0220 | 0.9940 |
| 8000 | 7.7958 | 2430.4 | 0.000967 | 0.0000* | 178.48 | −3.0667 | 1.8524 | 0.0219 | 0.9945 |
| 9000 | 7.6739 | 2151.5 | 0.000955 | 0.7013 | 183.77 | −3.1583 | 1.8762 | 0.0231 | 0.9956 |
| 10000 | 7.5639 | 1927.3 | 0.000940 | 0.0000* | 188.74 | −3.2362 | 1.8931 | 0.0228 | 0.9961 |

\* Grad norm = 0.0 due to report generation after `optimizer.zero_grad()` (bug fixed in commit `07108c4`).

### Epoch Summary

| Epoch | Train PPL | Eval PPL | Best |
|-------|-----------|----------|------|
| 1 | 1927.3 | **884.4** | ✓ best |

---

## Epoch 2 (steps 10000–20000, in progress)

### Step-Level Metrics

| Step | Loss | PPL | LR | Grad Norm | Embed Norm | Logits Mean | Logits Std | Hidden Mean | Hidden Std |
|------|------|-----|----|-----------|------------|-------------|------------|-------------|------------|
| 10100 | 6.3808 | 590.4 | 9.38e-04 | — | — | — | — | — | — |
| 10200 | 6.3696 | 583.8 | 9.37e-04 | — | — | — | — | — | — |
| 10300 | 6.3236 | 557.6 | 9.35e-04 | — | — | — | — | — | — |
| 10400 | 6.3241 | 557.9 | 9.33e-04 | — | — | — | — | — | — |
| 10500 | 6.3198 | 555.5 | 9.32e-04 | — | — | — | — | — | — |
| 10600 | 6.3138 | 552.1 | 9.30e-04 | — | — | — | — | — | — |
| 10700 | 6.3057 | 547.7 | 9.28e-04 | — | — | — | — | — | — |
| 10800 | 6.2992 | 544.1 | 9.27e-04 | — | — | — | — | — | — |
| 10900 | 6.2883 | 538.2 | 9.25e-04 | — | — | — | — | — | — |
| **11000** | **6.2877** | **537.9** | **9.23e-04** | **0.7334** | **194.00** | **−1.8797** | **2.0401** | **0.0638** | **0.9980** |
| 11100 | 6.2888 | 538.5 | 9.21e-04 | — | — | — | — | — | — |

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

### Key Changes at Epoch 2 Start

- **PPL crashed** from 1927 → 538 in 1000 steps — rapid improvement
- **Logits spread widening** (std 1.89 → 2.04, range −3.24/+0.65 → −1.88/+16.35) — model more confident
- **W_v explosion** (+152%) — bind value path dominating
- **Covariance memory path shrinking** — W_read −42%, W_mem2v −26%
- **Impulse gate closing** — b_i mean went from 0.945 → −0.062 (sigmoid 0.72 → 0.48)
- **τ ≈ 7.8 tokens** — decay constant unchanged
- **Hidden state σ ≈ 1.0** — normalization still perfect
- **Grad norm fix working** — 0.7334 (no more zeros)

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

1. **Loss drops consistently** from 9.86 → 7.56 (PPL 19208 → 1927) over 10K steps
2. **Eval PPL (884) < Train PPL (1927)** — model generalizes
3. **τ ≈ 8 tokens** from b_decay ≈ 1.906 (sigmoid → 0.871) — covariance memory horizon
4. **W_read grows strong** (+1514%) — memory read path actively used
5. **W_mem2v grows modestly** (+454%) — memory→value projection less critical at L=128
6. **b_i ≈ 0.95** (sigmoid→0.72) — impulse gate mostly open
7. **Grad norm ∼0.7–0.9** — stable training dynamics
8. **Hidden state σ ≈ 1.0** — consistent normalization maintained

---

## TODOs / Next Steps

- [ ] Record epoch 2 metrics here as they arrive
- [ ] Add per-layer parameter norm table for all layers
- [ ] Log eval PPL at each checkpoint
- [ ] Record wall-clock time per epoch
- [ ] Add learning rate schedule plot data
