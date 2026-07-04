# FCP — Fibonacci Covariance Processor

**MemBind**: multi-head covariance memory + bilinear bind + Fibonacci spectrum.
No softmax, no sigmoid gates, no attention, no transformers.

## Architecture

```
MemBindBlock:
  conv → norm → bind(u*v) + cov_memory(H heads) + spectral(V·λ·V^T)
```

Three parallel content-dependent mechanisms:
1. **Multi-head Covariance Memory** — M[t] = decay·M[t-1] + i·kᵀk, parallel scan O(log L)
2. **Bind Adaptation (FCF-inspired)** — h_adapt = h + (u * v_enh) @ W_out, replaces all gates
3. **γ-λ Spectral Operator** — V·diag(λ_k)·V^T, Fibonacci roots (1.618, 1.839, 1.927, 1.966)

## Key Results (5000 steps)

| Architecture | PPL | tok/s | vs CovGate |
|-------------|-----|-------|------------|
| CovGate (seq) | 438 | 260 | 100% |
| **MemBind (this)** | **466** | **962** | **94% quality, 3.7× speed** |

## Current Scale: 89M params

- D=896, L=24, bottleneck=896
- H=4 covariance heads, r=16 per head
- 108M total params (incl. frozen V matrices)
- Fits in 2GB VRAM (MX550)

## Training

```bash
python train_phase2.py --arch membind --n_layers 24 --bottleneck 896 \
  --train_chunks 20000 --epochs 10 --data russian
```

## Files

```
├── ld_model/core.py          — MemBindBlock, MemBindStack, parallel_prefix_scan
├── train_phase2.py           — training pipeline (gradient accum + warmup + cosine)
├── analyze_architecture.py   — generates model_analysis.html
├── test_membind.py           — original prototype (reference)
├── russian_tokenizer/        — custom BPE tokenizer (vocab=50000)
└── model_analysis.html       — full architecture report
```

## License

MIT
