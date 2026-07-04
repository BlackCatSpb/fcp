"""
explore_frequencies.py — Полный перебор частотных конфигураций.
"Покрути ручки": K, spacing, block structure, damping.
"""

import math, sys, os
import numpy as np
import torch

D = 896
N_LAYERS = 24

# ─── 1. Генерация разных семейств частот ───────────────────────────────

def fibonacci_roots_k(k):
    """λ_k — характеристический корень k-го порядка Фибоначчи."""
    lo, hi = 1.0, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        powers = mid ** torch.arange(k, -1, -1, dtype=torch.float64)
        f = powers[0] - powers[1:].sum()
        if f > 0: hi = mid
        else: lo = mid
    return float((lo + hi) / 2)

def freqs_fibonacci(K):
    """Standard Fibonacci roots λ₂..λ_{K+1}"""
    return torch.tensor([fibonacci_roots_k(k) for k in range(2, K + 2)], dtype=torch.float32)

def freqs_linear(K, lo=1.0, hi=2.0):
    """Evenly spaced in [lo, hi)"""
    return torch.linspace(lo, hi * (1 - 1e-6), K, dtype=torch.float32)

def freqs_geometric(K, lo=1.0, hi=2.0):
    """Evenly spaced in log space"""
    return torch.exp(torch.linspace(math.log(lo), math.log(hi * (1 - 1e-6)), K, dtype=torch.float32))

def freqs_chebyshev(K, lo=1.0, hi=2.0):
    """Chebyshev nodes (clustered near 1 and 2)"""
    k = torch.arange(1, K + 1, dtype=torch.float32)
    x = -torch.cos((2 * k - 1) / (2 * K) * math.pi)  # cos in [-1, 1]
    return lo + (hi - lo) * (x + 1) / 2

def freqs_powerphi(K, lo=1.0, hi=2.0):
    """Powers of golden ratio, normalized to [lo, hi)"""
    phi = 1.618033988749895
    raw = phi ** torch.arange(0, K, dtype=torch.float32)
    raw = raw / raw.max() * hi * 0.9999
    return raw.clamp(min=lo)

def freqs_sigmoid(K, lo=1.0, hi=2.0):
    """Logistic-sigmoid spaced (S-curve)"""
    x = torch.linspace(-4, 4, K, dtype=torch.float32)
    s = torch.sigmoid(x)
    return lo + (hi - lo) * s

def freqs_alternating(K, lo=1.2, hi=1.8):
    """Mix of low and high, no middle"""
    half = K // 2
    low = torch.linspace(lo, lo + 0.1, half, dtype=torch.float32)
    high = torch.linspace(hi - 0.1, hi, K - half, dtype=torch.float32)
    return torch.cat([low, high])

# ─── 2. Стратегии распределения block_size ──────────────────────────────

def blocks_equal(D, K):
    """Все блоки равного размера"""
    return [D // K] * K

def blocks_log(D, K, ascend=True):
    """Логарифмический рост/спад размеров блоков"""
    raw = torch.logspace(0, 2, K) if ascend else torch.logspace(2, 0, K)
    raw = raw / raw.sum() * D
    sizes = raw.round().int().tolist()
    diff = D - sum(sizes)
    sizes[-1] += diff
    return sizes

def blocks_exp(D, K, ascend=True):
    """Экспоненциальный рост/спад"""
    raw = torch.exp(torch.linspace(0, 3, K)) if ascend else torch.exp(torch.linspace(3, 0, K))
    raw = raw / raw.sum() * D
    sizes = raw.round().int().tolist()
    diff = D - sum(sizes)
    sizes[-1] += diff
    return sizes

# ─── 3. Анализ одной конфигурации ───────────────────────────────────────

def analyze_config(name, freqs, block_sizes, D=D, L=N_LAYERS):
    K = len(freqs)
    total_blocks = sum(block_sizes)
    
    # Range info
    λ_min = freqs.min().item()
    λ_max = freqs.max().item()
    λ_mean = freqs.mean().item()
    
    # Spacing
    if K > 1:
        sorted_f = freqs.sort()[0]
        spacing = [(sorted_f[i+1] - sorted_f[i]).item() for i in range(K-1)]
        mean_spacing = np.mean(spacing)
        min_spacing = min(spacing)
    else:
        spacing, mean_spacing, min_spacing = [], 0, 0
    
    # Condition number of diag(λ)
    cond = λ_max / max(λ_min, 1e-10)
    
    # Block info
    min_block = min(block_sizes)
    max_block = max(block_sizes)
    block_ratio = max_block / max(min_block, 1)
    
    # Effective spectral resolution (dimensions per unique λ)
    dims_per_lambda = [block_sizes[i] for i in range(K)]
    
    # Operator stability: total amplification across L layers
    # Each layer: V·diag(λ)·V^T → norm = max(λ)
    # L layers: max(λ)^L
    total_amplification = λ_max ** L
    
    # If any λ < 1, compute damping ratio
    damping_dims = sum(block_sizes[i] for i in range(K) if freqs[i] < 1.0)
    amplifying_dims = sum(block_sizes[i] for i in range(K) if freqs[i] > 1.0)
    
    # Shannon entropy of block distribution (information spread)
    probs = torch.tensor(block_sizes, dtype=torch.float32) / D
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_entropy = math.log(K)
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
    
    # New metric: spectral diversity (variance of λ)
    λ_var = freqs.var().item()
    
    # Metric: spectral tilt (skewness-like)
    λ_skew = ((freqs - λ_mean) ** 3).mean().item() / (λ_var ** 1.5 + 1e-10) if λ_var > 0 else 0
    
    return {
        'name': name, 'K': K,
        'λ_min': λ_min, 'λ_max': λ_max, 'λ_mean': λ_mean, 'λ_var': λ_var, 'λ_skew': λ_skew,
        'mean_spacing': mean_spacing, 'min_spacing': min_spacing,
        'cond': cond, 'total_amp': total_amplification,
        'total_blocks': total_blocks, 'min_block': min_block, 'max_block': max_block,
        'block_ratio': block_ratio,
        'damping_dims': damping_dims, 'amplifying_dims': amplifying_dims,
        'entropy': entropy, 'entropy_ratio': entropy_ratio,
        'dims_per_lambda': dims_per_lambda,
        'freqs_sorted': sorted(freqs.tolist()),
    }


# ─── 4. Полный перебор всех комбинаций ──────────────────────────────────

print("=" * 80)
print("FREQUENCY EXPLORATION: Chladni-Hilbert Spectral Analysis")
print("=" * 80)

all_results = []

# 4a. Vary K with Fibonacci roots, equal blocks
print("\n" + "-" * 80)
print("4a. VARY K (Fibonacci roots, equal blocks)")
print("-" * 80)
for K in [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]:
    freqs = freqs_fibonacci(K)
    blocks = blocks_equal(D, K)
    r = analyze_config(f"Fibonacci K={K}", freqs, blocks)
    all_results.append(r)
    print(f"  K={K:3d}: λ in [{r['λ_min']:.4f}, {r['λ_max']:.6f}]  "
          f"spacing={r['min_spacing']:.2e}  "
          f"block={r['min_block']}..{r['max_block']}  "
          f"entropy={r['entropy_ratio']:.3f}")

# 4b. Different spacing strategies at K=8
print("\n" + "-" * 80)
print("4b. DIFFERENT SPACING STRATEGIES (K=8, equal blocks)")
print("-" * 80)
spacings = [
    ("Fibonacci", freqs_fibonacci),
    ("Linear [1,2)", lambda K: freqs_linear(K, 1.0, 2.0)),
    ("Linear [1.5,2)", lambda K: freqs_linear(K, 1.5, 2.0)),
    ("Geometric", lambda K: freqs_geometric(K, 1.2, 2.0)),
    ("Chebyshev", lambda K: freqs_chebyshev(K, 1.2, 2.0)),
    ("Sigmoid", lambda K: freqs_sigmoid(K, 1.2, 2.0)),
    ("Power φ", lambda K: freqs_powerphi(K, 1.2, 2.0)),
    ("Alternating", lambda K: freqs_alternating(K, 1.2, 1.8)),
    ("All < 1", lambda K: freqs_linear(K, 0.5, 0.99)),
    ("Mixed <1 + >1", lambda K: torch.cat([
        freqs_linear(K//2, 0.5, 0.99),
        freqs_linear(K - K//2, 1.01, 1.8)
    ])),
]
for name, fn in spacings:
    freqs = fn(8)
    if len(freqs) < 8:  # skip if fewer than 8
        continue
    freqs = freqs[:8]  # trim to 8
    blocks = blocks_equal(D, 8)
    r = analyze_config(f"{name}", freqs, blocks)
    all_results.append(r)
    damp = f" (damping: {r['damping_dims']} dims)" if r['damping_dims'] > 0 else ""
    print(f"  {name:20s}: λ in [{r['λ_min']:.4f}, {r['λ_max']:.4f}]  "
          f"cond={r['cond']:.2f}  amp^L={r['total_amp']:.2e}{damp}")

# 4c. Different block structures at K=8, Fibonacci
print("\n" + "-" * 80)
print("4c. DIFFERENT BLOCK STRUCTURES (K=8, Fibonacci roots)")
print("-" * 80)
block_strategies = [
    ("Equal", lambda K: blocks_equal(D, K)),
    ("Log ↑ (small→large)", lambda K: blocks_log(D, K, ascend=True)),
    ("Log ↓ (large→small)", lambda K: blocks_log(D, K, ascend=False)),
    ("Exp ↑", lambda K: blocks_exp(D, K, ascend=True)),
    ("Exp ↓", lambda K: blocks_exp(D, K, ascend=False)),
]
for name, fn in block_strategies:
    freqs = freqs_fibonacci(8)
    blocks = fn(8)
    r = analyze_config(f"{name}", freqs, blocks)
    all_results.append(r)
    print(f"  {name:25s}: blocks={blocks}  "
          f"ratio={r['block_ratio']:.1f}  entropy={r['entropy_ratio']:.3f}")

# 4d. Very large K — what happens at the limit?
print("\n" + "-" * 80)
print("4d. VERY LARGE K (Fibonacci, limit behavior)")
print("-" * 80)
for K in [128, 256, 512, 896]:
    freqs = freqs_fibonacci(K)
    blocks = blocks_equal(D, K)
    r = analyze_config(f"Fibonacci K={K}", freqs, blocks)
    all_results.append(r)
    _, idx = freqs.sort(descending=True)
    top3 = freqs[idx[:3]].tolist()
    print(f"  K={K:3d}: top λ = {[f'{x:.8f}' for x in top3]}  "
          f"block_size = {r['min_block']}  "
          f"max-min = {r['λ_max']-r['λ_min']:.6f}")

# ─── 5. FINDINGS ────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Finding 1: λ asymptotics
print("\n[Finding 1] Fibonacci roots converge to 2.0 exponentially:")
print("  λ_k ≈ 2 - O(1/φ^k)")
for k in [2, 4, 8, 16, 32]:
    λ_val = float(fibonacci_roots_k(k))
    gap = 2.0 - λ_val
    print(f"  k={k:3d}: λ={λ_val:.8f},  gap to 2 = {gap:.2e}")

# Finding 2: At what K does λ become indistinguishable from 2?
print("\n[Finding 2] Resolution limits:")
for K in [8, 16, 32, 64]:
    freqs = freqs_fibonacci(K)
    λ_max = float(freqs.max())
    gap = 2.0 - λ_max
    _, top_idx = freqs.sort(descending=True)
    top_two_gap = float(freqs[top_idx[0]] - freqs[top_idx[1]])
    print(f"  K={K:2d}: λ_max = {λ_max:.8f}, gap to 2 = {gap:.2e}, "
          f"top-2 spread = {top_two_gap:.2e}")

# Finding 3: Non-Fibonacci alternatives
print("\n[Finding 3] Non-Fibonacci alternatives offer different spectral profiles:")
linear_freqs = freqs_linear(8, 1.0, 2.0)
fib_freqs = freqs_fibonacci(8)
print(f"  Linear[1,2): spread = {(linear_freqs[-1]-linear_freqs[0]):.4f}, "
      f"mean spacing = {(linear_freqs[-1]-linear_freqs[0])/7:.4f}")
print(f"  Fibonacci:  spread = {(fib_freqs[-1]-fib_freqs[0]):.4f}, "
      f"spacing varies {['{:.4f}'.format(x) for x in np.diff(fib_freqs.numpy())]}")

# Finding 4: Best configurations
print("\n[Finding 4] Recommended configurations (sorted by spectral diversity):")
sorted_results = sorted(all_results, key=lambda r: r['λ_var'], reverse=True)
for r in sorted_results[:10]:
    print(f"  {r['name']:30s}: λ_var={r['λ_var']:.6f}, "
          f"λ_range=[{r['λ_min']:.4f},{r['λ_max']:.4f}], "
          f"entropy={r['entropy_ratio']:.3f}, "
          f"damping_dims={r['damping_dims']}")

# ─── 6. SPECIAL RECOMMENDATIONS ─────────────────────────────────────────

print("\n" + "=" * 80)
print("SPECIAL CONFIGURATIONS WORTH TESTING")
print("=" * 80)

# Mixed damping+amplification
print("\n[A] Damped + Amplified (bipolar spectrum):")
for name, lo, hi in [("Mild damp", 0.8, 1.5), ("Strong damp", 0.5, 1.2), ("Balanced", 0.5, 1.8)]:
    mix = torch.cat([freqs_linear(4, lo, 0.99), freqs_linear(4, 1.01, hi)])
    r = analyze_config(f"{name}", mix, blocks_equal(D, 8))
    print(f"  {name:15s}: λ in [{r['λ_min']:.2f},{r['λ_max']:.2f}]  "
          f"{r['damping_dims']} damped + {r['amplifying_dims']} amplified dims  "
          f"cond={r['cond']:.1f}")

# Non-equal blocks with Fibonacci
print("\n[B] Unequal blocks (log ↑): more dims for low λ, fewer for high λ")
for K in [4, 8, 16]:
    freqs = freqs_fibonacci(K)
    blocks = blocks_log(D, K, ascend=True)
    r = analyze_config(f"Log↑ K={K}", freqs, blocks)
    print(f"  K={K:2d}: λ in [{r['λ_min']:.4f},{r['λ_max']:.4f}]  "
          f"min_block={r['min_block']} max_block={r['max_block']}  "
          f"entropy={r['entropy_ratio']:.3f}")

# Ultra-high K
print("\n[C] Very high K with tiny blocks (wavelet regime):")
for K in [64, 128, 256]:
    freqs = freqs_fibonacci(K)
    blocks = blocks_equal(D, K)
    r = analyze_config(f"Wavelet K={K}", freqs, blocks)
    # Count how many λ are essentially 2.0 (within float16 precision)
    near_two = (freqs > 1.999).sum().item()
    print(f"  K={K:3d}: {near_two}/{K} λ > 1.999  block_size={r['min_block']}")

# ─── 7. THE λ=1 SPECIAL POINT ──────────────────────────────────────────

print("\n" + "=" * 80)
print("THE λ=1 SINGULARITY")
print("=" * 80)
print("  λ = 1 → operator V·I·V^T = I (identity). No spectral transformation.")
print("  λ < 1 → damping (information decay). Acts as low-pass filter.")
print("  λ > 1 → amplification. Acts as high-pass filter (edge enhancement).")
print("  λ = 2 → critical point: operator norm = 2. Upper stability bound.")
print()
print("  Current config (K=4, Fibonacci): all λ > 1, all amplifying.")
print("  → Every dimension is amplified. No damping in the spectral path.")
print("  → Covariance memory + conv provide the only forgetting mechanisms.")
print()
print("  If we add λ < 1 dimensions, the spectral operator gains a")
print("  built-in forgetting/decay — like spectral low-pass filtering.")
print("  This would make the memory-cov path redundant for damping,")
print("  freeing it for higher-order correlations.")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
