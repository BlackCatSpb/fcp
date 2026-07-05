"""
explore_fib_spectra.py — Fibonacci sequence + linear hybrid spectra.
Тестируем: числа Фибоначчи, их отношения, комбинации c линейным.
"""

import math, sys, os
import numpy as np
import torch

D = 896
N_LAYERS = 24

def fibonacci_sequence(n):
    """First n Fibonacci numbers (F_1..F_n). F_1=1, F_2=1."""
    seq = [1, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq

def fibonacci_roots_k(k):
    """Characteristic root of k-th order Fibonacci recurrence."""
    lo, hi = 1.0, 2.0
    for _ in range(200):
        mid = (lo + hi) / 2
        powers = mid ** torch.arange(k, -1, -1, dtype=torch.float64)
        f = powers[0] - powers[1:].sum()
        if f > 0: hi = mid
        else: lo = mid
    return float((lo + hi) / 2)

def blocks_fib_seq(K):
    """Block sizes proportional to Fibonacci sequence F_2..F_{K+1}"""
    fibs = fibonacci_sequence(K + 1)[1:]  # F_2..F_{K+1}
    total = sum(fibs)
    sizes = [max(1, round(f * D / total)) for f in fibs]
    diff = D - sum(sizes)
    sizes[-1] += diff
    return sizes

def blocks_equal(K):
    return [D // K] * K

def norm_freqs(raw, lo, hi):
    """Normalize array to [lo, hi) range."""
    arr = torch.tensor(raw, dtype=torch.float32)
    if len(arr) == 1:
        return torch.tensor([(lo + hi) / 2])
    mn, mx = arr.min().item(), arr.max().item()
    if mx == mn:
        return torch.full_like(arr, (lo + hi) / 2)
    return lo + (arr - mn) / (mx - mn) * (hi - lo) * 0.9999

def analyze_config(name, freqs, block_sizes, L=N_LAYERS):
    K = len(freqs)
    λ_min = float(freqs.min())
    λ_max = float(freqs.max())
    λ_mean = float(freqs.mean())
    λ_var = float(freqs.var())

    if K > 1:
        sorted_f = freqs.sort()[0]
        spacing = [(sorted_f[i+1] - sorted_f[i]).item() for i in range(K-1)]
        min_sp = min(spacing)
        mean_sp = np.mean(spacing)
    else:
        min_sp = mean_sp = 0

    cond = λ_max / max(λ_min, 1e-10)
    damp_dims = sum(block_sizes[i] for i in range(K) if freqs[i] < 1.0)
    amp_dims = sum(block_sizes[i] for i in range(K) if freqs[i] > 1.0)
    total_amp = λ_max ** L

    probs = torch.tensor(block_sizes, dtype=torch.float32) / D
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_ent = math.log(K) if K > 0 else 1
    ent_ratio = entropy / max_ent if max_ent > 0 else 1.0

    print(f"  {name:40s}: K={K:2d}  "
          f"λ=[{λ_min:.4f},{λ_max:.4f}]  "
          f"var={λ_var:.5f}  "
          f"spacing_min={min_sp:.2e}  "
          f"damp={damp_dims}  amp={amp_dims}  "
          f"entropy={ent_ratio:.3f}  "
          f"cond={cond:.2f}")


# ══════════════════════════════════════════════════════════════════════
print("=" * 80)
print("FIBONACCI SPECTRA EXPLORATION")
print("=" * 80)

K = 8
HI = 1.8
LO = 0.8

print(f"\nK={K}, range=[{LO}, {HI}]")
print("-" * 80)

# 1. Fibonacci roots (current baseline)
roots = [fibonacci_roots_k(k) for k in range(2, K + 2)]
analyze_config("Fibonacci roots (current)", torch.tensor(roots), blocks_equal(K))

# 2. Fibonacci sequence normalized
fibs = fibonacci_sequence(K + 2)
# Try different Fibonacci subsequences
# a) F_2..F_{K+1}
fibs_a = fibonacci_sequence(K + 2)[1:-1]
analyze_config("Fibonacci seq F2..F9 normalized", norm_freqs(fibs_a, LO, HI),
               blocks_equal(K))

# b) F_3..F_{K+2} (skip the duplicate 1)
fibs_b = fibonacci_sequence(K + 3)[2:-1]
analyze_config("Fibonacci seq F3..F10 normalized", norm_freqs(fibs_b, LO, HI),
               blocks_equal(K))

# 3. Fibonacci RATIOS: F_{k+1}/F_k
ratios = [fibonacci_sequence(K + 3)[i+1] / fibonacci_sequence(K + 3)[i]
          for i in range(1, K + 1)]  # F_2/F_1 through F_{K+1}/F_K
analyze_config("Fibonacci ratios F_{k+1}/F_k", norm_freqs(ratios, LO, HI),
               blocks_equal(K))

# 4. Fibonacci ratios + linear hybrid (averaged)
ratio_vals = norm_freqs(ratios, LO, HI)
linear_vals = torch.linspace(LO, HI * 0.9999, K)
hybrid = (ratio_vals + linear_vals) / 2
analyze_config("Hybrid: avg(Fib ratio + linear)", hybrid, blocks_equal(K))

# 5. Fibonacci sequence + linear hybrid
fib_seq_a = norm_freqs(fibs_a, LO, HI)
hybrid2 = (fib_seq_a + linear_vals) / 2
analyze_config("Hybrid: avg(Fib seq + linear)", hybrid2, blocks_equal(K))

# 6. Fibonacci sequence for λ, Fibonacci numbers for block sizes
analyze_config("Fib λ = seq + Fib blocks", norm_freqs(fibs_a, LO, HI),
               blocks_fib_seq(K))

# 7. Fibonacci roots + Fib blocks
analyze_config("Fib λ = roots + Fib blocks", torch.tensor(roots),
               blocks_fib_seq(K))

# 8. Fibonacci ratios + Fib blocks
analyze_config("Fib λ = ratios + Fib blocks", norm_freqs(ratios, LO, HI),
               blocks_fib_seq(K))

# 9. CDF of Fibonacci numbers (cumulative distribution)
fib_cum = [sum(fibs_a[:i+1]) for i in range(len(fibs_a))]
analyze_config("Fib CDF (cumulative sum)", norm_freqs(fib_cum, LO, HI),
               blocks_equal(K))

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DETAILED BLOCK STRUCTURES (Fibonacci-sized blocks)")
print("=" * 80)
for K_test in [6, 8, 10]:
    fibs = fibonacci_sequence(K_test + 2)[1:-1]
    blocks = blocks_fib_seq(K_test)
    total = sum(blocks)
    print(f"\n  K={K_test}, blocks={blocks}, sum={total}")
    # Show which λ each block corresponds to (if using Fibonacci seq)
    λ_for_fibs = norm_freqs(fibs, LO, HI)
    for i in range(K_test):
        print(f"    λ_{i+2}={λ_for_fibs[i]:.4f} → {blocks[i]} dims ({blocks[i]/D*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CANDIDATE ANALYSIS: Full raw values for top candidates")
print("=" * 80)

# Candidate A: Fibonacci roots, equal blocks (current)
# Candidate B: Fibonacci sequence, equal blocks
# Candidate C: Fibonacci ratios, equal blocks  
# Candidate D: Hybrid (ratios + linear), equal blocks
# Candidate E: Fibonacci sequence, Fibonacci blocks
# Candidate F: Fibonacci ratios, Fibonacci blocks

for i, (name, λ_fn, blk_fn) in enumerate([
    ("A: Roots + equal blks", lambda K: torch.tensor([fibonacci_roots_k(k) for k in range(2, K+2)]),
     blocks_equal),
    ("B: Fib seq + equal blks", lambda K: norm_freqs(fibonacci_sequence(K+2)[1:-1], LO, HI),
     blocks_equal),
    ("C: Fib ratios + equal blks", lambda K: norm_freqs(
        [fibonacci_sequence(K+3)[i+1]/fibonacci_sequence(K+3)[i] for i in range(1, K+1)], LO, HI),
     blocks_equal),
    ("D: Hybrid ratio+lin + eq blks", lambda K: (norm_freqs(
        [fibonacci_sequence(K+3)[i+1]/fibonacci_sequence(K+3)[i] for i in range(1, K+1)], LO, HI)
        + torch.linspace(LO, HI*0.9999, K)) / 2, blocks_equal),
    ("E: Fib seq + Fib blocks", lambda K: norm_freqs(fibonacci_sequence(K+2)[1:-1], LO, HI),
     blocks_fib_seq),
    ("F: Fib ratios + Fib blocks", lambda K: norm_freqs(
        [fibonacci_sequence(K+3)[i+1]/fibonacci_sequence(K+3)[i] for i in range(1, K+1)], LO, HI),
     blocks_fib_seq),
    ("G: Linear alone", lambda K: torch.linspace(LO, HI*0.9999, K),
     blocks_equal),
]):
    K = 8
    freqs = λ_fn(K)
    blocks = blk_fn(K)
    print(f"\n  [{name}]")
    print(f"    λ = {[f'{x:.4f}' for x in freqs.tolist()]}")
    print(f"    blocks = {blocks}")
    print(f"    λ range: {float(freqs.min()):.4f}..{float(freqs.max()):.4f}, "
          f"var={float(freqs.var()):.5f}")
    print(f"    damping: {sum(blocks[i] for i in range(K) if freqs[i] < 1.0)} dims, "
          f"amplifying: {sum(blocks[i] for i in range(K) if freqs[i] > 1.0)} dims")

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
Best candidates for testing (ranked by spectral diversity + physical plausibility):

1. [C] Fibonacci ratios + equal blocks — clean mathematical structure,
   ratios naturally live in [1, 2), converge to φ=1.618, evenly balanced.

2. [D] Hybrid (ratios + linear) — combines the Fibonacci structure
   with uniform coverage. Best of both worlds.

3. [F] Fibonacci ratios + Fibonacci blocks — fully Fibonacci-consistent:
   both λ values AND block sizes follow Fibonacci. Each λ gets block size
   proportional to the corresponding Fibonacci number.

4. [A] Fibonacci roots + Fib blocks — current λ, but with Fibonacci-sized
   blocks instead of equal. More natural block distribution.

Recommendation for first test: [C] Ratios + equal blocks at K=8,
range [0.8, 1.8] with λ = [F_2/F_1, F_3/F_2, ..., F_9/F_8] normalized.
""")
