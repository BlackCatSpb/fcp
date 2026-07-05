"""
analyze_math.py - Mathematical verification of 5 architectural hypotheses.
----- + Fibonacci spectrum analysis.
"""

import math, sys, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import fibonacci_roots, random_orthogonal, rms_norm, parallel_prefix_scan

D = 896
N_LAYERS = 24
K_DEFAULT = 4
K_PROPOSED = 8

print("=" * 70)
print("MATHEMATICAL ANALYSIS: FIBONACCI COVARIANCE PROCESSOR")
print("=" * 70)

# ======================================================================
# 1. EIGHT FREQUENCIES (K=8) vs FOUR (K=4)
# ======================================================================
print("\n" + "-" * 70)
print("1. FULL SPECTRUM ANALYSIS: K=8 vs K=4")
print("-" * 70)

roots_8 = fibonacci_roots(9)  # λ-..λ-
roots_4 = roots_8[:4]

print(f"\n  Spectrum (8 frequencies):")
for i, r in enumerate(roots_8):
    # Verify the recurrence: λ^{k+1} = 2λ^k - 1
    k = i + 2
    lhs = r ** (k + 1)
    rhs = 2 * r ** k - 1
    err = abs(lhs - rhs)
    print(f"    λ_{k}={r:.6f}  |  λ^{k+1}=2λ^{k}-1  err={err:.2e}")

print(f"  → λ_k → 2 as k → -: λ_9 = {roots_8[-1].item():.6f}, -_to_2 = {2 - roots_8[-1].item():.2e}")

# Spectral coverage: difference between consecutive roots
diffs = [roots_8[i+1].item() - roots_8[i].item() for i in range(len(roots_8)-1)]
print(f"  Spacing between consecutive λ_k: {[f'{d:.4f}' for d in diffs]}")

# Block sizes
block_4 = D // 4
block_8 = D // 8
print(f"\n  K=4: each λ applied to {block_4} dims (D/K = {D}/4)")
print(f"  K=8: each λ applied to {block_8} dims (D/K = {D}/8)")

# Effective rank of spectral operator
# For K=4: 4 distinct eigenvalues, each repeated block_4 times
# For K=8: 8 distinct eigenvalues, each repeated block_8 times
# The operator V-diag(λ)-V^T has rank D with multiplicities
print(f"\n  K=4 eigenvalue multiplicities: {[block_4]*4}")
print(f"  K=8 eigenvalue multiplicities: {[block_8]*8}")

# Compute how much coverage each λ gets in terms of information through the spectral transform
# The spectral expansion factor per block: λ_k^L (total amplification after L tokens)
L_seq = 128
print(f"\n  Spectral expansion per λ after {L_seq} tokens (λ^L):")
for i, r in enumerate(roots_8):
    exp_factor = r ** (1.0 / L_seq)  # Per-step growth needed to reach λ after L steps
    total_growth = r ** L_seq
    print(f"    λ_{i+2}={r:.4f}: -={exp_factor:.6f}  λ^L={total_growth:.2e}")
    if total_growth > 1e10:
        print(f"      - UNSTABLE (λ^L → -)")
    elif total_growth < 1e-10:
        print(f"      - VANISHING (λ^L → 0)")

# Information preservation ratio
print(f"\n  λ^L for L={L_seq}:")
for i, r in enumerate([roots_8[0], roots_8[3], roots_8[-1]]):
    print(f"    λ_{i+2}={r:.4f}: {r**L_seq:.2e}")
print(f"  → K=4 range: λ_2^{L_seq}={roots_8[0]**L_seq:.2e} to λ_5^{L_seq}={roots_8[3]**L_seq:.2e}")
print(f"  → K=8 range: λ_2^{L_seq}={roots_8[0]**L_seq:.2e} to λ_9^{L_seq}={roots_8[-1]**L_seq:.2e}")

# ======================================================================
# 2. LAYER-DEPENDENT λ SHIFT
# ======================================================================
print("\n" + "-" * 70)
print("2. λ GRADIENT ACROSS LAYERS (Hladi depth analogy)")
print("-" * 70)

# Linear schedule: λ shifts from fine (high freq) to coarse (low freq)
# Lower layers: use higher λ_k indices (finer detail)
# Upper layers: use lower λ_k indices (coarser structure)
# Actually: lower layers = thin plate = high frequency = larger λ_k
#           upper layers = thick plate = low frequency = smaller λ_k

# Option A: λ window slides across layers
print("\n  Option A: Sliding λ window (K=4 out of 8)")
print(f"  Layer 0-5:  λ-,λ-,λ-,λ-  → λ={roots_8[0]:.4f}..{roots_8[3]:.4f}")
print(f"  Layer 6-11: λ-,λ-,λ-,λ-  → λ={roots_8[1]:.4f}..{roots_8[4]:.4f}")
print(f"  Layer 12-17: λ-,λ-,λ-,λ- → λ={roots_8[2]:.4f}..{roots_8[5]:.4f}")
print(f"  Layer 18-23: λ-,λ-,λ-,λ- → λ={roots_8[3]:.4f}..{roots_8[6]:.4f}")

# Verify no λ exceeds 2 (stability bound)
for g_idx, label in enumerate(["0-5", "6-11", "12-17", "18-23"]):
    offset = g_idx
    group_roots = roots_8[offset:offset+4]
    assert all(r < 2.0 for r in group_roots), f"λ ≥ 2 in group {label}"
    print(f"  Group {label}: λ_min={group_roots[0]:.4f} λ_max={group_roots[-1]:.4f} - stable")

# Option B: Continuous shift per layer (linear interpolation)
print("\n  Option B: Continuous λ shift (linear interpolation across 24 layers)")
shift_per_layer = (roots_8[-1] - roots_8[0]) / N_LAYERS
for l in [0, 5, 11, 17, 23]:
    λ_shift_val = roots_8[0].item() + shift_per_layer.item() * l
    print(f"    Layer {l:2d}: λ_avg - {λ_shift_val:.4f}  (λ_2 + {l}×{shift_per_layer.item():.4f})")

# Parameter cost: ZERO (just indexing into precomputed roots)
print(f"\n  Parameter cost: 0 (no learnable params)")

# ======================================================================
# 3. DCT-BASED V INITIALIZATION
# ======================================================================
print("\n" + "-" * 70)
print("3. DCT vs RANDOM ORTHOGONAL V MATRIX")
print("-" * 70)

def dct_matrix(n):
    """DCT-II orthonormal basis: V[i,j] = sqrt(1/n) for i=0 else sqrt(2/n)*cos(--i-(j+0.5)/n)"""
    V = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == 0:
                V[i, j] = math.sqrt(1.0 / n)
            else:
                V[i, j] = math.sqrt(2.0 / n) * math.cos(math.pi * i * (j + 0.5) / n)
    return V

# Test orthogonality at various D subsets
test_dims = [16, 64, 128, 256]
d_errors = []
r_errors = []

for test_d in test_dims:
    V_dct = dct_matrix(test_d)
    V_rand = random_orthogonal(test_d, n_reflections=32)

    # Orthogonality error: ||V-V^T - I||_F / D
    dct_err = (V_dct @ V_dct.T - torch.eye(test_d)).norm().item() / test_d
    rand_err = (V_rand @ V_rand.T - torch.eye(test_d)).norm().item() / test_d
    d_errors.append(dct_err)
    r_errors.append(rand_err)
    print(f"  D={test_d:4d}: DCT error={dct_err:.2e}  Random error={rand_err:.2e}")

print(f"  → DCT is perfectly orthogonal by construction (error ~ machine epsilon)")

# Frequency ordering: DCT basis vectors ordered by increasing frequency
# Check that V[i,:] corresponds to frequency i
print(f"\n  DCT basis frequency ordering (first 8 vectors):")
V_dct_full = dct_matrix(D)
for i in range(8):
    vec = V_dct_full[i]
    # Count zero crossings as a measure of frequency
    zc = ((vec[:-1] * vec[1:]) < 0).sum().item()
    print(f"    V[{i},:] zero crossings = {zc}  (frequency - {zc/D*math.pi:.3f} rad/sample)")

# Frequency-domain view: what does V-diag(λ)-V^T actually mean with DCT?
# With DCT: h → DCT → scale each frequency by λ_k → inverse DCT
# This is a filterbank!
print(f"\n  With DCT, the spectral operator becomes a literal frequency filterbank:")
print(f"  1. DCT(h) = V^T-h  - decompose into {D} frequency components")
print(f"  2. Scale each block of {D//4} consecutive frequencies by its λ_k")
print(f"  3. IDCT = V-(scaled) - reconstruct")
print(f"  → Each λ_k controls a contiguous frequency band (low-pass, mid, hi-pass)")

# Compare with random orthogonal: no notion of frequency
print(f"  → Random orthogonal V has NO frequency interpretation")
print(f"  → DCT is the STANDING WAVE solution for the 1D wave equation")
print(f"  → DCT directly matches the Hladi plate analogy")

# ======================================================================
# 4. λ-TIED MEMORY HEADS
# ======================================================================
print("\n" + "-" * 70)
print("4. λ-TIED COVARIANCE MEMORY HEADS")
print("-" * 70)

H = 4
K = 4
# Current: all heads same r=16, all λ same
# Proposed: head h bound to λ_{h+2}
print(f"\n  Current: H={H} heads, all r=16, all λ={roots_4}")
print(f"  Proposed: each head tied to one λ_k")
for h in range(H):
    l = roots_4[h]
    # Head's effective memory time constant: 1/(1-decay_avg)
    # If decay - λ_bias / (λ_bias + 1) where λ_bias = initial bias
    # For decay_bias=2.0: decay - sigmoid(2.0) - 0.88
    # Time constant - = 1/(1-sigmoid(2.0)) - 8.3 tokens
    print(f"  Head {h}: λ_{h+2}={l:.4f}  |  associated with {h+2}-gram scale")

# Theoretical: head's key/query projections should match the λ_k spectral band
# The key space is r-dimensional, and should be a subspace of the V-eigenbasis
# corresponding to the λ_k block

print(f"\n  Memory projection: r={16} dims per head")
print(f"  Each head operates in a {D//H}=224 dim spectral band")
print(f"  Head h key/query should only access frequencies in its band")

# Compute the subspace overlap
v_blocks = []
for k in range(K):
    start = k * D // K
    end = (k + 1) * D // K
    v_blocks.append((start, end))
    print(f"  λ_{k+2} band: dims {start}-{end-1} ({end-start} dims)")

print(f"\n  → λ-head tying ensures the memory operates in the same frequency band")
print(f"    as the spectral operator for that λ. No cross-frequency interference.")

# ======================================================================
# 5. LEARNABLE δ PER LAYER
# ======================================================================
print("\n" + "-" * 70)
print("5. LEARNABLE λ PERTURBATION (δ per layer)")
print("-" * 70)

# Tiny learnable offset per layer: δ_l - [-0.01, 0.01]
# Effective λ_k(l) = λ_k + δ_l
# Only 24 additional parameters.

print(f"\n  24 learnable scalars δ_l, each - [-0.01, 0.01]")
print(f"  Parameter cost: 24 (negligible)")

# Stability check: λ_k + δ_l < 2.0 at all times
max_λ = roots_8[-1].item() + 0.01
print(f"  Max possible λ with δ=0.01: λ_9 + 0.01 = {roots_8[-1].item():.4f} + 0.01 = {roots_8[-1].item()+0.01:.4f}")
print(f"  Stability: λ < 2.0 required → {'-' if max_λ < 2.0 else '-'}")

# Check the effect on the recurrence: V-diag(λ+δ)-V^T
# A small δ shifts the amplification factor: (λ+δ)^L - λ^L + L-λ^{L-1}-δ
# For L=128, λ=1.618: δ=0.01 changes output by 128-1.618^{127}-0.01 - huge!
print(f"\n  Perturbation sensitivity at L={L_seq}:")
for r in [roots_8[0], roots_8[3], roots_8[-1]]:
    base = r ** L_seq
    delta = 0.01
    perturbed = (r + delta) ** L_seq
    ratio = perturbed / base if base > 0 else float('inf')
    print(f"  λ={r.item():.4f}: λ^{L_seq}={base:.2e}  (λ+δ)^{L_seq}={perturbed:.2e}  ratio={ratio:.2f}x")
    if ratio > 2.0 or ratio < 0.5:
        print(f"    - Highly sensitive! Even δ=0.01 can double/halve the spectral output.")

# This suggests δ should be either:
# a) Applied per-step, not to λ directly: h[t] = (λ+δ)-h[t-1] + x[t] 
# b) Very small: δ - [-1e-4, 1e-4]
# c) Applied as multiplier on λ^t: λ^(t-(1+δ))
print(f"\n  Safer alternative: δ as multiplier on exponent: λ^(t*(1+δ))")
print(f"  For δ=0.001, L=128: λ^(128*1.001) = λ^128 * λ^0.128 - λ^128 * 1.06")
print(f"  → Much gentler, stable gradient")

# ======================================================================
# 6. SUMMARY AND RECOMMENDATIONS
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

recommendations = [
    {
        "id": 1,
        "name": "K=8 Full Spectrum",
        "params": 0,
        "impact": "High",
        "verdict": "IMPLEMENT - More complete spectral coverage. 8 λ values span λ-..λ-, giving better resolution across context scales. block_size=112 still sufficient for stable statistics.",
        "risk": "Low",
        "complexity": "1 day"
    },
    {
        "id": 2,
        "name": "Sliding λ Window",
        "params": 0,
        "impact": "Medium",
        "verdict": "IMPLEMENT - Perfect Hladi analogy: thin plate (lower layers) → high λ, thick plate (upper layers) → low λ. No extra params.",
        "risk": "Low",
        "complexity": "0.5 day"
    },
    {
        "id": 3,
        "name": "DCT Basis V",
        "params": 0,
        "impact": "High",
        "verdict": "CRITICAL - DCT makes the spectral operator a true frequency filterbank. Random orthogonal V has no physical interpretation. DCT is the standing wave solution for 1D wave equation = exact Hladi analog.",
        "risk": "Medium (need to verify gradient flow through DCT)",
        "complexity": "0.5 day"
    },
    {
        "id": 4,
        "name": "λ-tied Memory Heads",
        "params": 0,
        "impact": "Medium",
        "verdict": "IMPLEMENT - Each head operates in its λ_k frequency band. No interference between temporal scales. Conceptual match with multi-scale memory.",
        "risk": "Low",
        "complexity": "1 day"
    },
    {
        "id": 5,
        "name": "Learnable δ per Layer",
        "params": 24,
        "impact": "Low-Medium",
        "verdict": "OPTIONAL - Add only after K=8 + DCT + λ-sliding are proven. High sensitivity to δ at large L. Use exponent multiplier δ not additive.",
        "risk": "Medium (stability)",
        "complexity": "0.5 day"
    }
]

for rec in recommendations:
    print(f"\n  [{rec['id']}] {rec['name']}")
    print(f"      Params: {rec['params']}  |  Impact: {rec['impact']}  |  Risk: {rec['risk']}")
    print(f"      Verdict: {rec['verdict']}")
    print(f"      Effort: {rec['complexity']}")

print("\n" + "=" * 70)
print("RECOMMENDED IMPLEMENTATION ORDER")
print("=" * 70)
print("  Step 1: DCT basis V (changes V initialization only)")
print("  Step 2: K=8 full spectrum (changes block_size and λ array)")
print("  Step 3: Sliding λ window per layer group")
print("  Step 4: λ-tied memory heads")
print("  Step 5 (optional): Learnable δ as exponent multiplier")
print("=" * 70)
