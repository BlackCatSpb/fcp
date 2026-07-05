"""
Synthetic verification of λ_d architecture.
Tests: stability, gate differentiation, layer behaviour.
"""

import os, sys, math, time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, LDStack, fibonacci_roots, rms_norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ──────────────────────────────────────────────────────────────

# Small config for fast testing
cfg = LDConfig()
cfg.D = 256          # small hidden size for fast test
cfg.n_layers = 6     # 6 layers enough to test composition
cfg.n_modes = 4      # K=4 modes (roots for d=2,3,4,5)
cfg.vocab = 1000     # tiny vocab
cfg.intermediate = 1024
cfg.lora_rank = 32
cfg.use_lora = True

print(f'Config: D={cfg.D}, layers={cfg.n_layers}, modes={cfg.n_modes}')

# Precompute Fibonacci roots
lambdas = fibonacci_roots(cfg.n_modes + 1)
print(f'  λ_k: {[f"{l:.4f}" for l in lambdas.tolist()]}')
print(f'  φ = {(1 + 5**0.5)/2:.6f}')

# ─── Test 1: Stability ──────────────────────────────────────────────────

print('\n' + '=' * 60)
print('TEST 1: Stability (random input, no NaN)')
print('=' * 60)

# Create model
model = LDStack(cfg).to(DEVICE, torch.float16)
print(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable')
print(f'  Frozen: {sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e6:.2f}M')

# Random input
B, L = 2, 16
h = torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16)
h = F.normalize(h, dim=-1) * (cfg.D ** 0.5)

print(f'Input: shape={h.shape}, norm={h.norm().item():.2f}')

# Forward
t0 = time.perf_counter()
with torch.no_grad():
    h_out, gates = model(h, return_gates=True)
t_fwd = time.perf_counter() - t0

print(f'Output: shape={h_out.shape}, norm={h_out.norm().item():.2f}')
print(f'Forward time: {t_fwd*1000:.1f}ms')

# Checks
assert not torch.isnan(h_out).any(), 'NaN in output!'
assert not torch.isinf(h_out).any(), 'Inf in output!'
print('  [PASS] No NaN or Inf')

norm_ratio = h_out.norm().item() / h.norm().item()
print(f'  Norm ratio (out/in): {norm_ratio:.2f}')
assert norm_ratio < 10, f'Norm exploded: {norm_ratio}'
print('  [PASS] Norm stable')

# Per-layer norm check
print(f'\nPer-layer norms:')
h_test = torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16)
h_test = F.normalize(h_test, dim=-1) * (cfg.D ** 0.5)
for lidx in range(cfg.n_layers):
    h_test = model.layers[lidx](h_test)
    h_norm = h_test.norm().item()
    print(f'  Layer {lidx}: norm={h_norm:.2f}', end='')
    assert not torch.isnan(h_test).any(), f'NaN at layer {lidx}'
    assert not torch.isinf(h_test).any(), f'Inf at layer {lidx}'
    if h_norm > 1e4:
        print('  [EXPLODED]')
    else:
        print()
print('  [PASS] All layers stable')

# ─── Test 2: Gate Differentiation ──────────────────────────────────────

print('\n' + '=' * 60)
print('TEST 2: Gate Differentiation (α(h₁) ≠ α(h₂) for different inputs)')
print('=' * 60)

# Create two different "synthetic sentences"
h1 = torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16)
h2 = torch.randn(B, L, cfg.D, device=DEVICE, dtype=torch.float16)  # different seed

with torch.no_grad():
    _, gates1 = model(h1, return_gates=True)  # (n_layers, B, L, K)
    _, gates2 = model(h2, return_gates=True)

# Compute gate similarity
gate_cos = F.cosine_similarity(
    gates1.float().reshape(-1, cfg.n_modes),
    gates2.float().reshape(-1, cfg.n_modes),
    dim=-1
).mean().item()

print(f'  Mean gate cos(h₁, h₂): {gate_cos:.4f}', end='')
if gate_cos < 0.9:
    print('  [PASS] Gates differ')
else:
    print('  [COLLAPSE] Gates too similar')

# Gate entropy (should not be 0, which means one-hot collapse)
gate_entropy = -(gates1 * torch.log(gates1.clamp(min=1e-10))).sum(dim=-1).mean().item()
print(f'  Mean gate entropy: {gate_entropy:.4f} (max={math.log(cfg.n_modes):.4f})')

# Per-layer gate cos
print(f'\nPer-layer gate differentiation:')
for lidx in range(cfg.n_layers):
    g1 = gates1[lidx].float().reshape(-1, cfg.n_modes)
    g2 = gates2[lidx].float().reshape(-1, cfg.n_modes)
    cos = F.cosine_similarity(g1, g2, dim=-1).mean().item()
    ent = -(g1 * torch.log(g1.clamp(min=1e-10))).sum(dim=-1).mean().item()
    print(f'  Layer {lidx}: cos(h₁,h₂)={cos:.4f}, α-entropy={ent:.3f}')

# ─── Test 3: Layer Specialization ───────────────────────────────────────

print('\n' + '=' * 60)
print('TEST 3: Layer-wise gate distribution (mean α per layer)')
print('=' * 60)

with torch.no_grad():
    _, gates_all = model(h, return_gates=True)

for lidx in range(cfg.n_layers):
    g = gates_all[lidx].float().mean(dim=(0, 1))  # (K,) mean over batch and seq
    print(f'  Layer {lidx}: mean α = [{", ".join(f"{v.item():.3f}" for v in g)}]')
    dominant = g.argmax().item()
    print(f'           dominant mode: d={dominant + 2} (λ={lambdas[dominant]:.4f})')

# ─── Test 4: Sequential Generation Stability ────────────────────────────

print('\n' + '=' * 60)
print('TEST 4: Sequential (autoregressive) stability')
print('=' * 60)

# Simulate autoregressive generation
seq_len = 32
v = torch.randn(1, 1, cfg.D, device=DEVICE, dtype=torch.float16)
v = F.normalize(v, dim=-1) * (cfg.D ** 0.5)

norms = []
gate_trace = []
print(f'  Generating {seq_len} steps...')
t0 = time.perf_counter()
for step in range(seq_len):
    v_out, gates_step = model(v, return_gates=True)  # (1, 1, D)
    v = v_out
    norms.append(v.norm().item())
    gate_trace.append(gates_step[:, 0, 0, :].cpu())  # (n_layers, K)

t_gen = time.perf_counter() - t0
print(f'  Time: {t_gen*1000:.0f}ms ({t_gen/seq_len*1000:.1f}ms/token)')
print(f'  Norms: start={norms[0]:.2f}, end={norms[-1]:.2f}, min={min(norms):.2f}, max={max(norms):.2f}')

gate_var = torch.stack(gate_trace).var(dim=0).mean().item()
print(f'  Gate temporal variance: {gate_var:.6f} (0=never changes)')

if min(norms) > 0.1 and max(norms) < 1e4:
    print('  [PASS] Sequential generation stable')
else:
    print('  [FAIL] Norm bounds violated')

# ─── Summary ─────────────────────────────────────────────────────────────

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  Fibonacci roots: {[f"{l:.4f}" for l in lambdas.tolist()]}')
print(f'  Gate differentiation: {"PASS" if gate_cos < 0.9 else "COLLAPSE"}')
print(f'  Gate entropy: {gate_entropy:.3f} / {math.log(cfg.n_modes):.3f}')
print(f'  Sequential stability: norm range [{min(norms):.1f}, {max(norms):.1f}]')
print(f'  Forward time: {t_fwd*1000:.1f}ms (D={cfg.D}, L={L}, layers={cfg.n_layers})')
print()
print('All synthetic tests complete.')
