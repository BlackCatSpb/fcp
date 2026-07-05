"""
test_cov_gate.py — λ_d-Cov: Bind + Covariance Memory, no softmax, no sigmoid gate.

Три ключевые идеи из SOTA 2024-2025, объединённые в λ_d:

1. Bind pre-transformation (FCF → λ_d) — заменяет gate на u*v, без softmax
2. Covariance memory (mLSTM + GLA) — матричное состояние r×r, 
   обновляемое как M[t] = σ(W_decay·x)⊙M[t-1] + k^T⊗q
3. Экспоненциальный input gate — i = exp(W_i·x) / (1+exp(W_i·x)) — 
   позволяет >1 (усиление), стабилизирован нормализацией.

Архитектура:
    h_conv = causal_conv(h)
    h_norm = rms_norm(h + h_conv)
    
    # Bind adaptation (FCF-inspired, proven в BindGate)
    u = h_norm @ W_u; v = h_norm @ W_v
    h_adapt = h_norm + (u * v) @ W_out
    
    # Covariance memory
    k = h_adapt @ W_k       # (B, r) key
    q = h_adapt @ W_q       # (B, r) query
    i = exp(h_adapt @ W_i)  # input gate ∈ (0, ∞)
    M[t] = σ(d·h_norm)·M[t-1] + i·k^T⊗q
    
    # Memory read
    h_mem = (q @ M[t]) @ W_read  # content-dependent retrieval
    
    # Spectral operator with fixed λ
    h_total = h_adapt + h_mem      # main + memory paths
    Δ = V · diag(λ) · V^T · h_total
    h_out = h + Δ

Запуск: python test_cov_gate.py
"""

import os, sys, math, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 4
ACCUM_STEPS = 4
SEQ_LEN = 64
LR = 1e-3
WARMUP_STEPS = 250
N_STEPS = 5000
GRAD_CLIP = 1.0
LOG_EVERY = 100
DATA_FILE = 'russian_chunks.npy'
N_CHUNKS = 20000

COV_RANK = 16  # rank of covariance memory (r × r)

def fibonacci_roots(max_k=7):
    roots = []
    for k in range(2, max_k + 1):
        lo, hi = 1.0, 2.0
        for _ in range(100):
            mid = (lo + hi) / 2
            powers = mid ** torch.arange(k, -1, -1, dtype=torch.float64)
            f = powers[0] - powers[1:].sum()
            if f > 0: hi = mid
            else: lo = mid
        roots.append((lo + hi) / 2)
    return torch.tensor(roots, dtype=torch.float32)

LAMBDA_ROOTS = fibonacci_roots(N_MODES + 1)[:N_MODES]

def random_orthogonal(D, n_reflections=32):
    V = torch.eye(D)
    for _ in range(n_reflections):
        u = torch.randn(D)
        u = u / (u.norm() + 1e-10)
        V = V - 2 * torch.outer(V @ u, u)
    return V

def rms_norm(x, weight, eps=1e-6):
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x / rms.clamp(min=eps) * weight


# ─── Shared components ──────────────────────────────────────────────────
class CausalConv1d(nn.Module):
    def __init__(self, D, kernel_size=48):
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer('weight', torch.randn(D, 1, kernel_size) * 0.1)
        self.register_buffer('bias', torch.zeros(D))
    def forward(self, x):
        x_pad = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        return F.conv1d(x_pad, self.weight, bias=self.bias, groups=self.weight.shape[0]).transpose(1, 2)

class BottleneckMLP(nn.Module):
    def __init__(self, D, bottleneck=512):
        super().__init__()
        self.up = nn.Linear(D, bottleneck, bias=True)
        self.down = nn.Linear(bottleneck, D, bias=True)
    def forward(self, x):
        return self.down(F.silu(self.up(x)))


# ═════════════════════════════════════════════════════════════════════════
# CovLDBlock — Bind + Covariance Memory, no softmax/sigmoid gates
# ═════════════════════════════════════════════════════════════════════════
class CovLDBlock(nn.Module):
    """
    Bind pre-transformation (FCF u*v) + low-rank covariance memory (mLSTM×GLA).

    Per-layer forward (per token t):
      1. Bind adaptation:  h_adapt = h_norm + (W_u·h * W_v·h) @ W_out
      2. Memory update:    M = decay[t] · M + i[t] · k[t]^T @ q[t]
      3. Memory read:      h_mem = (q[t] @ M) @ W_read
      4. Spectral oper:    Δ = V·diag(λ)·V^T · (h_adapt + h_mem)
      5. Residual:         h_out = h + Δ

    decay[t] = sigmoid(W_decay · h_norm)   — learnable per-token forget
    i[t]     = exp(W_i · h_norm)            — exponential input gate (can be >1)
    """
    def __init__(self, layer_idx, cov_r=16):
        super().__init__()
        self.D = D
        self.K = N_MODES
        self.block_size = D // N_MODES
        self.r = cov_r

        self.conv = CausalConv1d(D)
        self.register_buffer('ln_w', torch.ones(D))

        V_init = random_orthogonal(D)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', LAMBDA_ROOTS)

        # 1. Bind adaptation (replaces gate)
        self.W_u = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_out = nn.Parameter(torch.zeros(self.r, D))

        # 2. Covariance memory: keys, queries, gates
        self.W_k = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_q = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_i = nn.Parameter(torch.randn(D, 1) * 0.01)     # exponential input gate
        self.b_i = nn.Parameter(torch.zeros(1))               # log(1) = 0 at init

        self.W_decay = nn.Parameter(torch.randn(D, 1) * 0.01)  # forget gate
        self.b_decay = nn.Parameter(torch.full((1,), 1.0))     # sigmoid(1) ≈ 0.73

        self.W_read = nn.Parameter(torch.zeros(self.r, D))     # zero init → h_mem=0 at start

    def forward(self, h):
        B, L, D = h.shape

        # ─── 1. Conv + Norm ────────────────────────────────────────────
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        # ─── 2. Bind adaptation (parallel over L) ───────────────────────
        u = h_norm @ self.W_u
        v = h_norm @ self.W_v
        h_adapt = h_norm + (u * v) @ self.W_out

        # ─── 3. Covariance memory keys/queries/gates (parallel) ────────
        K = h_norm @ self.W_k          # (B, L, r)
        Q = h_norm @ self.W_q          # (B, L, r)
        i_raw = h_norm @ self.W_i + self.b_i  # (B, L, 1)
        i_gate = torch.exp(i_raw)       # input gate ∈ (0, ∞), can amplify

        decay = torch.sigmoid(h_norm @ self.W_decay + self.b_decay)  # (B, L, 1), ∈ (0,1)

        # ─── 4. Sequential covariance update + memory read ────────────
        # M: (B, r, r) running covariance. Sequential over L but cheap (r×r).
        M = torch.zeros(B, self.r, self.r, device=h.device)
        mem_outputs = []

        for t in range(L):
            k_t = K[:, t, :].unsqueeze(-1)   # (B, r, 1)

            # Covariance update: M = decay · M + input_gate · k^T @ k
            # self-covariance (k^T@k) more stable than cross (k^T@q)
            decay_t = decay[:, t, :].unsqueeze(-1)  # (B, 1, 1)
            M = decay_t * M
            igate_t = i_gate[:, t, :].unsqueeze(-1)  # (B, 1, 1)
            M = M + igate_t * (k_t @ k_t.transpose(-2, -1))
            # (B, r, 1) @ (B, 1, r) → (B, r, r)

            # Memory read: h_mem = q @ M @ W_read
            q_t_2d = Q[:, t, :]  # (B, r)
            h_mem_t = (q_t_2d.unsqueeze(-2) @ M).squeeze(-2) @ self.W_read  # (B, D)
            mem_outputs.append(h_mem_t)

        h_mem = torch.stack(mem_outputs, dim=1)  # (B, L, D)

        # ─── 5. Spectral operator (parallel over L) ─────────────────────
        h_total = h_adapt + h_mem  # main path + memory path
        h_proj = h_total @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, self.D)
        delta = h_scaled @ self.V

        return h + delta, None  # no gates


# ═════════════════════════════════════════════════════════════════════════
# Stack + Model
# ═════════════════════════════════════════════════════════════════════════
class CovLDStack(nn.Module):
    def __init__(self, cov_r=16):
        super().__init__()
        self.register_buffer('final_norm_w', torch.ones(D))
        self.layers = nn.ModuleList([CovLDBlock(i, cov_r) for i in range(N_LAYERS)])
        self.mlps = nn.ModuleList([BottleneckMLP(D) for _ in range(N_LAYERS)])

    def forward(self, h):
        for lidx in range(N_LAYERS):
            h_layer, _ = self.layers[lidx](h)
            h_norm_mlp = rms_norm(h_layer, self.final_norm_w)
            h_mlp = h_layer + self.mlps[lidx](h_norm_mlp)
            h = h_mlp  # no adaptive gain (no gates to compute it from)
        return rms_norm(h, self.final_norm_w)


class CovModel(nn.Module):
    def __init__(self, cov_r=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        self.stack = CovLDStack(cov_r)
        self.lm_head = nn.Linear(D, VOCAB, bias=True)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed(input_ids)))


# ─── Data ────────────────────────────────────────────────────────────────
print(f'\nLoading data: {DATA_FILE}')
t0 = time.perf_counter()
arr = np.load(DATA_FILE)[:N_CHUNKS]
print(f'  {arr.shape[0]} chunks in {time.perf_counter()-t0:.1f}s')

n_train = int(N_CHUNKS * 0.95)
n_eval = N_CHUNKS - n_train
print(f'  Train: {n_train} chunks ({n_train*SEQ_LEN/1e6:.1f}M tok)  Eval: {n_eval}')

train_ids = torch.tensor(arr[:n_train], dtype=torch.long).to(DEVICE)
eval_ids  = torch.tensor(arr[n_train:], dtype=torch.long).to(DEVICE)

train_x, train_y = train_ids[:, :-1], train_ids[:, 1:]
eval_x,  eval_y  = eval_ids[:, :-1],  eval_ids[:, 1:]

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(TensorDataset(eval_x, eval_y),  batch_size=BATCH_SIZE)

# ─── Train ───────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'Training: COV (r={COV_RANK})')
print(f'{"="*60}')

model = CovModel(COV_RANK).to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_cov = sum(p.numel() for n, p in model.named_parameters() if 'W_k' in n or 'W_q' in n or 'W_i' in n or 'W_decay' in n or 'W_read' in n)
n_bind = sum(p.numel() for n, p in model.named_parameters() if 'W_u' in n or 'W_v' in n or 'W_out' in n)
print(f'  Params: {n_all/1e6:.2f}M | bind: {n_bind:,} | cov: {n_cov:,}')

# Sanity
model.eval()
with torch.no_grad():
    bx = next(iter(train_loader))[0][:1]
    h = model.embed(bx)
    print(f'  sanity: embed=[{h.min():.2f},{h.max():.2f}]')
    h = model.stack(h)
    nan, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
    print(f'  sanity: stack=[{h.min():.2f},{h.max():.2f}] nan={nan} inf={inf}')
    logits = model.lm_head(h)
    print(f'  sanity: logits=[{logits.min():.2f},{logits.max():.2f}]')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
step = 0
t_start = time.perf_counter()
optimizer.zero_grad()
data_iter = itertools.cycle(train_loader)

while step < N_STEPS:
    model.train()
    bx, by = next(data_iter)
    logits = model(bx)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f'  [SKIP] step {step}: nan/inf logits'); continue
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
    if torch.isnan(loss):
        print(f'  [SKIP] step {step}: nan loss'); continue
    (loss / ACCUM_STEPS).backward()
    step += 1
    if step % ACCUM_STEPS == 0 or step == N_STEPS:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = LR * min(step / WARMUP_STEPS, 1.0) if step < WARMUP_STEPS else LR * 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)))
        for g in optimizer.param_groups: g['lr'] = lr
        optimizer.step(); optimizer.zero_grad()
    if step % LOG_EVERY == 0:
        model.eval()
        with torch.no_grad():
            eval_loss = sum(F.cross_entropy(model(ebx).reshape(-1, VOCAB), eby.reshape(-1)).item() for ebx, eby in eval_loader) / len(eval_loader)
        tok_s = step * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN / (time.perf_counter() - t_start)
        print(f'  Step {step:4d}/{N_STEPS} | loss={loss.item():.4f} | eval_ppl={math.exp(eval_loss):.1f} | {tok_s:.0f} tok/s | {time.perf_counter()-t_start:.0f}s')

t_total = time.perf_counter() - t_start
print(f'\n  Done: {t_total:.0f}s, avg {(step * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN / t_total):.0f} tok/s')
print(f'  Total params: {n_all/1e6:.2f}M')
