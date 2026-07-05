"""
test_pscan_gate.py — CovGate + PScan prototype (reference).
Predecessor to MemBind. First to combine covariance memory + parallel scan.
Superseded by ld_model/core.py (MemBindBlock).
Full architecture description: LAMBDA_ARCHITECTURE.md
"""

import math, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

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
COV_RANK = 16

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


# ─── Parallel associative scan (Hillis-Steele) ─────────────────────────
def parallel_prefix_scan(a, b):
    """
    M[t] = a[t]·M[t-1] + b[t],  M[-1] = 0

    a: (B, L) decay scalars (in (0,1))
    b: (B, L, r, r) matrix increments
    Returns M: (B, L, r, r) — all prefix sums

    Hillis-Steele: O(log L) шагов.
    Без in-place writes — autograd-safe.
    Ассоциативность: combine((A1,B1),(A2,B2)) = (A1*A2, A2*B1+B2)
    """
    L = a.shape[1]
    # decays: (B, L, 1, 1),  increments: (B, L, r, r)
    A = a.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
    M = b  # (B, L, r, r)

    stride = 1
    while stride < L:
        # Все reads до любых writes — autograd-safe
        A_head, A_tail = A[:, :L-stride], A[:, stride:]
        M_head, M_tail = M[:, :L-stride], M[:, stride:]

        A_combined = A_head * A_tail          # (B, L-stride, 1, 1)
        M_combined = A_tail * M_head + M_tail # (B, L-stride, r, r) — broadcast A_tail

        # build new tensors via cat (новые тензоры, autograd graph сохранён)
        A = torch.cat([A[:, :stride], A_combined], dim=1)
        M = torch.cat([M[:, :stride], M_combined], dim=1)
        stride *= 2

    return M  # (B, L, r, r)


# ─── Components ─────────────────────────────────────────────────────────
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
# PScanLDBlock — Bind + Covariance Memory + Parallel Scan
# ═════════════════════════════════════════════════════════════════════════
class PScanLDBlock(nn.Module):
    """
    Bind pre-transformation + low-rank covariance memory + parallel scan.

    1. Bind adaptation:     h_adapt = h_norm + (W_u·h * W_v·h) @ W_out
    2. Covariance update:   M[t] = a[t]·M[t-1] + b[t]     (parallel scan)
       a[t] = sigmoid(W_decay·h_norm) ∈ (0,1)
       b[t] = exp(W_i·h_norm) · (k[t]^T @ k[t])
    3. Memory read:         h_mem[t] = q[t] @ M[t] @ W_read
    4. Spectral operator:   Δ = V · diag(λ) · V^T · (h_adapt + h_mem)
    5. Residual:            h_out = h + Δ

    Нет softmax, нет sigmoid gate, нет sequential loop.
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

        # Bind adaptation
        self.W_u = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_out = nn.Parameter(torch.zeros(self.r, D))

        # Covariance memory (all parallel: K, Q, decays, input gate)
        self.W_k = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_q = nn.Parameter(torch.randn(D, self.r) * 0.01)
        self.W_i = nn.Parameter(torch.randn(D, 1) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(1))
        self.W_decay = nn.Parameter(torch.randn(D, 1) * 0.01)
        self.b_decay = nn.Parameter(torch.full((1,), 1.0))
        self.W_read = nn.Parameter(torch.zeros(self.r, D))

    def forward(self, h):
        B, L, D = h.shape

        # Conv + Norm
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        # Bind adaptation (parallel over L)
        u = h_norm @ self.W_u
        v = h_norm @ self.W_v
        h_adapt = h_norm + (u * v) @ self.W_out

        # Covariance: keys, queries, gates (parallel over L)
        K = h_norm @ self.W_k          # (B, L, r)
        Q = h_norm @ self.W_q          # (B, L, r)

        # Decay a[t] ∈ (0,1) and input gate b[t] with exp amplification
        decay = torch.sigmoid(h_norm @ self.W_decay + self.b_decay).squeeze(-1)  # (B, L)
        i_gate = torch.exp(h_norm @ self.W_i + self.b_i).squeeze(-1)            # (B, L)

        # Covariance increments: b[t] = i_gate[t] · k[t]^T @ k[t]
        K_exp = K.unsqueeze(-1)   # (B, L, r, 1)
        delta = (K_exp @ K_exp.transpose(-2, -1)) * i_gate.unsqueeze(-1).unsqueeze(-1)  # (B, L, r, r)

        # Parallel prefix scan: M[t] = a[t]·M[t-1] + delta[t]
        M = parallel_prefix_scan(decay, delta)  # (B, L, r, r)

        # Memory read: h_mem[t] = q[t] @ M[t] @ W_read
        h_mem = (Q.unsqueeze(-2) @ M).squeeze(-2) @ self.W_read  # (B, L, D)

        # Spectral operator with fixed λ
        h_total = h_adapt + h_mem
        h_proj = h_total @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, self.D)
        delta_spec = h_scaled @ self.V

        return h + delta_spec, None


# ═════════════════════════════════════════════════════════════════════════
# Stack + Model
# ═════════════════════════════════════════════════════════════════════════
class PScanLDStack(nn.Module):
    def __init__(self, cov_r=16):
        super().__init__()
        self.register_buffer('final_norm_w', torch.ones(D))
        self.layers = nn.ModuleList([PScanLDBlock(i, cov_r) for i in range(N_LAYERS)])
        self.mlps = nn.ModuleList([BottleneckMLP(D) for _ in range(N_LAYERS)])

    def forward(self, h):
        for lidx in range(N_LAYERS):
            h_layer, _ = self.layers[lidx](h)
            h_norm_mlp = rms_norm(h_layer, self.final_norm_w)
            h = h_layer + self.mlps[lidx](h_norm_mlp)
        return rms_norm(h, self.final_norm_w)


class PScanModel(nn.Module):
    def __init__(self, cov_r=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        self.stack = PScanLDStack(cov_r)
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
print(f'  Train: {n_train} ({n_train*SEQ_LEN/1e6:.1f}M tok)  Eval: {n_eval}')

train_ids = torch.tensor(arr[:n_train], dtype=torch.long).to(DEVICE)
eval_ids  = torch.tensor(arr[n_train:], dtype=torch.long).to(DEVICE)
train_x, train_y = train_ids[:, :-1], train_ids[:, 1:]
eval_x,  eval_y  = eval_ids[:, :-1],  eval_ids[:, 1:]
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(TensorDataset(eval_x, eval_y),  batch_size=BATCH_SIZE)

# ─── Train ───────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'Training: PSCAN (r={COV_RANK}) — parallel scan covariance memory')
print(f'{"="*60}')

model = PScanModel(COV_RANK).to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
print(f'  Params: {n_all/1e6:.2f}M')

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
print(f'  Params: {n_all/1e6:.2f}M')
