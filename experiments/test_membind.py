"""
test_membind.py — MemBind prototype (reference).
First full implementation of multi-head covariance memory + bind feedback.
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
COV_HEADS = 4    # = N_MODES, одна память на моду
COV_R = 8        # rank per head → 4×8×8 = 256 элементов (как CovGate r=16)
BIND_R = 16      # bind rank (как в BindGate)

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


# ─── Parallel prefix scan (autograd-safe) ─────────────────────────────
def parallel_prefix_scan(a, b):
    """
    M[t] = a[t]·M[t-1] + b[t],  M[-1] = 0
    a: (B, L, H) decays
    b: (B, L, H, r, r) increments
    Returns M: (B, L, H, r, r)
    """
    L = a.shape[1]
    A = a.unsqueeze(-1).unsqueeze(-1)  # (B, L, H, 1, 1)
    M = b
    stride = 1
    while stride < L:
        A_left, A_right = A[:, :L-stride], A[:, stride:]
        M_left, M_right = M[:, :L-stride], M[:, stride:]
        A_combined = A_left * A_right
        M_combined = A_right * M_left + M_right
        A = torch.cat([A[:, :stride], A_combined], dim=1)
        M = torch.cat([M[:, :stride], M_combined], dim=1)
        stride *= 2
    return M


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
# MemBindBlock — Multi-head covariance + bind feedback + spectral operator
# ═════════════════════════════════════════════════════════════════════════
class MemBindBlock(nn.Module):
    """
    Цикл: память → bind → спектр.

    1. conv → norm
    2. u = h_norm @ W_u, v = h_norm @ W_v
    3. Multi-head covariance update (H=COV_HEADS, r=COV_R):
       k_h = h_norm @ W_k_h, q_h = h_norm @ W_q_h
       i_h = exp(W_i_h · h_norm), d_h = sigmoid(W_decay_h · h_norm)
       M_h[t] = d_h·M_h[t-1] + i_h·k_h^T@k_h             (parallel scan)
       mem_h = q_h @ M_h @ W_read_h
    4. Enhanced bind: v_enh = v + sum_h(mem_h)
       h_adapt = h_norm + (u * v_enh) @ W_out
    5. Spectral: Δ = V·diag(λ)·V^T·h_adapt
    6. h_out = h + Δ
    """
    def __init__(self, layer_idx):
        super().__init__()
        self.D = D
        self.K = N_MODES
        self.block_size = D // N_MODES
        self.H = COV_HEADS
        self.r = COV_R

        self.conv = CausalConv1d(D)
        self.register_buffer('ln_w', torch.ones(D))
        V_init = random_orthogonal(D)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', LAMBDA_ROOTS)

        # Bind adaptation
        self.W_u = nn.Parameter(torch.randn(D, BIND_R) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, BIND_R) * 0.01)
        self.W_out = nn.Parameter(torch.zeros(BIND_R, D))

        # Multi-head covariance memory (H heads, each r-dimensional)
        # Stack all head params for efficient matmul
        H_heads = self.H
        self.W_k = nn.Parameter(torch.randn(H_heads, D, self.r) * 0.01)
        self.W_q = nn.Parameter(torch.randn(H_heads, D, self.r) * 0.01)
        self.W_i = nn.Parameter(torch.randn(H_heads, D, 1) * 0.01)
        self.b_i = nn.Parameter(torch.zeros(H_heads, 1))
        self.W_decay = nn.Parameter(torch.randn(H_heads, D, 1) * 0.01)
        self.b_decay = nn.Parameter(torch.full((H_heads, 1), 2.0))
        self.W_read = nn.Parameter(torch.zeros(H_heads, self.r, D))
        self.W_mem2v = nn.Parameter(torch.zeros(D, BIND_R))  # mem_sum → v_enh projection

    def forward(self, h):
        B, L, D = h.shape
        H, r = self.H, self.r

        # 1. Conv + Norm
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        # 2. Bind projections (parallel over L)
        u = h_norm @ self.W_u     # (B, L, BIND_R)
        v = h_norm @ self.W_v     # (B, L, BIND_R)

        # 3. Multi-head covariance (all heads in parallel via einsum)
        # K, Q: (B, H, L, r) — H heads × L tokens × r-dim key/query
        K = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k)  # key
        Q = torch.einsum('bld,hdr->bhlr', h_norm, self.W_q)  # query

        i_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_i) + self.b_i.view(1, H, 1, 1)
        i_gate = torch.exp(i_raw)  # (B, H, L, 1) — exponential input gate

        decay_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_decay) + self.b_decay.view(1, H, 1, 1)
        decay = torch.sigmoid(decay_raw)  # (B, H, L, 1) ∈ (0,1)

        # Covariance increments: delta = i_gate · k^T @ k (per head, per token)
        K_e = K.unsqueeze(-1)  # (B, H, L, r, 1)
        delta = (K_e @ K_e.transpose(-2, -1)) * i_gate.unsqueeze(-1)  # (B, H, L, r, r)

        # Parallel scan over L: M[t] = decay[t]·M[t-1] + delta[t]
        # Reshape for scan: a (B, L, H), b (B, L, H, r, r)
        a_scan = decay.squeeze(-1).permute(0, 2, 1)     # (B, L, H)
        b_scan = delta.permute(0, 2, 1, 3, 4)            # (B, L, H, r, r)
        M_all = parallel_prefix_scan(a_scan, b_scan)     # (B, L, H, r, r)

        # Memory readout: mem_h[t] = q_h[t] @ M_h[t] @ W_read_h
        # Q: (B, H, L, r) → (B, L, H, r),  M: (B, L, H, r, r)
        Q_perm = Q.permute(0, 2, 1, 3)    # (B, L, H, r)
        mem_r = (Q_perm.unsqueeze(-2) @ M_all).squeeze(-2)  # (B, L, H, r)
        mem_D = torch.einsum('blhr,hro->blho', mem_r, self.W_read)  # (B, L, H, D)
        mem_sum = mem_D.sum(dim=2)  # (B, L, D) — aggregate over heads

        # 4. Memory feedback → enhanced bind
        # v_enh = v + W_mem2v · mem_sum  (memory modulates bind's v signal)
        v_enh = v + (mem_sum @ self.W_mem2v)

        # h_adapt = h_norm + (u * v_enh) @ W_out
        h_adapt = h_norm + (u * v_enh) @ self.W_out

        # 5. Spectral operator (fixed λ)
        h_proj = h_adapt @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, self.D)
        delta_spec = h_scaled @ self.V

        return h + delta_spec, None


# ═════════════════════════════════════════════════════════════════════════
# Stack + Model
# ═════════════════════════════════════════════════════════════════════════
class MemBindStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('final_norm_w', torch.ones(D))
        self.layers = nn.ModuleList([MemBindBlock(i) for i in range(N_LAYERS)])
        self.mlps = nn.ModuleList([BottleneckMLP(D) for _ in range(N_LAYERS)])

    def forward(self, h):
        for lidx in range(N_LAYERS):
            h_layer, _ = self.layers[lidx](h)
            h_norm_mlp = rms_norm(h_layer, self.final_norm_w)
            h = h_layer + self.mlps[lidx](h_norm_mlp)
        return rms_norm(h, self.final_norm_w)


class MemBindModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        self.stack = MemBindStack()
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
train_ids = torch.tensor(arr[:n_train], dtype=torch.long).to(DEVICE)
eval_ids  = torch.tensor(arr[n_train:], dtype=torch.long).to(DEVICE)
train_x, train_y = train_ids[:, :-1], train_ids[:, 1:]
eval_x,  eval_y  = eval_ids[:, :-1],  eval_ids[:, 1:]
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(TensorDataset(eval_x, eval_y),  batch_size=BATCH_SIZE)

# ─── Train ───────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'Training: MEMBIND (H={COV_HEADS}, r={COV_R}, bind_r={BIND_R})')
print(f'  Multi-head covariance + memory feedback + bind enhancement')
print(f'{"="*60}')

model = MemBindModel().to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_mem = sum(p.numel() for n, p in model.named_parameters() if 'W_k' in n or 'W_q' in n or 'W_i' in n or 'W_decay' in n or 'W_read' in n or 'W_mem' in n)
n_bind = sum(p.numel() for n, p in model.named_parameters() if 'W_u' in n or 'W_v' in n or 'W_out' in n)
print(f'  Params: {n_all/1e6:.2f}M | bind: {n_bind:,} | memory: {n_mem:,}')

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
