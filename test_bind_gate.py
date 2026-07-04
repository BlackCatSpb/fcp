"""
test_bind_gate.py — Сравнение sigmoid gate vs bind-based content adaptation.

Гипотеза: Gating (α·λ) можно заменить на content-dependent pre-transformation
через низкоранговый bind (u * v), инспирированный FCF.

BindGate не использует softmax / sigmoid / α.
Content-dependence — через билинейную операцию u*v (element-wise multiply).

Запуск:
    python test_bind_gate.py              # 5000 steps, r=16, sigmoid vs bind
    python test_bind_gate.py --bind_r 8   # bind rank=8
    python test_bind_gate.py --steps 2000
    python test_bind_gate.py --only bind  # только bind (без baseline)
"""

import os, sys, math, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
print(f'CUDA mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if DEVICE.type == 'cuda' else '')

# ─── Config ──────────────────────────────────────────────────────────────
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 4
ACCUM_STEPS = 4        # eff batch = 16 (as fresh start)
SEQ_LEN = 64           # chunks are 64 tokens
LR = 1e-3
WARMUP_STEPS = 250
N_STEPS = 5000
GRAD_CLIP = 1.0
LOG_EVERY = 100
DATA_FILE = 'russian_chunks.npy'
N_CHUNKS = 20000

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bind_r', type=int, default=16, help='Bind rank (default: 16)')
parser.add_argument('--steps', type=int, default=N_STEPS)
parser.add_argument('--only', type=str, default=None, choices=['sigmoid', 'bind'])
args = parser.parse_args()
BIND_R = args.bind_r
N_STEPS = args.steps

# ─── Fibonacci roots ────────────────────────────────────────────────────
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

# ─── Shared components ──────────────────────────────────────────────────
def rms_norm(x, weight, eps=1e-6):
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x / rms.clamp(min=eps) * weight

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
# SigmoidGate LDBlock (baseline) — current approach
# ═════════════════════════════════════════════════════════════════════════
class SigmoidGateBlock(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.D = D
        self.K = N_MODES
        self.block_size = D // N_MODES

        self.conv = CausalConv1d(D)
        self.register_buffer('ln_w', torch.ones(D))

        V_init = random_orthogonal(D)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', LAMBDA_ROOTS)

        # Gate (sigmoid)
        self.W_gate = nn.Parameter(torch.randn(D, N_MODES) * 0.01)
        self.b_gate = nn.Parameter(torch.full((N_MODES,), -1.1))
        self.gate_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        B, L, _ = h.shape
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        gate_logits = (h_norm @ self.W_gate + self.b_gate) * self.gate_scale
        alpha = torch.sigmoid(gate_logits)
        lambda_alpha = self.lambda_k * alpha

        h_proj = h_norm @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * lambda_alpha.unsqueeze(-1)).reshape(B, L, self.D)
        delta = h_scaled @ self.V

        return h + delta, alpha


# ═════════════════════════════════════════════════════════════════════════
# BindGate LDBlock — gate replacement via FCF-inspired bilinear binding
# ═════════════════════════════════════════════════════════════════════════
class BindGateBlock(nn.Module):
    """
    λ_d без gate. Content-dependence через bind двух низкоранговых проекций.

    h_norm → u, v (линейные проекции)
    bound = u * v                         # FCF bind (element-wise multiply)
    h_adapt = h_norm + bound @ W_out      # content-dependent pre-transform
    Δ = V · diag(λ) · V^T · h_adapt       # фиксированный спектральный оператор
    h_out = h + Δ                          # residual

    Никакого softmax/sigmoid/α.
    """
    def __init__(self, layer_idx, bind_r=16):
        super().__init__()
        self.D = D
        self.K = N_MODES
        self.block_size = D // N_MODES
        self.r = bind_r

        self.conv = CausalConv1d(D)
        self.register_buffer('ln_w', torch.ones(D))

        V_init = random_orthogonal(D)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', LAMBDA_ROOTS)

        # Bind weights — заменяют W_gate + b_gate + gate_scale
        self.W_u = nn.Parameter(torch.randn(D, bind_r) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, bind_r) * 0.01)
        self.W_out = nn.Parameter(torch.zeros(bind_r, D))  # zero init → identity at start

    def forward(self, h):
        B, L, _ = h.shape
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        # Content adaptation via bilinear bind (FCF-inspired)
        u = h_norm @ self.W_u       # (B, L, r)
        v = h_norm @ self.W_v       # (B, L, r)
        bound = u * v               # (B, L, r) — element-wise bind
        h_adapt = h_norm + (bound @ self.W_out)  # (B, L, D) — residual

        # Fixed spectral operator (λ_k unchanged, no gating)
        h_proj = h_adapt @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, self.D)
        delta = h_scaled @ self.V

        return h + delta, None  # no gates returned


# ═════════════════════════════════════════════════════════════════════════
# Stack (shared for both architectures)
# ═════════════════════════════════════════════════════════════════════════
class LDStack(nn.Module):
    def __init__(self, block_type='sigmoid', bind_r=16):
        super().__init__()
        self.block_type = block_type
        self.register_buffer('final_norm_w', torch.ones(D))

        if block_type == 'sigmoid':
            self.layers = nn.ModuleList([SigmoidGateBlock(i) for i in range(N_LAYERS)])
        elif block_type == 'bind':
            self.layers = nn.ModuleList([BindGateBlock(i, bind_r) for i in range(N_LAYERS)])
        else:
            raise ValueError(f'Unknown block_type: {block_type}')

        self.mlps = nn.ModuleList([BottleneckMLP(D) for _ in range(N_LAYERS)])

    def forward(self, h, return_gates=False):
        gates = [] if return_gates else None

        for lidx in range(N_LAYERS):
            h_layer, alpha = self.layers[lidx](h)

            # MLP с adaptive gain (только для sigmoid — есть alpha)
            h_norm_mlp = rms_norm(h_layer, self.final_norm_w)
            h_mlp = h_layer + self.mlps[lidx](h_norm_mlp)

            if self.block_type == 'sigmoid' and alpha is not None and lidx < N_LAYERS - 1:
                gain = alpha.mean(dim=-1, keepdim=True)  # (B, L, 1)
                h = h + gain * (h_mlp - h)
            else:
                h = h_mlp

            if return_gates:
                gates.append(alpha)

        h_out = rms_norm(h, self.final_norm_w)
        if return_gates:
            return h_out, torch.stack(gates, dim=0)
        return h_out


# ═════════════════════════════════════════════════════════════════════════
# Full model
# ═════════════════════════════════════════════════════════════════════════
class TestModel(nn.Module):
    def __init__(self, block_type='sigmoid', bind_r=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        # Embed init: uniform(-1/sqrt(D), 1/sqrt(D)) вместо N(0,1)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        self.stack = LDStack(block_type, bind_r)
        self.lm_head = nn.Linear(D, VOCAB, bias=True)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(self, input_ids, return_gates=False):
        h = self.embed(input_ids)
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        return self.lm_head(h)


# ─── Data ────────────────────────────────────────────────────────────────
print(f'\nLoading data: {DATA_FILE}')
t0 = time.perf_counter()
arr = np.load(DATA_FILE)
arr = arr[:N_CHUNKS]
print(f'  {arr.shape[0]} chunks loaded in {time.perf_counter()-t0:.1f}s')

n_train = int(N_CHUNKS * 0.95)
n_eval = N_CHUNKS - n_train
print(f'  Train: {n_train} chunks ({n_train * SEQ_LEN / 1e6:.1f}M tok)')
print(f'  Eval:  {n_eval} chunks ({n_eval * SEQ_LEN / 1e6:.1f}M tok)')

train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)

train_x = train_ids[:, :-1].to(DEVICE)
train_y = train_ids[:, 1:].to(DEVICE)
eval_x = eval_ids[:, :-1].to(DEVICE)
eval_y = eval_ids[:, 1:].to(DEVICE)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=BATCH_SIZE)

print(f'  Train batches/epoch: {len(train_loader)}')
print(f'  Steps: {N_STEPS}')


# ─── Training function ──────────────────────────────────────────────────
def train_model(block_type, bind_r=16):
    print(f'\n{"="*60}')
    print(f'Training: {block_type.upper()}' + (f' (r={bind_r})' if block_type == 'bind' else ''))
    print(f'{"="*60}')

    model = TestModel(block_type, bind_r).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_cayley = sum(p.numel() for n, p in model.named_parameters() if 'W_u' in n or 'W_v' in n or 'W_out' in n)
    n_gate = sum(p.numel() for n, p in model.named_parameters() if 'W_gate' in n or 'b_gate' in n or 'gate_scale' in n)
    print(f'  Params: {n_params/1e6:.2f}M | gate: {n_gate:,} | bind: {n_cayley:,}')

    # Sanity
    model.eval()
    with torch.no_grad():
        bx_test = next(iter(train_loader))[0][:1]
        h = model.embed(bx_test)
        print(f'  sanity: embed=[{h.min():.2f},{h.max():.2f}]')
        h = model.stack(h)
        nan, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
        print(f'  sanity: stack=[{h.min():.2f},{h.max():.2f}] nan={nan} inf={inf}')
        logits = model.lm_head(h)
        print(f'  sanity: logits=[{logits.min():.2f},{logits.max():.2f}]')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(N_STEPS - WARMUP_STEPS, 1)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    step = 0
    losses = []
    ppls = []
    times = []
    t_start = time.perf_counter()
    optimizer.zero_grad()

    data_iter = itertools.cycle(train_loader)
    while step < N_STEPS:
        model.train()
        bx, by = next(data_iter)
        logits = model(bx)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f'  [SKIP] step {step}: nan/inf in logits')
            continue

        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        if torch.isnan(loss):
            print(f'  [SKIP] step {step}: nan in loss')
            continue

        (loss / ACCUM_STEPS).backward()
        step += 1

        if step % ACCUM_STEPS == 0 or step == N_STEPS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                eval_loss = 0.0
                for ebx, eby in eval_loader:
                    elogits = model(ebx)
                    eval_loss += F.cross_entropy(elogits.reshape(-1, VOCAB), eby.reshape(-1)).item()
                eval_ppl = math.exp(eval_loss / len(eval_loader))

            elapsed = time.perf_counter() - t_start
            losses.append(loss.item())
            ppls.append(eval_ppl)
            times.append(elapsed)
            tok_per_sec = step * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN / elapsed
            print(f'  Step {step:4d}/{N_STEPS} | loss={loss.item():.4f} | eval_ppl={eval_ppl:.1f} | '
                  f'{tok_per_sec:.0f} tok/s | {elapsed:.0f}s')

    total_time = time.perf_counter() - t_start
    print(f'\n  Done: {total_time:.0f}s, avg {(step * BATCH_SIZE * ACCUM_STEPS * SEQ_LEN / total_time):.0f} tok/s')

    return {
        'type': block_type,
        'bind_r': bind_r if block_type == 'bind' else 0,
        'params': n_params,
        'time': total_time,
        'final_loss': losses[-1] if losses else None,
        'final_ppl': ppls[-1] if ppls else None,
        'losses': losses,
        'ppls': ppls,
        'model': model,
    }


# ═════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════
results = []

if args.only is None or args.only == 'sigmoid':
    results.append(train_model('sigmoid'))

if args.only is None or args.only == 'bind':
    results.append(train_model('bind', bind_r=BIND_R))

# ─── Summary ────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
for r in results:
    print(f'  {r["type"].upper():8s} | '
          f'{r["params"]/1e6:.2f}M | '
          f'time={r["time"]:.0f}s | '
          f'final loss={r["final_loss"]:.4f} | '
          f'final PPL={r["final_ppl"]:.1f}')

if len(results) == 2:
    s = results[0]
    b = results[1]
    if s['final_ppl'] and b['final_ppl']:
        ratio = b['final_ppl'] / s['final_ppl']
        print(f'\n  Bind/Sigmoid PPL ratio: {ratio:.3f}')
        if ratio < 1.0:
            print(f'  BindGate WINS by {100*(1-ratio):.1f}%')
        else:
            print(f'  SigmoidGate WINS by {100*(ratio-1):.1f}%')
