"""Benchmark model throughput, latency, memory usage."""
import torch, sys, time, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

# Minimal model class
class Phase2Model(torch.nn.Module):
    def __init__(self, use_gc=False, use_rc=False):
        super().__init__()
        self.use_gc = use_gc
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        cfg.kernel_size = 48
        cfg.use_global_context = use_gc
        cfg.recurrent_scan = use_rc
        cfg.weight_tying = True; cfg.lm_head_bias = True
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=cfg.lm_head_bias)
        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight
    def forward(self, x, return_gates=False):
        h = self.embed(x)
        if self.use_gc:
            ctx = self.stack(h).mean(dim=1)
            h2 = self.embed(x) + self.stack.ctx_proj(ctx).unsqueeze(1)
            h = self.stack(h2, context=ctx)
        elif return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        else:
            h = self.stack(h)
        return self.lm_head(h)

model = Phase2Model().to(DEVICE)
model.eval()

n = sum(p.numel() for p in model.parameters())
n_embed = sum(p.numel() for p in model.embed.parameters())
n_stack = sum(p.numel() for p in model.stack.parameters())
n_head = sum(p.numel() for p in model.lm_head.parameters())
print(f'\nParams: {n/1e6:.1f}M total')
print(f'  embed:   {n_embed/1e6:.1f}M')
print(f'  stack:   {n_stack/1e6:.1f}M')
print(f'  lm_head: {n_head/1e6:.1f}M')

if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    base_mem = torch.cuda.memory_allocated()
    print(f'Base VRAM: {base_mem/1e6:.1f} MB')

# Forward pass benchmark
configs = [
    (1, 64), (1, 512), (1, 1024), (1, 2048),
    (4, 64), (8, 64), (16, 64), (32, 64),
    (4, 128), (4, 256), (4, 512),
]

print(f'\n{"B":>4} {"L":>5} | {"latency":>8} {"tok/s":>8}', end='')
if DEVICE.type == 'cuda':
    print(f' {"VRAM":>10} {"VRAMpk":>10}')
else:
    print()
print('-' * 55)

for B, L in configs:
    x = torch.randint(0, VOCAB, (B, L), device=DEVICE)

    if DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(3):
        _ = model(x)

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

    n_runs = 20 if B * L <= 4096 else 10
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(x)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_runs

    tok_s = B * L / dt
    mem = torch.cuda.memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0
    mem_p = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0

    print(f'{B:>4} {L:>5} | {dt*1000:>6.1f}ms {tok_s:>8.0f}', end='')
    if DEVICE.type == 'cuda':
        print(f' {mem:>8.1f}MB {mem_p:>8.1f}MB')
    else:
        print()

# Generation benchmarks
print(f'\n--- Generation (B=1, prompt=128, gen=200 tok) ---')
print(f'  Method 1: full sequence each step (naive)')
x = torch.randint(0, VOCAB, (1, 128), device=DEVICE)
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for i in range(200):
        logits = model(x)[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
dt_gen = time.perf_counter() - t0
mem_gen = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0
print(f'    {200/dt_gen:.0f} tok/s, {dt_gen/200*1000:.1f} ms/tok')
print(f'    Peak VRAM: {mem_gen:.0f} MB')

# Generation with sliding window (causal conv = k=4 per layer = 48 effective)
print(f'  Method 2: sliding window (last 64 tokens, conv receptive=48)')
x = torch.randint(0, VOCAB, (1, 128), device=DEVICE)
WINDOW = 64
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

t0 = time.perf_counter()
with torch.no_grad():
    for i in range(200):
        ctx = x[:, -WINDOW:] if x.shape[1] > WINDOW else x
        logits = model(ctx)[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
dt_gen2 = time.perf_counter() - t0
mem_gen2 = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0
print(f'    {200/dt_gen2:.0f} tok/s, {dt_gen2/200*1000:.1f} ms/tok')
print(f'    Peak VRAM: {mem_gen2:.0f} MB')
print(f'    Speedup: {dt_gen/dt_gen2:.1f}x')

# Recurrent scan version benchmark
print(f'\n--- Recurrent scan (infinite context) ---')
model_rc = Phase2Model(use_gc=False, use_rc=True).to(DEVICE)
model_rc.eval()
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

x = torch.randint(0, VOCAB, (1, 128), device=DEVICE)
for _ in range(3):
    _ = model_rc(x)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

n_runs = 10
t0 = time.perf_counter()
for _ in range(n_runs):
    _ = model_rc(x)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
dt_rc = (time.perf_counter() - t0) / n_runs
mem_rc = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0
tok_rc = 128 / dt_rc
print(f'  B=1 L=128: {dt_rc*1000:.1f}ms, {tok_rc:.0f} tok/s')
print(f'  Peak VRAM: {mem_rc:.0f} MB')

# GC version benchmark
print(f'\n--- Global Context version ---')
model_gc = Phase2Model(use_gc=True).to(DEVICE)
model_gc.eval()
if DEVICE.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

x = torch.randint(0, VOCAB, (1, 128), device=DEVICE)
for _ in range(3):
    _ = model_gc(x)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

n_runs = 20
t0 = time.perf_counter()
for _ in range(n_runs):
    _ = model_gc(x)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
dt_gc = (time.perf_counter() - t0) / n_runs
mem_gc = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == 'cuda' else 0
tok_gc = 128 / dt_gc
print(f'  B=1 L=128: {dt_gc*1000:.1f}ms, {tok_gc:.0f} tok/s')
print(f'  Peak VRAM: {mem_gc:.0f} MB')
print(f'  Overhead: {(dt_gc/dt - 1)*100:.0f}% slower')

# Memory breakdown
if DEVICE.type == 'cuda':
    print(f'\n--- Memory breakdown (B=1, L=128) ---')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    def mem_snapshot(label):
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        print(f'  {label:>20}: cur={cur/1e6:.0f}MB peak={peak/1e6:.0f}MB')

    x = torch.randint(0, VOCAB, (1, 128), device=DEVICE)
    mem_snapshot('after input')
    h = model.embed(x)
    mem_snapshot('after embed')
    h = model.stack(h)
    mem_snapshot('after stack')
    logits = model.lm_head(h)
    mem_snapshot('after lm_head')
    del h, logits, x

print(f'\nDone.')
