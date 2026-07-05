"""
Test multi-timescale heads: разные τ на каждую голову.
Проверка: спектр τ работает лучше, чем единый τ для всех.
"""
import os, sys, math, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ld_model.core import LDConfig, MemBindBlock, MemBindStack, compute_spectrum, parallel_prefix_scan_1d

def make_synthetic_batch(B, L, vocab, rng):
    tokens = rng.randint(0, vocab, (B, L + 24))
    for b in range(B):
        for i in range(0, L - 2, 3):
            if rng.random() < 0.7:
                t = tokens[b, i]
                tokens[b, i+1] = (t + 1) % vocab
                tokens[b, i+2] = t
        for i in range(0, L - 8, 8):
            if rng.random() < 0.5:
                base = tokens[b, i]
                for j in range(1, 8):
                    tokens[b, i+j] = (base + j) % vocab
        for i in range(0, L - 24, 24):
            if rng.random() < 0.3:
                base = tokens[b, i]
                for j in range(1, 24, 2):
                    tokens[b, i+j] = base
    return torch.from_numpy(tokens[:, :L]), torch.from_numpy(tokens[:, 1:L+1])

def make_stack(cfg, timescales):
    """Create MemBindStack with per-head frozen b_decay."""
    lam, bs = compute_spectrum(cfg)
    # Build layers with overridden b_decay
    layers = nn.ModuleList()
    for i in range(cfg.n_layers):
        blk = MemBindBlock(cfg, i, lam, bs)
        # Override with per-head timescales
        d = 1.0 - 1.0 / torch.tensor(timescales, dtype=torch.float32)
        b = -torch.log(1.0 / d - 1.0)
        blk.register_buffer('b_decay', b.view(cfg.cov_heads, 1))
        blk._timescales = timescales
        layers.append(blk)
    # Wrap in a simple stack
    class SimpleStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers
            self.mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(cfg.D, cfg.bottleneck), nn.SiLU(),
                              nn.Linear(cfg.bottleneck, cfg.D))
                for _ in range(cfg.n_layers)
            ])
            self.fnw = nn.Parameter(torch.ones(cfg.D))
        def forward(self, h, state=None):
            if state is None: state = [None] * cfg.n_layers
            ns = []
            for lidx in range(cfg.n_layers):
                h, ls = self.layers[lidx](h, state[lidx])
                nh = h * (self.fnw / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt())
                h = h + self.mlps[lidx](nh)
                ns.append(ls)
            return h * (self.fnw / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt()), ns
    return SimpleStack()

def test_timescales(timescales, label, cfg_template):
    cfg = LDConfig()
    for k, v in cfg_template.__dict__.items():
        setattr(cfg, k, v)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    D, V, L = cfg.D, cfg.vocab, 256  # longer context
    stack = make_stack(cfg, timescales)
    model = nn.Sequential(nn.Embedding(V, D), stack, nn.Linear(D, V, bias=True)).to(dev)
    with torch.no_grad(): model[2].weight = model[0].weight
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.RandomState(42)
    losses = []
    for step in range(150):
        x, y = make_synthetic_batch(4, L, V, rng)
        x, y = x.to(dev).long(), y.to(dev).long()
        h = model[0](x); out, _ = model[1](h); logits = model[2](out)
        loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses[-1]

cfg_base = LDConfig(); cfg_base.D = 256; cfg_base.n_layers = 4; cfg_base.n_modes = 4
cfg_base.vocab = 200; cfg_base.cov_heads = 4; cfg_base.cov_r = 16; cfg_base.bind_r = 16
cfg_base.kernel_size = 16; cfg_base.cov_rf = True; cfg_base.cov_rf_dim = 32
cfg_base.cov_first_moment = True

print('=== Multi-timescale vs single-timescale (L=256, 4 layers, 150 steps) ===')
tests = [
    ('all 55',      [55, 55, 55, 55]),
    ('all 5',       [5,  5,  5,  5]),
    ('all 200',     [200,200,200,200]),
    ('multi 5-200', [5,  15, 55, 200]),
    ('multi 3-300', [3,  8,  30, 300]),
    ('multi 10-150',[10, 25, 70, 150]),
]
results = []
for label, ts in tests:
    t0 = time.time()
    loss = test_timescales(ts, label, cfg_base)
    elapsed = time.time() - t0
    results.append((label, loss))
    print(f'  {label:15s}  loss={loss:.4f}  ({elapsed:.0f}s)')
best = min(results, key=lambda r: r[1])
print(f'\n-> Best: {best[0]} ({best[1]:.4f})')
print('Multi wins' if best[0].startswith('multi') else 'Single wins')
