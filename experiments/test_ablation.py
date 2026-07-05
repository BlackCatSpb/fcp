"""
Ablation: test first moment and random features independently.
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ld_model.core import LDConfig, compute_spectrum, parallel_prefix_scan, dct_basis

from test_cov_enhanced import EnhancedMemBindBlock, parallel_prefix_scan_1d

def build_model(cfg, use_first_moment, use_rf, rf_dim=32):
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            lam, bs = compute_spectrum(cfg)
            self.layers = nn.ModuleList([
                EnhancedMemBindBlock(cfg, i, lam, bs,
                    use_first_moment=use_first_moment, use_rf=use_rf, rf_dim=rf_dim)
                for i in range(cfg.n_layers)
            ])
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.D, cfg.bottleneck), nn.SiLU(),
                    nn.Linear(cfg.bottleneck, cfg.D)
                ) for _ in range(cfg.n_layers)
            ])
            self.final_norm_w = nn.Parameter(torch.ones(cfg.D))
        def forward(self, h, state=None):
            if state is None: state = [None] * cfg.n_layers
            ns = []
            for lidx in range(cfg.n_layers):
                h, ls = self.layers[lidx](h, state[lidx])
                h = h + self.mlps[lidx](h * (self.final_norm_w / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt()))
                ns.append(ls)
            return h * (self.final_norm_w / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt()), ns
    D = cfg.D
    model = nn.Sequential(
        nn.Embedding(cfg.vocab, D),
        Stack(),
        nn.Linear(D, cfg.vocab, bias=True)
    )
    model[0].weight.data.uniform_(-0.1, 0.1)
    with torch.no_grad(): model[2].weight = model[0].weight
    return model

def test_config(name, use_first_moment, use_rf):
    cfg = LDConfig(); cfg.D=256; cfg.n_layers=4; cfg.n_modes=4; cfg.vocab=1000
    cfg.cov_heads=4; cfg.cov_r=16; cfg.bind_r=16; cfg.kernel_size=16; cfg.dct_basis=True
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg, use_first_moment, use_rf).to(dev)
    p = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(50):
        x = torch.randint(0, cfg.vocab, (2, 64), device=dev)
        y = torch.randint(0, cfg.vocab, (2, 64), device=dev)
        h = model[0](x); out, _ = model[1](h); logits = model[2](out)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    return losses[0], losses[-1], losses[-1] < losses[0], p

print('=== Ablation: µ vs RF vs both vs none ===')
for name, fm, rf in [('none',     False, False),
                      ('µ only',   True,  False),
                      ('RF only',  False, True),
                      ('µ + RF',   True,  True)]:
    l0, l1, ok, p = test_config(name, fm, rf)
    print(f'  {name:8s}  loss {l0:.2f} -> {l1:.2f}  drop={l0-l1:.2f}  ok={ok}  params={p/1e6:.2f}M')
print('Done.')
