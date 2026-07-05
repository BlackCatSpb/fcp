"""
Cognitive Mirror: модель смотрит на свои multi-head выходы,
обнаруживает disagreement между головами, и через bind-механизм
корректирует скрытое состояние — "самодиалог на уровне h → gate → h".

Без softmax, без sigmoid gates (кроме decay), чистый bind.
"""
import os, sys, math, time, numpy as np
from math import sqrt
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ld_model.core import (LDConfig, MemBindBlock, MemBindStack,
                           parallel_prefix_scan, parallel_prefix_scan_1d,
                           compute_spectrum, compute_timescales)


# ─── Mirror variants: как disagreement → correction ──────────────────
# Все варианты получают head_out: (B, L, H, D) и возвращают (B, L, D)

class MirrorLinear(nn.Module):
    """d @ W_linear — линейная проекция disagreement."""
    def __init__(self, D):
        super().__init__()
        self.W = nn.Parameter(torch.randn(D, D) * 0.01)
    def forward(self, head_out, h_norm=None):
        d = head_out.std(dim=2)
        return d @ self.W


class MirrorSelfBind(nn.Module):
    """(d@W_u) * (d@W_v) @ W_out — bind disagreement с собой."""
    def __init__(self, D, r_m=16):
        super().__init__()
        self.W_u = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_out = nn.Parameter(torch.randn(r_m, D) * 0.01)
    def forward(self, head_out, h_norm=None):
        d = head_out.std(dim=2)
        return ((d @ self.W_u) * (d @ self.W_v)) @ self.W_out


class MirrorHVBind(nn.Module):
    """(d@W_u) * (h_norm@W_v) @ W_out — disagreement × вход."""
    def __init__(self, D, r_m=16):
        super().__init__()
        self.W_u = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_out = nn.Parameter(torch.randn(r_m, D) * 0.01)
    def forward(self, head_out, h_norm):
        d = head_out.std(dim=2)
        return ((d @ self.W_u) * (h_norm @ self.W_v)) @ self.W_out


class MirrorStdBind(nn.Module):
    """(d@W_u) * (consensus@W_v) @ W_out — disagreement × консенсус."""
    def __init__(self, D, r_m=16):
        super().__init__()
        self.W_u = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_v = nn.Parameter(torch.randn(D, r_m) * 0.01)
        self.W_out = nn.Parameter(torch.randn(r_m, D) * 0.01)
    def forward(self, head_out, h_norm=None):
        d = head_out.std(dim=2)
        h = head_out.mean(dim=2)
        return ((d @ self.W_u) * (h @ self.W_v)) @ self.W_out


MIRROR_VARIANTS = {
    'none':      None,
    'linear':    MirrorLinear,
    'self_bind': MirrorSelfBind,
    'hv_bind':   MirrorHVBind,
    'std_bind':  MirrorStdBind,
}


# ─── Mirrored MemBindBlock ──────────────────────────────────────────
class MirroredMemBindBlock(MemBindBlock):
    """MemBindBlock + cognitive mirror."""
    def __init__(self, cfg, layer_idx, lambda_roots, block_sizes=None,
                 mirror_variant='none'):
        super().__init__(cfg, layer_idx, lambda_roots, block_sizes)
        if mirror_variant is not None and mirror_variant != 'none':
            if mirror_variant == 'linear':
                self.mirror = MirrorLinear(cfg.D)
            else:
                self.mirror = MIRROR_VARIANTS[mirror_variant](cfg.D, r_m=cfg.bind_r)
            self.mirror_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.mirror = None
        self.mirror_variant = mirror_variant

    def forward(self, h, state=None):
        if self.mirror is None:
            return super().forward(h, state)

        # Full forward from MemBindBlock, but capture head outputs
        B, L, D = h.shape
        H, r = self.H, self.r

        K = self.conv.kernel_size
        cov_state = None; mu_state = None; conv_state = None
        if state is not None:
            if self.cov_first_moment:
                cov_state, mu_state, conv_state = state
            else:
                cov_state, conv_state = state
        if conv_state is None:
            conv_state = torch.zeros(B, self.D, K - 1, device=h.device, dtype=h.dtype)

        h_conv, conv_state_out = self.conv(h, conv_state)
        h_norm = h + h_conv
        h_norm = h_norm * (self.ln_w / (h_norm.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt())

        u = h_norm @ self.W_u
        v = h_norm @ self.W_v

        if self.cov_rf:
            h_rf = h_norm @ self.R_frozen
            K_cov = torch.einsum('blp,hpr->bhlr', h_rf, self.W_k_rf)
            Q = torch.einsum('blp,hpr->bhlr', h_rf, self.W_q_rf)
        else:
            K_cov = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k)
            Q = torch.einsum('bld,hdr->bhlr', h_norm, self.W_q)

        i_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_i) + self.b_i.view(1, H, 1, 1)
        i_gate = torch.exp(i_raw)
        decay_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_decay) + self.b_decay.view(1, H, 1, 1)
        decay = torch.sigmoid(decay_raw)

        K_e = K_cov.unsqueeze(-1)
        delta = (K_e @ K_e.transpose(-2, -1)) * i_gate.unsqueeze(-1)
        a_scan = decay.squeeze(-1).permute(0, 2, 1)
        b_scan = delta.permute(0, 2, 1, 3, 4)
        M_all, final_cov_state = parallel_prefix_scan(a_scan, b_scan, cov_state)

        Q_perm = Q.permute(0, 2, 1, 3)
        mem_r = (Q_perm.unsqueeze(-2) @ M_all).squeeze(-2)

        # ── Здесь перехватываем head outputs для mirror ──
        head_out = torch.einsum('blhr,hro->blho', mem_r, self.W_read)  # (B, L, H, D)
        mem_sum = head_out.sum(dim=2)  # (B, L, D)

        # Cognitive mirror
        mirror_delta = self.mirror(head_out, h_norm) * self.mirror_scale
        mem_sum = mem_sum + mirror_delta

        # ── First moment µ ──
        if self.cov_first_moment:
            K_mu = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k_mu)
            b_mu = K_mu * i_gate
            mu_all, final_mu_state = parallel_prefix_scan_1d(
                a_scan, b_mu.permute(0, 2, 1, 3), mu_state)
            mu_read = torch.einsum('blhr,hrf->blhf', mu_all, self.q_mu).squeeze(-1)
            mem_sum = mem_sum + mu_read @ self.W_mu_mem
        else:
            final_mu_state = None

        # ── Bind enhance + spectral ──
        v_enh = v + (mem_sum @ self.W_mem2v)
        h_adapt = h_norm + (u * v_enh) @ self.W_out

        h_proj = h_adapt @ self.V_T
        if all(s == self.block_size for s in self.block_sizes):
            h_proj_r = h_proj.view(B, L, self.K, self.block_size)
            h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, self.D)
        else:
            h_blocks = list(torch.split(h_proj, self.block_sizes, dim=-1))
            h_scaled = torch.cat([b * lam for b, lam in zip(h_blocks, self.lambda_k)], dim=-1)
        delta_spec = h_scaled @ self.V

        if self.cov_first_moment:
            return h + delta_spec, (final_cov_state, final_mu_state, conv_state_out)
        return h + delta_spec, (final_cov_state, conv_state_out)


# ─── Synthetic data (patterns on 3 timescales) ──────────────────────
def synthetic_batch(B, L, V, rng):
    tok = rng.randint(0, V, (B, L + 24))
    for b in range(B):
        for i in range(0, L-2, 3):
            if rng.random() < 0.7:
                t = tok[b, i]; tok[b, i+1] = (t+1)%V; tok[b, i+2] = t
        for i in range(0, L-8, 8):
            if rng.random() < 0.5:
                base = tok[b, i]
                for j in range(1, 8): tok[b, i+j] = (base+j)%V
        for i in range(0, L-24, 24):
            if rng.random() < 0.3:
                base = tok[b, i]
                for j in range(1, 24, 2): tok[b, i+j] = base
    return torch.from_numpy(tok[:, :L]), torch.from_numpy(tok[:, 1:L+1])


# ─── Test runner ────────────────────────────────────────────────────
def test_mirror(mirror_variant, cfg_template, steps=100):
    cfg = LDConfig()
    for k, v in cfg_template.__dict__.items():
        setattr(cfg, k, v)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    D, V = cfg.D, cfg.vocab

    lam, bs = compute_spectrum(cfg)
    layers = nn.ModuleList([
        MirroredMemBindBlock(cfg, i, lam, bs, mirror_variant=mirror_variant)
        for i in range(cfg.n_layers)
    ])
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers
            self.mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(D, cfg.bottleneck), nn.SiLU(),
                              nn.Linear(cfg.bottleneck, D))
                for _ in range(cfg.n_layers)
            ])
            self.fnw = nn.Parameter(torch.ones(D))
        def forward(self, h, state=None):
            if state is None: state = [None]*cfg.n_layers
            ns = []
            for i in range(cfg.n_layers):
                h, ls = self.layers[i](h, state[i])
                nh = h * (self.fnw / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt())
                h = h + self.mlps[i](nh); ns.append(ls)
            return h * (self.fnw / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt()), ns

    emb = nn.Embedding(V, D)
    nn.init.normal_(emb.weight, 0, 1.0 / math.sqrt(D))
    model = nn.Sequential(emb, Stack(),
                          nn.Linear(D, V, bias=True)).to(dev)
    with torch.no_grad():
        model[2].weight = model[0].weight
        model[2].bias.zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.RandomState(42)
    losses = []
    for step in range(steps):
        x, y = synthetic_batch(4, 128, V, rng)
        x, y = x.to(dev).long(), y.to(dev).long()
        h = model[0](x); out, _ = model[1](h); logits = model[2](out)
        loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); losses.append(loss.item())
    return losses[0], losses[-1], losses


# ─── Run all variants ───────────────────────────────────────────────
cfg = LDConfig(); cfg.D=256; cfg.n_layers=3; cfg.n_modes=4; cfg.vocab=200
cfg.cov_heads=4; cfg.cov_r=16; cfg.bind_r=16; cfg.kernel_size=16
cfg.cov_rf=True; cfg.cov_rf_dim=32; cfg.cov_first_moment=True
cfg.cov_multi_timescale=True; cfg.cov_tau_lo=3; cfg.cov_tau_hi=200
cfg.dct_basis=True

print('=== Cognitive Mirror: all variants ===')
print(f'D={cfg.D} L=128 H={cfg.cov_heads} tau=[3,12,49,200] layers={cfg.n_layers}\n')

variants = ['none', 'linear', 'self_bind', 'hv_bind', 'std_bind']
results = []
for var in variants:
    t0 = time.time()
    l0, l1, hist = test_mirror(var, cfg, steps=100)
    elapsed = time.time() - t0
    drop = l0 - l1
    results.append((var, l1, drop, hist))
    print(f'  {var:12s}  {l0:.2f} -> {l1:.2f}  drop={drop:.2f}  ({elapsed:.0f}s)')

best = min(results, key=lambda r: r[1])
print(f'\n-> Best: {best[0]} ({best[1]:.4f})')
improvement = max(r[2] for r in results if r[0] != 'none')
base_drop = [r[2] for r in results if r[0] == 'none'][0]
print(f'  none drop={base_drop:.2f}, best drop={best[2]:.2f} (+{best[2]-base_drop:.2f})')
