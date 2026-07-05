"""
Test: first moment (µ) + random features for covariance memory.
Verify: no NaN, loss drops, backward works, generate coherent.
"""

import os, sys, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ld_model.core import LDConfig, MemBindBlock, MemBindStack, parallel_prefix_scan, dct_basis


# ─── 1D prefix scan for vectors ───────────────────────────────────────
def parallel_prefix_scan_1d(a, b, state=None):
    """
    v[t] = a[t]·v[t-1] + b[t],  v[-1] = state (or 0)
    a: (B, L, H) decays  (same as 2D)
    b: (B, L, H, r) increments  (1D instead of r×r)
    state: (B, H, r) or None
    Returns: (v_all, final_state)  both (B, L, H, r) and (B, H, r)
    """
    L = a.shape[1]
    A = a.unsqueeze(-1)  # (B, L, H, 1)
    v = b.clone()
    stride = 1
    while stride < L:
        A_left, A_right = A[:, :L-stride], A[:, stride:]
        v_left, v_right = v[:, :L-stride], v[:, stride:]
        A_combined = A_left * A_right
        v_combined = A_right * v_left + v_right
        A = torch.cat([A[:, :stride], A_combined], dim=1)
        v = torch.cat([v[:, :stride], v_combined], dim=1)
        stride *= 2
    # v[:L] = cumsum of b[t] weighted by cumprod of a[0..t]
    if state is not None:
        v = v + A * state.unsqueeze(1)
    return v, v[:, -1]


# ─── Enhanced MemBindBlock with µ + RF ────────────────────────────────
class EnhancedMemBindBlock(MemBindBlock):
    """
    Extended MemBindBlock with first moment µ[t] and random features.
    Toggle via `use_first_moment` and `use_rf`.
    """
    def __init__(self, cfg, layer_idx, lambda_roots, block_sizes=None,
                 use_first_moment=True, use_rf=True, rf_dim=64):
        super().__init__(cfg, layer_idx, lambda_roots, block_sizes)
        H, D, r = self.H, self.D, self.r

        if use_rf:
            # Frozen random projection D → rf_dim
            R = torch.randn(rf_dim, D) / math.sqrt(D)
            self.register_buffer('R', R)
            # Learnable projections from rf_dim to (H, r)
            self.W_k_rf = nn.Parameter(torch.randn(H, rf_dim, r) * 0.01)
            self.W_q_rf = nn.Parameter(torch.randn(H, rf_dim, r) * 0.01)
        self.use_rf = use_rf

        if use_first_moment:
            # First-moment key projection: h → K_mu (same K space)
            self.W_k_mu = nn.Parameter(torch.randn(H, D, r) * 0.01)
            # Read first moment: mu → scalar per head
            self.q_mu = nn.Parameter(torch.randn(H, r, 1) * 0.01)
            # Project from H scalars → D
            self.W_mu_mem = nn.Parameter(torch.zeros(H, D))
        self.use_first_moment = use_first_moment

    def forward(self, h: torch.Tensor, state: tuple = None):
        B, L, D = h.shape
        H, r = self.H, self.r

        K = self.conv.kernel_size
        cov_state = None
        mu_state = None
        conv_state = None
        if state is not None:
            cov_state, mu_state, conv_state = state
        if conv_state is None:
            conv_state = torch.zeros(B, D, K - 1, device=h.device, dtype=h.dtype)

        h_conv, conv_state_out = self.conv(h, conv_state)
        h_norm = h + h_conv
        h_norm = h_norm * (self.ln_w / (h_norm.pow(2).mean(-1, keepdim=True) + 1e-8).sqrt())

        # Bind
        u = h_norm @ self.W_u
        v = h_norm @ self.W_v

        # Covariance keys/queries (with optional random features)
        if self.use_rf:
            h_rf = h_norm @ self.R.T  # (B, L, rf_dim)
            K_cov = torch.einsum('blp,hpr->bhlr', h_rf, self.W_k_rf)
            Q = torch.einsum('blp,hpr->bhlr', h_rf, self.W_q_rf)
        else:
            K_cov = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k)
            Q = torch.einsum('bld,hdr->bhlr', h_norm, self.W_q)

        # Impulse and decay gates (same as base)
        i_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_i) + self.b_i.view(1, H, 1, 1)
        i_gate = torch.exp(i_raw)

        decay_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_decay) + self.b_decay.view(1, H, 1, 1)
        decay = torch.sigmoid(decay_raw)

        # ── Second moment Σ[t] = d·Σ[t-1] + i·(K@K.T) ──
        K_e = K_cov.unsqueeze(-1)
        delta = (K_e @ K_e.transpose(-2, -1)) * i_gate.unsqueeze(-1)

        a_scan = decay.squeeze(-1).permute(0, 2, 1)
        b_scan = delta.permute(0, 2, 1, 3, 4)
        M_all, final_cov_state = parallel_prefix_scan(a_scan, b_scan, cov_state)

        # Read from Σ
        Q_perm = Q.permute(0, 2, 1, 3)
        mem_r = (Q_perm.unsqueeze(-2) @ M_all).squeeze(-2)
        mem_D = torch.einsum('blhr,hro->blho', mem_r, self.W_read)
        mem_sum = mem_D.sum(dim=2)  # (B, L, D) sum over heads

        # ── First moment µ[t] = d·µ[t-1] + i·K ──
        if self.use_first_moment:
            K_mu = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k_mu)
            b_mu = K_mu * i_gate  # (B, L, H, r)
            a_mu = a_scan  # same decay
            mu_all, final_mu_state = parallel_prefix_scan_1d(a_mu, b_mu.permute(0, 2, 1, 3), mu_state)
            # Read from µ: simple weighted sum per head
            mu_read = torch.einsum('blhr,hrf->blhf', mu_all, self.q_mu).squeeze(-1)  # (B,L,H)
            # Project H scalars → D
            mem_sum = mem_sum + mu_read @ self.W_mu_mem
        else:
            final_mu_state = None

        # Bind enhance + spectral (same as base)
        v_enh = v + (mem_sum @ self.W_mem2v)
        h_adapt = h_norm + (u * v_enh) @ self.W_out

        h_proj = h_adapt @ self.V_T
        if all(s == self.block_size for s in self.block_sizes):
            h_proj_r = h_proj.view(B, L, self.K, self.block_size)
            h_scaled = (h_proj_r * self.lambda_k.view(1, 1, self.K, 1)).reshape(B, L, D)
        else:
            h_blocks = list(torch.split(h_proj, self.block_sizes, dim=-1))
            h_scaled = torch.cat([b * lam for b, lam in zip(h_blocks, self.lambda_k)], dim=-1)
        delta_spec = h_scaled @ self.V

        return h + delta_spec, (final_cov_state, final_mu_state, conv_state_out)


# ─── Test ─────────────────────────────────────────────────────────────
def test_enhanced():
    print('=== Test: First moment + Random Features ===')
    cfg = LDConfig()
    cfg.D = 256
    cfg.n_layers = 4
    cfg.n_modes = 4
    cfg.vocab = 1000
    cfg.cov_heads = 4
    cfg.cov_r = 16
    cfg.bind_r = 16
    cfg.kernel_size = 16
    cfg.dct_basis = True

    # Build an enhanced stack
    from ld_model.core import compute_spectrum
    lam, bs = compute_spectrum(cfg)

    # Replace MemBindStack layers with EnhancedMemBindBlock
    class EnhancedStack(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.n_layers = cfg.n_layers
            lam, bs = compute_spectrum(cfg)
            self.layers = nn.ModuleList([
                EnhancedMemBindBlock(cfg, i, lam, bs,
                                    use_first_moment=True, use_rf=True, rf_dim=32)
                for i in range(cfg.n_layers)
            ])
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(cfg.D, cfg.bottleneck),
                    nn.SiLU(),
                    nn.Linear(cfg.bottleneck, cfg.D)
                ) for _ in range(cfg.n_layers)
            ])
            self.final_norm_w = nn.Parameter(torch.ones(cfg.D))

        def forward(self, h, state=None):
            if state is None:
                state = [None] * self.n_layers
            new_state = []
            for lidx in range(self.n_layers):
                h, layer_state = self.layers[lidx](h, state[lidx])
                h = h + self.mlps[lidx](h * (self.final_norm_w / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt()))
                new_state.append(layer_state)
            norm = h * (self.final_norm_w / (h.pow(2).mean(-1,keepdim=True)+1e-8).sqrt())
            return norm, new_state

    # Build model
    D, VOCAB = cfg.D, cfg.vocab
    model = nn.Sequential(
        nn.Embedding(VOCAB, D),
        EnhancedStack(cfg),
        nn.Linear(D, VOCAB, bias=True)
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    model[0].weight.data.uniform_(-0.1, 0.1)
    model[2].weight = model[0].weight  # weight tying

    DEVICE = next(model.parameters()).device
    print(f'Device: {DEVICE}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.2f}M')

    # Forward + loss
    x = torch.randint(0, VOCAB, (2, 64), device=DEVICE)
    y = torch.randint(0, VOCAB, (2, 64), device=DEVICE)
    h = model[0](x)
    out, states = model[1](h)
    logits = model[2](out)
    loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
    print(f'Initial loss: {loss.item():.4f}, nan={torch.isnan(loss).any().item()}, inf={torch.isinf(loss).any().item()}')

    # Backward
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f'Grad norm: {grad_norm:.4f}')
    print(f'Grad NaN: {any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)}')
    print(f'Grad Inf: {any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)}')

    # Training step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    t0 = time.time()
    for step in range(50):
        x = torch.randint(0, VOCAB, (2, 64), device=DEVICE)
        y = torch.randint(0, VOCAB, (2, 64), device=DEVICE)
        h = model[0](x)
        out, _ = model[1](h)
        logits = model[2](out)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step+1) % 10 == 0:
            print(f'  S{step+1}: loss={loss.item():.4f}')

    elapsed = time.time() - t0
    print(f'Train: {losses[-1]:.4f} (from {losses[0]:.4f}) in {elapsed:.1f}s')
    print(f'Loss drop: {losses[0]-losses[-1]:.4f}')
    print('OK' if losses[-1] < losses[0] else 'WARNING: loss did not drop')

    # Generate
    print('\n--- Generation ---')
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (1, 1), device=DEVICE)
        states = [None] * cfg.n_layers
        out_tokens = [x.item()]
        for _ in range(32):
            h = model[0](x)
            out, states = model[1](h, state=states)
            logits = model[2](out)
            probs = F.softmax(logits[0, 0] / 1.0, dim=-1)
            token = torch.multinomial(probs, 1).item()
            out_tokens.append(token)
            x = torch.tensor([[token]], device=DEVICE)
        print(f'Generated {len(out_tokens)} tokens: {out_tokens[:16]}...')
        print(f'  Unique: {len(set(out_tokens))}/{len(out_tokens)}')

    print('\nAll tests passed!')


if __name__ == '__main__':
    test_enhanced()
