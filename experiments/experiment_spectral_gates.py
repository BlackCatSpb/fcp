"""
Experiment: replace learned gates (W_gate, b_gate) with spectral energy gates.
α_k = ||V_k^T · h_norm||² / Σⱼ ||V_j^T · h_norm||²

0 learnable params per gate. Pure geometry of h in V-basis.
Compares PPL vs original learned gates on the same checkpoint.
"""

import os, torch, sys, math, time
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack, rms_norm, random_orthogonal, fibonacci_roots, CausalConv1d

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

D, VOCAB, N_LAYERS, N_MODES = 896, 50000, 12, 4
block_size = D // N_MODES  # 224

# ─── Original LDBlock (learned gates) ────────────────────────────────────

class LDBlockOriginal(torch.nn.Module):
    def __init__(self, cfg, layer_idx, lambda_roots):
        super().__init__()
        self.D = cfg.D; self.K = cfg.n_modes; self.block_size = cfg.D // cfg.n_modes
        self.conv = CausalConv1d(cfg.D, kernel_size=4)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.W_gate = torch.nn.Parameter(torch.randn(cfg.D, cfg.n_modes) * 0.01)
        self.b_gate = torch.nn.Parameter(torch.zeros(cfg.n_modes))
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])
        self.register_buffer('input_ln_w', torch.ones(cfg.D))

    def forward(self, h, return_gates=False, residual=True):
        B, L, D = h.shape
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.input_ln_w)
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * 4.0
        alpha = torch.softmax(gate_logits, dim=-1)
        lambda_alpha = self.lambda_k * alpha
        h_proj = h_norm @ self.V_T
        h_proj_r = h_proj.view(B, L, self.K, self.block_size)
        h_scaled = (h_proj_r * lambda_alpha.unsqueeze(-1)).reshape(B, L, self.D)
        delta = h_scaled @ self.V_T.T
        h_out = h + delta if residual else delta
        if return_gates:
            return h_out, alpha
        return h_out

# ─── Spectral-gate LDBlock (0 learnable gate params) ────────────────────

class LDBlockSpectral(torch.nn.Module):
    """Spectral gates using L∞ norm per V-block (max resonance).
    α_k = softmax(τ · max_i |(V_k^T · h_norm)_i|)"""
    def __init__(self, cfg, layer_idx, lambda_roots, tau=15.0):
        super().__init__()
        self.D = cfg.D; self.K = cfg.n_modes; self.block_size = cfg.D // cfg.n_modes
        self.tau = tau
        self.conv = CausalConv1d(cfg.D, kernel_size=4)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])
        self.register_buffer('input_ln_w', torch.ones(cfg.D))

    def forward(self, h, return_gates=False, residual=True):
        B, L, D = h.shape
        h_conv = self.conv(h)
        h_norm = rms_norm(h + h_conv, self.input_ln_w)

        # L∞ gates: α_k = softmax(τ · max_i |proj_{V_k}(h)_i|)
        h_V = h_norm @ self.V_T  # (B, L, D)
        h_V_blocks = h_V.reshape(B, L, self.K, self.block_size)  # (B, L, K, M)
        max_abs = h_V_blocks.abs().max(dim=-1).values  # (B, L, K) — L∞ per block
        alpha = torch.softmax(self.tau * max_abs, dim=-1)

        lambda_alpha = self.lambda_k * alpha
        h_scaled = (h_V_blocks * lambda_alpha.unsqueeze(-1)).reshape(B, L, self.D)
        delta = h_scaled @ self.V_T.T

        if residual:
            h_out = h + delta
        else:
            h_out = delta

        if return_gates:
            return h_out, alpha
        return h_out

# ─── Phase2Model wrapper ─────────────────────────────────────────────────

class Phase2Model(torch.nn.Module):
    def __init__(self, block_cls):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        lambda_roots = fibonacci_roots(N_MODES + 1)
        self.stack = LDStack(cfg)
        # Replace each LDBlock with the experimental one
        self.stack.layers = torch.nn.ModuleList([
            block_cls(cfg, i, lambda_roots) for i in range(N_LAYERS)
        ])
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)

    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

# ─── Load checkpoint weights ─────────────────────────────────────────────

def load_weights(model, ckpt_path, is_spectral=False):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = {k: v.float() for k, v in ckpt['model_fp16'].items()}
    # Filter out gate params for spectral model
    if is_spectral:
        sd = {k: v for k, v in sd.items()
              if not any(g in k for g in ['W_gate', 'b_gate'])}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if is_spectral:
        expected_missing = [k for k in missing if 'W_gate' in k or 'b_gate' in k]
        other_missing = [k for k in missing if 'W_gate' not in k and 'b_gate' not in k]
        if other_missing:
            print('  WARN: unexpected missing keys:', other_missing)
        print('  Expected missing (gate params): %d' % len(expected_missing))
    else:
        if missing:
            print('  WARN: missing keys:', missing)
    return ckpt['step'], ckpt['epoch']

# ─── Eval PPL ────────────────────────────────────────────────────────────

def eval_ppl(model, data_path, n_eval=500, batch_size=8):
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    arr = np.load(data_path, mmap_mode='r')
    n_total = arr.shape[0]
    n_eval = min(n_eval, n_total // 20)
    start = n_total - n_eval

    eval_x = torch.from_numpy(arr[start:, :-1].copy()).to(torch.long)
    eval_y = torch.from_numpy(arr[start:, 1:].copy()).to(torch.long)
    loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=batch_size)

    model.eval()
    loss_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            loss_total += loss.item()
            n_batches += 1
    ppl = math.exp(loss_total / n_batches)
    return ppl

def analyze_alphas(model, data_path, n_samples=8):
    """Compute mean and std of alpha across layers."""
    import numpy as np
    arr = np.load(data_path, mmap_mode='r')
    bx = torch.from_numpy(arr[:n_samples, :-1].copy()).to(torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        h = model.embed(bx)
        h, alphas = model.stack(h, return_gates=True)
    # alphas: (n_layers, B, L, K)
    alphas = alphas.float()
    for k in range(alphas.shape[-1]):
        vals = alphas[:, :, :, k].cpu().numpy().ravel()
        print('  Mode %d: mean=%.4f std=%.4f' % (k, vals.mean(), vals.std()))
    print('  Entropy: %.4f' % (-(alphas * torch.log(alphas + 1e-10)).sum(dim=-1).mean().item()))


# ─── Main ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ckpt_path = 'checkpoints/model_step20000.pt'
    data_path = 'russian_chunks.npy'

    if not os.path.exists(ckpt_path):
        ckpt_path = 'checkpoints/model_step5000.pt'
    if not os.path.exists(ckpt_path):
        print('No checkpoint found!')
        sys.exit(1)

    print(f'\nCheckpoint: {ckpt_path}')
    print(f'Data: {data_path}')

    # Load checkpoint weights once
    ckpt_full = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    ckpt_sd = {k: v.float() for k, v in ckpt_full['model_fp16'].items()}

    # 1. Original (learned gates)
    print()
    print('--- Original (learned gates) ---')
    model_orig = Phase2Model(LDBlockOriginal).to(DEVICE)
    sd_orig = {k: v for k, v in ckpt_sd.items()}
    missing, unexpected = model_orig.load_state_dict(sd_orig, strict=False)
    if missing:
        print('  WARN: missing:', missing)
    t0 = time.time()
    ppl_orig = eval_ppl(model_orig, data_path)
    t_orig = time.time() - t0
    print('  PPL: %.1f (%.0fs)' % (ppl_orig, t_orig))
    analyze_alphas(model_orig, data_path)

    # 2. Spectral gates — L∞ with various τ
    sd_spec = {k: v for k, v in ckpt_sd.items()
               if not any(g in k for g in ['W_gate', 'b_gate', 'V_T'])}
    for tau in [5, 10, 15, 20, 30]:
        print()
        print('--- Spectral Linf gates (tau=%d) ---' % tau)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES; cfg.vocab = VOCAB; cfg.bottleneck = 512
        lambda_roots = fibonacci_roots(N_MODES + 1)
        class Phase2ModelFlex(torch.nn.Module):
            def __init__(self, block_cls, **kwargs):
                super().__init__()
                self.embed = torch.nn.Embedding(VOCAB, D)
                c = LDConfig()
                c.D = D; c.n_layers = N_LAYERS; c.n_modes = N_MODES; c.vocab = VOCAB; c.bottleneck = 512
                lr = fibonacci_roots(N_MODES + 1)
                self.stack = LDStack(c)
                self.stack.layers = torch.nn.ModuleList([
                    block_cls(c, i, lr, **kwargs) for i in range(N_LAYERS)
                ])
                self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
            def forward(self, x):
                return self.lm_head(self.stack(self.embed(x)))
        model_spec = Phase2ModelFlex(LDBlockSpectral, tau=tau).to(DEVICE)
        model_spec.load_state_dict(sd_spec, strict=False)
        t0 = time.time()
        ppl_spec = eval_ppl(model_spec, data_path)
        t_spec = time.time() - t0
        print('  PPL: %.1f (%.0fs)' % (ppl_spec, t_spec))
        analyze_alphas(model_spec, data_path)

    # 4. Compare
    print()
    print('--- Comparison ---')
    delta = ppl_spec - ppl_orig
    pct = delta / ppl_orig * 100
    if delta > 0:
        print('Spectral is %.1f PPL worse (%.1f%%)' % (delta, pct))
    else:
        print('Spectral is %.1f PPL BETTER (%.1f%%)' % (-delta, -pct))

    print()
    print('Params in gates (12 layers):')
    print('  Learned: 12 x (%dx%d + %d) = %d' % (D, N_MODES, N_MODES, 12 * (D*N_MODES + N_MODES)))
    print('  Spectral: 0 (pure geometry)')
