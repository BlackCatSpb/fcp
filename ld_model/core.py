"""MemBind: Multi-head covariance memory + bilinear bind + spectral operator.
No softmax, no sigmoid gates, no attention.

Архитектуры:
  - MemBindBlock: multi-head covariance memory + bind feedback (основная)
  - LDBlock: sigmoid-gated spectral transform (legacy)

Полное описание: LAMBDA_ARCHITECTURE.md
"""

import math
import torch
import torch.nn.functional as F


class LDConfig:
    D: int = 2560
    n_layers: int = 36
    n_modes: int = 6
    vocab: int = 146260
    bottleneck: int = 512       # MLP bottleneck dim
    kernel_size: int = 48       # causal conv kernel (RF = 1 + (k-1)*n_layers)
    adaptive_gain: bool = True  # modulate update by gate spread
    learnable_V: bool = True    # eigenbasis rotation via Cayley
    V_rank: int = 16            # Cayley generator rank
    use_global_context: bool = False
    recurrent_scan: bool = False
    weight_tying: bool = True
    lm_head_bias: bool = True
    trainable_conv: bool = False
    # MemBind-specific
    arch: str = 'ld'       # 'ld' or 'membind'
    cov_heads: int = 4     # multi-head covariance heads (MemBind)
    cov_r: int = 16        # covariance rank per head (MemBind)
    bind_r: int = 16       # bind projection rank (MemBind)
    # Spectrum options
    spectrum_type: str = 'fib_root'  # 'fib_root', 'fib_seq', 'fib_ratio', 'linear', 'hybrid'
    spec_lo: float = 0.8   # lower bound for λ range
    spec_hi: float = 1.8   # upper bound for λ range


# ─── Fibonacci roots ────────────────────────────────────────────────────

def fibonacci_roots(max_k: int = 7) -> torch.Tensor:
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


# ─── Orthogonal V ───────────────────────────────────────────────────────

def fibonacci_sequence(n: int) -> list:
    """First n Fibonacci numbers: F_1=1, F_2=1, ..."""
    seq = [1, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

def compute_spectrum(cfg: LDConfig) -> tuple:
    """Return (lambda_k_tensor, block_sizes_list) based on cfg.spectrum_type."""
    K = cfg.n_modes
    D = cfg.D
    lo, hi = cfg.spec_lo, cfg.spec_hi

    if cfg.spectrum_type == 'fib_root':
        lambda_roots = fibonacci_roots(K + 1)
        block_sizes = [D // K] * K
        return lambda_roots, block_sizes

    elif cfg.spectrum_type == 'fib_seq':
        fibs = fibonacci_sequence(K + 2)[1:-1]  # F_2..F_{K+1}
        lambdas = torch.tensor(fibs, dtype=torch.float32)
        if len(fibs) == 1:
            lambdas = torch.tensor([(lo + hi) / 2])
        else:
            mn, mx = min(fibs), max(fibs)
            lambdas = lo + (lambdas - mn) / (mx - mn) * (hi - lo) * 0.9999
        total_fib = sum(fibs)
        block_sizes = [max(1, round(f * D / total_fib)) for f in fibs]
        diff = D - sum(block_sizes)
        block_sizes[-1] += diff
        return lambdas, block_sizes

    elif cfg.spectrum_type == 'fib_ratio':
        fibs = fibonacci_sequence(K + 3)
        ratios = [fibs[i+1] / fibs[i] for i in range(1, K + 1)]
        lambdas = torch.tensor(sorted(ratios), dtype=torch.float32)
        mn, mx = min(ratios), max(ratios)
        lambdas = lo + (lambdas - mn) / (mx - mn) * (hi - lo) * 0.9999 if mx != mn else torch.full_like(lambdas, (lo + hi) / 2)
        block_sizes = [D // K] * K
        return lambdas, block_sizes

    elif cfg.spectrum_type == 'linear':
        lambdas = torch.linspace(lo, hi * 0.9999, K)
        block_sizes = [D // K] * K
        return lambdas, block_sizes

    elif cfg.spectrum_type == 'hybrid':
        # fib_ratio + linear averaged
        fibs = fibonacci_sequence(K + 3)
        ratios = sorted([fibs[i+1] / fibs[i] for i in range(1, K + 1)])
        lambdas_r = torch.tensor(ratios, dtype=torch.float32)
        mn_r, mx_r = min(ratios), max(ratios)
        lambdas_r = lo + (lambdas_r - mn_r) / (mx_r - mn_r) * (hi - lo) * 0.9999 if mx_r != mn_r else torch.full_like(lambdas_r, (lo + hi) / 2)
        lambdas_l = torch.linspace(lo, hi * 0.9999, K)
        lambdas = (lambdas_r + lambdas_l) / 2
        block_sizes = [D // K] * K
        return lambdas, block_sizes

    else:
        raise ValueError(f"Unknown spectrum_type={cfg.spectrum_type}")


def random_orthogonal(D: int, n_reflections: int | None = None) -> torch.Tensor:
    if n_reflections is None:
        n_reflections = min(32, D)
    V = torch.eye(D, dtype=torch.float32)
    for _ in range(n_reflections):
        u = torch.randn(D, dtype=torch.float32)
        u = u / (u.norm() + 1e-10)
        V = V - 2 * torch.outer(V @ u, u)
    return V


# ─── RMS Norm ───────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    rms = rms.clamp(min=eps)
    return x / rms * weight


# ─── Causal 1D Convolution ──────────────────────────────────────────────

class CausalConv1d(torch.nn.Module):
    """Depthwise causal 1D conv: provides local n-gram mixing per channel.
    Kernel size k, padding = k-1 (left-only), groups = D (depthwise).
    """
    def __init__(self, D: int, kernel_size: int = 16, trainable: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        w = torch.randn(D, 1, kernel_size) * 0.1
        if trainable:
            self.weight = torch.nn.Parameter(w)
            self.bias = torch.nn.Parameter(torch.zeros(D))
        else:
            self.register_buffer('weight', w)
            self.register_buffer('bias', torch.zeros(D))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple:
        x_perm = x.transpose(1, 2)
        k = self.kernel_size
        if state is not None:
            x_perm = torch.cat([state, x_perm], dim=-1)
        x_pad = F.pad(x_perm, (k - 1, 0))
        out = F.conv1d(x_pad, self.weight, bias=self.bias,
                       groups=self.weight.shape[0])
        # out: (B, D, L_out). Without state: L_out=L.  With state: L_out=L+k-1
        if state is not None:
            L = x.shape[1]
            out = out[:, :, k-1:k-1+L]
        out = out.transpose(1, 2)
        if state is not None:
            new_state = x_perm[:, :, -(k - 1):]
            return out, new_state
        return out, None


# ─── LDBlock: conv → rms_norm → V·Λ·Vᵀ ─────────────────────────────────

class LDBlock(torch.nn.Module):
    """λ_d layer: causal conv → norm → content-dependent spectral transform.

    Forward:
        h_conv = causal_conv1d(h)
        h_norm = rms_norm(h + h_conv)
        α = sigmoid(scale · W_gate · h_norm)        # independent per mode, no sum-to-1
        Λ̂ = diag(α⊙λ spread across D/K blocks)       # view+broadcast, no repeat_interleave
        Δ = V_eff · Λ̂ · V_effᵀ · h_norm
        h_out = h + Δ

    When learnable_V is enabled, V_eff = V_frozen @ R (Cayley rotation, R ∈ O(D)).
    """
    def __init__(self, cfg: LDConfig, layer_idx: int, lambda_roots: torch.Tensor):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.block_size = cfg.D // cfg.n_modes
        self.r = cfg.V_rank
        self.recurrent_scan = cfg.recurrent_scan

        # Causal conv (cross-token mixing)
        self.conv = CausalConv1d(cfg.D, kernel_size=cfg.kernel_size,
                                 trainable=cfg.trainable_conv)

        # Eigenbasis (frozen base)
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())

        # Learnable V via explicit Cayley rotation on skew-symmetric generator
        # V_eff = V_frozen @ R, R = (I-S)^{-1}(I+S) ∈ O(D), S = A·B^T - B·A^T
        # R computed explicitly once per forward (solve D×D, O(D³)).
        self.learnable_V = cfg.learnable_V
        if self.learnable_V:
            init_scale = 0.001
            self.V_cay_A = torch.nn.Parameter(torch.randn(cfg.D, cfg.V_rank) * init_scale)
            self.V_cay_B = torch.nn.Parameter(torch.randn(cfg.D, cfg.V_rank) * init_scale)
        else:
            self.V_cay_A = None
            self.V_cay_B = None

        # Gate
        self.W_gate = torch.nn.Parameter(torch.randn(cfg.D, cfg.n_modes) * 0.01)
        self.b_gate = torch.nn.Parameter(torch.full((cfg.n_modes,), -1.1))  # sigmoid(-1.1)≈0.25
        self.gate_scale = torch.nn.Parameter(torch.tensor(1.0))  # sigmoid steepness

        # λ roots (frozen)
        self.register_buffer('lambda_k', lambda_roots[:cfg.n_modes])

        # RMS norm weight
        self.register_buffer('input_ln_w', torch.ones(cfg.D))

    def compute_R(self) -> torch.Tensor:
        """Explicit Cayley rotation matrix R = (I-S)^{-1}(I+S) ∈ O(D).
        
        S = A·B^T - B·A^T (skew-symmetric, rank 2r). O(D³) — один solve на forward.
        """
        A, B = self.V_cay_A, self.V_cay_B
        S = A @ B.T - B @ A.T
        I = torch.eye(self.D, device=S.device, dtype=S.dtype)
        return torch.linalg.solve(I - S, I + S)

    def forward(self, h: torch.Tensor, return_gates: bool = False,
                residual: bool = True) -> torch.Tensor:
        B, L, D = h.shape

        # 1. Causal conv → local mixing
        h_conv = self.conv(h)

        # 2. Pre-norm
        h_norm = rms_norm(h + h_conv, self.input_ln_w)

        # 3. Gate: sigmoid (independent per mode, no sum-to-1 constraint)
        gate_logits = (h_norm @ self.W_gate) + self.b_gate
        gate_logits = gate_logits * self.gate_scale
        alpha = torch.sigmoid(gate_logits)

        # 4. Spectral transform: V_eff @ diag(α·λ) @ V_eff^T
        lambda_alpha = self.lambda_k * alpha  # (B, L, K)
        if self.learnable_V:
            R = self.compute_R()
            V_eff_T = R.T @ self.V_T   # (D,D) — V_eff^T = R^T @ V^T
            V_eff = self.V @ R          # (D,D) — V_eff = V @ R
        else:
            V_eff_T = self.V_T
            V_eff = self.V

        if self.recurrent_scan:
            # λ_d token recurrence: h_t = V·diag(α_t·λ)·V⁻¹·h_{t-1} + x_t
            h_state = torch.zeros(B, D, device=h.device, dtype=h.dtype)
            h_scan = torch.empty_like(h_norm)
            for t in range(L):
                x_t = h_norm[:, t, :]
                alpha_t = lambda_alpha[:, t, :]  # (B, K)
                h_proj_t = h_state @ V_eff_T
                # Broadcast λ·α across block_size dims
                h_proj_r = h_proj_t.view(B, self.K, self.block_size)
                h_scaled_t = (h_proj_r * alpha_t.unsqueeze(-1)).reshape(B, self.D)
                h_out_t = h_scaled_t @ V_eff
                h_state = h_out_t + x_t
                h_scan[:, t, :] = h_state
            h_out = h + h_scan if residual else h_scan
        else:
            # Parallel per-token spectral transform: Δ = V_eff · Λ̂ · V_eff^T · h_norm
            h_proj = h_norm @ V_eff_T  # (B, L, D)
            # Avoid repeat_interleave: reshape → broadcast → merge
            h_proj_r = h_proj.view(B, L, self.K, self.block_size)
            h_scaled = (h_proj_r * lambda_alpha.unsqueeze(-1)).reshape(B, L, self.D)
            delta = h_scaled @ V_eff

            if residual:
                h_out = h + delta
            else:
                h_out = delta

        if return_gates:
            return h_out, alpha
        return h_out


# ─── Dense Bottleneck MLP ───────────────────────────────────────────────

class BottleneckMLP(torch.nn.Module):
    """Dense bottleneck MLP: D → bottleneck → D. Fully trainable."""
    def __init__(self, D: int, bottleneck: int = 512):
        super().__init__()
        self.up = torch.nn.Linear(D, bottleneck, bias=True)
        self.down = torch.nn.Linear(bottleneck, D, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


# ─── λ_d Stack ───────────────────────────────────────────────────────────

# ─── Parallel prefix scan (autograd-safe, Hillis-Steele) ────────────────

def parallel_prefix_scan(a, b, state=None):
    """
    M[t] = a[t]·M[t-1] + b[t],  M[-1] = state (or 0)
    a: (B, L, H) decays
    b: (B, L, H, r, r) increments
    state: (B, H, r, r) or None — initial state M[-1]
    Returns: (M_all, final_state)
      M_all: (B, L, H, r, r) — M[t] for each token
      final_state: (B, H, r, r) — M[L-1], for next inference step
    """
    L = a.shape[1]
    A = a.unsqueeze(-1).unsqueeze(-1)
    M = b.clone()
    stride = 1
    while stride < L:
        A_left, A_right = A[:, :L-stride], A[:, stride:]
        M_left, M_right = M[:, :L-stride], M[:, stride:]
        A_combined = A_left * A_right
        M_combined = A_right * M_left + M_right
        A = torch.cat([A[:, :stride], A_combined], dim=1)
        M = torch.cat([M[:, :stride], M_combined], dim=1)
        stride *= 2
    # After scan: M[t] = recurrence from M[-1]=0, A[t] = cumprod(a[0..t])
    if state is not None:
        M = M + A * state.unsqueeze(1)
    return M, M[:, -1]


# ─── MemBindBlock: multi-head covariance memory + bind feedback ─────────

class MemBindBlock(torch.nn.Module):
    """
    Один слой MemBind: conv → norm → bind → covariance scan → bind enhance → spectral.

    State seq: (cov_state, conv_state) для инкрементального инференса.
    Полное описание: LAMBDA_ARCHITECTURE.md
    """
    def __init__(self, cfg: LDConfig, layer_idx: int, lambda_roots: torch.Tensor,
                 block_sizes: list | None = None):
        super().__init__()
        self.D = cfg.D
        self.K = cfg.n_modes
        self.block_size = cfg.D // cfg.n_modes
        self.block_sizes = block_sizes if block_sizes else [self.block_size] * self.K
        self.H = cfg.cov_heads
        self.r = cfg.cov_r
        bind_r = cfg.bind_r

        self.conv = CausalConv1d(cfg.D, kernel_size=cfg.kernel_size,
                                 trainable=cfg.trainable_conv)
        self.register_buffer('ln_w', torch.ones(cfg.D))
        V_init = random_orthogonal(cfg.D, n_reflections=32)
        self.register_buffer('V', V_init)
        self.register_buffer('V_T', V_init.T.contiguous())
        self.register_buffer('lambda_k', lambda_roots)
        self.register_buffer('block_sizes_t', torch.tensor(self.block_sizes, dtype=torch.long))

        # Bind adaptation (FCF-inspired u*v interaction, no gates)
        self.W_u = torch.nn.Parameter(torch.randn(cfg.D, bind_r) * 0.01)
        self.W_v = torch.nn.Parameter(torch.randn(cfg.D, bind_r) * 0.01)
        self.W_out = torch.nn.Parameter(torch.zeros(bind_r, cfg.D))

        # Multi-head covariance memory
        H = self.H
        self.W_k = torch.nn.Parameter(torch.randn(H, cfg.D, self.r) * 0.01)
        self.W_q = torch.nn.Parameter(torch.randn(H, cfg.D, self.r) * 0.01)
        self.W_i = torch.nn.Parameter(torch.randn(H, cfg.D, 1) * 0.01)
        self.b_i = torch.nn.Parameter(torch.zeros(H, 1))
        self.W_decay = torch.nn.Parameter(torch.randn(H, cfg.D, 1) * 0.01)
        self.b_decay = torch.nn.Parameter(torch.full((H, 1), 2.0))
        self.W_read = torch.nn.Parameter(torch.randn(H, self.r, cfg.D) * 0.01)
        self.W_mem2v = torch.nn.Parameter(torch.randn(cfg.D, bind_r) * 0.01)

    def forward(self, h: torch.Tensor, state: tuple | None = None) -> tuple:
        B, L, D = h.shape
        H, r = self.H, self.r

        K = self.conv.kernel_size
        cov_state = None
        conv_state = None
        if state is not None:
            cov_state, conv_state = state
        if conv_state is None:
            conv_state = torch.zeros(B, self.D, K - 1, device=h.device, dtype=h.dtype)

        h_conv, conv_state_out = self.conv(h, conv_state)
        h_norm = rms_norm(h + h_conv, self.ln_w)

        u = h_norm @ self.W_u
        v = h_norm @ self.W_v

        K = torch.einsum('bld,hdr->bhlr', h_norm, self.W_k)
        Q = torch.einsum('bld,hdr->bhlr', h_norm, self.W_q)

        i_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_i) + self.b_i.view(1, H, 1, 1)
        i_gate = torch.exp(i_raw)

        decay_raw = torch.einsum('bld,hdi->bhli', h_norm, self.W_decay) + self.b_decay.view(1, H, 1, 1)
        decay = torch.sigmoid(decay_raw)

        K_e = K.unsqueeze(-1)
        delta = (K_e @ K_e.transpose(-2, -1)) * i_gate.unsqueeze(-1)

        a_scan = decay.squeeze(-1).permute(0, 2, 1)
        b_scan = delta.permute(0, 2, 1, 3, 4)
        M_all, final_cov_state = parallel_prefix_scan(a_scan, b_scan, cov_state)

        Q_perm = Q.permute(0, 2, 1, 3)
        mem_r = (Q_perm.unsqueeze(-2) @ M_all).squeeze(-2)
        mem_D = torch.einsum('blhr,hro->blho', mem_r, self.W_read)
        mem_sum = mem_D.sum(dim=2)

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

        return h + delta_spec, (final_cov_state, conv_state_out)


# ─── MemBindStack ────────────────────────────────────────────────────────

class MemBindStack(torch.nn.Module):
    def __init__(self, cfg: LDConfig):
        super().__init__()
        self.cfg = cfg
        self.D = cfg.D
        self.n_layers = cfg.n_layers
        lambda_k, block_sizes = compute_spectrum(cfg)
        self.layers = torch.nn.ModuleList([
            MemBindBlock(cfg, i, lambda_k, block_sizes) for i in range(cfg.n_layers)
        ])
        self.mlps = torch.nn.ModuleList([
            BottleneckMLP(cfg.D, cfg.bottleneck) for _ in range(cfg.n_layers)
        ])
        self.register_buffer('final_norm_w', torch.ones(cfg.D))

    def forward(self, h: torch.Tensor, state: list | None = None) -> tuple:
        if state is None:
            state = [None] * self.n_layers
        new_state = []
        for lidx in range(self.n_layers):
            h_layer, layer_state = self.layers[lidx](h, state[lidx])
            h_norm_mlp = rms_norm(h_layer, self.final_norm_w)
            h = h_layer + self.mlps[lidx](h_norm_mlp)
            new_state.append(layer_state)
        return rms_norm(h, self.final_norm_w), new_state


class LDStack(torch.nn.Module):
    def __init__(self, cfg: LDConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers
        self.D = cfg.D

        lambda_roots = fibonacci_roots(cfg.n_modes + 1)

        self.layers = torch.nn.ModuleList([
            LDBlock(cfg, i, lambda_roots) for i in range(cfg.n_layers)
        ])
        self.mlps = torch.nn.ModuleList([
            BottleneckMLP(cfg.D, cfg.bottleneck) for _ in range(cfg.n_layers)
        ])
        self.register_buffer('final_norm_w', torch.ones(cfg.D))

        # Adaptive gain: modulate update by gate decisiveness (all tokens, all layers)
        self.adaptive_gain = cfg.adaptive_gain

        # Global context feedback: aggregate → inject
        self.use_global_context = cfg.use_global_context
        if self.use_global_context:
            self.ctx_proj = torch.nn.Linear(cfg.D, cfg.D, bias=False)
            self.norm_before_context = rms_norm
            # stores context from the first pass (set by forward)
            self.register_buffer('_global_context', None)

    def forward(self, h: torch.Tensor, return_gates: bool = False,
                context: torch.Tensor | None = None) -> torch.Tensor:
        gates = [] if return_gates else None

        # Global context injection (from a previous pass)
        if context is not None:
            h = h + self.ctx_proj(context).unsqueeze(1)

        for lidx in range(self.n_layers):
            h_layer, alpha = self.layers[lidx](h, return_gates=True)

            # MLP
            h_norm = rms_norm(h_layer, self.final_norm_w)
            h_mlp = h_layer + self.mlps[lidx](h_norm)

            # Adaptive gain: scale update by mean gate activation (fraction of spectral power in use)
            if self.adaptive_gain and lidx < self.n_layers - 1:
                gain = alpha.mean(dim=-1, keepdim=True)  # (B, L, 1), ∈ (0,1)
                h = h + gain * (h_mlp - h)
            else:
                h = h_mlp

            if return_gates:
                gates.append(alpha)

        h_out = rms_norm(h, self.final_norm_w)

        # Store global context for the next call (if user wants it)
        if self.use_global_context and not self.training:
            self._global_context = h_out.mean(dim=1).detach()

        if return_gates:
            return h_out, torch.stack(gates, dim=0)
        return h_out
