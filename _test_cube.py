"""
Cube analogy: V is a D×D cube with K=8 fib faces.
Each weight W: D×d_out uses one face combination.
Test: block-mean approximation in V-space (NOT row truncation).
"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack, dct_basis, compute_spectrum

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D, VOCAB = 896, 50000

# ─── Compute fib block sizes (same as model) ───
K = 8
cfg_spec = LDConfig()
cfg_spec.n_modes = K; cfg_spec.D = D
cfg_spec.spectrum_type = 'fib_seq'; cfg_spec.spec_lo = 0.8; cfg_spec.spec_hi = 1.8
lambdas, block_sizes = compute_spectrum(cfg_spec)
print(f'Fib block sizes: {block_sizes}')
print(f'Sum: {sum(block_sizes)} (should be {D})')
print(f'Lambdas: {lambdas}')

# V basis (DCT)
V = dct_basis(D).to(DEVICE)  # (D, D) orthonormal

# ─── Build model ───
cfg = LDConfig()
cfg.D = D; cfg.n_layers = 24; cfg.n_modes = K; cfg.vocab = VOCAB
cfg.bottleneck = 896; cfg.kernel_size = 48
cfg.weight_tying = True; cfg.lm_head_bias = True
cfg.arch = 'membind'; cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
cfg.dct_basis = True; cfg.lambda_sliding = True
cfg.cov_first_moment = True; cfg.cov_rf = True; cfg.cov_rf_dim = 64
cfg.cov_multi_timescale = True; cfg.cov_mirror = True

model = torch.nn.Module()
model.embed = torch.nn.Embedding(VOCAB, D)
model.stack = MemBindStack(cfg)
model.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
model.lm_head.weight = model.embed.weight

ckpt = torch.load('checkpoints/ACTION_step2500.pt', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(DEVICE).eval()

# Build block index: start positions of each block
block_starts = [0]
for bs in block_sizes:
    block_starts.append(block_starts[-1] + bs)

def factorize_weight_blockmean(W, V, block_sizes, block_starts):
    """
    Factorize W: (D, d_out) through V.
    1. W_proj = V^T @ W  (project into V basis)
    2. Split rows into K fib blocks
    3. Within each block, compute mean across rows → (d_out,)
    4. Store: W_enc: (K, d_out)
    5. Reconstruct: expand mean to fill block, then W = V @ expanded
    """
    d_in, d_out = W.shape
    W_proj = V.T @ W  # (D, d_out) in eigenbasis
    
    W_enc = []
    for k in range(len(block_sizes)):
        start, end = block_starts[k], block_starts[k+1]
        block_rows = W_proj[start:end, :]  # (block_size, d_out)
        block_mean = block_rows.mean(dim=0, keepdim=True)  # (1, d_out)
        W_enc.append(block_mean)
    W_enc = torch.cat(W_enc, dim=0)  # (K, d_out)
    
    # Reconstruct
    expanded = torch.zeros(D, d_out, device=W.device)
    for k in range(len(block_sizes)):
        start, end = block_starts[k], block_starts[k+1]
        expanded[start:end, :] = W_enc[k:k+1, :]  # broadcast to fill block
    
    W_recon = V @ expanded  # (D, d_out)
    return W_enc, W_recon

def factorize_weight_rev(W, V, block_sizes, block_starts):
    """
    Factorize W: (d_in, D) where d_in != D.
    W in V-space: W_proj = W @ V → blocks on columns → mean → expand.
    """
    d_in, d_out = W.shape
    W_proj = W @ V  # (d_in, D) in eigenbasis
    
    W_enc = []
    for k in range(len(block_sizes)):
        start, end = block_starts[k], block_starts[k+1]
        block_cols = W_proj[:, start:end]  # (d_in, block_size)
        block_mean = block_cols.mean(dim=1, keepdim=True)  # (d_in, 1)
        W_enc.append(block_mean)
    W_enc = torch.cat(W_enc, dim=1)  # (d_in, K)
    
    expanded = torch.zeros(d_in, D, device=W.device)
    for k in range(len(block_sizes)):
        start, end = block_starts[k], block_starts[k+1]
        expanded[:, start:end] = W_enc[:, k:k+1]  # broadcast
    
    W_recon = expanded @ V.T  # (d_in, D)
    return W_enc, W_recon

# Test all weights
lines = ['Cube analogy: block-mean factorization through V']
lines.append(f'V basis: DCT, D={D}, K={K}')
lines.append(f'Block sizes: {block_sizes}')
lines.append(f'')
lines.append(f'{"Name":35s} {"Shape":14s} {"Compress":10s} {"Rel Err":12s} {"Type":8s}')
lines.append('-' * 80)

total_params_orig = 0
total_params_enc = 0
logit_errors = []

for name, W in model.named_parameters():
    if W.dim() != 2:
        continue
    
    d_in, d_out = W.shape
    
    if d_in == D and d_out != D:
        # Forward: D → d_out
        W_enc, W_recon = factorize_weight_blockmean(W, V, block_sizes, block_starts)
        rel_err = (W - W_recon).norm().item() / W.norm().item()
        # Compression: store K×d_out instead of D×d_out
        params_inv = K * d_out  # W_enc
        # Plus V is shared (amortized over all weights)
        comp = d_in * d_out / params_inv
        lines.append(f'{name:35s} ({d_in:4d},{d_out:4d})  {comp:6.1f}×    {rel_err:.6f}    D→')
        total_params_orig += d_in * d_out
        total_params_enc += params_inv
        
    elif d_out == D and d_in != D:
        # Reverse: d_in → D
        W_enc, W_recon = factorize_weight_rev(W, V, block_sizes, block_starts)
        rel_err = (W - W_recon).norm().item() / W.norm().item()
        params_inv = d_in * K
        comp = d_in * d_out / params_inv
        lines.append(f'{name:35s} ({d_in:4d},{d_out:4d})  {comp:6.1f}×    {rel_err:.6f}    →D')
        total_params_orig += d_in * d_out
        total_params_enc += params_inv
        
    elif d_in == D and d_out == D:
        # D×D: factorize both sides
        W_proj = V.T @ W @ V  # (D, D) full eigenbasis
        W_enc = torch.zeros(K, K, device=W.device)
        for ki in range(K):
            si, ei = block_starts[ki], block_starts[ki+1]
            for kj in range(K):
                sj, ej = block_starts[kj], block_starts[kj+1]
                W_enc[ki, kj] = W_proj[si:ei, sj:ej].mean()
        # Reconstruct
        expanded = torch.zeros(D, D, device=W.device)
        for ki in range(K):
            si, ei = block_starts[ki], block_starts[ki+1]
            for kj in range(K):
                sj, ej = block_starts[kj], block_starts[kj+1]
                expanded[si:ei, sj:ej] = W_enc[ki, kj]
        W_recon = V @ expanded @ V.T
        rel_err = (W - W_recon).norm().item() / W.norm().item()
        params_inv = K * K
        comp = D * D / params_inv
        lines.append(f'{name:35s} ({d_in:4d},{d_out:4d})  {comp:6.1f}×    {rel_err:.6f}    D×D')
        total_params_orig += d_in * d_out
        total_params_enc += params_inv

# Embedding: special case — vocab × D
name = 'embed.weight'
W = model.embed.weight  # (50000, 896)
d_in, d_out = W.shape
W_enc, W_recon = factorize_weight_rev(W, V, block_sizes, block_starts)
rel_err = (W - W_recon).norm().item() / W.norm().item()
params_inv = d_in * K
comp = d_in * d_out / params_inv
lines.append(f'{name:35s} ({d_in:5d},{d_out:4d}) {comp:6.1f}×    {rel_err:.6f}    →D')
total_params_orig += d_in * d_out
total_params_enc += params_inv

lines.append(f'')
lines.append(f'Total params original: {total_params_orig/1e6:.1f}M')
lines.append(f'Total params encoded:  {total_params_enc/1e6:.1f}M')
lines.append(f'Overall compression:   {total_params_orig/total_params_enc:.1f}×')
lines.append(f'Plus V basis (shared): {D*D/1e6:.1f}M ({D*D*4/1e6:.1f}MB)')
lines.append(f'')
lines.append(f'Plus buffers (41M): 164MB (unchanged)')
lines.append(f'Estimated VRAM: {(total_params_enc*4 + D*D*4 + 41e6*4)/1e6:.0f}MB')

with open('outputs/_cube_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Done.')
