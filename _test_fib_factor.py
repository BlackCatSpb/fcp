"""
Test: factorize all model weights through the spectral basis V.
W (D×d_out) → V^T @ W → keep top K rows → V[:,:K] @ W_enc → W_recon
Measure reconstruction error per weight.
"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack, dct_basis

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D, VOCAB = 896, 50000

# ─── V basis (DCT, matches training) ───
V = dct_basis(D).to(DEVICE)  # (D, D) orthonormal

# ─── Build model and load checkpoint ───
cfg = LDConfig()
cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = VOCAB
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

# ─── Test: factorize each weight through V ───
K = 8  # fib modes
results = []
logit_diffs = []

# For all named parameters
for name, W in model.named_parameters():
    if W.dim() != 2:
        continue  # skip biases, scalars
    
    d_in, d_out = W.shape
    
    # Case 1: W is (D, d_out) — project rows through V^T, keep top K
    if d_in == D:
        # Full projection: W_proj = V^T @ W  (D, d_out)
        W_proj = V.T @ W  # (D, d_out)
        W_enc = W_proj[:K, :]  # (K, d_out) — first K fib modes
        W_recon = V[:, :K] @ W_enc  # (D, d_out)
        rel_err = (W - W_recon).norm().item() / W.norm().item()
        compression = d_in * d_out / (K * d_out)  # D / K
        results.append((name, d_in, d_out, K, compression, rel_err, 'D→'))
    
    # Case 2: W is (d_in, D) — project cols through V, keep top K
    elif d_out == D:
        W_proj = W @ V  # (d_in, D) — project columns into V basis
        W_enc = W_proj[:, :K]  # (d_in, K)
        W_recon = W_enc @ V[:, :K].T  # (d_in, D)
        rel_err = (W - W_recon).norm().item() / W.norm().item()
        compression = d_in * d_out / (d_in * K)  # D / K
        results.append((name, d_in, d_out, K, compression, rel_err, '→D'))
    
    # Case 3: neither dim is D — skip (e.g., RF projections)
    else:
        results.append((name, d_in, d_out, 0, 0, 0, 'other'))

# ─── Also test: full forward pass with factorized weights ───
# Replace all D→* weights with factorized versions, measure logit change
model_factorized = torch.nn.Module()
model_factorized.embed = torch.nn.Embedding(VOCAB, D)
model_factorized.stack = MemBindStack(cfg)
model_factorized.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
model_factorized.lm_head.weight = model_factorized.embed.weight

# Copy all non-D×* weights directly
sd_factorized = {}
for name, W in model.named_parameters():
    if W.dim() != 2:
        sd_factorized[name] = W.detach().clone()
        continue
    d_in, d_out = W.shape
    if d_in == D and d_out != D:
        W_proj = V.T @ W
        W_enc = W_proj[:K, :]
        sd_factorized[name] = (V[:, :K] @ W_enc).detach().clone()
    elif d_out == D and d_in != D:
        W_proj = W @ V
        W_enc = W_proj[:, :K]
        sd_factorized[name] = (W_enc @ V[:, :K].T).detach().clone()
    elif d_in == D and d_out == D:
        # Full D×D — factorize both sides
        W_proj = V.T @ W @ V  # (D, D) in eigenbasis
        W_enc = W_proj[:K, :K]  # (K, K)
        W_recon = V[:, :K] @ W_enc @ V[:, :K].T
        sd_factorized[name] = W_recon.detach().clone()
    else:
        sd_factorized[name] = W.detach().clone()

model_factorized.load_state_dict(sd_factorized, strict=False)
model_factorized.to(DEVICE).eval()
del sd_factorized

# Forward pass comparison
x = torch.randint(0, 1000, (2, 128), device=DEVICE)
with torch.no_grad():
    h_orig = model.embed(x)
    h_orig = model.stack(h_orig)[0]
    logits_orig = model.lm_head(h_orig)
    
    h_fact = model_factorized.embed(x)
    h_fact = model_factorized.stack(h_fact)[0]
    logits_fact = model_factorized.lm_head(h_fact)
    
    logit_diff = (logits_orig - logits_fact).abs().mean().item()
    logit_max = (logits_orig - logits_fact).abs().max().item()
    hidden_diff = (h_orig - h_fact).abs().mean().item()

# ─── Report ───
lines = []
lines.append(f'V basis: D×D = {D}×{D}, truncated to K={K} modes')
lines.append(f'DCT basis: yes')
lines.append(f'')
lines.append(f'Per-weight reconstruction error (factorized = V[:,:K] @ W_enc):')
lines.append(f'{"Name":30s} {"Shape":16s} {"K":4s} {"Compress":10s} {"Rel Err":10s} {"Type":8s}')
lines.append('-' * 80)

best = []
worst = []
for name, d_in, d_out, k, comp, err, wtype in results:
    if wtype == 'D→' or wtype == '→D':
        best.append((err, name, comp))
        worst.append((-err, name, comp))
        lines.append(f'{name:30s} ({d_in:4d},{d_out:4d}) K={k:2d} {comp:6.1f}×  {err:.6f}  {wtype:8s}')

lines.append('')
best.sort(key=lambda x: x[0])
worst.sort(key=lambda x: x[0])

lines.append('Best 5 (lowest error):')
for err, name, comp in best[:5]:
    lines.append(f'  {name:30s} err={err:.6f} comp={comp:.0f}×')

lines.append('')
lines.append('Worst 5 (highest error):')
for err, name, comp in worst[-5:]:
    lines.append(f'  {name:30s} err={-err:.6f} comp={comp:.0f}×')

lines.append('')
lines.append(f'Forward pass comparison (original vs factorized, K={K}):')
lines.append(f'  Logit mean diff:  {logit_diff:.6f}')
lines.append(f'  Logit max diff:   {logit_max:.6f}')
lines.append(f'  Hidden mean diff: {hidden_diff:.6f}')

# Compare logit distributions
lines.append('')
lines.append(f'  Logits orig: mean={logits_orig.mean().item():.4f} std={logits_orig.std().item():.4f}')
lines.append(f'  Logits fact: mean={logits_fact.mean().item():.4f} std={logits_fact.std().item():.4f}')

# ─── Also test K=16, K=32 ───
for K_test in [16, 32, 64]:
    model_tmp = torch.nn.Module()
    model_tmp.embed = torch.nn.Embedding(VOCAB, D)
    model_tmp.stack = MemBindStack(cfg)
    model_tmp.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
    model_tmp.lm_head.weight = model_tmp.embed.weight
    
    sd_tmp = {}
    for name, W in model.named_parameters():
        if W.dim() != 2:
            sd_tmp[name] = W.detach().clone()
            continue
        d_in, d_out = W.shape
        if d_in == D and d_out != D:
            W_enc = (V.T @ W)[:K_test, :]
            sd_tmp[name] = (V[:, :K_test] @ W_enc).detach().clone()
        elif d_out == D and d_in != D:
            W_enc = (W @ V)[:, :K_test]
            sd_tmp[name] = (W_enc @ V[:, :K_test].T).detach().clone()
        elif d_in == D and d_out == D:
            W_enc = (V.T @ W @ V)[:K_test, :K_test]
            sd_tmp[name] = (V[:, :K_test] @ W_enc @ V[:, :K_test].T).detach().clone()
        else:
            sd_tmp[name] = W.detach().clone()
    
    model_tmp.load_state_dict(sd_tmp, strict=False)
    model_tmp.to(DEVICE).eval()
    del sd_tmp
    
    with torch.no_grad():
        h_tmp = model_tmp.stack(model_tmp.embed(x))[0]
        logits_tmp = model_tmp.lm_head(h_tmp)
        diff = (logits_orig - logits_tmp).abs().mean().item()
    
    lines.append(f'  K={K_test:2d}: logit mean diff = {diff:.6f}')
    del model_tmp

with open('outputs/_fib_factor_test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Done. Check outputs/_fib_factor_test.txt')
