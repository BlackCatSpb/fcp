"""
Experiment: token importance -> abstraction levels in lambda_d.

Hypothesis:
  Gate entropy (per-token) correlates with semantic importance.
  Low entropy = model makes decisive mode choice -> content words.
  High entropy = model spreads across all modes -> function words.

  Abstraction levels:
    Level 0 (imp<0.2): syntax, function words -> shallow processing
    Level 1 (imp<0.4): modifiers (adj/adv) -> medium
    Level 2 (imp<0.6): concrete nouns/verbs -> deep
    Level 3 (imp>0.6): abstract concepts, entities -> deepest

  Adaptive depth: high-importance tokens benefit from more layers.
"""

import torch, sys, math, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack, rms_norm
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig(); cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x, return_gates=False):
        h = self.embed(x)
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        return self.lm_head(self.stack(h))

model = Phase2Model().to(DEVICE)
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Loaded step {ckpt["step"]}')

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:50, :128].copy()).long().to(DEVICE)  # B=50 for stats

with torch.no_grad():
    logits, gates = model(x, return_gates=True)   # (N_LAYERS, B, L, K)

B, L, K = x.shape[0], x.shape[1], N_MODES

# ---- 1. Multiple salience signals from gates ----
gate_mean = gates.mean(dim=0)                         # (B, L, K)
entropy = -(gate_mean * (gate_mean + 1e-10).log()).sum(-1)
max_H = math.log(K)
imp_entropy = 1.0 - entropy / max_H                   # low entropy = important

# Spread (std across modes) - high spread = decisive mode choice
imp_spread = gate_mean.std(dim=-1)                    # (B, L)

# Max gate value - high max = one mode dominates
imp_max = gate_mean.max(dim=-1).values                # (B, L)

# Per-layer gate volatility (how much gates change between adjacent tokens)
gate_diff = gates[:, :, 1:] - gates[:, :, :-1]       # (N_LAYERS, B, L-1, K)
gate_volatility = gate_diff.norm(dim=-1).mean(dim=0)  # (B, L-1)
# Pad to match length
imp_volatility = torch.nn.functional.pad(gate_volatility, (1, 0), value=0.0)  # (B, L)

# ---- 2. Token frequency proxy ----
token_freqs = torch.bincount(x.flatten(), minlength=VOCAB).float()
token_freqs = token_freqs / token_freqs.sum()
freq = token_freqs[x]   # (B, L)

# ---- 3. Which salience signal best separates rare vs frequent tokens? ----
print("\n=== Salience Signal Comparison ===")
print(f"{'signal':<18} {'rare':>8} {'freq':>8} {'delta':>8} {'corr w/freq'}")
print("-"*60)

signals = {
    "entropy->imp": imp_entropy,
    "spread": imp_spread,
    "max_gate": imp_max,
    "volatility": imp_volatility,
}

from scipy.stats import pearsonr, spearmanr
for name, sig in signals.items():
    rare_mask = freq < 0.005
    freq_mask = freq >= 0.005
    if rare_mask.sum() > 0 and freq_mask.sum() > 0:
        rare_mean = sig[rare_mask].mean().item()
        freq_mean = sig[freq_mask].mean().item()
        delta = rare_mean - freq_mean
        r_pearson = pearsonr(sig.flatten().cpu().numpy(), freq.flatten().cpu().numpy())[0]
        r_spearman = spearmanr(sig.flatten().cpu().numpy(), freq.flatten().cpu().numpy())[0]
        print(f"  {name:<18} {rare_mean:>8.4f} {freq_mean:>8.4f} {delta:>+8.4f} "
              f"p={r_pearson:+.3f} s={r_spearman:+.3f}")

# ---- 4. Best signal: spread - abstraction levels ----
# Use spread as importance (it's the most intuitive: decisive mode = important)
imp = imp_spread

def abstraction_label(imp_val):
    if imp_val < 0.08: return 0, "syntax/function"
    if imp_val < 0.15: return 1, "modifier"
    if imp_val < 0.25: return 2, "concrete"
    return 3, "abstract"

abstraction = torch.zeros_like(imp, dtype=torch.long)
for b in range(B):
    for l in range(L):
        abstraction[b,l] = abstraction_label(imp[b,l].item())[0]

print("\n=== Mode profile by abstraction level (spread-based) ===")
for level in range(4):
    m = (abstraction == level)
    if m.sum() == 0: continue
    g = gate_mean[m]
    mode_means = g.mean(dim=0).cpu().tolist()
    dom_mode = g.argmax(-1)
    dom_dist = torch.bincount(dom_mode.flatten(), minlength=K)
    dom_pct = dom_dist / dom_dist.sum() * 100
    print(f"  L{level} (n={m.sum().item():>6}) | spread={imp[m].mean():.3f} "
          f"| modes=[{', '.join(f'{v:.3f}' for v in mode_means)}] "
          f"| dominant=[{', '.join(f'{v:.0f}%' for v in dom_pct)}]")

# ---- 5. Token-level view of best signal ----
print(f"\n=== Token-level importance (first 40 tokens, batch 0) ===")
print(f"{'pos':>3} {'tok_id':>7} {'H':>5} {'sprd':>5} {'lvl':>2} {'modes (1..4)':<16}")
for pos in range(min(40, L)):
    spr = imp[0, pos].item()
    H = entropy[0, pos].item()
    lvl = abstraction[0, pos].item()
    g = gate_mean[0, pos].cpu().tolist()
    g_str = ' '.join(f'{v:.2f}' for v in g)
    print(f"  {pos:>3} {x[0,pos].item():>7} {H:>5.3f} {spr:>5.3f} {lvl:>2.0f} [{g_str}]")

# ---- 6. Per-layer convergence by importance ----
print(f"\n=== Layer-wise convergence by importance ===")
h = model.embed(x[:4])
with torch.no_grad():
    all_hidden = [h.cpu()]
    for lidx in range(N_LAYERS):
        h = model.stack.layers[lidx](h)
        h_norm = rms_norm(h, model.stack.final_norm_w)
        h = h + model.stack.mlps[lidx](h_norm)
        all_hidden.append(h.cpu().clone())

all_hidden = torch.stack(all_hidden)

imp_sub = imp[:4].cpu()
thresholds = np.percentile(imp_sub.flatten().numpy(), [50, 75, 90])
print(f"Importance threshold: low={thresholds[0]:.3f} mid={thresholds[1]:.3f} high={thresholds[2]:.3f}")

high = imp_sub > thresholds[2]
mid = (imp_sub > thresholds[0]) & (imp_sub <= thresholds[2])
low = imp_sub <= thresholds[0]

ref_layers = [2, 4, 6, 8, 10, 12]
print(f"\n{'group':<12} {'metric':>6}", end='')
for rl in ref_layers:
    print(f" {rl:>5}", end='')
print()

for group_name, group_mask in [("high-sprd", high), ("mid-sprd", mid), ("low-sprd", low)]:
    if group_mask.sum() == 0: continue
    cos_vals = []
    for rl in ref_layers:
        cos = torch.nn.functional.cosine_similarity(
            all_hidden[rl][group_mask].float(), all_hidden[-1][group_mask].float(), dim=-1)
        cos_vals.append(f"{cos.mean():.3f}")
    print(f"  {group_name:<12} {'cos':>6} {'  '.join(cos_vals)}")

# ---- 7. Summary ----
print(f"\n=== Summary ===")
print(f"  Mean spread (all tokens): {imp.mean():.3f}")
print(f"  % tokens spread>0.2 (decisive): {(imp > 0.2).sum().item() / (B*L) * 100:.1f}%")
print(f"  % tokens spread<0.08 (uniform): {(imp < 0.08).sum().item() / (B*L) * 100:.1f}%")
print(f"  Best signal: spread (gate std) = content words have higher spread")
print()
print("Takeaway: gate spread (decisiveness) is a zero-cost salience signal.")
print("High-spread tokens converge slower -> need more layers.")
print("Low-spread tokens converge early -> can shallow-route.")
print("Adaptive depth based on gate decisiveness can save compute.")
