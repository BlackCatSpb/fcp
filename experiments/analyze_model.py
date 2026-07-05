"""
Per-layer model analyzer for lambda_d.
Usage:
  python analyze_model.py                          # latest checkpoint, ascii
  python analyze_model.py --html                   # HTML report
  python analyze_model.py --html --output out.html # custom path
  python analyze_model.py --plot                   # + matplotlib viz
  python analyze_model.py --ckpt path/to/model.pt  # specific checkpoint
"""

import os, sys, argparse, math, time, io, textwrap
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Config (mirrors train_phase2) ---------------------------------------
D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 2
SEQ_LEN = 128
CKPT_DIR = 'checkpoints'

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path')
parser.add_argument('--html', action='store_true', help='Generate HTML report')
parser.add_argument('--output', type=str, default=None, help='Output file (default: model_report.html)')
parser.add_argument('--plot', action='store_true', help='Show matplotlib plots')
parser.add_argument('--save_plot', type=str, default=None, help='Save plot to file')
args = parser.parse_args()

if args.ckpt is None:
    import glob
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_*.pt')))
    if not files:
        files = sorted(glob.glob(os.path.join(CKPT_DIR, 'model_step*.pt')))
    args.ckpt = files[-1] if files else None

if args.ckpt is None or not os.path.exists(args.ckpt):
    print(f'Checkpoint not found: {args.ckpt}')
    sys.exit(1)

print(f'Loading checkpoint: {args.ckpt}')
ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
cfg_dict = ckpt.get('config', {})
D = cfg_dict.get('D', D)
VOCAB = cfg_dict.get('VOCAB', VOCAB)
N_MODES = cfg_dict.get('N_MODES', N_MODES)
N_LAYERS = cfg_dict.get('N_LAYERS', N_LAYERS)

# --- Build model ---------------------------------------------------------
class Phase2Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        ld_cfg = LDConfig()
        ld_cfg.D = D; ld_cfg.n_layers = N_LAYERS; ld_cfg.n_modes = N_MODES
        ld_cfg.vocab = VOCAB; ld_cfg.bottleneck = 512; ld_cfg.kernel_size = 48
        ld_cfg.weight_tying = True; ld_cfg.lm_head_bias = True
        self.stack = LDStack(ld_cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=ld_cfg.lm_head_bias)
        if ld_cfg.weight_tying:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, return_gates=False, return_all=False):
        h = self.embed(input_ids)
        if return_all:
            h, gates, hiddens = self.stack(h, return_gates=True, return_hiddens=True)
            return self.lm_head(h), gates, hiddens
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        return self.lm_head(h)

model = Phase2Model().to(DEVICE)
sd = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt.get('model_fp16')
if sd is None:
    raise KeyError(f'No model state dict found in checkpoint keys: {list(ckpt.keys())[:10]}')
model.load_state_dict(sd, strict=False)
model.eval()
print(f'  Step {ckpt.get("step", "?")}, epoch {ckpt.get("epoch", "?")}')

# --- Patch LDStack to return hiddens -------------------------------------
original_forward = LDStack.forward

def patched_forward(self, h, return_gates=False, return_hiddens=False,
                    context=None):
    gates_list = [] if (return_gates or return_hiddens) else None
    hiddens_list = [] if return_hiddens else None

    # Global context injection
    if context is not None:
        h = h + self.ctx_proj(context).unsqueeze(1)

    for lidx in range(self.n_layers):
        h_layer, alpha = self.layers[lidx](h, return_gates=True)
        h_norm = rms_norm_fn(h_layer, self.final_norm_w)
        h_mlp = h_layer + self.mlps[lidx](h_norm)

        # Adaptive gain: all tokens always pass, scaled by gate decisiveness
        if self.adaptive_gain and lidx < self.n_layers - 1:
            gain = alpha.std(dim=-1, keepdim=True).expand(-1, -1, 1)
            gain = gain.clamp(0.0, 1.0)
            h = h + gain * (h_mlp - h)
        else:
            h = h_mlp

        if return_gates:
            gates_list.append(alpha)
        if return_hiddens:
            hiddens_list.append(h.detach())

    h_out = rms_norm_fn(h, self.final_norm_w)
    if return_hiddens:
        return h_out, torch.stack(gates_list, dim=0) if gates_list else None, hiddens_list
    if return_gates:
        return h_out, torch.stack(gates_list, dim=0)
    return h_out

def rms_norm_fn(x, weight, eps=1e-6):
    rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    return x / rms.clamp(min=eps) * weight

import ld_model.core as core_mod
core_mod.rms_norm_fn = rms_norm_fn
LDStack.forward = patched_forward

# --- HTML report writer ---------------------------------------------------
def write_html_report(data, path):
    g = data['gates_avg']
    H_per_layer = [-(g[l] * np.log(g[l] + 1e-10)).sum() for l in range(data['n_layers'])]
    max_ent = math.log(data['K'])
    pass_rates = data.get('pass_rates', [])
    cayley_norms = data.get('cayley_norms', [])
    orth_errs = data.get('orth_errs', [])
    h_norms = data.get('h_norms', [])
    h_deltas = data.get('h_deltas', [])
    eff_ranks = data.get('eff_ranks', [])
    sv_tops = data.get('sv_tops', [])

    mode_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    def tr(*cells, tag='td'):
        return '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>lambda_d Model Report - Step {data['step']}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; color: #222; }}
h1, h2, h3 {{ color: #111; }}
.summary {{ background: #fff; padding: 16px 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.12); margin-bottom: 16px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 20px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.12); border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 6px 12px; text-align: right; font-size: 13px; border-bottom: 1px solid #eee; }}
th {{ background: #fafafa; font-weight: 600; }}
td:first-child, th:first-child {{ text-align: left; }}
.gate-bar {{ display: inline-block; height: 16px; border-radius: 2px; vertical-align: middle; }}
.pass-bg {{ display: inline-block; width: 80px; height: 14px; border-radius: 2px; vertical-align: middle; background: #eee; }}
.pass-fg {{ display: inline-block; height: 14px; border-radius: 2px; vertical-align: middle; background: #4daf4a; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.metric {{ background: #fff; padding: 12px 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }}
.metric .val {{ font-size: 24px; font-weight: 700; }}
.metric .label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: .5px; }}
.gate-th {{ width: 16%; }}
</style></head>
<body>
<h1 style="display:flex;align-items:center;gap:12px;">
  <span style="background:#222;color:#fff;padding:4px 12px;border-radius:6px;font-size:14px;">lambda_d</span>
  Model Report <span style="font-size:14px;color:#666;font-weight:400;">Step {data['step']}</span>
</h1>

<div class="grid">
  <div class="metric"><div class="val">{data['ppl']:.0f}</div><div class="label">Perplexity</div></div>
  <div class="metric"><div class="val">{data['loss']:.2f}</div><div class="label">Loss</div></div>
  <div class="metric"><div class="val">{data['n_params']/1e6:.1f}M</div><div class="label">Parameters</div></div>
  <div class="metric"><div class="val">{data['D']}</div><div class="label">Dimension</div></div>
  <div class="metric"><div class="val">{data['n_layers']}</div><div class="label">Layers</div></div>
  <div class="metric"><div class="val">{data['K']}</div><div class="label">Modes (K)</div></div>
</div>

<h2>Gate Composition per Layer</h2>
<div style="overflow-x:auto;">
<table>
<tr><th>Layer</th>
'''
    for k in range(data['K']):
        html += f'<th>m{k}</th>'
    html += '<th>Entropy</th><th>Spread</th><th>Gain>0.5</th><th>Visual</th></tr>\n'
    total_w = 120
    for lidx in range(data['n_layers']):
        row = g[lidx]
        H = H_per_layer[lidx]
        spread = data['gates_spread'][lidx]
        bar = ''
        for k in range(data['K']):
            w = max(int(row[k] * total_w), 1)
            bar += f'<span class="gate-bar" style="width:{w}px;background:{mode_colors[k%len(mode_colors)]};"></span>'
        pr = pass_rates[lidx] if lidx < len(pass_rates) else 1.0
        pr_pct = f'{pr*100:.0f}%'
        pass_fill = f'<div class="pass-bg"><div class="pass-fg" style="width:{pr*80:.0f}px;"></div></div>' if lidx == 0 or pr > 0 else '<span style="color:#999;">-</span>'
        html += tr(f'L{lidx}', *(f'{row[k]:.3f}' for k in range(data['K'])),
                   f'{H:.2f}/{max_ent:.2f}', f'{spread:.4f}', pr_pct,
                   f'<span style="display:inline-block;width:{total_w}px;height:16px;background:#eee;border-radius:2px;">{bar}</span>')
    html += '</table></div>\n'

    if cayley_norms:
        html += '<h2>Cayley Rotation (Learnable V)</h2>\n<div style="overflow-x:auto;"><table><tr><th>Layer</th><th>|A+B|</th><th>|S|</th><th>Orth error</th><th>Eff rank</th><th>Top SV</th><th>cond(I-S)</th></tr>\n'
        for lidx in range(data['n_layers']):
            rank_str = f'{eff_ranks[lidx]:.1f}' if eff_ranks else '-'
            top_sv_str = f'{sv_tops[lidx]:.4f}' if sv_tops else '-'
            html += tr(f'L{lidx}', f'{cayley_norms[lidx]:.4f}', f'{data["S_norms"][lidx]:.4f}',
                       f'{orth_errs[lidx]:.2e}', rank_str, top_sv_str, f'{data["cond_est"][lidx]:.2f}')
        html += '</table></div>\n'

    if h_norms:
        html += '<h2>Hidden State Dynamics</h2>\n<div style="overflow-x:auto;"><table><tr><th>Layer</th><th>||h|| mean</th><th>h std</th><th>||delta||</th><th>delta/|h|</th></tr>\n'
        for lidx in range(data['n_layers']):
            html += tr(f'L{lidx}', f'{h_norms[lidx]:.2f}', f'{data["h_stds"][lidx]:.4f}',
                       f'{h_deltas[lidx]:.4f}', f'{data["h_delta_ratios"][lidx]:.4f}')
        html += '</table></div>\n'

    html += '<h2>Parameter Breakdown</h2>\n<div style="overflow-x:auto;"><table><tr><th>Component</th><th>Params</th><th>%</th></tr>\n'
    for name, count, pct in data['param_breakdown']:
        t = 'th' if name == 'Total' else 'td'
        html += tr(name, f'{count:,}', f'{pct:.2f}%', tag=t)
    html += '</table></div>\n'

    html += '<div style="color:#888;font-size:11px;text-align:center;margin-top:40px;">Generated by analyze_model.py</div>\n'
    html += '</body></html>'

    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'HTML report saved: {path}')

# --- Load data -----------------------------------------------------------
data_file = 'russian_chunks.npy'
if not os.path.exists(data_file):
    data_file = 'wikitext_chunks.npy'
    if not os.path.exists(data_file):
        print('No data file found. Generating random input.')
        bx = torch.randint(10, VOCAB-10, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
        by = bx.clone()
    else:
        arr = np.load(data_file)
        bx = torch.tensor(arr[:BATCH_SIZE, :-1], dtype=torch.long, device=DEVICE)
        by = torch.tensor(arr[:BATCH_SIZE, 1:], dtype=torch.long, device=DEVICE)
else:
    arr = np.load(data_file)
    bx = torch.tensor(arr[:BATCH_SIZE, :-1], dtype=torch.long, device=DEVICE)
    by = torch.tensor(arr[:BATCH_SIZE, 1:], dtype=torch.long, device=DEVICE)

# --- Forward pass with all diagnostics -----------------------------------
with torch.no_grad():
    logits, gates, hiddens = model(bx, return_all=True)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), by.reshape(-1))
    ppl = math.exp(loss.item())

# --- Per-layer analysis ---------------------------------------------------
gates_np = gates.cpu().numpy()  # (n_layers, B, L, K)
report_data = {}

print(f'\n{"="*90}')
print(f'  lambda_d Model Analysis - Step {ckpt.get("step", "?")}')
print(f'  PPL: {ppl:.1f} | Loss: {loss.item():.4f}')
print(f'  D={D}, L={N_LAYERS} layers, K={N_MODES} modes, B={BATCH_SIZE}, seq={SEQ_LEN}')
print(f'{"="*90}')
print()

# 1. Gate distributions per layer
gate_means = gates_np.mean(axis=(1, 2))  # (L, K)
gates_spread = [gates_np[lidx].std(axis=-1).mean() for lidx in range(N_LAYERS)]

print(f'  {"Layer":>6} | {"Mode weights (avg)":>30} | {"Entropy":>8} | {"Spread":>8} | {"Gain>0.5":>8}')
print(f'  {"-"*6}-+-{"-"*30}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}')
max_ent = math.log(N_MODES)
pass_rates = []
for lidx in range(N_LAYERS):
    gate_avg = gate_means[lidx]
    gate_str = ' '.join(f'{g:.3f}' for g in gate_avg)
    H = -(gate_avg * np.log(gate_avg + 1e-10)).sum()
    spread = gates_spread[lidx]
    if lidx < N_LAYERS - 1:
        thresh = 0.5  # gain threshold (adaptive gain clamps to [0, 1])
        pr = (gates_np[lidx].std(axis=-1).mean(axis=-1) > thresh).mean()
    else:
        thresh = 0
        pr = 1.0
    pass_rates.append(pr)
    print(f'  L{lidx:>3}   | {gate_str:>30} | {H:.3f}/{max_ent:.3f} | {spread:.4f} | {pr*100:>6.0f}%')

print()

# 2. Cayley |A+B| norms
cayley_norms_arr = []
S_norms_arr = []
orth_errs_arr = []
cond_est_arr = []
if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
    print(f'  {"Layer":>6} | {"|A+B|":>10} | {"|S|":>10} | {"orth_err":>12} | {"R_cond":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*12}-+-{"-"*10}')
    for lidx in range(N_LAYERS):
        l = model.stack.layers[lidx]
        A, B = l.V_cay_A, l.V_cay_B
        n_ab = (A.norm().item() + B.norm().item())
        S = A @ B.T - B @ A.T
        n_S = S.norm().item()
        R = l.compute_R()
        I = torch.eye(D, device=R.device, dtype=R.dtype)
        orth_err = (R @ R.T - I).norm().item() / D
        try:
            cond = torch.linalg.cond(I - S).item()
        except:
            cond = float('nan')
        cayley_norms_arr.append(n_ab)
        S_norms_arr.append(n_S)
        orth_errs_arr.append(orth_err)
        cond_est_arr.append(cond)
        print(f'  L{lidx:>3}   | {n_ab:>10.4f} | {n_S:>10.4f} | {orth_err:.2e}    | {cond:>10.2f}')

# 3. Hidden state dynamics
h_norms_arr = []
h_stds_arr = []
h_deltas_arr = []
h_delta_ratios_arr = []
if hiddens:
    print(f'\n  {"Layer":>6} | {"h_norm":>10} | {"h_std":>10} | {"delta_norm":>12} | {"delta/|h|":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*12}-+-{"-"*10}')
    prev_h = model.embed(bx)
    for lidx in range(N_LAYERS):
        h_cur = hiddens[lidx]
        h_n = h_cur.norm(dim=-1).mean().item()
        h_s = h_cur.std(dim=-1).mean().item()
        delta = h_cur - prev_h
        d_n = delta.norm(dim=-1).mean().item()
        ratio = d_n / (h_n + 1e-10)
        h_norms_arr.append(h_n)
        h_stds_arr.append(h_s)
        h_deltas_arr.append(d_n)
        h_delta_ratios_arr.append(ratio)
        print(f'  L{lidx:>3}   | {h_n:>10.4f} | {h_s:>10.4f} | {d_n:>12.4f} | {ratio:>10.4f}')
        prev_h = h_cur

# 4. Cayley effective rank
eff_ranks_arr = []
sv_tops_arr = []
if hasattr(model.stack.layers[0], 'V_cay_A') and model.stack.layers[0].V_cay_A is not None:
    print(f'\n  {"Layer":>6} | {"svd_rank":>10} | {"top_sv":>10} | {"bot_sv":>10} | {"spect_gap":>10}')
    print(f'  {"-"*6}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}-+-{"-"*10}')
    for lidx in range(N_LAYERS):
        l = model.stack.layers[lidx]
        A, B = l.V_cay_A, l.V_cay_B
        S = A @ B.T - B @ A.T
        sv = torch.linalg.svdvals(S)
        eff_rank = (sv / sv.max()).sum().item()
        eff_ranks_arr.append(eff_rank)
        sv_tops_arr.append(sv[0].item())
        print(f'  L{lidx:>3}   | {eff_rank:>10.2f} | {sv[0]:>10.4f} | {sv[-1]:>10.4f} | {(sv[0]-sv[-1]):>10.4f}')

# 5. Model stats
n_all = sum(p.numel() for p in model.parameters())
n_emb = sum(p.numel() for p in model.embed.parameters())
n_stack = sum(p.numel() for p in model.stack.parameters())
n_head = sum(p.numel() for p in model.lm_head.parameters())
n_cayley = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
n_gates = sum(p.numel() for n, p in model.named_parameters() if 'W_gate' in n or 'b_gate' in n)
print(f'\n  {"Component":>20} | {"Params":>12} | {"%":>8}')
print(f'  {"-"*20}-+-{"-"*12}-+-{"-"*8}')
print(f'  {"Embed":>20} | {n_emb:>12,} | {n_emb/n_all*100:>7.2f}%')
print(f'  {"Stack (LDBlock+MLP)":>20} | {n_stack:>12,} | {n_stack/n_all*100:>7.2f}%')
print(f'  {"  | Cayley (A,B)":>20} | {n_cayley:>12,} | {n_cayley/n_all*100:>7.2f}%')
print(f'  {"  | Gates (W,b)":>20} | {n_gates:>12,} | {n_gates/n_all*100:>7.2f}%')
print(f'  {"lm_head":>20} | {n_head:>12,} | {n_head/n_all*100:>7.2f}%')
print(f'  {"-"*20}-+-{"-"*12}-+-{"-"*8}')
print(f'  {"Total":>20} | {n_all:>12,} | {"100%":>8}')

# --- Build report data -----------------------------------------------------
report_data = {
    'step': ckpt.get('step', '?'),
    'ppl': ppl, 'loss': loss.item(),
    'D': D, 'n_layers': N_LAYERS, 'K': N_MODES,
    'n_params': n_all,
    'gates_avg': gate_means,
    'gates_spread': gates_spread,
    'pass_rates': pass_rates,
    'cayley_norms': cayley_norms_arr,
    'S_norms': S_norms_arr,
    'orth_errs': orth_errs_arr,
    'cond_est': cond_est_arr,
    'h_norms': h_norms_arr,
    'h_stds': h_stds_arr,
    'h_deltas': h_deltas_arr,
    'h_delta_ratios': h_delta_ratios_arr,
    'eff_ranks': eff_ranks_arr,
    'sv_tops': sv_tops_arr,
    'param_breakdown': [
        ('Embed', n_emb, n_emb/n_all*100),
        ('Stack (LDBlock+MLP)', n_stack, n_stack/n_all*100),
        ('  | Cayley (A,B)', n_cayley, n_cayley/n_all*100),
        ('  | Gates (W,b)', n_gates, n_gates/n_all*100),
        ('lm_head', n_head, n_head/n_all*100),
        ('Total', n_all, 100.0),
    ],
    'ckpt': True,
}

# --- HTML output ----------------------------------------------------------
if args.html:
    out_path = args.output if args.output else 'model_report.html'
    write_html_report(report_data, out_path)

# --- Plot -----------------------------------------------------------------
if args.plot or args.save_plot:
    try:
        import matplotlib
        if args.save_plot:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'lambda_d Model Analysis - Step {ckpt.get("step", "?")}', fontsize=14)

        # 1. Gate composition per layer
        ax = axes[0, 0]
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        bottom = np.zeros(N_LAYERS)
        for k in range(N_MODES):
            ax.bar(range(N_LAYERS), gate_means[:, k], bottom=bottom,
                   color=colors[k % len(colors)], label=f'mode {k}', width=0.7)
            bottom += gate_means[:, k]
        ax.set_xlabel('Layer'); ax.set_ylabel('Gate weight')
        ax.set_title('Gate composition')
        ax.legend(fontsize=8)
        ax.set_xticks(range(N_LAYERS))

        # 2. Gate entropy
        ax = axes[0, 1]
        H_plot = [-(gate_means[l] * np.log(gate_means[l] + 1e-10)).sum() for l in range(N_LAYERS)]
        ax.plot(range(N_LAYERS), H_plot, 'o-', color='#e41a1c')
        ax.axhline(max_ent, color='gray', linestyle='--', label=f'max={max_ent:.2f}')
        ax.set_xlabel('Layer'); ax.set_ylabel('Entropy')
        ax.set_title('Gate entropy')
        ax.legend()
        ax.set_xticks(range(N_LAYERS))

        # 3. Cayley |A+B| norms
        ax = axes[0, 2]
        if cayley_norms_arr:
            ax.bar(range(N_LAYERS), cayley_norms_arr, color='#377eb8', width=0.7)
            ax.set_xlabel('Layer'); ax.set_ylabel('|A| + |B|')
            ax.set_title('Cayley update magnitude')
            ax.set_xticks(range(N_LAYERS))

        # 4. Adaptive depth pass rate
        ax = axes[1, 0]
        if pass_rates:
            ax.bar(range(N_LAYERS - 1), pass_rates[:-1], color='#4daf4a', width=0.7)
            ax.set_xlabel('Layer (entry)'); ax.set_ylabel('% tokens passing')
            ax.set_title('Adaptive depth pass rate')
            ax.set_xticks(range(N_LAYERS - 1))
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

        # 5. Hidden state norm
        ax = axes[1, 1]
        if h_norms_arr:
            ax.plot(range(N_LAYERS), h_norms_arr, 'o-', color='#984ea3')
            ax.set_xlabel('Layer'); ax.set_ylabel('||h|| mean')
            ax.set_title('Hidden state norm')
            ax.set_xticks(range(N_LAYERS))

        # 6. Cayley effective rank
        ax = axes[1, 2]
        if eff_ranks_arr:
            ax.bar(range(N_LAYERS), eff_ranks_arr, color='#ff7f00', width=0.7)
            ax.set_xlabel('Layer'); ax.set_ylabel('Effective rank')
            ax.set_title(f'Cayley S effective rank (r={model.stack.layers[0].r})')
            ax.axhline(model.stack.layers[0].r, color='gray', linestyle='--',
                       label=f'max r={model.stack.layers[0].r}')
            ax.legend()
            ax.set_xticks(range(N_LAYERS))

        plt.tight_layout()
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f'\nPlot saved: {args.save_plot}')
        if args.plot:
            plt.show()
    except ImportError:
        print('\nmatplotlib not available - install with: pip install matplotlib')
