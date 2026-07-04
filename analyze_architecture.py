"""
analyze_architecture.py — Generate HTML report of MemBind architecture.
"""

import os, math, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack, MemBindBlock

D = 896
VOCAB = 50000
N_LAYERS = 24
BOTTLENECK = 896
COV_HEADS = 4
COV_R = 16
BIND_R = 16

cfg = LDConfig()
cfg.D = D
cfg.n_layers = N_LAYERS
cfg.n_modes = 4
cfg.vocab = VOCAB
cfg.bottleneck = BOTTLENECK
cfg.kernel_size = 48
cfg.weight_tying = True
cfg.lm_head_bias = True
cfg.arch = 'membind'
cfg.cov_heads = COV_HEADS
cfg.cov_r = COV_R
cfg.bind_r = BIND_R

layer = MemBindBlock(cfg, 0, torch.tensor([1.618, 1.839, 1.927, 1.966]))
layer_params = {n: p.numel() for n, p in layer.named_parameters()}

def p(name): return sum(v for n, v in layer_params.items() if name in n)

bind_params = p('W_u') + p('W_v') + p('W_out')
cov_kq = p('W_k') + p('W_q')
cov_gate = p('W_i') + p('W_decay') + p('b_i') + p('b_decay')
cov_read = p('W_read')
cov_fb = p('W_mem2v')
conv_params = p('conv')
mlp_up = D * BOTTLENECK + BOTTLENECK
mlp_down = BOTTLENECK * D + D
v_matrix = D * D

per_layer_train = bind_params + cov_kq + cov_gate + cov_read + cov_fb + conv_params + mlp_up + mlp_down
embed_params = VOCAB * D
lm_bias = VOCAB
v_matrices = N_LAYERS * v_matrix
total_trainable = embed_params + N_LAYERS * per_layer_train + lm_bias
total_frozen = v_matrices

S = dict(
    D=str(D), V=f'{VOCAB:,}', L=str(N_LAYERS), B=str(BOTTLENECK),
    H=str(COV_HEADS), R=str(COV_R), BR=str(BIND_R),
    bind=f'{bind_params:,}',
    cov_kq=f'{cov_kq:,}', cov_gate=f'{cov_gate:,}',
    cov_read=f'{cov_read:,}', cov_fb=f'{cov_fb:,}',
    conv=f'{conv_params:,}',
    mlp_up=f'{mlp_up:,}', mlp_down=f'{mlp_down:,}',
    v=f'{v_matrix:,}',
    per_layer=f'{per_layer_train:,}',
    L_per_layer=f'{N_LAYERS * per_layer_train:,}',
    embed=f'{embed_params:,}',
    frozen=f'{v_matrices:,}',
    bias=f'{lm_bias:,}',
    total_train=f'{total_trainable:,}',
    total_train_m=f'{total_trainable/1e6:.1f}',
    total_all_m=f'{(total_trainable + v_matrices)/1e6:.1f}',
    ptrain=f'{embed_params/total_trainable*100:.1f}',
    player=f'{N_LAYERS * per_layer_train/total_trainable*100:.1f}',
    pbias=f'{lm_bias/total_trainable*100:.1f}',
)

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EVA-Ai MemBind Architecture Analysis</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 20px; background: #0d1117; color: #e6edf3; }}
h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
h2 {{ color: #58a6ff; margin-top: 30px; }}
h3 {{ color: #79c0ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px 0; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
th {{ background: #161b22; color: #8b949e; }}
tr:nth-child(even) {{ background: #161b22; }}
tr.total {{ background: #1f2937; font-weight: bold; }}
.code {{ font-family: 'Courier New', monospace; background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; white-space: pre; font-size: 13px; line-height: 1.5; }}
.metric {{ display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 20px; margin: 6px; text-align: center; }}
.metric .val {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
.metric .lbl {{ font-size: 12px; color: #8b949e; }}
.grid {{ display: flex; flex-wrap: wrap; }}
</style>
</head>
<body>

<h1>EVA-Ai: MemBind Architecture Analysis</h1>
<p>Next-gen language model &mdash; no softmax, no sigmoid gates, no attention.</p>

<div class="grid">
<div class="metric"><div class="val">{S['total_train_m']}M</div><div class="lbl">Trainable Params</div></div>
<div class="metric"><div class="val">{S['total_all_m']}M</div><div class="lbl">Total Params</div></div>
<div class="metric"><div class="val">{S['D']}</div><div class="lbl">Hidden Dim (D)</div></div>
<div class="metric"><div class="val">{S['L']}</div><div class="lbl">Layers</div></div>
<div class="metric"><div class="val">{S['H']}</div><div class="lbl">Cov Heads</div></div>
<div class="metric"><div class="val">{S['R']}</div><div class="lbl">Cov Rank (r)</div></div>
<div class="metric"><div class="val">{S['V']}</div><div class="lbl">Vocabulary</div></div>
</div>

<h2>1. Architecture Overview</h2>
<p><b>MemBind</b> combines three distinct content-dependent mechanisms:</p>
<ul>
<li><b>Multi-head Covariance Memory</b> (H={S['H']} heads, r={S['R']} each): tracks input covariance over time via parallel associative scan. Each head has its own learned decay rate, creating multi-scale memory.</li>
<li><b>Bind Adaptation</b> (FCF-inspired bilinear u*v): replaces all gating with elementwise product of learned projections. No sigmoid, no softmax, no attention.</li>
<li><b>&gamma;-&lambda; Spectral Operator</b> (Fibonacci spectrum &lambda;_k): fixed content-independent recurrence via V&middot;diag(&lambda;)&middot;V^T. The same for every token.</li>
</ul>
<p>Key innovation: <b>Memory &rarr; Bind feedback loop</b>. Memory readout modulates the bind v signal: <code>v_enh = v + W_mem2v &middot; mem_sum</code>, creating direct feedback from stored past information to current token transformation.</p>

<h2>2. Per-Layer Parameter Breakdown</h2>
<table class="params">
<tr><th>Component</th><th>Params</th></tr>
<tr><td>Bind: W_u, W_v, W_out (D x bind_r each)</td><td style="text-align:right">{S['bind']}</td></tr>
<tr><td>Cov: W_k, W_q per head (H x D x r each)</td><td style="text-align:right">{S['cov_kq']}</td></tr>
<tr><td>Cov: input/decay gates (W_i, W_d, biases)</td><td style="text-align:right">{S['cov_gate']}</td></tr>
<tr><td>Cov: W_read (r x D per head)</td><td style="text-align:right">{S['cov_read']}</td></tr>
<tr><td>Cov: W_mem2v feedback (D x bind_r)</td><td style="text-align:right">{S['cov_fb']}</td></tr>
<tr><td>Spectral: V matrix (D x D, frozen)</td><td style="text-align:right">{S['v']}</td></tr>
<tr><td>Causal Conv1d (depthwise, k=48)</td><td style="text-align:right">{S['conv']}</td></tr>
<tr><td>MLP up / down (D x B, B x D)</td><td style="text-align:right">{S['mlp_up']} + {S['mlp_down']}</td></tr>
<tr class="total"><td><b>Total trainable per layer</b></td><td style="text-align:right"><b>{S['per_layer']}</b></td></tr>
</table>

<h3>Full Model</h3>
<table class="params">
<tr><th>Component</th><th>Params</th><th>% Trainable</th></tr>
<tr><td>Embedding (V x D, weight-tied)</td><td style="text-align:right">{S['embed']}</td><td style="text-align:right">{S['ptrain']}%</td></tr>
<tr><td>{S['L']} x MemBindBlock (trainable)</td><td style="text-align:right">{S['L_per_layer']}</td><td style="text-align:right">{S['player']}%</td></tr>
<tr><td>{S['L']} x V matrix (frozen)</td><td style="text-align:right">{S['frozen']}</td><td style="text-align:right">0%</td></tr>
<tr><td>LM Head bias</td><td style="text-align:right">{S['bias']}</td><td style="text-align:right">{S['pbias']}%</td></tr>
<tr class="total"><td><b>Trainable Total</b></td><td style="text-align:right"><b>{S['total_train']}</b></td><td style="text-align:right">100%</td></tr>
<tr class="total"><td><b>Total (incl. frozen)</b></td><td style="text-align:right"><b>{int(total_trainable + v_matrices):,}</b></td><td style="text-align:right">&mdash;</td></tr>
</table>

<h2>3. Memory / VRAM Estimation (fp32)</h2>
<table class="params">
<tr><th>Component</th><th>Size</th></tr>
<tr><td>Model parameters (trainable)</td><td style="text-align:right">{total_trainable*4/1e9:.2f} GB</td></tr>
<tr><td>Frozen V matrices</td><td style="text-align:right">{v_matrices*4/1e9:.2f} GB</td></tr>
<tr><td>AdamW states (2 x trainable)</td><td style="text-align:right">{total_trainable*2*4/1e9:.2f} GB</td></tr>
<tr><td>Gradients</td><td style="text-align:right">{total_trainable*4/1e9:.2f} GB</td></tr>
<tr class="total"><td><b>Total (w/o activations)</b></td><td style="text-align:right"><b>{(total_trainable*4 + total_trainable*8 + v_matrices*4)/1e9:.2f} GB</b></td></tr>
<tr><td>Estimated activations (B=4, L=128)</td><td style="text-align:right">~0.3-0.5 GB</td></tr>
<tr class="total"><td><b>Estimated total</b></td><td style="text-align:right"><b>~{((total_trainable*4 + total_trainable*8 + v_matrices*4 + 0.4e9))/1e9:.2f} GB</b></td></tr>
</table>
<p style="color:#f85149;">Warning: Requires ~2GB GPU VRAM. With B=2, estimated ~1.5-1.7 GB.</p>

<h2>4. Forward Pass Flow</h2>
<div class="code">
h[t-1]  -----------------┬------------------------------------------> h[t]
                         |
                     [Conv1d]   depthwise causal conv (k=48)
                         |
                 h + h_conv
                         |
                   [RMS Norm]
                         |
                    ----+----
                    |        |
           [Bind Projections]|
             u = h * W_u     |   [Spectral Operator]
             v = h * W_v     |     h_spec = V * diag(lambda) * V^T * h
                    |        |
       -------------+        |
       |   [Cov Memory]      |
       |   H heads:          |
       |   k = h * W_k       |
       |   q = h * W_q       |
       |   i = exp(h * W_i)  |
       |   d = sigma(h*W_d)  |
       |   M[t] = d*M + i*k^T*k  (parallel scan)
       |   mem = q * M * W_read |
       |            |           |
       |     mem_sum (sum over H)|
       |            |           |
       +------------+           |
                    |           |
         v_enh = v + W_mem2v * mem
         h_adapt = h + (u * v_enh) * W_out
                    |           |
                    +-----+-----+
                          |
                   h + h_spec    residual
                          |
                   [RMS Norm]
                          |
                    [MLP]
                   SiLU(h * W_up) * W_down
                          |
                    h + MLP    residual
</div>

<h2>5. Fibonacci Spectrum (&lambda;_k)</h2>
<table class="params">
<tr><th>k</th><th>&lambda;_k</th><th>Description</th></tr>
<tr><td>2</td><td>1.618</td><td>Golden ratio &mdash; &phi; = (1 + &radic;5)/2</td></tr>
<tr><td>3</td><td>1.839</td><td>Tribonacci &mdash; root of x^3 = x^2 + x + 1</td></tr>
<tr><td>4</td><td>1.927</td><td>Tetranacci &mdash; root of x^4 = x^3 + x^2 + x + 1</td></tr>
<tr><td>5</td><td>1.966</td><td>Pentanacci &mdash; root of x^5 = x^4 + x^3 + x^2 + x + 1</td></tr>
</table>

<h2>6. Comparison with Alternatives</h2>
<table class="params">
<tr><th>Architecture</th><th>PPL (5000)</th><th>tok/s</th><th>Quality</th><th>Speed</th></tr>
<tr><td>CovGate (sequential)</td><td>438</td><td>260</td><td>100%</td><td>1x</td></tr>
<tr><td>BindGate (no memory)</td><td>793</td><td>1175</td><td>55%</td><td>4.5x</td></tr>
<tr><td>PScan (parallel cov)</td><td>688</td><td>1212</td><td>64%</td><td>4.7x</td></tr>
<tr><td><b>MemBind (this)</b></td><td><b>466</b></td><td><b>962</b></td><td><b>94%</b></td><td><b>3.7x</b></td></tr>
</table>

<h2>7. Training Configuration</h2>
<table class="params">
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Hidden dim (D)</td><td>{S['D']}</td></tr>
<tr><td>Layers</td><td>{S['L']}</td></tr>
<tr><td>MLP bottleneck</td><td>{S['B']}</td></tr>
<tr><td>Cov heads (H)</td><td>{S['H']}</td></tr>
<tr><td>Cov rank per head (r)</td><td>{S['R']}</td></tr>
<tr><td>Bind rank</td><td>{S['BR']}</td></tr>
<tr><td>Vocab size</td><td>{S['V']}</td></tr>
<tr><td>Seq length</td><td>128</td></tr>
<tr><td>Batch size</td><td>4</td></tr>
<tr><td>Gradient accumulation</td><td>8</td></tr>
<tr><td>Effective batch</td><td>32</td></tr>
<tr><td>Weight tying</td><td>Yes</td></tr>
<tr><td>Weight init</td><td>U(-1/sqrt(D), 1/sqrt(D))</td></tr>
<tr><td>Optimizer</td><td>AdamW (WD=0.01)</td></tr>
<tr><td>LR schedule</td><td>Warmup + Cosine</td></tr>
<tr><td>Gradient clipping</td><td>1.0</td></tr>
</table>

<h2>8. Code Location</h2>
<div class="code">
ld_model/core.py:
  MemBindBlock          -- multi-head covariance + bind feedback + spectral
  MemBindStack          -- N x MemBindBlock + MLPs + final norm
  parallel_prefix_scan  -- Hillis-Steele O(log L) associative scan

train_phase2.py         -- training loop with --arch membind / ld
</div>

</body>
</html>
"""

with open('model_analysis.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'HTML saved: model_analysis.html ({os.path.getsize("model_analysis.html"):,} bytes)')
