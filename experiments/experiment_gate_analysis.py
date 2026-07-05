"""
Gate analysis on checkpoint model_step25000.pt.
Measures: entropy per layer, mode specialization, dominant modes.
"""
import torch, sys, math, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
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
x = torch.from_numpy(arr[:50, :128].copy()).long().to(DEVICE)

with torch.no_grad():
    logits, gates = model(x, return_gates=True)

K = N_MODES
H_per_layer = []; mode_weights = []
for l in range(N_LAYERS):
    g = gates[l]
    mode_mean = g.mean(dim=(0, 1))
    mode_weights.append(mode_mean.cpu().tolist())
    H = -(g * (g + 1e-10).log()).sum(dim=-1).mean().item()
    H_per_layer.append(H)

max_H = math.log(K)
print(f'\n=== Gate Analysis (step {ckpt["step"]}) ===')
print(f'Max possible entropy: {max_H:.3f}')
print(f'Layer | Mean H | Specialization | Mode weights')
for l in range(N_LAYERS):
    spec = max(mode_weights[l]) / sum(mode_weights[l]) * 100
    w = [f'{w:.3f}' for w in mode_weights[l]]
    print(f'  {l:2d}   | {H_per_layer[l]:.3f} | {spec:5.1f}%     | [{", ".join(w)}]')

mean_H = sum(H_per_layer) / len(H_per_layer)
print(f'\nMean entropy: {mean_H:.3f} / max={max_H:.3f} (ratio={mean_H/max_H:.3f})')
print(f'\nDominant mode per layer:')
for l in range(N_LAYERS):
    dom = np.argmax(mode_weights[l])
    print(f'  Layer {l:2d}: Mode {dom} ({mode_weights[l][dom]:.3f})')

# V-energy analysis (compare to earlier result at step 37500 of wikitext)
print(f'\n=== V-energy anisotropy check ===')
embed_w = model.embed.weight  # (VOCAB, D)
V0 = model.stack.layers[0].V  # (D, D)
h_embed = embed_w[:1000] @ V0.T  # project 1000 token embeddings into V-basis
energy_embed = (h_embed ** 2).sum(dim=0)
print(f'V-energy (embed, first 1000 tokens): min={energy_embed.min().item():.0f}, '
      f'max={energy_embed.max().item():.0f}, ratio={energy_embed.max()/energy_embed.min():.2f}')

# Last layer
V_last = model.stack.layers[-1].V
h_last = model.stack(model.embed(x[:1])) @ V_last.T
energy_last = (h_last[0, -1] ** 2)
print(f'V-energy (layer 11, last token): min={energy_last.min().item():.0f}, '
      f'max={energy_last.max().item():.0f}, ratio={energy_last.max()/energy_last.min():.2f}')
