"""Verify adaptive gain works with the actual checkpoint model."""
import torch, sys, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
cfg.vocab = VOCAB; cfg.bottleneck = 512
cfg.adaptive_gain = True

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x, return_gates=False):
        h = self.embed(x)
        if return_gates:
            return self.stack(h, return_gates=True)
        return self.stack(h)

model = Phase2Model().to(DEVICE)
model.eval()

ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model_sd = model.state_dict()
compat_sd = {}
for k, v in sd.items():
    if k in model_sd and model_sd[k].shape == v.shape:
        compat_sd[k] = v
    elif k in model_sd:
        print(f'skip {k}: shape mismatch {model_sd[k].shape} vs {v.shape}')
missed = model.load_state_dict(compat_sd, strict=False)
if missed.unexpected_keys:
    print(f'unexpected: {missed.unexpected_keys[:3]}')
if missed.missing_keys:
    print(f'missing (expected - new params): {missed.missing_keys[:5]}')
del ckpt, sd

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:2, :128].copy()).long().to(DEVICE)
h = model.embed(x)

with torch.no_grad():
    h_out, gates = model.stack(h, return_gates=True)

print(f'Output: {h_out.shape}')
print(f'Gates:  {gates.shape}')

print(f'\n=== Adaptive gain on checkpoint model ===')
for lidx in range(N_LAYERS):
    spread = gates[lidx].std(dim=-1)
    gain = spread.mean().item()
    print(f'  Layer {lidx}: spread={spread.mean():.3f} gain={gain:.3f}')
