"""End-to-end test: learnable V Cayley on real data with cross-entropy."""
import torch, sys, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 6

cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
cfg.vocab = VOCAB; cfg.bottleneck = 512
cfg.adaptive_depth = False
cfg.learnable_V = True; cfg.V_rank = 16

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        h = self.embed(x)
        return self.lm_head(self.stack(h))

model = Phase2Model().to(DEVICE)
model.train()

# Load checkpoint for warm-start
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model_sd = model.state_dict()
compat_sd = {}
for k, v in sd.items():
    if k in model_sd and model_sd[k].shape == v.shape:
        compat_sd[k] = v
model.load_state_dict(compat_sd, strict=False)
del ckpt, sd

print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
v_params = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
print(f'Cayley params: {v_params:,}')

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:200, :128].copy()).long().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

model.eval()
with torch.no_grad():
    logits = model(x[:16])
    loss0 = F.cross_entropy(logits.view(-1, VOCAB), x[:16].view(-1))
    ppl0 = torch.exp(loss0).item()
print(f'Before training: loss={loss0.item():.4f} ppl={ppl0:.1f}')
model.train()

for step in range(20):
    opt.zero_grad()
    logits = model(x[:16])
    loss = F.cross_entropy(logits.view(-1, VOCAB), x[:16].view(-1))
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 5 == 0:
        cayley_norms = [f'{(l.V_cay_A.norm().item() + l.V_cay_B.norm().item()):.4f}'
                       for l in model.stack.layers if l.V_cay_A is not None]
        ppl = torch.exp(loss).item()
        print(f'step {step:2d}: loss={loss.item():.4f} ppl={ppl:.1f} gn={gn:.4f} |A+B|={cayley_norms}')

for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NaN: {name}')
        break
else:
    print(f'\nNo NaN. PPL delta: {ppl0:.1f} -> {torch.exp(loss).item():.1f}')

# Verify orthogonality of Cayley rotation
l0 = model.stack.layers[0]
v = torch.randn(1, D, device=DEVICE)
with torch.no_grad():
    R = l0.compute_R()
    V_eff_T = R.T @ l0.V_T
    v_eff = v @ V_eff_T @ l0.V @ R  # v @ V_eff^T @ V_eff = v
    ratio = (v_eff / v.norm()).norm().item()
print(f'Cayley orthogonality check: ||V_eff·V_eff^T·v||/||v|| = {ratio:.6f} (target=1.0)')
