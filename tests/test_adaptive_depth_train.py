"""Test adaptive gain training (gradient flow)."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack

cfg = LDConfig()
cfg.D = 128; cfg.n_layers = 4; cfg.n_modes = 4; cfg.vocab = 1000
cfg.bottleneck = 32; cfg.adaptive_gain = True

model = LDStack(cfg)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B, L = 4, 32
x = torch.randn(B, L, 128)

for step in range(10):
    opt.zero_grad()
    h, gates = model(x, return_gates=True)
    loss = h.norm() + gates.mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    gains = []
    for lidx in range(cfg.n_layers - 1):
        spread = gates[lidx].std(dim=-1)
        gains.append(f'{spread.mean():.3f}')

    print(f'step {step}: loss={loss.item():.3f} grad={grad_norm:.3f} gain=[{" ".join(gains)}]')

print('Training works.')
