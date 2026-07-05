"""Fast Cayley divergence test on random data, small dim.

Tests whether different layers learn different Cayley rotations
(divergence of R matrices across layers), analogous to V_delta divergence.
"""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
import torch.nn as nn

cfg = LDConfig()
cfg.D = 256; cfg.n_layers = 6; cfg.n_modes = 4; cfg.vocab = 50000
cfg.bottleneck = 64; cfg.adaptive_depth = False
cfg.learnable_V = True; cfg.V_rank = 16

model = LDStack(cfg)
model.train()
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
cayley_params = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
print(f'Cayley params: {cayley_params:,}')

lm_head = nn.Linear(256, 50000, bias=False)
opt = torch.optim.Adam(list(model.parameters()) + list(lm_head.parameters()), lr=1e-3)

x = torch.randint(0, 50000, (8, 32))

for step in range(200):
    opt.zero_grad()
    h = model(lm_head.weight[x])
    logits = lm_head(h)
    loss = nn.functional.cross_entropy(logits.view(-1, 50000), x.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 50 == 0:
        # Measure divergence: cos similarity between R matrices of adjacent layers
        coses = []
        for i in range(len(model.layers) - 1):
            with torch.no_grad():
                Ri = model.layers[i].compute_R()
                Rj = model.layers[i+1].compute_R()
                cos = (Ri.flatten() * Rj.flatten()).sum().item() / (Ri.norm().item() * Rj.norm().item() + 1e-10)
                coses.append(f'{cos:.3f}')
        cayley_norms = [f'{(l.V_cay_A.norm().item() + l.V_cay_B.norm().item()):.4f}' for l in model.layers]
        print(f'step {step:3d}: loss={loss.item():.3f} norms={cayley_norms}')
        print(f'  cos(adjacent R): [{", ".join(coses)}]')

print(f'\nFinal Cayley norms:')
for i, l in enumerate(model.layers):
    a_norm = l.V_cay_A.norm().item()
    b_norm = l.V_cay_B.norm().item()
    print(f'  L{i}: |A|={a_norm:.4f} |B|={b_norm:.4f} |A+B|={a_norm + b_norm:.4f}')
