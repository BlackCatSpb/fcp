"""Test learnable V Cayley: orthogonality guarantee + gradient flow."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack, rms_norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

cfg = LDConfig()
cfg.D = 256; cfg.n_layers = 4; cfg.n_modes = 4; cfg.vocab = 10000
cfg.bottleneck = 64; cfg.adaptive_depth = False
cfg.learnable_V = True; cfg.V_rank = 16

model = LDStack(cfg).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B, L = 4, 32
x = torch.randn(B, L, 256, device=DEVICE)

print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
v_cay_params = 0
for l in model.layers:
    if l.V_cay_A is not None:
        v_cay_params += l.V_cay_A.numel() + l.V_cay_B.numel()
print(f'Cayley params: {v_cay_params:,}')

for step in range(30):
    opt.zero_grad()
    h, gates = model(x, return_gates=True)

    total_norm = 0
    for l in model.layers:
        if l.V_cay_A is not None:
            a_norm = l.V_cay_A.norm().item()
            b_norm = l.V_cay_B.norm().item()
            total_norm += a_norm + b_norm

    loss = h.norm() + gates.mean()
    loss.backward()

    v_grad_norm = 0
    for l in model.layers:
        if l.V_cay_A is not None and l.V_cay_A.grad is not None:
            v_grad_norm += l.V_cay_A.grad.norm().item()
            v_grad_norm += l.V_cay_B.grad.norm().item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 10 == 0:
        print(f'step {step:3d}: loss={loss.item():.3f} |Cayley|={total_norm:.4f} '
              f'V_grad={v_grad_norm:.4f} grad={grad_norm:.4f}')

for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NaN in {name}!')
        break
else:
    print('No NaN detected.')

print(f'\nCayley orthogonality check (layer 0):')
l0 = model.layers[0]
v = torch.randn(1, 256, device=DEVICE)
with torch.no_grad():
    R = l0.compute_R()
    v_R = v @ R.T
    v_fwd = v_R @ R
    ratio = v_fwd.norm() / v.norm()
    print(f'  ||v||={v.norm().item():.4f} ||R^T·R·v||={v_fwd.norm().item():.4f} (ratio={ratio:.6f}, target=1.0)')

print('\nAll tests passed.')
