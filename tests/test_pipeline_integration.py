"""Verify pipeline integration: learnable_V + adaptive_gain + global_context."""
import torch, sys, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 6

cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
cfg.vocab = VOCAB; cfg.bottleneck = 512
cfg.adaptive_gain = True
cfg.use_global_context = True

print(f'Defaults: learnable_V={cfg.learnable_V}, V_rank={cfg.V_rank}')
print(f'Defaults: adaptive_gain={cfg.adaptive_gain}, global_context={cfg.use_global_context}')

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

model = Model().to(DEVICE)
model.train()

ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model_sd = model.state_dict()
compat_sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
missed = model.load_state_dict(compat_sd, strict=False)
del ckpt, sd

v_params = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
print(f'Params: {sum(p.numel() for p in model.parameters()):,} total, {v_params:,} Cayley')
if missed.missing_keys:
    print(f'Missing (expected): {missed.missing_keys[:5]}')

arr = np.load('russian_chunks.npy')
x = torch.from_numpy(arr[:8, :128].copy()).long().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(5):
    opt.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), x.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    cayley_norms = [(l.V_cay_A.norm().item() + l.V_cay_B.norm().item())
               for l in model.stack.layers if l.V_cay_A is not None]
    print(f'step {step}: loss={loss.item():.2f} |A+B|={[f"{n:.4f}" for n in cayley_norms]}')

for name, p in model.named_parameters():
    if torch.isnan(p).any():
        print(f'NaN: {name}')
        break
else:
    print('No NaN. Pipeline integration OK.')
