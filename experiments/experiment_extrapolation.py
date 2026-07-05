"""Context extrapolation test: PPL at L=128, 256, 512, 1024."""
import torch, sys, math, numpy as np
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack
import torch.nn as nn, torch.nn.functional as F

D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig(); cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

model = Phase2Model().to(DEVICE)
ckpt = torch.load('checkpoints/model_step25000.pt', map_location=DEVICE, weights_only=True)
sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
model.load_state_dict(sd, strict=False)
model.eval()
print(f'Loaded step {ckpt["step"]}')

arr = np.load('russian_chunks.npy')
# Concatenate chunks to build longer sequence
lengths = [128, 256, 512, 1024]
n_concat = max(lengths) // 128 + 2
seq = torch.from_numpy(arr[:n_concat, :].flatten().copy()).long().to(DEVICE)
print(f'Loaded {n_concat} chunks = {len(seq)} tokens')

lengths = [128, 256, 512, 1024]
print(f'\n=== Context Extrapolation ===')
print(f'Length | PPL')
for L in lengths:
    if L + 1 > len(seq): break
    x = seq[:L].unsqueeze(0)
    y = seq[1:L+1].unsqueeze(0)
    with torch.no_grad():
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
    ppl = math.exp(loss.item())
    print(f'  {L:5d} | {ppl:.1f}')

print(f'\nTraining length was L=128. Model is recurrent (no position embedding)')
print(f'so extrapolation should work by design.')
