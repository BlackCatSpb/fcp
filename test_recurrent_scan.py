"""Тест λ_d token recurrence: parallel vs recurrent_scan.
Быстрый: 300 шагов, замер экстраполяции."""
import os, sys, math, time, torch, numpy as np
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'  {torch.cuda.get_device_name(0)}, VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

class Phase2Model(torch.nn.Module):
    def __init__(self, recurrent=False):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512; cfg.kernel_size = 48
        cfg.recurrent_scan = recurrent
        cfg.weight_tying = True; cfg.lm_head_bias = True
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=cfg.lm_head_bias)
        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight
    def forward(self, x, return_gates=False):
        h = self.embed(x)
        if return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        return self.lm_head(h)

arr = np.load('russian_chunks.npy')
ids = torch.tensor(arr[:3000].copy(), dtype=torch.long)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(ids[:, :-1], ids[:, 1:]),
    batch_size=4, shuffle=True)

BATCH_SIZE, ACCUM_STEPS, TOTAL_STEPS = 4, 4, 300

def train_and_eval(recurrent, name):
    model = Phase2Model(recurrent=recurrent).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    step = 0
    t0 = time.perf_counter()
    print(f'\n--- {name} ---')

    while step < TOTAL_STEPS:
        for bx, by in loader:
            if step >= TOTAL_STEPS: break
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = F.cross_entropy(model(bx).reshape(-1, VOCAB), by.reshape(-1)) / ACCUM_STEPS
            loss.backward()
            step += 1
            if step % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f'  Step {step:3d} | loss={loss.item()*ACCUM_STEPS:.4f} | ppl={math.exp(loss.item()*ACCUM_STEPS):.1f} | {time.perf_counter()-t0:.0f}s', flush=True)

    dt = time.perf_counter() - t0
    print(f'  Train time: {dt:.0f}s ({TOTAL_STEPS/dt:.0f} steps/s)')

    # Extrapolation test
    model.eval()
    with torch.no_grad():
        for L in [128, 256, 512, 1024, 2048]:
            n = max(1, L // 128) + 1
            seq = ids[n*2:n*2+1, :L+1].to(DEVICE) if L < 128 else \
                  torch.cat([ids[i:i+1] for i in range(n*2, n*2+n)], dim=1)[0, :L+1].to(DEVICE)
            # Actually simpler: just use one long chunk
            bx, by = seq[:L].unsqueeze(0), seq[1:L+1].unsqueeze(0)
            loss = F.cross_entropy(model(bx).reshape(-1, VOCAB), by.reshape(-1))
            ppl = math.exp(loss.item())
            print(f'  Extrap L={L:>5}: PPL={ppl:.1f}')

train_and_eval(False, 'PARALLEL')
train_and_eval(True, 'RECURRENT SCAN')

print('\nDone.')
