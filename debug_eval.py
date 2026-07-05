import json, numpy as np, torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import Dataset, DataLoader
from ld_model.core import LDConfig, MemBindStack

DEVICE = 'cuda'
S, BS, V = 128, 2, 0.05

ckpt = torch.load('checkpoints/best_ACTION.pt', map_location=DEVICE, weights_only=True)
sd = ckpt['model_state_dict']

idx = json.load(open('token_index.json'))
e = idx['ACTION']
arr = np.memmap(e['file'], dtype=np.int32, mode='r')
l = e['length']; nt = int(l*(1-V)); nv = l-nt
nt -= nt%S; nv -= nv%S
print(f'train offset=0 count={nt}  val offset={nt} count={nv}')

class EpochDataset(Dataset):
    def __init__(self, arr, n_tokens, seq_len, rng_seed=42, offset=0):
        self.arr = arr; self.offset = offset
        self.n_windows = n_tokens // seq_len; self.seq_len = seq_len
        self.rng = np.random.RandomState(rng_seed)
        self.order = self.rng.permutation(self.n_windows).tolist()
    def __len__(self): return self.n_windows
    def __getitem__(self, idx):
        start = self.offset + self.order[idx] * self.seq_len
        chunk = self.arr[start:start+self.seq_len+1].copy()
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))

print('Building model...')
cfg = LDConfig()
cfg.D = 896; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = 50000
cfg.bottleneck = 896; cfg.kernel_size = 48
cfg.weight_tying = True; cfg.lm_head_bias = True; cfg.arch = 'membind'
cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
cfg.dct_basis = True; cfg.lambda_sliding = True
cfg.cov_first_moment = True; cfg.cov_rf = True; cfg.cov_rf_dim = 64
cfg.cov_multi_timescale = True; cfg.cov_tau_lo = 3; cfg.cov_tau_hi = 200
cfg.cov_mirror = True

embed = torch.nn.Embedding(50000, 896).to(DEVICE)
stack = MemBindStack(cfg).to(DEVICE)
lm_head = torch.nn.Linear(896, 50000, bias=True).to(DEVICE)
embed.weight.data.copy_(sd['embed.weight'])
lm_head.weight.data.copy_(sd['lm_head.weight'])
lm_head.bias.data.copy_(sd['lm_head.bias'])
stack.load_state_dict({k:v for k,v in sd.items() if k.startswith('stack.')}, strict=False)
model = lambda x: lm_head(stack(embed(x))[0])

val_ds = EpochDataset(arr, nv, S, rng_seed=0, offset=nt)
val_loader = DataLoader(val_ds, BS, shuffle=False, drop_last=True)

print('Evaluating val...')
total_loss = 0.0; count = 0
with torch.no_grad():
    for i, (x, y) in enumerate(val_loader):
        if i >= 50: break
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = torch.nn.functional.cross_entropy(model(x).view(-1, 50000), y.view(-1))
        total_loss += loss.item(); count += 1
        if i < 5:
            first_logits = model(x)[0,0,:10].tolist()
            print(f'  batch {i}: loss={loss.item():.4f}  first_logits[0,:10]={[f"{v:.1f}" for v in first_logits]}')
avg = total_loss / max(count, 1)
print(f'\nEval result: avg_loss={avg:.4f} ppl={math.exp(avg):.1f}')
