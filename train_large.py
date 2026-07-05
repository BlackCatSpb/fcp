"""
Large-scale MemBind training on full Russian corpus.
Deterministic per-epoch pass: every token seen exactly once, no random skip.
"""

import os, sys, math, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='token_stream.npy')
parser.add_argument('--seq_len', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--accum_steps', type=int, default=16)
parser.add_argument('--d_model', type=int, default=896)
parser.add_argument('--n_layers', type=int, default=24)
parser.add_argument('--n_modes', type=int, default=8)
parser.add_argument('--bottleneck', type=int, default=896)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--log_every', type=int, default=50)
parser.add_argument('--ckpt_every', type=int, default=500)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--eval_tokens', type=int, default=5_000_000)
parser.add_argument('--eval_batches', type=int, default=50)
parser.add_argument('--spectrum', default='fib_seq')
parser.add_argument('--spec_lo', type=float, default=0.8)
parser.add_argument('--spec_hi', type=float, default=1.8)
parser.add_argument('--dct', action='store_true', help='Use DCT basis')
parser.add_argument('--slide', action='store_true', help='Use lambda sliding')
args = parser.parse_args()

D = args.d_model
VOCAB = 50000
N_MODES = args.n_modes
N_LAYERS = args.n_layers
BS = args.batch_size
ACCUM = args.accum_steps
SEQ_LEN = args.seq_len
LR = args.lr
WARMUP = args.warmup_steps
EPOCHS = args.epochs

# ─── Load token stream ────────────────────────────────────────────────
DATA_PATH = args.data
if not os.path.exists(DATA_PATH):
    print(f'[ERROR] {DATA_PATH} not found. Run prepare_corpus.py first.')
    sys.exit(1)

print(f'Loading {DATA_PATH}...')
t0 = time.perf_counter()
arr = np.load(DATA_PATH, mmap_mode='r')
total_tokens = len(arr)
n_eval_tok = min(args.eval_tokens, total_tokens // 20)
n_train_tok = total_tokens - n_eval_tok
print(f'  {total_tokens//1e6:.0f}M tokens, '
      f'train={n_train_tok//1e6:.0f}M eval={n_eval_tok//1e6:.0f}M '
      f'({time.perf_counter()-t0:.1f}s)')

# Pre-compute non-overlapping windows
train_windows = n_train_tok // SEQ_LEN
eval_windows = n_eval_tok // SEQ_LEN
print(f'Train windows: {train_windows}, Eval windows: {eval_windows}')

# Shuffle indices ONCE at epoch start (done in dataset __init__)
class EpochDataset(Dataset):
    """Deterministic dataset: every window seen exactly once per epoch."""
    def __init__(self, arr, n_tokens, seq_len, rng_seed=42):
        self.arr = arr
        self.n_windows = n_tokens // seq_len
        self.seq_len = seq_len
        self.rng = np.random.RandomState(rng_seed)
        self.order = None
        self.reset()

    def reset(self):
        """Shuffle window order. Call at epoch start."""
        self.order = self.rng.permutation(self.n_windows).tolist()

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        window_idx = self.order[idx]
        start = window_idx * self.seq_len
        chunk = self.arr[start:start + self.seq_len + 1].copy()
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


train_data = EpochDataset(arr[:n_train_tok], n_train_tok, SEQ_LEN, rng_seed=42)
eval_data = EpochDataset(arr[n_train_tok:n_train_tok + n_eval_tok],
                         n_eval_tok, SEQ_LEN, rng_seed=0)

train_loader = DataLoader(train_data, batch_size=BS, shuffle=False,
                          drop_last=True, pin_memory=False)
eval_loader = DataLoader(eval_data, batch_size=BS, shuffle=False,
                         drop_last=True, pin_memory=False)

BATCHES_PER_EPOCH = len(train_loader)
EVAL_BATCHES = min(args.eval_batches, len(eval_loader))
print(f'Batches/epoch: {BATCHES_PER_EPOCH}')
print(f'Eval batches:  {EVAL_BATCHES}')

# ─── Model ─────────────────────────────────────────────────────────────
cfg = LDConfig()
cfg.D = D
cfg.n_layers = N_LAYERS
cfg.n_modes = N_MODES
cfg.vocab = VOCAB
cfg.bottleneck = args.bottleneck
cfg.kernel_size = 48
cfg.weight_tying = True
cfg.lm_head_bias = True
cfg.arch = 'membind'
cfg.cov_heads = 4
cfg.cov_r = 16
cfg.bind_r = 16
cfg.spectrum_type = args.spectrum
cfg.spec_lo = args.spec_lo
cfg.spec_hi = args.spec_hi
cfg.dct_basis = args.dct
cfg.lambda_sliding = args.slide


class Phase2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        self.stack = MemBindStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=True)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        h = self.embed(input_ids)
        h = self.stack(h)[0]
        return self.lm_head(h)


model = Phase2Model(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params/1e6:.1f}M params')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                              betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

os.makedirs('checkpoints', exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────
def get_lr(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    total = BATCHES_PER_EPOCH * EPOCHS
    progress = (step - WARMUP) / max(total - WARMUP, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def evaluate(loader, n_batches):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
            total_loss += loss.item()
    model.train()
    avg = total_loss / n_batches
    return avg, math.exp(avg)


# ─── Training ─────────────────────────────────────────────────────────
total_steps = BATCHES_PER_EPOCH * EPOCHS
print(f'Total steps: {total_steps}')

step = 0
best_ppl = float('inf')
train_t0 = time.time()
ckpt_dir = 'checkpoints'

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_data.reset()  # new shuffle
    epoch_loss = 0.0
    batch_count = 0
    optimizer.zero_grad()
    epoch_t0 = time.perf_counter()

    # Generator over all batches
    loader_iter = iter(train_loader)
    for batch_idx in range(BATCHES_PER_EPOCH):
        x, y = next(loader_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        loss = loss / ACCUM
        loss.backward()

        if (batch_idx + 1) % ACCUM == 0 or (batch_idx + 1) == BATCHES_PER_EPOCH:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * ACCUM
        batch_count += 1
        step += 1

        if step % args.log_every == 0:
            avg = epoch_loss / batch_count
            tok_s = (batch_count * BS * SEQ_LEN) / (time.perf_counter() - epoch_t0 + 1e-10)
            print(f'  E{epoch} S{step}: loss={avg:.3f} ppl={math.exp(avg):.0f} '
                  f'lr={get_lr(step):.2e} tok/s={tok_s:.0f} '
                  f'[{batch_count}/{BATCHES_PER_EPOCH}]')

        # Eval
        if step % args.ckpt_every == 0:
            eval_loss, eval_ppl = evaluate(eval_loader, EVAL_BATCHES)
            train_ppl = math.exp(epoch_loss / max(batch_count, 1))
            print(f'  === EVAL E{epoch} S{step}: train_ppl={train_ppl:.0f} '
                  f'eval_ppl={eval_ppl:.0f} ===')

            if eval_ppl < best_ppl:
                best_ppl = eval_ppl
                ckpt = {'step': step, 'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': args.__dict__}
                torch.save(ckpt, os.path.join(ckpt_dir, 'best.pt'))
                print(f'  [NEW BEST ppl={best_ppl:.1f}]')

            torch.save({'step': step, 'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': args.__dict__},
                       os.path.join(ckpt_dir, f'step_{step}.pt'))
            print(f'  [CKPT] step_{step}.pt')

    # Epoch done
    epoch_time = time.perf_counter() - epoch_t0
    avg_loss = epoch_loss / max(batch_count, 1)
    print(f'Epoch {epoch}: loss={avg_loss:.3f} ppl={math.exp(avg_loss):.0f} '
          f'({epoch_time:.0f}s, '
          f'{batch_count*BS*SEQ_LEN/epoch_time/1e3:.0f}K tok/s)')

    torch.save({'step': step, 'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': args.__dict__},
               os.path.join(ckpt_dir, f'epoch_{epoch}.pt'))
    print(f'  [EPOCH] epoch_{epoch}.pt')

print(f'Done in {time.perf_counter()-train_t0:.0f}s')
