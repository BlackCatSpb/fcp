"""
Phase 2: MemBind training with gradient accumulation + warmup.
--arch ld for legacy sigmoid architecture.
Full architecture description: LAMBDA_ARCHITECTURE.md
"""

import os, sys, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack, MemBindStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Argparse ────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='membind', choices=['ld', 'membind'],
                    help='Architecture: ld (sigmoid-gated) or membind (next-gen, no sigmoid)')
parser.add_argument('--data', default='russian',
                    choices=['wikitext', 'russian', 'auto'],
                    help='Dataset to use')
parser.add_argument('--train_chunks', type=int, default=None)
parser.add_argument('--eval_chunks', type=int, default=500)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--n_layers', type=int, default=24,
                    help='Number of layers (default 24 for scaling)')
parser.add_argument('--bottleneck', type=int, default=896,
                    help='MLP bottleneck dim (default = D for dense expansion)')
parser.add_argument('--seq_len', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--accum_steps', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup_frac', type=float, default=0.05)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--log_every', type=int, default=100)
parser.add_argument('--spectrum', default='fib_seq',
                    help='Spectrum type: fib_root, fib_seq, fib_ratio, linear, hybrid')
parser.add_argument('--spec_lo', type=float, default=0.8,
                    help='Lower bound for spectrum range')
parser.add_argument('--spec_hi', type=float, default=1.8,
                    help='Upper bound for spectrum range')
parser.add_argument('--n_modes', type=int, default=8,
                    help='Number of spectral modes / frequencies (K)')
parser.add_argument('--ckpt_every', type=int, default=1000,
                    help='Save checkpoint every N steps (0 to disable)')
args = parser.parse_args()

D = 896
VOCAB = 50000
N_MODES = args.n_modes
N_LAYERS = args.n_layers
BATCH_SIZE = args.batch_size
ACCUM_STEPS = args.accum_steps
SEQ_LEN = args.seq_len
LR = args.lr
WARMUP_FRAC = args.warmup_frac
EPOCHS = args.epochs
GRAD_CLIP = args.grad_clip
LOG_EVERY = args.log_every
BOTTLENECK = args.bottleneck
ARCH = args.arch
CKPT_DIR = 'checkpoints'

# ─── Load data ───────────────────────────────────────────────────────────
def choose_data(data_choice):
    if data_choice == 'auto':
        if os.path.exists('russian_chunks.npy'):
            return 'russian_chunks.npy'
        return 'wikitext_chunks.npy'
    return {'wikitext': 'wikitext_chunks.npy', 'russian': 'russian_chunks.npy'}[data_choice]

data_file = choose_data(args.data)
print(f'Data file: {data_file}')
t0 = time.perf_counter()
arr = np.load(data_file)
print(f'  Loaded {arr.shape[0]} chunks in {time.perf_counter()-t0:.1f}s')

N_CHUNKS_TRAIN = args.train_chunks if args.train_chunks else arr.shape[0] - args.eval_chunks
N_CHUNKS_EVAL = min(args.eval_chunks, arr.shape[0] - 10000) if args.train_chunks is None else args.eval_chunks

n_total = min(arr.shape[0], N_CHUNKS_TRAIN + N_CHUNKS_EVAL)
n_train = min(N_CHUNKS_TRAIN, n_total - N_CHUNKS_EVAL)
n_eval = min(N_CHUNKS_EVAL, n_total - n_train)

train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)
print(f'Train: {n_train} chunks ({n_train * SEQ_LEN / 1e6:.1f}M tok), '
      f'Eval: {n_eval} chunks ({n_eval * SEQ_LEN / 1e6:.1f}M tok)')

class ChunkDataset(torch.utils.data.Dataset):
    """Держит данные на CPU, отдаёт батчи на GPU по запросу."""
    def __init__(self, ids, seq_len):
        self.ids = ids  # (N, seq_len) на CPU
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return self.ids[idx]

def collate_chunk(batch):
    ids = torch.stack(batch).to(DEVICE)  # (B, seq_len) single GPU alloc
    return ids[:, :-1], ids[:, 1:]

train_loader = DataLoader(ChunkDataset(train_ids, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_chunk)
eval_loader = DataLoader(ChunkDataset(eval_ids, SEQ_LEN), batch_size=BATCH_SIZE,
                         collate_fn=collate_chunk)

# ─── Build config ─────────────────────────────────────────────────────────
cfg = LDConfig()
cfg.D = D
cfg.n_layers = N_LAYERS
cfg.n_modes = N_MODES
cfg.vocab = VOCAB
cfg.bottleneck = BOTTLENECK
cfg.kernel_size = 48
cfg.weight_tying = True
cfg.lm_head_bias = True
cfg.arch = ARCH
cfg.cov_heads = 4
cfg.cov_r = 16
cfg.bind_r = 16
cfg.spectrum_type = args.spectrum
cfg.spec_lo = args.spec_lo
cfg.spec_hi = args.spec_hi

# ─── Model ───────────────────────────────────────────────────────────────
class Phase2Model(nn.Module):
    def __init__(self, cfg, arch='membind'):
        super().__init__()
        self.arch = arch
        self.embed = nn.Embedding(VOCAB, D)
        nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))
        if arch == 'ld':
            self.stack = LDStack(cfg)
        else:
            self.stack = MemBindStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=True)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, return_gates=False):
        h = self.embed(input_ids)
        if self.arch == 'ld' and return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        h = self.stack(h)
        if self.arch == 'membind':
            h = h[0]  # MemBindStack returns (h, states)
        return self.lm_head(h)

model = Phase2Model(cfg, arch=ARCH).to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model: {n_all/1e6:.1f}M params ({n_t/1e6:.1f}M trainable, arch={ARCH})')
if ARCH == 'membind':
    sample_l = model.stack.layers[0].lambda_k
    sample_bs = model.stack.layers[0].block_sizes
    print(f'  Spectrum: {args.spectrum}  K={N_MODES}  lam in [{float(sample_l.min()):.4f},{float(sample_l.max()):.4f}]')
    print(f'  Blocks: {sample_bs}')

# ─── Checkpoint helpers ──────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, step, epoch, best_ppl, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl, 'stats': stats,
        'config': {
            'D': D, 'VOCAB': VOCAB, 'N_MODES': N_MODES,
            'N_LAYERS': N_LAYERS, 'BATCH_SIZE': BATCH_SIZE,
            'ACCUM_STEPS': ACCUM_STEPS, 'SEQ_LEN': SEQ_LEN, 'LR': LR,
            'EPOCHS': EPOCHS, 'ARCH': ARCH, 'BOTTLENECK': BOTTLENECK,
            'SPECTRUM': args.spectrum, 'SPEC_LO': args.spec_lo, 'SPEC_HI': args.spec_hi,
        }
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(ckpt, path)
    print(f'  [CKPT] Saved {path}')

def find_latest_ckpt():
    files = sorted(glob.glob(os.path.join(CKPT_DIR, 'phase2_*.pt')))
    return files[-1] if files else None

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt

# ─── Sanity ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx_test = next(iter(train_loader))[0][:1]
    h = model.embed(bx_test)
    print(f'  sanity: embed range=[{h.min():.4f},{h.max():.4f}]', flush=True)
    h, _ = model.stack(h)
    n_val, inf_val = torch.isnan(h).any().item(), torch.isinf(h).any().item()
    print(f'  sanity: stack range=[{h.min():.4f},{h.max():.4f}] nan={n_val} inf={inf_val}', flush=True)
    logits = model.lm_head(h)
    print(f'  sanity: logits range=[{logits.min():.4f},{logits.max():.4f}]', flush=True)

# ─── Training ────────────────────────────────────────────────────────────
total_steps = (n_train // BATCH_SIZE) * EPOCHS
warmup_steps = int(total_steps * WARMUP_FRAC)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def get_lr(step):
    if step < warmup_steps:
        return LR * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

step = 0
start_epoch = 0
best_ppl = float('inf')
t_start = time.perf_counter()

# Resume
ckpt_path = find_latest_ckpt()
if ckpt_path:
    print(f'Resuming from {ckpt_path}...')
    ckpt = load_checkpoint(ckpt_path, model, optimizer)
    step = ckpt['step']
    start_epoch = ckpt['epoch']
    best_ppl = ckpt['best_ppl']
    print(f'  Resumed step={step}, epoch={start_epoch}, best_ppl={best_ppl:.1f}')

# ─── Analysis helper ─────────────────────────────────────────────────────
@torch.no_grad()
def generate_report(step, loss, ppl, model, loader, ckpt_path):
    """Generate HTML report with model state analysis."""
    model.eval()
    # Collect stats from a small eval batch
    bx, by = next(iter(loader))
    h = model.embed(bx)
    h, states = model.stack(h)
    logits = model.lm_head(h)

    # Compute grad norms
    total_grad = 0.0
    n_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.norm().item() ** 2
            n_grad += 1
    grad_norm = math.sqrt(total_grad) if n_grad else 0.0

    # Embedding stats
    embed_norm = model.embed.weight.norm().item()

    # Parameter stats
    param_norms = []
    for name, p in model.named_parameters():
        param_norms.append((name, p.norm().item(), p.mean().item(), p.std().item()))

    # Build HTML
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MemBind Report - Step {step}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 0.3em; }}
h2 {{ color: #16213e; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 0.5em 0.8em; text-align: left; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
th {{ background: #1a1a2e; color: white; }}
tr:hover {{ background: #f0f0f0; }}
.metric {{ font-weight: bold; color: #e94560; }}
.good {{ color: #2ecc71; }}
.warn {{ color: #f39c12; }}
.bad {{ color: #e74c3c; }}
</style></head><body>
<h1>MemBind Training Report — Step {step}</h1>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Cross Entropy</td><td class="metric">{loss:.4f}</td></tr>
<tr><td>Perplexity</td><td class="metric">{ppl:.1f}</td></tr>
<tr><td>Learning Rate</td><td class="metric">{get_lr(step):.6f}</td></tr>
<tr><td>Gradient Norm</td><td class="metric">{grad_norm:.4f}</td></tr>
<tr><td>Embedding Norm</td><td>{embed_norm:.4f}</td></tr>
<tr><td>Checkpoint</td><td>{os.path.basename(ckpt_path)}</td></tr>
</table>
<h2>Logits</h2>
<table>
<tr><th>Stat</th><th>Value</th></tr>
<tr><td>Mean</td><td>{logits.mean().item():.4f}</td></tr>
<tr><td>Std</td><td>{logits.std().item():.4f}</td></tr>
<tr><td>Min</td><td>{logits.min().item():.4f}</td></tr>
<tr><td>Max</td><td>{logits.max().item():.4f}</td></tr>
</table>
<h2>Hidden State (stack output)</h2>
<table>
<tr><th>Stat</th><th>Value</th></tr>
<tr><td>Mean</td><td>{h.mean().item():.4f}</td></tr>
<tr><td>Std</td><td>{h.std().item():.4f}</td></tr>
<tr><td>Min</td><td>{h.min().item():.4f}</td></tr>
<tr><td>Max</td><td>{h.max().item():.4f}</td></tr>
</table>
<h2>Key Parameter Norms</h2>
<table>
<tr><th>Parameter</th><th>Norm</th><th>Mean</th><th>Std</th></tr>"""
    for name, n, m, s in param_norms:
        short = name.replace('stack.layers.','l').replace('stack.mlps.','m')
        html += f"<tr><td>{short}</td><td>{n:.4f}</td><td>{m:.6f}</td><td>{s:.6f}</td></tr>"
    html += """</table></body></html>"""

    report_path = os.path.join(CKPT_DIR, f'phase2_step{step}_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  [REPORT] Saved {report_path}')
    model.train()


print(f'\nTraining: {n_train//BATCH_SIZE} steps/epoch, {EPOCHS} epochs')
print(f'  total_steps={total_steps}, warmup_steps={warmup_steps}')

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for bx, by in train_loader:
        logits = model(bx)

        if torch.isnan(logits).any():
            print(f'  [NAN] logits at step {step}, skipping')
            continue

        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))

        if torch.isnan(loss):
            print(f'  [NAN] loss at step {step}, skipping')
            continue

        loss = loss / ACCUM_STEPS
        loss.backward()

        epoch_loss += loss.item() * ACCUM_STEPS
        n_batches += 1
        step += 1

        if step % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

        # Step-based checkpoint (outside ACCUM_STEPS guard — triggers on exact step)
        ckpt_every = args.ckpt_every
        if ckpt_every > 0 and step % ckpt_every == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'phase2_step{step}.pt')
            ppl_now = math.exp(epoch_loss / max(n_batches, 1))
            save_checkpoint(ckpt_path, model, optimizer, None, step, epoch+1, best_ppl,
                            {'train_loss': epoch_loss/max(n_batches,1), 'train_ppl': ppl_now})
            generate_report(step, epoch_loss/max(n_batches,1), ppl_now, model, train_loader, ckpt_path)

        if step % LOG_EVERY == 0:
            ppl = math.exp(epoch_loss / max(n_batches, 1))
            lr_now = optimizer.param_groups[0]['lr']
            v_info = ''
            if ARCH == 'ld' and hasattr(model.stack, 'cfg') and model.stack.cfg.learnable_V:
                norms = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                         for l in model.stack.layers if hasattr(l, 'V_cay_A') and l.V_cay_A is not None]
                v_info = f' |A+B|~[{", ".join(f"{n:.2f}" for n in norms)}]'
            print(f'  Step {step:5d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr_now:.2e}{v_info}')

    # Flush accumulated gradients at epoch end
    if step % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        train_ppl = math.exp(epoch_loss / n_batches)
    else:
        train_ppl = float('inf')
        print(f'  [WARN] No valid batches in epoch {epoch+1}')

    model.eval()
    eval_loss = 0.0
    all_gate_info = []
    with torch.no_grad():
        for bx, by in eval_loader:
            if ARCH == 'ld':
                logits, gates = model(bx, return_gates=True)
                H = -(gates * (gates + 1e-10).log()).sum(dim=-1).mean(dim=(1, 2))
                all_gate_info.append(H)
            else:
                logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
            eval_loss += loss.item()
    eval_ppl = math.exp(eval_loss / len(eval_loader))

    if ARCH == 'ld' and all_gate_info:
        avg_entropy = torch.stack(all_gate_info).mean(dim=0).cpu().tolist()
        max_entropy = math.log(N_MODES)
        print(f'  Gate entropy per layer: {[f"{e:.3f}" for e in avg_entropy]}')
        print(f'  Max possible H={max_entropy:.3f}, mean H={sum(avg_entropy)/len(avg_entropy):.3f}')

    print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')
    is_best = eval_ppl < best_ppl
    if is_best:
        best_ppl = eval_ppl
        save_checkpoint(os.path.join(CKPT_DIR, 'phase2_best.pt'),
                        model, optimizer, None, step, epoch+1, best_ppl,
                        {'train_ppl': train_ppl, 'eval_ppl': eval_ppl})
    save_checkpoint(os.path.join(CKPT_DIR, f'phase2_epoch{epoch+1}.pt'),
                    model, optimizer, None, step, epoch+1, best_ppl,
                    {'train_ppl': train_ppl, 'eval_ppl': eval_ppl, 'is_best': is_best})

print(f'\nTime: {time.perf_counter()-t_start:.0f}s')
print(f'Best eval PPL: {best_ppl:.1f}')
print('Phase 2 complete.')
