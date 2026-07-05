"""
MemBind training: per-genre with early stopping.
Each genre trains until validation PPL plateaus, then moves to next.
"""

import os, sys, math, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack
from ld_model.readout import ZeckendorfReadout

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

parser = argparse.ArgumentParser()
parser.add_argument('--index', default='token_index.json')
parser.add_argument('--genre', default='all', help='Genre name or "all"')
parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint for the genre')
parser.add_argument('--seq_len', type=int, default=128)
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
parser.add_argument('--ckpt_every', type=int, default=1000, help='Save checkpoint + HTML report every N steps')
parser.add_argument('--eval_every', type=int, default=500, help='Eval every N steps')
parser.add_argument('--patience', type=int, default=3, help='Early stop after N evals without PPL improvement')
parser.add_argument('--eval_batches', type=int, default=50)
parser.add_argument('--val_split', type=float, default=0.05, help='Fraction of genre tokens for validation')
parser.add_argument('--spectrum', default='fib_seq')
parser.add_argument('--spec_lo', type=float, default=0.8)
parser.add_argument('--spec_hi', type=float, default=1.8)
parser.add_argument('--dct', action='store_true', help='Use DCT basis')
parser.add_argument('--slide', action='store_true', help='Use lambda sliding')
parser.add_argument('--no-first-moment', action='store_true', dest='no_first_moment', default=False)
parser.add_argument('--no-rf', action='store_true', dest='no_rf', default=False)
parser.add_argument('--rf_dim', type=int, default=64)
parser.add_argument('--no-multi-tau', action='store_true', dest='no_multi_tau', default=False)
parser.add_argument('--tau-lo', type=int, default=3, dest='tau_lo')
parser.add_argument('--tau-hi', type=int, default=200, dest='tau_hi')
parser.add_argument('--no-mirror', action='store_true', dest='no_mirror', default=False)
parser.add_argument('--factorized', action='store_true', default=False,
                    help='Use V_shared (DxK) + W_code (Kxd_out) weight factorization')
parser.add_argument('--factorized-K', type=int, default=0, dest='factorized_K',
                    help='Override Zeckendorf K (0=auto from VOCAB)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enable automatic mixed precision (FP16)')
parser.add_argument('--zeckendorf', action='store_true', default=False,
                    help='Use Zeckendorf tree readout instead of lm_head (D-space centroids)')
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

# ─── Load index ──────────────────────────────────────────────────────
INDEX_PATH = args.index
if not os.path.exists(INDEX_PATH):
    print(f'[ERROR] {INDEX_PATH} not found. Run build_index.py first.')
    sys.exit(1)

with open(INDEX_PATH) as f:
    index = json.load(f)
genres = sorted(index.keys())
total_tokens = sum(index[g]['length'] for g in genres)
print(f'Index: {len(genres)} genres, {total_tokens//1e6:.0f}M tokens')
for g in genres:
    e = index[g]
    print(f'  {g:15s}  {e["length"]//1e6:.0f}M tok  file={e["file"]}')
print()

# ─── Dataset (deterministic per-epoch) ─────────────────────────────────
class EpochDataset(Dataset):
    def __init__(self, arr, n_tokens, seq_len, rng_seed=42, offset=0):
        self.arr = arr
        self.offset = offset
        self.n_windows = n_tokens // seq_len
        self.seq_len = seq_len
        self.rng = np.random.RandomState(rng_seed)
        self.order = None
        self.reset()

    def reset(self):
        self.order = self.rng.permutation(self.n_windows).tolist()

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        window_idx = self.order[idx]
        start = self.offset + window_idx * self.seq_len
        chunk = self.arr[start:start + self.seq_len + 1].copy()
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long))


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
cfg.cov_first_moment = not args.no_first_moment
cfg.cov_rf = not args.no_rf
cfg.cov_rf_dim = args.rf_dim
cfg.cov_multi_timescale = not args.no_multi_tau
cfg.cov_tau_lo = args.tau_lo
cfg.cov_tau_hi = args.tau_hi
cfg.cov_mirror = not args.no_mirror
cfg.factorized = args.factorized

# Auto-compute factorized_K from vocab
if cfg.factorized:
    from ld_model.core import fibonacci_bases
    _fibs = fibonacci_bases(VOCAB)
    cfg.factorized_K = args.factorized_K if args.factorized_K > 0 else len(_fibs)
    print(f'Factorized: K={cfg.factorized_K} (vocab={VOCAB}, fib levels={len(_fibs)})')


class Phase2Model(nn.Module):
    def __init__(self, cfg, use_zeckendorf=False):
        super().__init__()
        self.factorized = cfg.factorized
        self.use_zeckendorf = use_zeckendorf

        if self.factorized:
            from ld_model.core import zeckendorf_codes
            K = cfg.factorized_K
            E_code = zeckendorf_codes(VOCAB)
            if E_code.shape[1] < K:
                E_code = F.pad(E_code, (0, K - E_code.shape[1]))
            elif E_code.shape[1] > K:
                E_code = E_code[:, :K]
            self.register_buffer('E_code', E_code)
            self.lm_head_bias = nn.Parameter(torch.zeros(VOCAB))
        else:
            self.embed = nn.Embedding(VOCAB, D)
            nn.init.uniform_(self.embed.weight, -1.0 / math.sqrt(D), 1.0 / math.sqrt(D))

        self.stack = MemBindStack(cfg)

        if use_zeckendorf:
            self.readout = ZeckendorfReadout(cfg)
            self.lm_head = None
        elif self.factorized:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(D, VOCAB, bias=True)
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        if self.factorized:
            h = F.embedding(input_ids, self.E_code).float() @ self.stack.V_shared.T
            h = self.stack(h)[0]
            if self.use_zeckendorf:
                return h
            return (h @ self.stack.V_shared) @ self.E_code.T + self.lm_head_bias
        else:
            h = self.embed(input_ids)
            h = self.stack(h)[0]
            if self.use_zeckendorf:
                return h
            return self.lm_head(h)

    def compute_loss(self, h, targets):
        if self.use_zeckendorf:
            B, L, D = h.shape
            log_probs = self.readout.log_probs_for_target(
                h.reshape(-1, D), targets.reshape(-1))
            return -log_probs.mean()
        elif self.factorized:
            logits = (h @ self.stack.V_shared) @ self.E_code.T + self.lm_head_bias
            return F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        else:
            return F.cross_entropy(self.lm_head(h).reshape(-1, VOCAB), targets.reshape(-1))


USE_AMP = args.amp and DEVICE.type == 'cuda'
USE_ZECK = args.zeckendorf

model = Phase2Model(cfg, use_zeckendorf=USE_ZECK).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params/1e6:.1f}M params')
if USE_AMP:
    print(f'AMP: enabled (FP16)')
if USE_ZECK:
    print(f'ZeckendorfReadout: enabled')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                              betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
scaler = GradScaler('cuda', enabled=USE_AMP)

os.makedirs('checkpoints', exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────
def get_lr(step, warmup_steps=WARMUP, peak_lr=LR):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    return peak_lr

def evaluate(model, loader, n_batches, debug=False):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            if debug and i == 0:
                print(f'  [EVAL DEBUG] x[0,:5]={x[0,:5].tolist()}  y[0,:5]={y[0,:5].tolist()}')
            with autocast('cuda', enabled=USE_AMP):
                if USE_ZECK:
                    hd = model(x)
                    loss = model.compute_loss(hd, y)
                else:
                    loss = F.cross_entropy(model(x).view(-1, VOCAB), y.view(-1))
            total_loss += loss.item()
            count += 1
    model.train()
    avg = total_loss / max(count, 1)
    return avg, math.exp(avg)


def generate_report(step, loss, ppl, model, loader, ckpt_path, genre):
    """HTML report with model state (mirrors train_phase2.py format)."""
    model.eval()
    bx, by = next(iter(loader))
    bx, by = bx.to(DEVICE), by.to(DEVICE)
    if model.factorized:
        V = model.stack.V_shared
        h = F.embedding(bx, model.E_code).float() @ V.T
        h = model.stack(h)[0]
        logits = h if model.use_zeckendorf else (h @ V) @ model.E_code.T + model.lm_head_bias
        embed_norm = model.E_code.norm().item()
    else:
        h = model.embed(bx)
        h = model.stack(h)[0]
        logits = h if model.use_zeckendorf else model.lm_head(h)
        embed_norm = model.embed.weight.norm().item()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = math.sqrt(grad_norm)

    param_rows = []
    for name, p in model.named_parameters():
        param_rows.append((name, p.norm().item(), p.mean().item(), p.std().item()))

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MemBind Report - {genre} Step {step}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 0.3em; }}
h2 {{ color: #16213e; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 0.5em 0.8em; text-align: left; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
th {{ background: #1a1a2e; color: white; }}
tr:hover {{ background: #f0f0f0; }}
.metric {{ font-weight: bold; color: #e94560; }}
</style></head><body>
<h1>MemBind Report — {genre} Step {step}</h1>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Cross Entropy</td><td class="metric">{loss:.4f}</td></tr>
<tr><td>Perplexity</td><td class="metric">{ppl:.1f}</td></tr>
<tr><td>Learning Rate</td><td class="metric">{get_lr(step):.6f}</td></tr>
<tr><td>Gradient Norm</td><td class="metric">{grad_norm:.4f}</td></tr>
<tr><td>Embedding Norm</td><td>{embed_norm:.4f}</td></tr>
<tr><td>Checkpoint</td><td>{os.path.basename(ckpt_path)}</td></tr>
</table>
<h2>{'Logits' if not model.use_zeckendorf else 'Hidden State'}</h2>
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
<tr><th>Parameter</th><th>Norm</th><th>Mean</th><th>Std</th></tr>'''
    for name, n, m, s in param_rows:
        short = name.replace('stack.layers.', 'l').replace('stack.mlps.', 'm')
        html += f'<tr><td>{short}</td><td>{n:.4f}</td><td>{m:.6f}</td><td>{s:.6f}</td></tr>'
    html += '</table></body></html>'

    report_path = ckpt_path.replace('.pt', '_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    model.train()


# ─── Overwrite tracking ──────────────────────────────────────────────
_prev_step_ckpt = None  # path to previous step checkpoint for overwrite

def save_step_ckpt(genre, step, model, opt, batch_idx=0):
    global _prev_step_ckpt
    path = f'checkpoints/{genre}_step{step}.pt'
    torch.save({'genre': genre, 'step': step, 'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, path)
    # Delete previous step checkpoint — keep only latest + best
    if _prev_step_ckpt is not None and os.path.exists(_prev_step_ckpt):
        os.remove(_prev_step_ckpt)
        print(f'  [CLEAN] removed {os.path.basename(_prev_step_ckpt)}')
    _prev_step_ckpt = path
    return path


def train_on_genre(genre_name, model, opt, step_counter):
    """Train on one genre until early stopping. Returns (step_counter, best_ppl)."""
    global _prev_step_ckpt
    _prev_step_ckpt = None

    entry = index[genre_name]
    filepath = entry['file']
    length = entry['length']

    print(f'\n{"="*50}')
    print(f'Genre: {genre_name}  file={filepath}  tokens={length//1e6:.0f}M')

    t0 = time.perf_counter()
    arr = np.memmap(filepath, dtype=np.int32, mode='r')
    print(f'  Loaded in {time.perf_counter()-t0:.1f}s')

    n_train = int(length * (1 - args.val_split))
    n_val = length - n_train
    n_train -= n_train % SEQ_LEN
    n_val -= n_val % SEQ_LEN
    print(f'  train={n_train//1e6:.0f}M  val={n_val//1e6:.0f}M')

    train_ds = EpochDataset(arr, n_train, SEQ_LEN, rng_seed=42)
    val_ds = EpochDataset(arr, n_val, SEQ_LEN, rng_seed=0, offset=n_train)
    train_loader = DataLoader(train_ds, BS, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, BS, shuffle=False, drop_last=True)

    EVAL_BATCHES = min(args.eval_batches, len(val_loader))
    batches_per_epoch = len(train_loader)
    print(f'  batches/epoch={batches_per_epoch}, val_batches={EVAL_BATCHES}')

    step = step_counter
    best_val_ppl = float('inf')
    no_improve_count = 0
    genre_t0 = time.perf_counter()

    # Resume from checkpoint
    resume_batch_idx = 0
    if args.resume:
        best_path = f'checkpoints/best_{genre_name}.pt'
        ckpt = None
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
        else:
            import glob
            step_ckpts = sorted(glob.glob(f'checkpoints/{genre_name}_step*.pt'))
            if step_ckpts:
                ckpt = torch.load(step_ckpts[-1], map_location=DEVICE, weights_only=True)
        if ckpt is not None:
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            step = ckpt['step'] + 1
            resume_batch_idx = (ckpt.get('batch_idx') or 0) + 1
            best_val_ppl = ckpt.get('best_ppl', float('inf'))
            print(f'  [RESUME] step={step-1} batch_idx={resume_batch_idx-1} best_ppl={best_val_ppl:.1f}')

    for local_epoch in range(1, 999):  # infinite until early stop
        train_ds.reset()
        epoch_loss = 0.0
        batch_count = 0
        opt.zero_grad()
        epoch_t0 = time.perf_counter()
        loader_iter = iter(train_loader)

        for batch_idx in range(batches_per_epoch):
            if batch_idx < resume_batch_idx:
                # Skip batches already processed before resume
                try: next(loader_iter)
                except: pass
                continue

            x, y = next(loader_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast('cuda', enabled=USE_AMP):
                hd = model(x)
                loss = model.compute_loss(hd, y) if USE_ZECK else F.cross_entropy(hd.view(-1, VOCAB), y.view(-1))
            scaler.scale(loss / ACCUM).backward()

            if (batch_idx + 1) % ACCUM == 0 or (batch_idx + 1) == batches_per_epoch:
                scaler.unscale_(opt)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                lr = get_lr(step)
                for pg in opt.param_groups:
                    pg['lr'] = lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            epoch_loss += loss.item() * ACCUM
            batch_count += 1
            step += 1

            if step % args.log_every == 0:
                avg = epoch_loss / max(batch_count, 1)
                real_loss = avg / ACCUM
                tok_s = (batch_count * BS * SEQ_LEN) / (time.perf_counter() - epoch_t0 + 1e-10)
                print(f'  [{genre_name}] S{step}: loss={avg:.3f} ppl={math.exp(avg):.0f} '
                      f'real_ppl={math.exp(real_loss):.0f} tok/s={tok_s:.0f} [{batch_count}/{batches_per_epoch}]')

            # Save step checkpoint + HTML report every ckpt_every steps
            if step % args.ckpt_every == 0:
                avg = epoch_loss / max(batch_count, 1)
                real_loss = avg / ACCUM
                ckpt_path = save_step_ckpt(genre_name, step, model, opt, batch_idx=batch_idx)
                generate_report(step, avg, math.exp(avg), model, val_loader, ckpt_path, genre_name)
                print(f'  [CKPT+REPORT] {os.path.basename(ckpt_path)}')

            # Eval with early stopping
            if step % args.eval_every == 0:
                val_loss, val_ppl = evaluate(model, val_loader, EVAL_BATCHES, debug=(step == args.eval_every))
                train_ppl = math.exp(epoch_loss / max(batch_count, 1))
                print(f'  === [{genre_name}] S{step}: train_ppl={train_ppl:.0f} '
                      f'val_ppl={val_ppl:.0f} '
                      f'real_train_ppl={math.exp(epoch_loss/max(batch_count,1)/ACCUM):.0f} ===')

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    no_improve_count = 0
                    ckpt = {'genre': genre_name, 'step': step, 'batch_idx': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'best_ppl': best_val_ppl}
                    torch.save(ckpt, f'checkpoints/best_{genre_name}.pt')
                    print(f'  [NEW BEST {genre_name}: ppl={best_val_ppl:.1f}]')
                else:
                    no_improve_count += 1
                    print(f'  [no improvement {no_improve_count}/{args.patience}]')
                    if no_improve_count >= args.patience:
                        print(f'  Early stopping {genre_name} (patience={args.patience})')
                        elapsed = time.perf_counter() - genre_t0
                        print(f'  Genre done: {elapsed:.0f}s, best PPL={best_val_ppl:.1f}')
                        return step, best_val_ppl

        resume_batch_idx = 0  # reset after first epoch
        epoch_time = time.perf_counter() - epoch_t0
        avg_loss = epoch_loss / max(batch_count, 1)
        tok_rate = batch_count * BS * SEQ_LEN / epoch_time
        print(f'  [{genre_name}] Epoch done: loss={avg_loss:.3f} ppl={math.exp(avg_loss):.0f} '
              f'({epoch_time:.0f}s, {tok_rate/1e3:.0f}K tok/s)')

    return step, best_val_ppl


# ─── Main training loop ──────────────────────────────────────────────
selected = [args.genre] if args.genre != 'all' else list(genres)
step = 0
g = None  # for interrupt handler
print(f'Training genres: {selected}\n')

try:
    for g in selected:
        if g not in index:
            print(f'[WARN] Unknown genre "{g}", skipping')
            continue
        step, best = train_on_genre(g, model, optimizer, step)
        print(f'  [{g}] finished with best PPL={best:.1f}')
except KeyboardInterrupt:
    print(f'\n[INTERRUPT] Saving checkpoint...')
    scaler.unscale_(optimizer)
    if g is not None:
        ckpt_path = save_step_ckpt(g, step, model, optimizer)
        torch.save({'genre': g, 'step': step, 'interrupted': True,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   f'checkpoints/interrupt_{g}_step{step}.pt')
        print(f'  Saved interrupt_{g}_step{step}.pt')

print(f'\nAll done.')
print(f'Total steps: {step}')
