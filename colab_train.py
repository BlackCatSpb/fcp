"""
Colab training for Phase 2.
Optimised for free tier: Drive checkpointing, fp16 ckpts, auto-resume, streaming data.

Usage:
  python colab_train.py --drive /content/drive/MyDrive/lambda --data russian_chunks.npy
"""

import os, sys, math, time, glob, argparse, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
SEQ_LEN = 128
LR = 1e-3
WARMUP_FRAC = 0.05
EPOCHS = 3
GRAD_CLIP = 1.0
LOG_EVERY = 100
KEEP_CKPTS = 3           # keep only N most recent checkpoints

# ─── Model ───────────────────────────────────────────────────────────────
class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed(input_ids)))

# ─── Checkpoint utils ────────────────────────────────────────────────────
def save_light(path, model, optimizer, step, epoch, best_ppl):
    """fp16 weights only — ~190 MB."""
    sd = {k: v.half() if v.dtype == torch.float32 else v
          for k, v in model.state_dict().items()}
    torch.save({
        'model_fp16': sd,
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl,
    }, path)

def save_full(path, model, optimizer, step, epoch, best_ppl):
    """fp32 + optimizer — ~1.1 GB."""
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step, 'epoch': epoch, 'best_ppl': best_ppl,
    }, path)

def load_light(path, model):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if 'model_fp16' in ckpt:
        sd = {k: v.float() if v.dtype == torch.float16 else v
              for k, v in ckpt['model_fp16'].items()}
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    return ckpt.get('step', 0), ckpt.get('epoch', 0), ckpt.get('best_ppl', float('inf'))

def find_latest_ckpt(ckpt_dir):
    pattern = re.compile(r'model(?:_interrupt)?_step(\d+)\.pt')
    best = None
    best_num = -1
    if not os.path.exists(ckpt_dir):
        return None
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if not m and f == 'model_best.pt' and best is None:
            best = os.path.join(ckpt_dir, f)
        if m:
            n = int(m.group(1))
            if n > best_num:
                best_num = n
                best = os.path.join(ckpt_dir, f)
    return best

def clean_old_ckpts(ckpt_dir, keep=KEEP_CKPTS):
    """Delete all but the `keep` most recent checkpoints."""
    pattern = re.compile(r'model(?:_interrupt)?_step(\d+)\.pt')
    files = []
    if not os.path.exists(ckpt_dir):
        return
    for f in os.listdir(ckpt_dir):
        m = pattern.match(f)
        if m:
            files.append((int(m.group(1)), os.path.join(ckpt_dir, f)))
    files.sort(reverse=True)
    for _, path in files[keep:]:
        os.remove(path)

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints')
    parser.add_argument('--drive', default=None,
                        help='Google Drive mount path for persistent checkpoints')
    parser.add_argument('--data', default='russian_chunks.npy')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_chunks', type=int, default=None,
                        help='Limit chunks for testing (e.g. 50000)')
    args = parser.parse_args()

    CKPT_DIR = args.ckpt_dir
    os.makedirs(CKPT_DIR, exist_ok=True)
    DRIVE_DIR = args.drive
    if DRIVE_DIR:
        os.makedirs(DRIVE_DIR, exist_ok=True)

    # ─── Data ─────────────────────────────────────────────────────────
    print(f'Loading data: {args.data}')
    t0 = time.time()
    arr = np.load(args.data)
    print(f'  {arr.shape[0]} chunks, {time.time()-t0:.1f}s')

    n_total = arr.shape[0]
    if args.max_chunks:
        n_total = min(n_total, args.max_chunks)
    n_eval = min(500, n_total // 20)
    n_train = n_total - n_eval

    train_ids = torch.tensor(arr[:n_train], dtype=torch.long)
    eval_ids = torch.tensor(arr[n_train:n_train + n_eval], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(train_ids[:, :-1].to(DEVICE), train_ids[:, 1:].to(DEVICE)),
        batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(
        TensorDataset(eval_ids[:, :-1].to(DEVICE), eval_ids[:, 1:].to(DEVICE)),
        batch_size=args.batch_size)

    # ─── Model ────────────────────────────────────────────────────────
    model = Phase2Model().to(DEVICE)
    total_steps = (n_train // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def get_lr(step):
        if step < warmup_steps:
            return LR * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ─── Resume ───────────────────────────────────────────────────────
    step, epoch, best_ppl = 0, 0, float('inf')
    ckpt_path = None
    if DRIVE_DIR:
        ckpt_path = find_latest_ckpt(DRIVE_DIR)
        if ckpt_path is None:
            ckpt_path = find_latest_ckpt(CKPT_DIR)
    if ckpt_path is None:
        ckpt_path = find_latest_ckpt(CKPT_DIR)

    if ckpt_path:
        print(f'Resuming from {ckpt_path}...')
        step, epoch, best_ppl = load_light(ckpt_path, model)
        # Resume step one batch behind (will inc at first batch)
        print(f'  step={step}, epoch={epoch}, best_ppl={best_ppl:.1f}')
        # Copy to local for speed
        import shutil
        shutil.copy2(ckpt_path, os.path.join(CKPT_DIR, os.path.basename(ckpt_path)))

    # ─── Sanity ───────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        bx = next(iter(train_loader))[0][:1]
        h = model.embed(bx)
        h = model.stack(h)
        n, inf = torch.isnan(h).any().item(), torch.isinf(h).any().item()
        print(f'  sanity: stack [{h.min():.2f},{h.max():.2f}] nan={n} inf={inf}')

    # ─── Graceful interrupt ───────────────────────────────────────────
    import signal
    _saved = [False]
    def _on_sigint(sig, frame):
        if _saved[0]: return
        _saved[0] = True
        print('\n  [SIGINT] Saving...')
        fname = f'model_interrupt_step{step}.pt'
        save_light(os.path.join(CKPT_DIR, fname), model, optimizer, step, epoch, best_ppl)
        if DRIVE_DIR:
            import shutil
            shutil.copy2(os.path.join(CKPT_DIR, fname), os.path.join(DRIVE_DIR, fname))
        sys.exit(0)
    signal.signal(signal.SIGINT, _on_sigint)

    # ─── Train loop ───────────────────────────────────────────────────
    print(f'\nTrain: {n_train} chunks, Eval: {n_eval} chunks')
    print(f'Total steps: {total_steps}, warmup: {warmup_steps}')
    t_start = time.time()

    for epoch in range(epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))

            if torch.isnan(loss).item():
                print(f'  [NAN] step {step}, skipping')
                continue

            loss.backward()

            # Simple training loop (no grad accum — T4 has 16 GB, batch=8 is fine)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1
            n_batches += 1

            if step % LOG_EVERY == 0:
                ppl = math.exp(epoch_loss / n_batches)
                v_info = ''
                if model.stack.cfg.learnable_V:
                    norms = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                             for l in model.stack.layers if l.V_cay_A is not None]
                    v_info = f' |A+B|≈[{" ".join(f"{n:.2f}" for n in norms)}]'
                print(f'  Step {step:5d} | loss={epoch_loss/n_batches:.4f} | ppl={ppl:.1f} | lr={lr:.2e}{v_info}')

            # ── Checkpoint ──
            if step % 5000 == 0:
                fname = f'model_step{step}.pt'
                save_light(os.path.join(CKPT_DIR, fname), model, optimizer, step, epoch, best_ppl)
                if DRIVE_DIR:
                    import shutil
                    shutil.copy2(os.path.join(CKPT_DIR, fname), os.path.join(DRIVE_DIR, fname))
                clean_old_ckpts(CKPT_DIR)
                if DRIVE_DIR:
                    clean_old_ckpts(DRIVE_DIR)

        # ── Epoch end ──
        train_ppl = math.exp(epoch_loss / n_batches) if n_batches else float('inf')

        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for bx, by in eval_loader:
                loss = F.cross_entropy(model(bx).reshape(-1, VOCAB), by.reshape(-1))
                eval_loss += loss.item()
        eval_ppl = math.exp(eval_loss / len(eval_loader))

        print(f'>> Epoch {epoch+1}: train_ppl={train_ppl:.1f}, eval_ppl={eval_ppl:.1f}')

        # Save best
        is_best = eval_ppl < best_ppl
        if is_best:
            best_ppl = eval_ppl
            save_light(os.path.join(CKPT_DIR, 'model_best.pt'), model, optimizer, step, epoch+1, best_ppl)
            if DRIVE_DIR:
                import shutil
                shutil.copy2(os.path.join(CKPT_DIR, 'model_best.pt'), os.path.join(DRIVE_DIR, 'model_best.pt'))

        # Save epoch
        fname = f'model_epoch{epoch+1}.pt'
        save_light(os.path.join(CKPT_DIR, fname), model, optimizer, step, epoch+1, best_ppl)
        if DRIVE_DIR:
            import shutil
            shutil.copy2(os.path.join(CKPT_DIR, fname), os.path.join(DRIVE_DIR, fname))

    print(f'\nTime: {time.time()-t_start:.0f}s')
    print(f'Best eval PPL: {best_ppl:.1f}')
    return best_ppl

if __name__ == '__main__':
    main()
