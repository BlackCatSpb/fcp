"""
Distill ZeckendorfReadout from a trained Phase2 model.
Freezes the stack, trains only the tree-structured decoder.

Usage:
  python distill_zeckendorf.py --checkpoint checkpoints/phase2_best.pt
"""

import os, sys, math, time, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig
from ld_model.readout import ZeckendorfReadout

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 896
VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
BATCH_SIZE = 16
SEQ_LEN = 128
LR = 5e-4
EPOCHS = 5
N_CHUNKS_EVAL = 500

def main(ckpt_path, output_path='zeckendorf_readout.pt'):
    # ─── Load frozen model ────────────────────────────────────────────
    from train_phase2 import Phase2Model
    model = Phase2Model().to(DEVICE)
    train_ppl = float('inf')
    step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        step = ckpt['step']
        train_ppl = ckpt.get('stats', {}).get('train_ppl', float('inf'))
        print(f'Loaded {ckpt_path}: step={step}, train_ppl={train_ppl:.1f}')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ─── Zeckendorf readout ───────────────────────────────────────────
    cfg = LDConfig()
    cfg.D = D
    cfg.vocab = VOCAB
    readout = ZeckendorfReadout(cfg).to(DEVICE)

    # ─── Data ─────────────────────────────────────────────────────────
    print('Loading chunks...')
    arr = np.load('wikitext_chunks.npy')
    n_total = arr.shape[0]
    n_eval = min(N_CHUNKS_EVAL, n_total // 2)
    eval_ids = torch.tensor(arr[:n_eval], dtype=torch.long)
    train_ids = torch.tensor(arr[n_eval:], dtype=torch.long)

    eval_loader = DataLoader(
        TensorDataset(eval_ids[:, :-1].to(DEVICE), eval_ids[:, 1:].to(DEVICE)),
        batch_size=BATCH_SIZE)

    # ─── Extract hidden states ────────────────────────────────────────
    @torch.no_grad()
    def extract_hidden(loader):
        all_h = []
        all_y = []
        for bx, by in loader:
            h = model.embed(bx)
            h = model.stack(h)
            all_h.append(h.reshape(-1, D))
            all_y.append(by.reshape(-1))
        return torch.cat(all_h, dim=0), torch.cat(all_y, dim=0)

    print('Extracting hidden states...')
    train_h, train_y = extract_hidden(DataLoader(
        TensorDataset(train_ids[:, :-1].to(DEVICE), train_ids[:, 1:].to(DEVICE)),
        batch_size=BATCH_SIZE))
    eval_h, eval_y = extract_hidden(eval_loader)

    print(f'Train: {train_h.shape[0]} tokens, Eval: {eval_h.shape[0]} tokens')

    # ─── Validate Zeckendorf coverage ─────────────────────────────────
    valid_vocab = readout.codes.shape[0]
    print(f'Zeckendorf covers {valid_vocab}/{VOCAB} tokens ({(valid_vocab/VOCAB)*100:.1f}%)')
    # Filter tokens outside Zeckendorf range
    train_mask = train_y < valid_vocab
    eval_mask = eval_y < valid_vocab
    train_h, train_y = train_h[train_mask], train_y[train_mask]
    eval_h, eval_y = eval_h[eval_mask], eval_y[eval_mask]
    print(f'  Train valid: {train_h.shape[0]}, Eval valid: {eval_h.shape[0]}')

    # ─── Train readout ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(readout.parameters(), lr=LR, weight_decay=0.01)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        readout.train()
        perm = torch.randperm(train_h.shape[0])
        train_h, train_y = train_h[perm], train_y[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, train_h.shape[0], BATCH_SIZE):
            bh = train_h[i:i + BATCH_SIZE]
            by = train_y[i:i + BATCH_SIZE]

            log_probs = readout.forward_log_probs(bh)  # (B, V')
            loss = F.nll_loss(log_probs, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_ppl_z = math.exp(train_loss)

        # Eval
        readout.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for i in range(0, eval_h.shape[0], BATCH_SIZE):
                bh = eval_h[i:i + BATCH_SIZE]
                by = eval_y[i:i + BATCH_SIZE]
                log_probs = readout.forward_log_probs(bh)
                loss = F.nll_loss(log_probs, by)
                eval_loss += loss.item()
        eval_loss /= max(1, eval_h.shape[0] // BATCH_SIZE)
        eval_ppl_z = math.exp(eval_loss)

        # Compare with lm_head
        with torch.no_grad():
            lm_logits = model.lm_head(eval_h[:1024])
            lm_ppl = math.exp(F.cross_entropy(lm_logits, eval_y[:1024]).item())

        print(f'Epoch {epoch+1}: train_ppl={train_ppl_z:.1f}, eval_ppl={eval_ppl_z:.1f}, lm_head_ppl={lm_ppl:.1f}')

        if eval_ppl_z < best_loss:
            best_loss = eval_ppl_z
            torch.save({
                'readout_state_dict': readout.state_dict(),
                'eval_ppl': eval_ppl_z,
                'lm_head_ppl': lm_ppl,
                'cfg': {'D': D, 'vocab': VOCAB},
            }, output_path)
            print(f'  Saved {output_path} (eval_ppl={eval_ppl_z:.1f})')

    print(f'\nBest Zeckendorf eval PPL: {best_loss:.1f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/phase2_best.pt')
    parser.add_argument('--output', default='zeckendorf_readout.pt')
    args = parser.parse_args()
    main(args.checkpoint, args.output)
