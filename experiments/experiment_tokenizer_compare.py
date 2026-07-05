"""
Standalone: compare ruadapt (Qwen3, 146K) vs Mistral (131K) tokenizers
on Russian text for λ_d architecture.

Measures:
  - Compression (tokens/chunk, tokens/word)
  - Short fine-tune PPL trajectory (← tests λ_d compatibility)

Does NOT modify any existing files or architecture.
"""

import os, sys, math, time, json
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ─── Config ───────────────────────────────────────────────────────────────
D = 256              # reduced for MX550 2GB (large vocabs = big embed)
N_MODES = 4
N_LAYERS = 12
SEQ_LEN = 64
SAMPLE_CHUNKS = 500      # for tokenizer analysis
TRAIN_CHUNKS = 500       # for fine-tune
EVAL_CHUNKS = 100        # for eval
FT_STEPS = 50            # fine-tune steps
FT_LR = 1e-4
TRAIN_BS = 8            # batch for fine-tune

# ─── Model ───────────────────────────────────────────────────────────────
class Phase2Model(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = vocab_size; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.embed.weight.normal_(0, 0.02)
            self.lm_head.weight.normal_(0, 0.02)

    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed(input_ids)))


# ─── Tokenizer loader ────────────────────────────────────────────────────
def load_hf_tokenizer(path_or_name: str):
    """Load HF tokenizer from path or HF hub name."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
        return tok
    except Exception as e:
        print(f'  HF load failed ({e}), trying raw tokenizers...')
        # Fallback for local ruadapt tokenizer (tokenizers.json)
        if os.path.isdir(path_or_name):
            json_path = os.path.join(path_or_name, 'tokenizer.json')
            if os.path.exists(json_path):
                from tokenizers import Tokenizer as HFTokenizer
                tok_raw = HFTokenizer.from_file(json_path)
                # Wrap in a minimal compat layer
                class Wrap:
                    def __init__(self, t):
                        self.t = t
                        self.vocab_size = t.get_vocab_size()
                    def encode(self, text):
                        return self.t.encode(text).ids
                    def decode(self, ids):
                        return self.t.decode(ids)
                    def __call__(self, text, **kw):
                        ids = self.t.encode(text).ids
                        return {'input_ids': [ids]}
                return Wrap(tok_raw)
        raise


def sample_russian_text(n_chunks=1000):
    """Sample Russian text from russian_chunks.npy via our 50K tokenizer."""
    from tokenizers import Tokenizer as HFTokenizer
    our_tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
    arr = np.load('russian_chunks.npy', mmap_mode='r')
    total = min(arr.shape[0], 200000)
    indices = np.random.choice(total, n_chunks, replace=False)
    texts = []
    for idx in indices:
        chunk = arr[idx].tolist()
        text = our_tok.decode([t for t in chunk if t < 50000])
        texts.append(text)
    return texts, indices


def analyze_tokenizer(name, texts, tok, vocab_size, max_samples=500):
    """Measure compression metrics for a tokenizer."""
    total_tokens = 0
    total_chars = 0
    n_analyzed = 0
    for text in texts[:max_samples]:
        ids = tok.encode(text).ids
        total_tokens += len(ids)
        total_chars += len(text)
        n_analyzed += 1

    chars_per_token = total_chars / max(total_tokens, 1)
    tokens_per_chunk = total_tokens / max(n_analyzed, 1)
    print(f'  [{name}] vocab={vocab_size}, '
          f'{tokens_per_chunk:.0f} tok/chunk, '
          f'{chars_per_token:.2f} char/tok, '
          f'{total_tokens} total tok')

    return {
        'name': name,
        'vocab': vocab_size,
        'total_tokens': total_tokens,
        'tokens_per_chunk': tokens_per_chunk,
        'chars_per_token': chars_per_token,
        'n_chunks': n_analyzed,
    }


# ─── Data loader builder ─────────────────────────────────────────────────
def build_data(texts, seq_len, tok, vocab_size):
    """Tokenize texts and build flat list of (in, out) sequences."""
    all_x, all_y = [], []
    skipped = 0
    for text in texts:
        ids = tok.encode(text).ids
        ids = [min(i, vocab_size - 1) for i in ids]
        if len(ids) > seq_len:
            for i in range(0, len(ids) - seq_len, seq_len // 2):
                chunk = ids[i:i+seq_len+1]
                if len(chunk) > seq_len:
                    all_x.append(chunk[:seq_len])
                    all_y.append(chunk[1:seq_len+1])
        elif len(ids) >= 16:
            padded = (ids + [0] * (seq_len + 1 - len(ids)))[:seq_len+1]
            all_x.append(padded[:seq_len])
            all_y.append(padded[1:seq_len+1])
        else:
            skipped += 1

    if not all_x:
        return None, 0, 0

    t = torch.tensor(all_x, dtype=torch.long)
    u = torch.tensor(all_y, dtype=torch.long)
    ds = TensorDataset(t, u)
    loader = DataLoader(ds, batch_size=TRAIN_BS, shuffle=True)
    return loader, len(all_x), len(loader)


def eval_ppl(model, loader, vocab_size):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(DEVICE); by = by.to(DEVICE)
            logits = model(bx)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), by.reshape(-1))
            total_loss += loss.item()
            n += 1
    return math.exp(total_loss / max(n, 1)) if n else float('inf')


# ─── Main ────────────────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('Tokenizer comparison: ruadapt (Qwen3, 146K) vs Mistral (131K)')
    print('=' * 60)

    # 1. Load tokenizers
    print('\n=== 1. Load tokenizers ===')
    from tokenizers import Tokenizer as HFTokenizer

    # Our baseline
    our_tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
    VOCAB_OUR = 50000

    # ruadapt (local archive)
    ruadapt_raw = HFTokenizer.from_file(
        '_archive/models/ruadapt_qwen3_4b_openvino_ModelB/tokenizer.json')
    VOCAB_RU = ruadapt_raw.get_vocab_size()  # 151678
    print(f'  ruadapt: {VOCAB_RU} vocab')
    ruadapt_tok = ruadapt_raw

    # Mistral (131K via Nemo — download from HF)
    print('  Downloading Mistral tokenizer (Nemo, 131K)...')
    try:
        mistral_raw = HFTokenizer.from_pretrained('mistralai/Mistral-Nemo-Base-2407')
        VOCAB_MISTRAL = mistral_raw.get_vocab_size()
        print(f'  Mistral Nemo: {VOCAB_MISTRAL} vocab')
        mistral_tok = mistral_raw
    except Exception as e:
        print(f'  Mistral download failed: {e}')
        print('  Falling back to Qwen 2.5 tokenizer (151K)')
        mistral_raw = HFTokenizer.from_file(
            'eva_ai/mlearning/eva_models/qwen2.5-0.5b/tokenizer.json')
        VOCAB_MISTRAL = mistral_raw.get_vocab_size()
        mistral_tok = mistral_raw

    # 2. Sample Russian text
    print('\n=== 2. Sample Russian text ===')
    arr = np.load('russian_chunks.npy', mmap_mode='r')
    total_chunks = min(arr.shape[0], 200000)
    sample_indices = np.random.choice(total_chunks, SAMPLE_CHUNKS, replace=False)
    texts = []
    for idx in sample_indices:
        chunk = arr[idx].tolist()
        text = our_tok.decode([t for t in chunk if t < 50000])
        texts.append(text)
    print(f'  Sampled {len(texts)} texts')

    # 3. Compression analysis
    print('\n=== 3. Compression analysis ===')
    results = []
    for name, tok, vs in [
        ('our 50K', our_tok, VOCAB_OUR),
        ('ruadapt 146K', ruadapt_tok, VOCAB_RU),
        ('Mistral 131K', mistral_tok, VOCAB_MISTRAL),
    ]:
        r = analyze_tokenizer(name, texts, tok, vs, max_samples=500)
        results.append(r)

    # Best compression
    best = min(results, key=lambda r: r['tokens_per_chunk'])
    print(f'\n  Best compression: {best["name"]} '
          f'({best["tokens_per_chunk"]:.0f} tok/chunk, '
          f'{best["chars_per_token"]:.2f} char/tok)')

    # 4. Split data for fine-tune
    print('\n=== 4. Prepare fine-tune data ===')
    ft_indices = np.random.choice(total_chunks, TRAIN_CHUNKS, replace=False)
    ft_texts = []
    for idx in ft_indices:
        chunk = arr[idx].tolist()
        text = our_tok.decode([t for t in chunk if t < 50000])
        ft_texts.append(text)

    eval_indices = np.random.choice(total_chunks, EVAL_CHUNKS, replace=False)
    eval_texts = []
    for idx in eval_indices:
        chunk = arr[idx].tolist()
        text = our_tok.decode([t for t in chunk if t < 50000])
        eval_texts.append(text)

    # 5. Fine-tune: ruadapt
    print('\n=== 5. Fine-tune: ruadapt (Qwen3, 146K) ===')
    ru_loader, ru_n, ru_batches = build_data(ft_texts, SEQ_LEN, ruadapt_tok, VOCAB_RU)
    ru_eval_loader, ru_en, _ = build_data(eval_texts, SEQ_LEN, ruadapt_tok, VOCAB_RU)
    print(f'  Train examples: {ru_n}, {ru_batches} batches/epoch')

    if ru_loader is not None:
        ru_model = Phase2Model(VOCAB_RU).to(DEVICE)
        ru_opt = torch.optim.AdamW(ru_model.parameters(), lr=FT_LR, weight_decay=0.01)
        ru_ppl_history = []
        ru_iter = iter(ru_loader)

        for step in range(1, FT_STEPS + 1):
            ru_model.train()
            # Grab one batch (cycle loader if exhausted)
            try:
                bx, by = next(ru_iter)
            except StopIteration:
                ru_iter = iter(ru_loader)
                bx, by = next(ru_iter)

            bx = bx.to(DEVICE); by = by.to(DEVICE)
            ru_opt.zero_grad()
            loss = F.cross_entropy(ru_model(bx).reshape(-1, VOCAB_RU), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ru_model.parameters(), 1.0)
            ru_opt.step()

            if step % 20 == 0:
                ev = eval_ppl(ru_model, ru_eval_loader, VOCAB_RU)
                ru_ppl_history.append((step, math.exp(loss.item()), ev))
                print(f'    Step {step}: train_ppl={math.exp(loss.item()):.1f}, eval_ppl={ev:.1f}')

        final_ru = eval_ppl(ru_model, ru_eval_loader, VOCAB_RU)
    else:
        final_ru = None
        ru_ppl_history = []

    # 6. Fine-tune: Mistral
    print('\n=== 6. Fine-tune: Mistral (131K) ===')
    mi_loader, mi_n, mi_batches = build_data(ft_texts, SEQ_LEN, mistral_tok, VOCAB_MISTRAL)
    mi_eval_loader, mi_en, _ = build_data(eval_texts, SEQ_LEN, mistral_tok, VOCAB_MISTRAL)
    print(f'  Train examples: {mi_n}, {mi_batches} batches/epoch')

    if mi_loader is not None:
        mi_model = Phase2Model(VOCAB_MISTRAL).to(DEVICE)
        mi_opt = torch.optim.AdamW(mi_model.parameters(), lr=FT_LR, weight_decay=0.01)
        mi_ppl_history = []
        mi_iter = iter(mi_loader)

        for step in range(1, FT_STEPS + 1):
            mi_model.train()
            try:
                bx, by = next(mi_iter)
            except StopIteration:
                mi_iter = iter(mi_loader)
                bx, by = next(mi_iter)

            bx = bx.to(DEVICE); by = by.to(DEVICE)
            mi_opt.zero_grad()
            loss = F.cross_entropy(mi_model(bx).reshape(-1, VOCAB_MISTRAL), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mi_model.parameters(), 1.0)
            mi_opt.step()

            if step % 20 == 0:
                ev = eval_ppl(mi_model, mi_eval_loader, VOCAB_MISTRAL)
                mi_ppl_history.append((step, math.exp(loss.item()), ev))
                print(f'    Step {step}: train_ppl={math.exp(loss.item()):.1f}, eval_ppl={ev:.1f}')

        final_mi = eval_ppl(mi_model, mi_eval_loader, VOCAB_MISTRAL)
    else:
        final_mi = None
        mi_ppl_history = []

    # 7. Summary
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)

    print(f'\n--- Compression ---')
    for r in results:
        print(f'  {r["name"]:15s}: {r["tokens_per_chunk"]:6.0f} tok/chunk, '
              f'{r["chars_per_token"]:.2f} char/tok')

    print(f'\n--- Fine-tune PPL (from scratch, {FT_STEPS} steps) ---')
    if final_ru is not None:
        print(f'  ruadapt 146K: eval_ppl = {final_ru:.1f}')
    if final_mi is not None:
        print(f'  Mistral 131K: eval_ppl = {final_mi:.1f}')

    # Comparison with our 50K (very crude — different data split)
    # Just show relative ordering
    print(f'\n--- Trajectory (step, train_ppl, eval_ppl) ---')
    if ru_ppl_history:
        print(f'  ruadapt:')
        for s, tr, ev in ru_ppl_history:
            print(f'    step {s:3d}: train={tr:.0f}, eval={ev:.0f}')
    if mi_ppl_history:
        print(f'  Mistral:')
        for s, tr, ev in mi_ppl_history:
            print(f'    step {s:3d}: train={tr:.0f}, eval={ev:.0f}')

    print(f'\n=== Done ===')
    return results, final_ru, final_mi


if __name__ == '__main__':
    main()
