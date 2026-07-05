"""
Prepare full Russian corpus for MemBind training.
Two-pass: count tokens → allocate exact array → fill.
Usage:
  python prepare_corpus.py
  python prepare_corpus.py --dry-run
"""

import os, sys, time, json, argparse, re, math
import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 1024
CHUNK_CHARS = 50_000_000  # chars per temp chunk (avoid max mem)
SRC_ROOT = r'C:\Users\black\OneDrive\Desktop\fcp\main-russian'


def clean_text(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'\.{4,}', '...', line)
        line = re.sub(r'!{2,}', '!', line)
        line = re.sub(r'\?{2,}', '?', line)
        lines.append(line)
    return '\n'.join(lines)


def collect_files(src):
    """Return list of (genre, filepath) for all txt files."""
    items = []
    for entry in sorted(os.scandir(src), key=lambda e: e.name):
        if entry.is_dir():
            genre = entry.name
            for fn in sorted(os.listdir(entry.path)):
                if fn.endswith('.txt'):
                    items.append((genre, os.path.join(entry.path, fn)))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--src', default=SRC_ROOT)
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f'[ERROR] {args.src} not found')
        sys.exit(1)

    # Tokenizer
    tok_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'russian_tokenizer', 'tokenizer.json')
    if not os.path.exists(tok_path):
        print(f'[ERROR] tokenizer not found at {tok_path}')
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)
    print(f'Tokenizer: vocab={tokenizer.get_vocab_size()}')

    # Collect files
    items = collect_files(args.src)
    total_files = len(items)
    print(f'Files: {total_files}')

    # ─── Pass 1: count tokens per file ────────────────────────────────
    print('\n[Pass 1/2] Counting tokens...')
    token_counts = []
    t0 = time.perf_counter()
    for idx, (genre, path) in enumerate(items):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()
        except Exception as e:
            print(f'\n  [WARN] {genre}/{os.path.basename(path)}: {e}')
            token_counts.append(0)
            continue
        text = clean_text(raw)
        n_tok = len(tokenizer.encode(text).ids)
        token_counts.append(n_tok)
        if (idx + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            print(f'  [{idx+1}/{total_files}] {rate:.0f} files/s, '
                  f'[{genre}] {n_tok//1e3:.0f}K tok')
    elapsed = time.perf_counter() - t0
    total_tokens = sum(token_counts)
    print(f'  Done: {total_tokens//1e6:.0f}M tokens from {total_files} files '
          f'in {elapsed:.0f}s ({total_tokens/elapsed/1e6:.2f}M tok/s)')

    if args.dry_run:
        print(f'\nSEQ_LEN={SEQ_LEN}: ~{total_tokens//SEQ_LEN//1e6:.0f}M windows')
        print(f'Est .npy size: {total_tokens*4/1e9:.2f} GB')
        return

    # ─── Pass 2: fill via memmap (no RAM spike) ────────────────────────
    print('\n[Pass 2/2] Tokenizing + saving (memmap)...')
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'token_stream.bin')
    if os.path.exists(out_path):
        os.remove(out_path)
    arr = np.memmap(out_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
    pos = 0
    t0 = time.perf_counter()
    for idx, (genre, path) in enumerate(items):
        n_tok = token_counts[idx]
        if n_tok == 0:
            continue
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()
        except:
            continue
        text = clean_text(raw)
        ids = tokenizer.encode(text).ids
        arr[pos:pos + n_tok] = ids
        pos += n_tok
        if (idx + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = pos / elapsed
            print(f'  [{idx+1}/{total_files}] {pos//1e6:.0f}M tokens, '
                  f'{rate/1e6:.2f}M tok/s, [{genre}]')

    arr.flush()
    del arr
    elapsed = time.perf_counter() - t0
    assert pos == total_tokens, f'pos={pos} != total={total_tokens}'
    size_gb = os.path.getsize(out_path) / 1e9
    print(f'  Done: {pos//1e6:.0f}M tokens, {size_gb:.2f} GB, '
          f'{elapsed:.0f}s ({pos/elapsed/1e6:.2f}M tok/s)')
    print('Done.')


if __name__ == '__main__':
    main()
