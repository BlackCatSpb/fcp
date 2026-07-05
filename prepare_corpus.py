"""
Prepare Russian corpus for MemBind training — one genre at a time.
Each genre → separate file (token_stream_{GENRE}.bin) to avoid massive 47GB memmap.
Usage:
  python prepare_corpus.py --genre ACTION
  python prepare_corpus.py --genre all       # all genres (sequential, one file per genre)
  python prepare_corpus.py --dry-run
"""

import os, sys, time, json, argparse, re, math
import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 1024
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


def collect_files(src, genre_filter=None):
    """Return list of (genre, filepath). If genre_filter set, only that genre."""
    items = []
    for entry in sorted(os.scandir(src), key=lambda e: e.name):
        if entry.is_dir():
            genre = entry.name
            if genre_filter is not None and genre != genre_filter:
                continue
            for fn in sorted(os.listdir(entry.path)):
                if fn.endswith('.txt'):
                    items.append((genre, os.path.join(entry.path, fn)))
    return items


def process_genre(genre_name, tokenizer, src):
    """Count + tokenize one genre into token_stream_{genre}.bin. Returns (n_files, n_tokens)."""
    items = collect_files(src, genre_filter=genre_name)
    if not items:
        print(f'  [WARN] No files found for genre "{genre_name}"')
        return 0, 0

    n_files = len(items)
    print(f'\n  Files: {n_files}')

    # ─── Count tokens ────────────────────────────────────────────────
    token_counts = []
    t0 = time.perf_counter()
    for idx, (genre, path) in enumerate(items):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()
        except Exception as e:
            print(f'\n  [WARN] {os.path.basename(path)}: {e}')
            token_counts.append(0)
            continue
        text = clean_text(raw)
        n_tok = len(tokenizer.encode(text).ids)
        token_counts.append(n_tok)
        if (idx + 1) % 250 == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / elapsed
            done_tok = sum(token_counts)
            print(f'    [{idx+1}/{n_files}] {rate:.0f} files/s, '
                  f'{done_tok//1e3:.0f}K tok')

    total_tokens = sum(token_counts)
    elapsed = time.perf_counter() - t0
    print(f'  Counted: {total_tokens//1e6:.0f}M tokens from {n_files} files '
          f'in {elapsed:.0f}s ({total_tokens/elapsed/1e6:.2f}M tok/s)')

    # ─── Save to binary ──────────────────────────────────────────────
    out_path = f'token_stream_{genre_name}.bin'
    print(f'  Writing {out_path} ({total_tokens*4/1e9:.1f} GB)...')
    t0 = time.perf_counter()
    arr = np.memmap(out_path, dtype=np.int32, mode='w+', shape=(total_tokens,))
    pos = 0
    for idx, (genre, path) in enumerate(items):
        n_tok = token_counts[idx]
        if n_tok == 0:
            continue
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                raw = f.read()
        except:
            continue
        ids = tokenizer.encode(clean_text(raw)).ids
        arr[pos:pos + n_tok] = ids
        pos += n_tok
        if (idx + 1) % 250 == 0:
            elapsed = time.perf_counter() - t0
            rate = pos / elapsed
            print(f'    [{idx+1}/{n_files}] {pos//1e6:.0f}M tok, {rate/1e6:.2f}M tok/s')

    arr.flush()
    del arr
    elapsed = time.perf_counter() - t0
    size_gb = os.path.getsize(out_path) / 1e9
    print(f'  Saved: {pos//1e6:.0f}M tokens, {size_gb:.2f} GB, '
          f'{elapsed:.0f}s ({pos/elapsed/1e6:.2f}M tok/s)')
    return n_files, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--src', default=SRC_ROOT)
    parser.add_argument('--genre', default='all',
                        help='Genre name (e.g. ACTION) or "all"')
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

    # Determine genres to process
    if args.genre == 'all':
        # Collect all genre dirs
        genres = sorted(e.name for e in os.scandir(args.src) if e.is_dir())
    else:
        genres = [args.genre]

    print(f'Genres to process: {genres}')

    if args.dry_run:
        for g in genres:
            items = collect_files(args.src, genre_filter=g)
            est_tok = 0
            for _, path in items:
                try:
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        est_tok += len(f.read()) // 3  # rough est
                except:
                    pass
            print(f'  {g}: {len(items)} files, ~{est_tok//1e6:.0f}M tok est')
        return

    total_files = 0
    total_tokens = 0
    for g in genres:
        nf, nt = process_genre(g, tokenizer, args.src)
        total_files += nf
        total_tokens += nt

    print(f'\n{"="*50}')
    print(f'All done: {total_files} files, {total_tokens//1e6:.0f}M tokens')


if __name__ == '__main__':
    main()
