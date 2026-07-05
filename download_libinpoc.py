"""
Download ru-libinpoc-11k (11.5K Russian books, 2.66 GB, MIT license),
tokenize with existing BPE, append to russian_chunks.npy.

Usage:
  python download_libinpoc.py          # download + tokenize
  python download_libinpoc.py --skip-download  # use cached .txt files
  python download_libinpoc.py --dry-run # count only
"""

import os, sys, time, json, argparse, struct, io, itertools
import numpy as np
from tokenizers import Tokenizer

SEQ_LEN = 128
SAVE_EVERY = 500000
TEMP_FILE = 'russian_chunks_libinpoc_temp.npy'
STATE_FILE = 'libinpoc_state.json'

def download_7z(url: str, out_path: str):
    """Stream download a file with progress."""
    import requests
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    t0 = time.time()
    with open(out_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                elapsed = time.time() - t0
                speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                print(f'\r  {pct:.0f}% ({downloaded/1e9:.2f}/{total/1e9:.2f} GB, {speed:.0f} MB/s)', end='')
    print()
    return out_path


def extract_7z(archive_path: str, out_dir: str) -> list[str]:
    """Extract 7z and return list of .txt file paths."""
    try:
        import py7zr
    except ImportError:
        print('[ERROR] Need py7zr: pip install py7zr')
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    with py7zr.SevenZipFile(archive_path, mode='r') as sz:
        sz.extractall(path=out_dir)
    elapsed = time.time() - t0
    txt_files = []
    for root, dirs, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith('.txt'):
                txt_files.append(os.path.join(root, fn))
    print(f'  Extracted {len(txt_files)} txt files in {elapsed:.0f}s')
    return txt_files


def count_tokens(txt_files: list[str], tokenizer: Tokenizer) -> int:
    """Count total tokens in all txt files (dry-run)."""
    total = 0
    t0 = time.time()
    for i, path in enumerate(txt_files):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        ids = tokenizer.encode(text).ids
        total += len(ids)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'\r  [{i+1}/{len(txt_files)}] {total//1e6:.0f}M tok, {elapsed:.0f}s', end='')
    print()
    return total


def tokenize_and_chunk(txt_files: list[str], tokenizer: Tokenizer,
                       existing_chunks: list | None = None) -> np.ndarray:
    """Tokenize texts, chunk into SEQ_LEN+1 sliding windows, return array."""
    all_chunks = list(existing_chunks) if existing_chunks else []
    chunk_count = len(all_chunks)
    t0 = time.time()
    lap = t0
    for i, path in enumerate(txt_files):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except Exception as e:
            print(f'\n  [WARN] skip {path}: {e}')
            continue
        if len(text) < 100:
            continue
        ids = tokenizer.encode(text).ids
        while len(ids) >= SEQ_LEN + 1:
            chunk = ids[:SEQ_LEN + 1]
            all_chunks.append(chunk)
            ids = ids[SEQ_LEN:]

        # Periodic save + report
        if len(all_chunks) - chunk_count >= SAVE_EVERY:
            now = time.time()
            rate = (len(all_chunks) - chunk_count) / (now - lap)
            print(f'  [{i+1}/{len(txt_files)}] {len(all_chunks)} chunks '
                  f'({len(all_chunks)*SEQ_LEN//1e6:.0f}M tok), {rate:.0f} chunks/s')
            # Save temp
            arr = np.array(all_chunks, dtype=np.int32)
            np.save(TEMP_FILE, arr)
            with open(STATE_FILE, 'w') as f:
                json.dump({'files_processed': i + 1, 'chunks': len(all_chunks)}, f)
            lap = now
            chunk_count = len(all_chunks)

    arr = np.array(all_chunks, dtype=np.int32)
    return arr


def main():
    parser = argparse.ArgumentParser(description='Download & tokenize ru-libinpoc-11k')
    parser.add_argument('--skip-download', action='store_true',
                        help='Use cached extracted txt files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count tokens only, do not save')
    parser.add_argument('--merge', action='store_true',
                        help='Merge with existing russian_chunks.npy')
    args = parser.parse_args()

    # Tokenizer
    tokenizer_path = 'russian_tokenizer/tokenizer.json'
    if not os.path.exists(tokenizer_path):
        print('[ERROR] russian_tokenizer/tokenizer.json not found. Run prepare_russian_data.py first.')
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f'Tokenizer: vocab={tokenizer.get_vocab_size()}')

    # Download / extract
    if not args.skip_download:
        url = 'https://huggingface.co/datasets/4eJIoBek/ru-libinpoc-11k/resolve/main/main-russian.7z'
        archive_path = 'main-russian.7z'
        if not os.path.exists(archive_path):
            print(f'Downloading {url}...')
            download_7z(url, archive_path)
        else:
            print(f'Using cached {archive_path}')

        txt_dir = 'libinpoc_txt'
        print('Extracting...')
        txt_files = extract_7z(archive_path, txt_dir)
    else:
        txt_dir = 'libinpoc_txt'
        if not os.path.exists(txt_dir):
            print(f'[ERROR] --skip-download but {txt_dir} not found')
            sys.exit(1)
        txt_files = []
        for root, dirs, files in os.walk(txt_dir):
            for fn in files:
                if fn.endswith('.txt'):
                    txt_files.append(os.path.join(root, fn))
        print(f'Found {len(txt_files)} txt files in {txt_dir}')

    if not txt_files:
        print('[ERROR] No txt files found')
        sys.exit(1)

    if args.dry_run:
        print('Counting tokens (dry-run)...')
        total_tok = count_tokens(txt_files, tokenizer)
        total_chunks = total_tok // SEQ_LEN
        print(f'Total tokens: {total_tok//1e6:.0f}M')
        print(f'Equivalent chunks (SEQ_LEN={SEQ_LEN}): {total_chunks} ({total_chunks*SEQ_LEN//1e6:.0f}M tok)')
        print(f'Estimated .npy size: {total_chunks*4*(SEQ_LEN+1)/1e9:.2f} GB')
        return

    # Resume
    existing_chunks = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        if os.path.exists(TEMP_FILE):
            arr = np.load(TEMP_FILE)
            existing_chunks = [list(row) for row in arr]
            # Skip already processed files
            processed = state.get('files_processed', 0)
            txt_files = txt_files[processed:]
            print(f'Resumed: {len(existing_chunks)} chunks, {processed} files done, {len(txt_files)} remaining')

    # Tokenize & chunk
    print(f'Tokenizing {len(txt_files)} files...')
    arr = tokenize_and_chunk(txt_files, tokenizer, existing_chunks)

    # Load existing russian_chunks.npy if merging
    if args.merge and os.path.exists('russian_chunks.npy'):
        existing = np.load('russian_chunks.npy')
        print(f'Existing: {existing.shape}')
        combined = np.concatenate([existing, arr], axis=0)
        print(f'Combined: {combined.shape}')
        arr = combined

    # Save
    out_path = 'russian_chunks_libinpoc.npy' if not args.merge else 'russian_chunks.npy'
    print(f'Saving to {out_path}...')
    np.save(out_path, arr)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f'Shape: {arr.shape}, size: {size_gb:.2f} GB')

    # Clean temp
    for f in [TEMP_FILE, STATE_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print('Done.')


if __name__ == '__main__':
    main()
