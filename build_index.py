"""
Build token_index.json: genre → {file, length} mapping for per-genre streams.
Scans for token_stream_{GENRE}.bin files.

Usage:
  python build_index.py
"""
import os, json, argparse, glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='token_index.json')
    args = parser.parse_args()

    files = sorted(glob.glob('token_stream_*.bin'))
    if not files:
        print('[ERROR] No token_stream_*.bin files found. Run prepare_corpus.py first.')
        return

    index = {}
    total_tokens = 0
    for fp in files:
        genre = fp.replace('token_stream_', '').replace('.bin', '')
        size = os.path.getsize(fp)
        length = size // 4  # int32
        index[genre] = {'file': fp, 'length': length}
        total_tokens += length

    print(f'Found {len(index)} genre files, {total_tokens//1e6:.0f}M tokens')
    for g, v in sorted(index.items(), key=lambda x: x[1]['length'], reverse=True):
        print(f'  {g:15s}  {v["length"]//1e6:.0f}M tok  file={v["file"]}')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f'\nSaved {args.output}')
    print('Done.')

if __name__ == '__main__':
    main()
