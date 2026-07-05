"""
Prepare clean Russian text dataset for λ_d training.
Trains a BPE tokenizer (vocab=50000) on Russian data, then tokenises + chunks.
Saves intermediate checkpoints for crash recovery.
Output: russian_chunks.npy + russian_tokenizer/
"""

import os, sys, math, time, json, argparse, itertools
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

SEQ_LEN = 128
VOCAB = 50000
SAVE_EVERY = 100000  # save partial every N chunks
PARTIAL_DIR = 'russian_chunks_temp'
STATE_FILE = 'russian_chunks_temp/state.json'

# ─── streaming helpers ─────────────────────────────────────────────────

def stream_wikipedia(skip=0, max_chars=None):
    from datasets import load_dataset
    wiki = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train', streaming=True)
    if skip > 0:
        wiki = wiki.skip(skip)
    chars = 0
    for article in wiki:
        text = article['text'].strip()
        if len(text) < 200:
            continue
        yield text
        chars += len(text)
        if max_chars and chars >= max_chars:
            break

def stream_books(skip=0, max_chars=None):
    from datasets import load_dataset
    try:
        rpd = load_dataset('PleIAs/Russian-PD', split='train', streaming=True)
        if skip > 0:
            rpd = rpd.skip(skip)
        chars = 0
        for book in rpd:
            text = book['text'].strip()
            if len(text) < 500:
                continue
            yield text
            chars += len(text)
            if max_chars and chars >= max_chars:
                break
    except Exception as e:
        print('  [WARN] Russian-PD not available: %s' % e)

# ─── save/load partial ────────────────────────────────────────────────

def save_partial(chunks_list, state):
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    # save chunks
    arr = np.array(chunks_list, dtype=np.int32)
    np.save(os.path.join(PARTIAL_DIR, 'chunks.npy'), arr)
    # save state
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_partial():
    if not os.path.exists(STATE_FILE):
        return [], None
    with open(STATE_FILE) as f:
        state = json.load(f)
    arr = np.load(os.path.join(PARTIAL_DIR, 'chunks.npy'))
    chunks = [list(row) for row in arr]
    print('  Resumed: %d chunks from partial save' % len(chunks))
    return chunks, state

# ─── main ──────────────────────────────────────────────────────────────

def main(sample_size=None):
    # ─── Tokenizer ────────────────────────────────────────────────────
    tokenizer_path = 'russian_tokenizer/tokenizer.json'
    if os.path.exists(tokenizer_path):
        print('[1/4] Loading existing tokenizer from %s' % tokenizer_path)
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print('[1/4] Training BPE tokenizer (vocab=%d)...' % VOCAB)
        t0 = time.time()
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB,
            special_tokens=['<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>'],
            min_frequency=2,
            show_progress=True,
        )
        sample_limit = min(sample_size, 200_000_000) if sample_size else 200_000_000
        def text_stream():
            n_chars = 0
            for text in stream_wikipedia(max_chars=sample_limit):
                yield text
                n_chars += len(text)
            for text in stream_books(max_chars=sample_limit - n_chars):
                yield text
                n_chars += len(text)
            print('  Sampled %.1fM chars for tokenizer training' % (n_chars / 1e6))
        tokenizer.train_from_iterator(text_stream(), trainer=trainer)
        print('  Trained in %.1fs, vocab=%d' % (time.time() - t0, tokenizer.get_vocab_size()))
        os.makedirs('russian_tokenizer', exist_ok=True)
        tokenizer.save(tokenizer_path)

    # ─── Tokenize & chunk ─────────────────────────────────────────────
    print('\n[2/4] Streaming and tokenizing...')

    # Resume?
    chunks, state = load_partial()
    if state:
        wiki_skip = state.get('wiki_articles', 0)
        book_skip = state.get('book_articles', 0)
        source = state.get('source', 'wikipedia')
        print('  Resuming from %s (wiki_skip=%d, book_skip=%d, chunks=%d)' % (
            source, wiki_skip, book_skip, len(chunks)))
        all_chunks = chunks
        partial_chunks = []
    else:
        wiki_skip = 0
        book_skip = 0
        source = 'wikipedia'
        all_chunks = []
        partial_chunks = []

    # ── Wikipedia ──
    if source == 'wikipedia':
        print('  Wikipedia (skip=%d)...' % wiki_skip)
        t0 = time.time()
        n_articles = 0
        for text in stream_wikipedia(skip=wiki_skip):
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            while len(ids) >= SEQ_LEN + 1:
                chunk = ids[:SEQ_LEN + 1]
                all_chunks.append(chunk)
                partial_chunks.append(chunk)
                ids = ids[SEQ_LEN:]
                if len(partial_chunks) >= SAVE_EVERY:
                    save_partial(all_chunks, {
                        'source': 'wikipedia',
                        'wiki_articles': n_articles + wiki_skip,
                        'book_articles': book_skip,
                    })
                    print('    [CKPT] %d chunks (%dM tok) in %.0fs' % (
                        len(all_chunks), len(all_chunks) * SEQ_LEN // 1000000, time.time() - t0))
                    partial_chunks = []
            n_articles += 1
            if sample_size and len(all_chunks) * SEQ_LEN * 4 >= sample_size:
                break

        # Wikipedia done → save checkpoint
        save_partial(all_chunks, {
            'source': 'books',
            'wiki_articles': n_articles + wiki_skip,
            'book_articles': 0,
        })
        print('    [CKPT] Wikipedia done: %d chunks in %.0fs' % (len(all_chunks), time.time() - t0))
        book_skip = 0  # already in state
        source = 'books'

    # ── Books ──
    if source == 'books':
        print('  Books (skip=%d)...' % book_skip)
        t0 = time.time()
        n_books = 0
        for text in stream_books(skip=book_skip):
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            while len(ids) >= SEQ_LEN + 1:
                chunk = ids[:SEQ_LEN + 1]
                all_chunks.append(chunk)
                partial_chunks.append(chunk)
                ids = ids[SEQ_LEN:]
                if len(partial_chunks) >= SAVE_EVERY:
                    save_partial(all_chunks, {
                        'source': 'books',
                        'wiki_articles': 0,  # not tracked after wiki done
                        'book_articles': n_books + book_skip,
                    })
                    print('    [CKPT] %d chunks (%dM tok) in %.0fs' % (
                        len(all_chunks), len(all_chunks) * SEQ_LEN // 1000000, time.time() - t0))
                    partial_chunks = []
            n_books += 1
            if sample_size and len(all_chunks) * SEQ_LEN * 4 >= sample_size:
                break

    print('  Total: %d chunks, ~%dM tokens' % (len(all_chunks), len(all_chunks) * SEQ_LEN // 1000000))

    if len(all_chunks) == 0:
        print('[ERROR] No chunks generated!')
        return

    # ─── Save final ──────────────────────────────────────────────────
    print('\n[3/4] Saving to russian_chunks.npy...')
    arr = np.array(all_chunks, dtype=np.int32)
    np.save('russian_chunks.npy', arr)
    size_mb = os.path.getsize('russian_chunks.npy') / 1e6
    print('  Shape: %s' % str(arr.shape))
    print('  Size: %.0f MB' % size_mb)

    # Clean temp
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        os.remove(os.path.join(PARTIAL_DIR, 'chunks.npy'))
        try: os.rmdir(PARTIAL_DIR)
        except: pass

    # ─── Stats ────────────────────────────────────────────────────────
    print('\n[4/4] Stats:')
    n_train = int(len(all_chunks) * 0.99)
    n_eval = len(all_chunks) - n_train
    print('  Train chunks: %d (~%dM tok)' % (n_train, n_train * SEQ_LEN // 1000000))
    print('  Eval chunks:  %d (~%dM tok)' % (n_eval, n_eval * SEQ_LEN // 1000000))
    print('  Total:        %d (~%dM tok)' % (len(all_chunks), len(all_chunks) * SEQ_LEN // 1000000))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample size in chars for testing')
    args = parser.parse_args()
    main(sample_size=args.sample)
