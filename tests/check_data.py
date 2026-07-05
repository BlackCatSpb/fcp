"""Check available datasets."""
from datasets import load_dataset

# Try wikitext-103 (small, fast to download)
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
count = 0
total_chars = 0
for x in ds:
    if len(x['text'].strip()) > 0:
        count += 1
        total_chars += len(x['text'])
    if count >= 10000:
        break
print(f'wikitext-103: {count} docs, {total_chars} chars')
print(f'First doc: {ds[0]["text"][:100]}')
print(f'Total size: {len(ds)} rows')
