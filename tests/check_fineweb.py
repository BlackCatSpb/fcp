"""Check FineWeb dataset availability."""
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb', split='train', streaming=True, trust_remote_code=True)
for i, x in enumerate(ds):
    if i >= 3: break
    print(f'Doc {i}: {len(x["text"])} chars, preview={x["text"][:80]}')
