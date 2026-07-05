import sys, json
from tokenizers import Tokenizer

tok = Tokenizer.from_file('russian_tokenizer/tokenizer.json')

# Decode specific IDs
ids = [175, 127, 125, 224]
results = []
for id_val in ids:
    decoded = tok.decode([id_val], skip_special_tokens=False)
    results.append(f'ID {id_val}: {repr(decoded)}')

# Check prompt encoding
enc = tok.encode('Он вошёл в тёмную комнату и')
results.append(f'Prompt IDs: {enc.ids}')
results.append(f'Prompt re-encoded: {repr(tok.decode(enc.ids))}')

# Check some random tokens
check = [0, 1, 2, 10, 100, 1000, 5000, 10000, 25000, 49999]
for i in check:
    d = tok.decode([i], skip_special_tokens=False)
    results.append(f'ID {i}: {repr(d)}')

with open('_tok_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print('Done. Check _tok_result.txt')
