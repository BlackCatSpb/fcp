from tokenizers import Tokenizer
tok = Tokenizer.from_file('russian_tokenizer/tokenizer.json')

lines = []
seq = [224, 175, 127, 125, 224, 175, 127, 125]
decoded = tok.decode(seq)
lines.append(f'Cycle decoded: {repr(decoded)}')
decoded3 = tok.decode([175, 127, 125])
lines.append(f'3-byte seq: {repr(decoded3)}')

common = ['на', 'по', 'в', 'с', 'к', 'от', 'из', 'у', 'за', 'о', 'об', 'при', 'про', 'до', 'для']
for word in common:
    ids = tok.encode(word).ids
    lines.append(f'{word}: {ids}')

# Check the model's prompt
prompt = 'Он вошёл в тёмную комнату и'
enc = tok.encode(prompt)
dec = tok.decode(enc.ids)
lines.append(f'Prompt: {prompt}')
lines.append(f'Encoded: {enc.ids}')
lines.append(f'Re-decoded: {repr(dec)}')

with open('_tok2.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Done')
