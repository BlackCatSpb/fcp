"""Diagnose which tokens have extreme logits."""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import AutoTokenizer
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

D, VOCAB, K = 256, 10000, 4
TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tok.pad_token = tok.eos_token
VOCAB = min(VOCAB, tok.vocab_size)

cfg = LDConfig()
cfg.D = D; cfg.n_layers = 1; cfg.n_modes = K; cfg.vocab = VOCAB
cfg.intermediate = 1024; cfg.lora_rank = 0; cfg.use_lora = False
lambdas = fibonacci_roots(K + 1)

model = torch.nn.Sequential(
    torch.nn.Embedding(VOCAB, D),
    LDBlock(cfg, 0, lambdas),
    torch.nn.LayerNorm(D, eps=1e-6),
).cuda()
embed = model[0]

ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
texts = [x['text'] for x in ds if len(x['text'].strip()) > 0][:10]
enc = tok(texts, truncation=True, max_length=129, padding=False)
all_ids = []
for ids in enc['input_ids']:
    all_ids.extend([min(i, VOCAB-1) for i in ids])
chunks = [all_ids[i:i+129] for i in range(0, len(all_ids)-129, 64)]

ids = torch.tensor(chunks[0], dtype=torch.long).unsqueeze(0).cuda()
x, y = ids[:, :-1], ids[:, 1:]

with torch.no_grad():
    h = embed(x)
    h_out, a = model[1](h, return_gates=True)
    h_normed = model[2](h_out.float())
    logits = h_normed @ embed.weight.T.float()

print(f'logits shape: {logits.shape}')  # (1, 128, 10000)
print(f'logits std: {logits.std().item():.2f}')

# Find top-5 logit values overall
vals, idxs = logits.reshape(-1).topk(10)
print(f'Top-10 logit values: {vals.tolist()}')
for v, i in zip(vals.tolist(), idxs.tolist()):
    pos = i // VOCAB
    tok_id = i % VOCAB
    correct = 'CORRECT' if tok_id == y[0, pos].item() else 'WRONG'
    print(f'  pos={pos:3d}, tok={tok_id:5d}, logit={v:8.2f} {correct}')

# Also check: embed weight norms vs logit magnitude
embed_norms = embed.weight.norm(dim=-1)
print(f'\nembed.weight max norm: {embed_norms.max().item():.2f}')
print(f'embed.weight min norm: {embed_norms.min().item():.2f}')
print(f'embed.weight mean norm: {embed_norms.mean().item():.2f}')
w_max = embed_norms.argmax()
print(f'Max norm token: {w_max.item()}, norm={embed_norms[w_max].item():.2f}')

# Check if extreme tokens have large norms
for v, i in zip(vals.tolist(), idxs.tolist()):
    tok_id = i % VOCAB
    print(f'  Extreme tok={tok_id}: embed norm={embed_norms[tok_id].item():.2f}')
