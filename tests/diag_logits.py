"""Diagnose logit scale in Phase 1 training."""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import AutoTokenizer
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

D, VOCAB, K = 256, 10000, 4
TOKENIZER_PATH = 'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b'
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
tok.pad_token = tok.eos_token
VOCAB = min(VOCAB, tok.vocab_size)
print(f'Vocab: {VOCAB}')

cfg = LDConfig()
cfg.D = D; cfg.n_layers = 1; cfg.n_modes = K; cfg.vocab = VOCAB
cfg.intermediate = 1024; cfg.lora_rank = 0; cfg.use_lora = False
lambdas = fibonacci_roots(K + 1)

class DiagModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.block = LDBlock(cfg, 0, lambdas)
    def forward(self, x):
        h = self.embed(x)
        h_out, a = self.block(h, return_gates=True)
        logits = h_out.float() @ self.embed.weight.T.float()
        return logits, a, h_out, h

model = DiagModel().cuda()

# Load one batch of data
ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
texts = [x['text'] for x in ds if len(x['text'].strip()) > 0][:10]
enc = tok(texts, truncation=True, max_length=129, padding=False)
all_ids = []
for ids in enc['input_ids']:
    all_ids.extend([min(i, VOCAB-1) for i in ids])
chunks = [all_ids[i:i+129] for i in range(0, len(all_ids)-129, 64)]

ids = torch.tensor(chunks[0], dtype=torch.long).unsqueeze(0).cuda()
x, y = ids[:, :-1], ids[:, 1:]
print(f'Input shape: {x.shape}, labels shape: {y.shape}')

with torch.no_grad():
    logits, a, h_out, h_in = model(x)

print(f'h_in  norm: {h_in.norm(dim=-1).mean().item():.2f}')
print(f'h_out norm: {h_out.norm(dim=-1).mean().item():.2f}')
print(f'logits shape: {logits.shape}')
print(f'logits mean: {logits.mean().item():.2f}')
print(f'logits std:  {logits.std().item():.2f}')
print(f'logits max:  {logits.max().item():.2f}')
print(f'logits min:  {logits.min().item():.2f}')

# Correct token logit stats
correct = logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
print(f'correct logit mean: {correct.mean().item():.2f}')
print(f'correct logit std:  {correct.std().item():.2f}')

# CE loss
loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
print(f'CE loss: {loss.item():.4f}')
print(f'Expected random: {math.log(VOCAB):.4f}')

# Try with temperature scaling
for T in [1, 5, 10, 20, 50]:
    loss_T = F.cross_entropy(logits.reshape(-1, VOCAB) / T, y.reshape(-1))
    ppl_T = math.exp(loss_T.item())
    print(f'  T={T:3d}: loss={loss_T.item():.4f}, ppl={ppl_T:.2f}')

# Embed weight stats
print(f'\nembed.weight mean: {model.embed.weight.mean().item():.2f}')
print(f'embed.weight std:  {model.embed.weight.std().item():.2f}')
print(f'embed.weight max:  {model.embed.weight.max().item():.2f}')
print(f'embed.weight min:  {model.embed.weight.min().item():.2f}')
