import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys.stdout, 'buffer'):
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
from ld_model.core import LDConfig, MemBindStack
from tokenizers import Tokenizer as HFTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D, VOCAB = 896, 50000

cfg = LDConfig()
cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = VOCAB
cfg.bottleneck = 896; cfg.kernel_size = 48
cfg.weight_tying = True; cfg.lm_head_bias = True
cfg.arch = 'membind'; cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
cfg.dct_basis = True; cfg.lambda_sliding = True
cfg.cov_first_moment = True; cfg.cov_rf = True; cfg.cov_rf_dim = 64
cfg.cov_multi_timescale = True; cfg.cov_mirror = True

model = torch.nn.Module()
model.embed = torch.nn.Embedding(VOCAB, D)
model.stack = MemBindStack(cfg)
model.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
model.lm_head.weight = model.embed.weight

ckpt = torch.load('checkpoints/ACTION_step2500.pt', map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.to(DEVICE)
model.eval()

tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
n = sum(p.numel() for p in model.parameters())
print(f'Model: {n/1e6:.1f}M params | ACTION_step2500', flush=True)

prompt = 'Он вошёл в тёмную комнату и'
enc = tok.encode(prompt)
ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
print(f'\nPrompt: {prompt}')
print(f'Tokens: {len(enc.ids)} IDs', flush=True)

k, l, rf = 48, 24, 1 + 47 * 24
top_k, temp = 40, 0.8

for _ in range(100):
    ctx = ids[:, -rf:] if ids.shape[1] > rf else ids
    logits = model.lm_head(model.stack(model.embed(ctx))[0])[:, -1, :]
    if top_k > 0:
        vals, _ = torch.topk(logits, top_k)
        logits[logits < vals[:, -1:]] = -float('Inf')
    probs = F.softmax(logits / temp, dim=-1)
    nxt = torch.multinomial(probs, 1)
    ids = torch.cat([ids, nxt], dim=1)

text = tok.decode(ids[0].tolist())
print(f'\n--- Generated (100 tok, t={temp}, top_k={top_k}) ---')
print(text)
print('---')

logits = model.lm_head(model.stack(model.embed(ids))[0])[:, -1, :]
probs = F.softmax(logits, dim=-1)
H = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
max_H = math.log(VOCAB)
print(f'\nEntropy: {H:.3f}/{max_H:.3f} ({H/max_H*100:.0f}% of max)')
