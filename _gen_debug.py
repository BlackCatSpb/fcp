import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
sd = ckpt['model_state_dict']
print(f'State dict keys: {len(sd)}')
print(f'First 20 keys: {list(sd.keys())[:20]}')
print(f'Has embed.weight: {"embed.weight" in sd}')
print(f'Has stack.layers.0.b_decay: {"stack.layers.0.b_decay" in sd}')
print(f'Has stack.layers.0.W_k_rf: {"stack.layers.0.W_k_rf" in sd}')

model.load_state_dict(sd, strict=False)
model.to(DEVICE)
model.eval()

# Check buffer b_decay
buf = model.stack.layers[0].b_decay
print(f'\nb_decay layer 0: {buf} (registered buffer)')

# Generate and check raw IDs
tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
prompt = 'Он вошёл в тёмную комнату и'
enc = tok.encode(prompt)
ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
print(f'\nPrompt IDs: {ids.tolist()}')
print(f'Prompt decoded: {tok.decode(ids[0].tolist())}')

k, l, rf = 48, 24, 1 + 47 * 24
top_k, temp = 40, 0.8

for step in range(50):
    ctx = ids[:, -rf:] if ids.shape[1] > rf else ids
    logits = model.lm_head(model.stack(model.embed(ctx))[0])[:, -1, :]
    if top_k > 0:
        vals, _ = torch.topk(logits, top_k)
        logits[logits < vals[:, -1:]] = -float('Inf')
    probs = F.softmax(logits / temp, dim=-1)
    nxt = torch.multinomial(probs, 1)
    ids = torch.cat([ids, nxt], dim=1)
    if step < 20 or step >= 40:
        print(f'  step {step}: ID={nxt.item():5d}')

text = tok.decode(ids[0].tolist())
final_ids = ids[0, 7:].tolist()  # after prompt
print(f'\nGenerated IDs (first 20): {final_ids[:20]}')
print(f'\n--- Text ---')
print(text)
print('---')
