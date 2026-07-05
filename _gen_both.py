"""
Generate from both models: Phase 2 (old, no extra features) and ACTION (new, all features).
Writes results to _gen_results.txt.
"""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack
from tokenizers import Tokenizer as HFTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D, VOCAB = 896, 50000
OUTFILE = '_gen_results.txt'

def make_action_cfg():
    cfg = LDConfig()
    cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = VOCAB
    cfg.bottleneck = 896; cfg.kernel_size = 48
    cfg.weight_tying = True; cfg.lm_head_bias = True
    cfg.arch = 'membind'; cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
    cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
    cfg.dct_basis = True; cfg.lambda_sliding = True
    cfg.cov_first_moment = True; cfg.cov_rf = True; cfg.cov_rf_dim = 64
    cfg.cov_multi_timescale = True; cfg.cov_mirror = True
    return cfg

def make_base_cfg():
    cfg = LDConfig()
    cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = VOCAB
    cfg.bottleneck = 896; cfg.kernel_size = 48
    cfg.weight_tying = True; cfg.lm_head_bias = True
    cfg.arch = 'membind'; cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
    cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
    cfg.cov_first_moment = False; cfg.cov_rf = False; cfg.cov_multi_timescale = False; cfg.cov_mirror = False
    return cfg

def build_model(cfg):
    m = torch.nn.Module()
    m.embed = torch.nn.Embedding(VOCAB, D)
    m.stack = MemBindStack(cfg)
    m.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
    m.lm_head.weight = m.embed.weight
    return m

def generate_text(model, tok, prompt, n_tokens=200, temp=0.8, top_k=40):
    k, l, rf = 48, 24, 1 + 47 * 24
    enc = tok.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
    for _ in range(n_tokens):
        ctx = ids[:, -rf:] if ids.shape[1] > rf else ids
        logits = model.lm_head(model.stack(model.embed(ctx))[0])[:, -1, :]
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = -float('Inf')
        probs = F.softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    text = tok.decode(ids[0].tolist())
    logits = model.lm_head(model.stack(model.embed(ids))[0])[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    H = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
    max_H = math.log(VOCAB)
    return text, H, max_H, ids

tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
prompts = [
    'Он вошёл в тёмную комнату и',
    'В начале было Слово, и Слово было',
    'Однажды в студёную зимнюю пору',
]

results = []

# ─── Phase 2 model ───
cfg_old = make_base_cfg()
m_old = build_model(cfg_old)
ckpt_old = torch.load('checkpoints/phase2_best.pt', map_location=DEVICE, weights_only=True)
sd_old = ckpt_old.get('model_state_dict') or ckpt_old.get('model')
m_old.load_state_dict(sd_old, strict=False)
m_old.to(DEVICE).eval()
n_old = sum(p.numel() for p in m_old.parameters())
results.append(f'=== Phase 2 (base, {n_old/1e6:.1f}M params) ===')

for prompt in prompts:
    text, H, max_H, ids = generate_text(m_old, tok, prompt, 100, 0.8, 40)
    results.append(f'\nPrompt: {prompt}')
    results.append(f'IDs after prompt: {ids[0, len(tok.encode(prompt).ids):].tolist()[:30]}')
    results.append(f'Output: {text}')
    results.append(f'Entropy: {H:.3f}/{max_H:.3f} ({H/max_H*100:.0f}%)')

# ─── ACTION model ───
cfg_new = make_action_cfg()
m_new = build_model(cfg_new)
ckpt_new = torch.load('checkpoints/ACTION_step2500.pt', map_location=DEVICE, weights_only=True)
m_new.load_state_dict(ckpt_new['model_state_dict'], strict=False)
m_new.to(DEVICE).eval()
n_new = sum(p.numel() for p in m_new.parameters())
results.append(f'\n\n=== ACTION (all features, {n_new/1e6:.1f}M params) ===')

for prompt in prompts:
    text, H, max_H, ids = generate_text(m_new, tok, prompt, 100, 0.8, 40)
    results.append(f'\nPrompt: {prompt}')
    results.append(f'IDs after prompt: {ids[0, len(tok.encode(prompt).ids):].tolist()[:30]}')
    results.append(f'Output: {text}')
    results.append(f'Entropy: {H:.3f}/{max_H:.3f} ({H/max_H*100:.0f}%)')

with open(OUTFILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))
print(f'Done — results in {OUTFILE}')
