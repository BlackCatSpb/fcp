"""
Lightweight generation — loads checkpoint on CPU, model to GPU.
VRAM: ~526 MB instead of ~2 GB.
"""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, MemBindStack
from tokenizers import Tokenizer as HFTokenizer

DEVICE = 'cuda'
D, VOCAB = 896, 50000
OUT = 'outputs/_gen_result.txt'

def build_model(cfg_updates=None):
    cfg = LDConfig()
    cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8; cfg.vocab = VOCAB
    cfg.bottleneck = 896; cfg.kernel_size = 48
    cfg.weight_tying = True; cfg.lm_head_bias = True
    cfg.arch = 'membind'; cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
    cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
    cfg.dct_basis = True; cfg.lambda_sliding = True
    cfg.cov_first_moment = True; cfg.cov_rf = True; cfg.cov_rf_dim = 64
    cfg.cov_multi_timescale = True; cfg.cov_mirror = True
    m = torch.nn.Module()
    m.embed = torch.nn.Embedding(VOCAB, D)
    m.stack = MemBindStack(cfg)
    m.lm_head = torch.nn.Linear(D, VOCAB, bias=True)
    m.lm_head.weight = m.embed.weight
    return m

def load_model_light(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd = ckpt['model_state_dict']
    model = build_model()
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()
    return model

def generate(model, tok, prompt, n_tokens=200, temp=0.8, top_k=40):
    k, l, rf = 48, 24, 1 + 47 * 24
    enc = tok.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(n_tokens):
            ctx = ids[:, -rf:] if ids.shape[1] > rf else ids
            logits = model.lm_head(model.stack(model.embed(ctx))[0])[:, -1, :]
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = -float('Inf')
            probs = F.softmax(logits / temp, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0].tolist())

tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')

lines = []
for name, path in [('Phase 2', 'checkpoints/phase2_best.pt'), ('ACTION', 'checkpoints/ACTION_step2500.pt')]:
    model = load_model_light(path)
    n = sum(p.numel() for p in model.parameters())
    alloc = torch.cuda.memory_allocated() / 1e6
    lines.append(f'=== {name} ({n/1e6:.1f}M params, {alloc:.0f}MB VRAM) ===')
    
    for prompt in ['Он вошёл в тёмную комнату и', 'В начале было Слово, и Слово было']:
        text = generate(model, tok, prompt, 100, 0.8, 40)
        lines.append(f'\nPrompt: {prompt}')
        lines.append(f'Output: {text}')
    
    lines.append('')
    del model
    torch.cuda.empty_cache()

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Results written to {OUT}')
