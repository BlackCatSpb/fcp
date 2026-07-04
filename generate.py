"""Генерация русского текста из чекпоинта λ_d."""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack
from tokenizers import Tokenizer as HFTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

class Phase2Model(torch.nn.Module):
    def __init__(self, recurrent=False):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512; cfg.kernel_size = 48
        cfg.recurrent_scan = recurrent
        cfg.weight_tying = True; cfg.lm_head_bias = True
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=cfg.lm_head_bias)
        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

def load_model(path, recurrent=False):
    model = Phase2Model(recurrent=recurrent).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    sd = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt.get('model_fp16', ckpt)
    if sd is ckpt.get('model_fp16', None):
        sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@torch.no_grad()
def generate(model, ids, n_tokens=200, temp=0.8, top_k=40):
    ids = ids.to(DEVICE)
    # Receptive field for sliding window (parallel mode)
    k = model.stack.cfg.kernel_size
    l = model.stack.n_layers
    rf = 1 + (k - 1) * l  # causal conv receptive field
    for _ in range(n_tokens):
        if model.stack.cfg.recurrent_scan:
            # Stateful — only need last token
            logits = model(ids[:, -1:])[:, -1, :]
        else:
            # Sliding window for efficiency
            ctx = ids[:, -rf:] if ids.shape[1] > rf else ids
            logits = model(ctx)[:, -1, :]
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = -float('Inf')
        probs = F.softmax(logits / temp, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/model_step30000.pt')
    parser.add_argument('--tokens', type=int, default=200)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--prompt', type=str, default='Привет, как дела?')
    parser.add_argument('--recurrent', action='store_true', help='use λ_d token recurrence (infinite context)')
    args = parser.parse_args()

    tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
    model = load_model(args.ckpt, recurrent=args.recurrent)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params | {args.ckpt}', flush=True)

    # Tokenize prompt
    enc = tok.encode(args.prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)
    print(f'\nPrompt: {args.prompt}')
    print(f'Tokens: {len(enc.ids)} IDs', flush=True)

    # Generate
    ids = generate(model, ids, n_tokens=args.tokens, temp=args.temp, top_k=args.top_k)
    text = tok.decode(ids[0].tolist())
    print(f'\n--- Generated ({args.tokens} tok, t={args.temp}, top_k={args.top_k}) ---')
    print(text)
    print('---')

    # Output entropy
    with torch.no_grad():
        logits = model(ids)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        H = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
        max_H = math.log(VOCAB)
    print(f'\nEntropy: {H:.3f}/{max_H:.3f} ({H/max_H*100:.0f}% of max)')
