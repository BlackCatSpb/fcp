"""Генерация текста из чекпоинта model_step25000.pt"""
import torch, sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB, N_MODES, N_LAYERS = 896, 50000, 4, 12

class Phase2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)
    def forward(self, x):
        return self.lm_head(self.stack(self.embed(x)))

def load(path):
    model = Phase2Model().to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if 'model_fp16' in ckpt:
        sd = {k: v.float() if v.dtype==torch.float16 else v for k,v in ckpt['model_fp16'].items()}
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def generate(model, prompt, n_tokens=100, temp=0.8, top_k=50):
    model.eval()
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(prompt)[:, -1, :]  # (1, V)
            # Top-k filtering
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = -float('Inf')
            probs = F.softmax(logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            prompt = torch.cat([prompt, next_tok], dim=1)
    return prompt

def show_stats(model):
    n = sum(p.numel() for p in model.parameters())
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n/1e6, n_t/1e6

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/model_step25000.pt')
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--temp', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--prompt_len', type=int, default=20)
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    model = load(args.ckpt)
    n, n_t = show_stats(model)
    print(f'Model: {n:.1f}M total, {n_t:.1f}M trainable')

    # Random prompt seed
    prompt = torch.randint(0, min(VOCAB, 1000), (1, args.prompt_len)).to(DEVICE)
    print(f'Prompt: {prompt[0].tolist()}')

    out = generate(model, prompt, n_tokens=args.tokens, temp=args.temp, top_k=args.top_k)
    print(f'\nGenerated {args.tokens} tokens (temp={args.temp}, top_k={args.top_k}):')
    print(f'  IDs: {out[0, args.prompt_len:].tolist()}')
    print(f'  Full: {out[0].tolist()}')

    # Quick quality check: entropy of output distribution
    with torch.no_grad():
        logits = model(out)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        H = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
        max_H = math.log(VOCAB)
    print(f'\nOutput entropy: {H:.3f} / max={max_H:.3f} (conf={"low" if H/max_H > 0.5 else "high"})')
