"""
Тест DCT-базиса V и lambda-sliding для MemBind.
Флаги: --dct, --slide, --both, --generate
Использует чекпоинт phase2_step30000.pt без дообучения.

Результаты (step 30000, 2.6M tok, 89M params):
  DCT:     loss=14.6  nan=0  inf=0  OK
  slide:   loss=14.6  nan=0  inf=0  OK
  both:    loss=14.3  nan=0  inf=0  OK генерация связанная
  baseline loss=14.9  nan=0  inf=0  OK
"""
import os, sys, math, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ld_model.core import LDConfig, MemBindStack, compute_spectrum

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D, VOCAB = 896, 50000

def dct_basis(D: int) -> torch.Tensor:
    n = torch.arange(D, dtype=torch.float32)
    k = torch.arange(D, dtype=torch.float32).unsqueeze(1)
    V = torch.cos(math.pi / D * k * (n + 0.5))
    V[0] *= 1.0 / math.sqrt(2)
    V /= math.sqrt(D / 2)
    return V

def slide_lambda(lambda_k: torch.Tensor, layer_idx: int, n_layers: int) -> torch.Tensor:
    t = layer_idx / max(n_layers - 1, 1)
    scale = 0.5 + t
    return lambda_k * scale

def build_model(dct=False, slide=False):
    cfg = LDConfig()
    cfg.D = D; cfg.n_layers = 24; cfg.n_modes = 8
    cfg.vocab = VOCAB; cfg.bottleneck = 896; cfg.kernel_size = 48
    cfg.cov_heads = 4; cfg.cov_r = 16; cfg.bind_r = 16
    cfg.spectrum_type = 'fib_seq'; cfg.spec_lo = 0.8; cfg.spec_hi = 1.8
    cfg.recurrent_scan = False; cfg.weight_tying = True; cfg.lm_head_bias = True

    embed = torch.nn.Embedding(VOCAB, D).to(DEVICE)
    stack = MemBindStack(cfg).to(DEVICE)
    lm_head = torch.nn.Linear(D, VOCAB, bias=True).to(DEVICE)
    lm_head.weight = embed.weight

    ckpt = torch.load('checkpoints/phase2_step30000.pt', map_location=DEVICE, weights_only=True)
    sd = ckpt.get('model_state_dict') or ckpt.get('model') or ckpt
    embed.load_state_dict({'weight': sd['embed.weight']})
    lm_head.bias.data = sd['lm_head.bias']
    stack.load_state_dict({k.replace('stack.', ''): v for k, v in sd.items() if k.startswith('stack.')}, strict=False)

    if dct:
        V = dct_basis(D).to(DEVICE)
        for layer in stack.layers:
            layer.register_buffer('V', V)
            layer.register_buffer('V_T', V.T.contiguous())

    if slide:
        for i, layer in enumerate(stack.layers):
            layer.register_buffer('lambda_k', slide_lambda(layer.lambda_k, i, cfg.n_layers))

    embed.eval(); stack.eval(); lm_head.eval()
    return embed, stack, lm_head

@torch.no_grad()
def test_forward(embed, stack, lm_head, label=''):
    x = torch.randint(0, VOCAB, (2, 128), device=DEVICE)
    h = embed(x[:, :-1])
    h, _ = stack(h)
    logits = lm_head(h)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    nan = torch.isnan(logits).any().item()
    inf = torch.isinf(logits).any().item()
    print(f'  [{label}] loss={loss.item():.4f}  nan={nan}  inf={inf}  '
          f'logits mean={logits.mean().item():.2f} std={logits.std().item():.2f}')
    return not (nan or inf)

@torch.no_grad()
def test_generate(embed, stack, lm_head, prompt_ids, n_tokens=100, label=''):
    from tokenizers import Tokenizer as HFTokenizer
    tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
    ids = prompt_ids.to(DEVICE)
    state = None
    for _ in range(n_tokens):
        h = embed(ids[:, -1:])
        h, state = stack(h, state)
        logits = lm_head(h)[:, -1, :]
        vals, _ = torch.topk(logits, 40)
        logits[logits < vals[:, -1:]] = -float('Inf')
        probs = torch.nn.functional.softmax(logits / 0.9, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    text = tok.decode(ids[0].tolist())
    print(f'\n[{label}]')
    print(text[:600])
    print('...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dct', action='store_true')
    parser.add_argument('--slide', action='store_true')
    parser.add_argument('--both', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    args.dct = args.dct or args.both
    args.slide = args.slide or args.both

    print(f'DCT={args.dct}  SLIDE={args.slide}')
    embed, stack, lm_head = build_model(dct=args.dct, slide=args.slide)
    ok = test_forward(embed, stack, lm_head, 'modified' if (args.dct or args.slide) else 'baseline')
    print(f'  Forward: {"OK" if ok else "FAIL"}')

    print('\n  lambda per layer (first 3 modes):')
    for i in range(min(3, len(stack.layers))):
        lam = stack.layers[i].lambda_k.cpu().tolist()
        print(f'    layer {i:2d}: {[f"{l:.4f}" for l in lam[:3]]}...')

    if args.generate:
        from tokenizers import Tokenizer as HFTokenizer
        tok = HFTokenizer.from_file('russian_tokenizer/tokenizer.json')
        prompt = 'В начале было Слово, и Слово было'
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=DEVICE)
        label = 'DCT+slide' if (args.dct and args.slide) else ('DCT' if args.dct else 'slide')
        test_generate(embed, stack, lm_head, ids, n_tokens=200, label=label)
