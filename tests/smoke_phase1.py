"""Quick smoke test for Phase 1 training."""
import os, sys, math, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from ld_model.core import LDConfig, LDBlock, fibonacci_roots

D, VOCAB, K = 256, 10000, 4
tok = AutoTokenizer.from_pretrained(
    'C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/mlearning/eva_models/qwen3.5-0.8b',
    trust_remote_code=True
)
tok.pad_token = tok.eos_token
VOCAB = min(VOCAB, tok.vocab_size)
print(f'Vocab: {VOCAB}')

cfg = LDConfig()
cfg.D = D
cfg.n_layers = 1
cfg.n_modes = K
cfg.vocab = VOCAB
cfg.intermediate = 1024
cfg.lora_rank = 0
cfg.use_lora = False
lambdas = fibonacci_roots(K + 1)

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.block = LDBlock(cfg, 0, lambdas)
    def forward(self, x):
        h = self.embed(x)
        h_out, a = self.block(h, return_gates=True)
        logits = h_out.float() @ self.embed.weight.T.float()
        return logits, a

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TestModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

ds = load_dataset('wikitext', 'wikitext-103-v1', split='train')
texts = [x['text'] for x in ds if len(x['text'].strip()) > 0][:200]
enc = tok(texts, truncation=True, max_length=129, padding=False)
all_ids = []
for ids in enc['input_ids']:
    all_ids.extend(ids)
chunks = [all_ids[i:i+129] for i in range(0, len(all_ids)-129, 64)]
print(f'Chunks: {len(chunks)}')

for step in range(5):
    ids = torch.tensor(chunks[step], dtype=torch.long).unsqueeze(0).to(DEVICE)
    ids = ids.clamp(max=VOCAB - 1)
    x, y = ids[:, :-1], ids[:, 1:]
    logits, a = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    H = -(a * torch.log(a.clamp(min=1e-10))).sum(dim=-1).mean().item()
    print(f'Step {step}: loss={loss.item():.3f}, ppl={math.exp(loss.item()):.1f}, H(a)={H:.3f}')

print('Smoke test PASSED')
