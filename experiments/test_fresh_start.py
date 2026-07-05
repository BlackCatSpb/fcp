"""Fresh start: train Phase2Model from scratch with all features.
Tests whether adaptive_gain + learnable_V + global_context produce balanced
layer usage when trained from step 0 (vs loaded old checkpoint)."""

import os, sys, math, time, glob, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

D = 896; VOCAB = 50000; N_MODES = 4; N_LAYERS = 12
BATCH_SIZE = 4; ACCUM_STEPS = 4; SEQ_LEN = 64
EFF_BATCH = BATCH_SIZE * ACCUM_STEPS
LR = 1e-3; WARMUP_FRAC = 0.05; EPOCHS = 1; GRAD_CLIP = 1.0
TOTAL_STEPS = 5000; LOG_EVERY = 100; CKPT_EVERY = 100

# ─── Data: first N chunks from russian ───────────────────────────────────
arr = np.load('russian_chunks.npy')
N_CHUNKS = 20000  # enough for 5000 steps
ids = torch.tensor(arr[:N_CHUNKS], dtype=torch.long)
x = ids[:, :-1].to(DEVICE)
y = ids[:, 1:].to(DEVICE)
loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)
print(f'Data: {N_CHUNKS} chunks of {SEQ_LEN}, {N_CHUNKS*SEQ_LEN/1e6:.1f}M tok')

# ─── Model (fresh, all features enabled) ─────────────────────────────────
class Phase2Model(torch.nn.Module):
    def __init__(self, use_global_context=False):
        super().__init__()
        self.use_global_context = use_global_context
        self.embed = torch.nn.Embedding(VOCAB, D)
        # Weight tying: embed.weight ≈ lm_head.weight. Init N(0,1) даёт logits в ±900,
        # убивает softmax. Стандартный Linear init: U(-1/√D, 1/√D), σ=1/√(3D)≈0.019.
        torch.nn.init.uniform_(self.embed.weight, -1/math.sqrt(D), 1/math.sqrt(D))
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512; cfg.kernel_size = 48
        cfg.use_global_context = use_global_context
        cfg.weight_tying = True; cfg.lm_head_bias = True
        self.stack = LDStack(cfg)
        self.lm_head = torch.nn.Linear(D, VOCAB, bias=cfg.lm_head_bias)
        if cfg.weight_tying:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, return_gates=False):
        h = self.embed(input_ids)
        if self.use_global_context:
            ctx = self.stack(h).mean(dim=1)       # first pass → pool
            h2 = self.embed(input_ids) + self.stack.ctx_proj(ctx).unsqueeze(1)
            h = self.stack(h2, context=ctx)        # second pass with context
        elif return_gates:
            h, gates = self.stack(h, return_gates=True)
            return self.lm_head(h), gates
        else:
            h = self.stack(h)
        return self.lm_head(h)

model = Phase2Model(use_global_context=False).to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_cayley = sum(p.numel() for n, p in model.named_parameters() if 'V_cay' in n)
n_ctx = sum(p.numel() for n, p in model.named_parameters() if 'ctx_proj' in n)
print(f'Model: {n_all/1e6:.1f}M params | Cayley: {n_cayley:,} | ctx_proj: {n_ctx:,}')

# ─── Sanity ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx = next(iter(loader))[0][:1]
    h = model.embed(bx)
    h = model.stack(h)
    logits = model.lm_head(h)
    print(f'sanity: stack=[{h.min():.2f},{h.max():.2f}] logits=[{logits.min():.2f},{logits.max():.2f}]')

# ─── Training ────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

def get_lr(step):
    if step < TOTAL_STEPS * WARMUP_FRAC:
        return LR * (step + 1) / max(TOTAL_STEPS * WARMUP_FRAC, 1)
    progress = (step - TOTAL_STEPS * WARMUP_FRAC) / max(TOTAL_STEPS - TOTAL_STEPS * WARMUP_FRAC, 1)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

step = 0; n_batches = 0; epoch_loss = 0.0
optimizer.zero_grad()
t0 = time.perf_counter()

print(f'\nTraining: {TOTAL_STEPS} steps, B={BATCH_SIZE}, accum={ACCUM_STEPS} (eff={EFF_BATCH})')
print(f'  warmup={int(TOTAL_STEPS*WARMUP_FRAC)} steps')

while step < TOTAL_STEPS:
    for bx, by in loader:
        if step >= TOTAL_STEPS: break
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        loss = loss / ACCUM_STEPS
        loss.backward()
        epoch_loss += loss.item() * ACCUM_STEPS
        n_batches += 1
        step += 1

        if step % ACCUM_STEPS == 0 or step == TOTAL_STEPS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr = get_lr(step)
            for g in optimizer.param_groups: g['lr'] = lr
            optimizer.step(); optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            ppl = math.exp(min(avg_loss, 80)) if avg_loss < 700 else float('inf')
            norms = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                     for l in model.stack.layers if l.V_cay_A is not None]
            spreads = [l.V_cay_A.norm().item() + l.V_cay_B.norm().item()
                      for l in model.stack.layers if l.V_cay_A is not None] if model.stack.adaptive_gain else []
            v = f' |A+B|~[{", ".join(f"{n:.2f}" for n in norms[:4])}...]' if norms else ''
            if model.use_global_context:
                v += ' ctx=ON'
            print(f'  Step {step:4d} | loss={epoch_loss/max(n_batches,1):.4f} | ppl={ppl:.1f} | lr={lr:.2e}{v}')

        if step % CKPT_EVERY == 0:
            ckpt_path = f'checkpoints/fresh_start_{step}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'step': step, 'loss': epoch_loss / max(n_batches,1)}, ckpt_path)

# ─── Final evaluation ───────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    bx_eval, by_eval = next(iter(loader))
    logits, gates = model(bx_eval), None
    # We need gates: recreate forward with return_gates
    # Quick hack: use the model's stack directly
    h = model.embed(bx_eval)
    h, gates = model.stack(h, return_gates=True)
    logits = model.lm_head(h)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), by_eval.reshape(-1))
    final_loss = loss.item()
    final_ppl = math.exp(min(final_loss, 80)) if final_loss < 700 else float('inf')
    print(f'\n>> Final: PPL={final_ppl:.1f}, Loss={final_loss:.4f}')

    # Adaptive gain analysis (sigmoid gates: mean activation = gain per token)
    gates_np = gates.cpu().numpy()
    K = gates_np.shape[-1]
    print(f'\n  Layer | Mean α | Std α  | Min α  | Max α  | Tokens α>0.5')
    for lidx in range(N_LAYERS):
        g = gates_np[lidx]
        mu = g.mean()
        s = g.std()
        mn = g.min()
        mx = g.max()
        frac = (g.mean(axis=-1) > 0.5).mean()  # fraction of tokens with mean gate > 0.5
        print(f'  L{lidx:>3} | {mu:.3f}  | {s:.3f}  | {mn:.3f}  | {mx:.3f}  | {frac*100:>5.1f}%')

print(f'\nTime: {time.perf_counter()-t0:.0f}s')
print('Fresh start test complete.')

# Save checkpoint for analyzer
torch.save({'model_state_dict': model.state_dict(), 'step': step},
           'checkpoints/fresh_start_test.pt')
print('Saved checkpoints/fresh_start_test.pt')
