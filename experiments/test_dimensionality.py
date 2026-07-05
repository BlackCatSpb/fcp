"""
Тест обучаемости λ_d модели на разных размерностях D.
Загружает чекпоинт phase2_best.pt (D=896), тестирует D ∈ {512,768,896,1024,1536,2048}.
Без сохранения — чисто проверка стабильности.

Запуск:
    python test_dimensionality.py
    python test_dimensionality.py --Ds 512,768,1024 --steps 100
"""

import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ld_model.core import LDConfig, LDStack

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}\n')

VOCAB = 50000
N_MODES = 4
N_LAYERS = 12
SEQ_LEN = 128
BATCH_SIZE = 8
LR = 1e-3
STEPS = 300
LOG_EVERY = 50
GRAD_CLIP = 1.0
DATA_FILE = 'russian_chunks.npy'
CKPT_FILE = 'checkpoints/phase2_best.pt'


class Phase2Model(nn.Module):
    def __init__(self, D: int):
        super().__init__()
        self.D = D
        self.embed = nn.Embedding(VOCAB, D)
        cfg = LDConfig()
        cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = N_MODES
        cfg.vocab = VOCAB; cfg.bottleneck = 512
        self.stack = LDStack(cfg)
        self.lm_head = nn.Linear(D, VOCAB, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed(input_ids)))

    def count_params(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n


def load_data(n_chunks: int = 500):
    print(f'  Loading {n_chunks} chunks from {DATA_FILE}...', end=' ')
    t0 = time.time()
    arr = np.load(DATA_FILE)
    arr = arr[:n_chunks]
    x = torch.tensor(arr[:, :-1], dtype=torch.long)
    y = torch.tensor(arr[:, 1:], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)
    print(f'{arr.shape[0]} chunks, {time.time()-t0:.1f}s')
    return loader


def load_checkpoint_into(model: Phase2Model, ckpt_path: str):
    """Загружает state_dict чекпоинта в модель, если размерности совпадают."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    sd_ckpt = ckpt['model_state_dict']
    # Проверяем совпадение D
    ckpt_D = sd_ckpt['embed.weight'].shape[1]
    if ckpt_D != model.D:
        print(f'  [SKIP] checkpoint D={ckpt_D} != model D={model.D}')
        return False
    missing, unexpected = model.load_state_dict(sd_ckpt, strict=False)
    if missing:
        print(f'  [WARN] missing keys: {missing}')
    if unexpected:
        print(f'  [WARN] unexpected keys: {unexpected}')
    print(f'  [OK] loaded checkpoint: step={ckpt["step"]}, epoch={ckpt["epoch"]}, best_ppl={ckpt["best_ppl"]:.1f}')
    return True


def test_dimension(D: int, loader, ckpt_path: str = None, steps: int = STEPS):
    print(f'\n{"="*60}')
    print(f'  D = {D}')
    print(f'{"="*60}')

    model = Phase2Model(D).to(DEVICE)
    n_params = model.count_params()
    print(f'  Params: {n_params/1e6:.2f}M')

    # Загружаем чекпоинт если размерность совпадает
    loaded = False
    if ckpt_path and os.path.exists(ckpt_path):
        loaded = load_checkpoint_into(model, ckpt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    model.train()
    losses = []
    nan_count = 0
    grad_norms = []
    t_start = time.time()

    loader_iter = iter(loader)
    for step in range(1, steps + 1):
        try:
            bx, by = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            bx, by = next(loader_iter)

        bx, by = bx.to(DEVICE), by.to(DEVICE)

        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))

        if torch.isnan(loss).item():
            nan_count += 1
            if nan_count <= 3:
                print(f'  [NAN] step {step}')
            continue

        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norms.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)

        optimizer.step()

        losses.append(loss.item())

        if step % LOG_EVERY == 0:
            avg_loss = sum(losses[-LOG_EVERY:]) / min(LOG_EVERY, len(losses))
            ppl = math.exp(avg_loss)
            avg_grad = sum(grad_norms[-LOG_EVERY:]) / min(LOG_EVERY, len(grad_norms))
            print(f'    step {step:4d} | loss={avg_loss:.4f} | ppl={ppl:.1f} | grad={avg_grad:.4f}')

    elapsed = time.time() - t_start
    valid = len(losses)

    if valid > 0:
        first_loss = losses[0]
        last_loss = losses[-1]
        last_ppl = math.exp(sum(losses[-min(LOG_EVERY, valid):]) / min(LOG_EVERY, valid))
        mean_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    else:
        first_loss = last_loss = last_ppl = mean_grad = float('nan')

    return {
        'D': D,
        'params_M': n_params / 1e6,
        'loaded_ckpt': loaded,
        'steps': steps,
        'valid_steps': valid,
        'nan_count': nan_count,
        'first_loss': first_loss,
        'last_loss': last_loss,
        'last_ppl': last_ppl,
        'mean_grad': mean_grad,
        'time_s': elapsed,
        'ok': nan_count == 0 and valid == steps,
    }


def print_summary(results):
    print(f'\n{"="*70}')
    print('  Svodka: test obuchaemosti lambda_d na raznyh D')
    print(f'{"="*70}')
    print(f'  {"D":>6s} | {"Params":>8s} | {"Ckpt":>5s} | {"Shagi":>5s} | {"NaN":>3s} | '
          f'{"Loss->":>8s} | {"Loss|":>8s} | {"PPL|":>6s} | {"Grad":>7s} | {"Status":>10s}')
    print(f'  {"-"*6} | {"-"*8} | {"-"*5} | {"-"*5} | {"-"*3} | '
          f'{"-"*8} | {"-"*8} | {"-"*6} | {"-"*7} | {"-"*10}')
    for r in results:
        status = 'OK' if r['ok'] else 'NAN!' if r['nan_count'] > 0 else 'PARTIAL'
        print(f'  {r["D"]:>6d} | {r["params_M"]:>7.2f}M | '
              f'{"yes" if r["loaded_ckpt"] else "no":>5s} | '
              f'{r["valid_steps"]:>5d} | {r["nan_count"]:>3d} | '
              f'{r["first_loss"]:>8.4f} | {r["last_loss"]:>8.4f} | '
              f'{r["last_ppl"]:>6.1f} | {r["mean_grad"]:>7.4f} | {status:>10s}')

    all_ok = all(r['ok'] for r in results)
    print(f'\n  >> Itogo: {"VSE TESTY PROJDENY" if all_ok else "EST PROBLEMY"}')


def main():
    parser = argparse.ArgumentParser(description='Тест обучаемости λ_d на разных D')
    parser.add_argument('--Ds', default='512,768,896,1024,1536,2048',
                        help='Список размерностей через запятую')
    parser.add_argument('--steps', type=int, default=STEPS,
                        help='Количество шагов обучения на каждую D')
    parser.add_argument('--chunks', type=int, default=500,
                        help='Количество чанков данных для теста')
    parser.add_argument('--data', default=DATA_FILE)
    parser.add_argument('--ckpt', default=CKPT_FILE)
    args = parser.parse_args()

    Ds = [int(d) for d in args.Ds.split(',')]

    # Загружаем данные один раз
    loader = load_data(n_chunks=args.chunks)

    results = []
    for D in Ds:
        # Проверяем что D делится на N_MODES
        if D % N_MODES != 0:
            print(f'\n  [SKIP] D={D}: не делится на n_modes={N_MODES}')
            continue
        r = test_dimension(D, loader, ckpt_path=args.ckpt, steps=args.steps)
        results.append(r)

    print_summary(results)


if __name__ == '__main__':
    main()
