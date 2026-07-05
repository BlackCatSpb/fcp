"""Test adaptive gain in LDStack."""
import torch, sys
sys.path.insert(0, '.')
from ld_model.core import LDConfig, LDStack

D, K, N_LAYERS = 256, 4, 6
cfg = LDConfig()
cfg.D = D; cfg.n_layers = N_LAYERS; cfg.n_modes = K; cfg.vocab = 10000
cfg.bottleneck = 64
cfg.adaptive_gain = True

model = LDStack(cfg)
model.eval()

B, L = 2, 16
x = torch.randn(B, L, D)

with torch.no_grad():
    h, gates = model(x, return_gates=True)
    print(f"Output shape: {h.shape}")
    print(f"Gates shape:  {gates.shape}")

    # Spread (gain) per layer
    print(f"\n=== Per-layer gain ===")
    for lidx in range(N_LAYERS):
        spread = gates[lidx].std(dim=-1)
        gain = spread.mean().item()
        print(f"  Layer {lidx}: spread={spread.mean():.3f} gain={gain:.3f}")

    # Test global context
    print(f"\n=== Global context ===")
    cfg2 = LDConfig()
    cfg2.D = D; cfg2.n_layers = N_LAYERS; cfg2.n_modes = K; cfg2.vocab = 10000
    cfg2.bottleneck = 64
    cfg2.use_global_context = True
    model2 = LDStack(cfg2)
    model2.eval()
    ctx = torch.randn(B, D)
    h_ctx = model2(x, context=ctx)
    print(f"With context: {h_ctx.shape}")

    # Adaptive gain off
    cfg3 = LDConfig()
    cfg3.D = D; cfg3.n_layers = N_LAYERS; cfg3.n_modes = K; cfg3.vocab = 10000
    cfg3.bottleneck = 64
    cfg3.adaptive_gain = False
    model3 = LDStack(cfg3)
    model3.eval()
    h3 = model3(x)
    print(f"Adaptive gain off: {h3.shape}")

print("\nAll tests passed.")
