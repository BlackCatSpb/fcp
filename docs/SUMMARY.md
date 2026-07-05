# MemBind: Summary

**Multi-head covariance memory + bilinear bind + spectral operator.**
No softmax, no sigmoid gates, no attention.

## Current status (July 2026)

- **Architecture**: MemBind, validated
- **Params**: 89.1M trainable (D=896, L=24, H=4, r=16)
- **Training**: fib_seq spectrum, 2.6M tok/epoch, ~2.9s/step on MX550
- **PPL**: ~5,000 at 384K tokens (cold start, improving)
- **Context**: unlimited at inference (96KB covariance state)

## Key results

| Metric | Value |
|--------|-------|
| Training speed | ~962 tok/s |
| Inference speed (fp32) | ~370K tok/s |
| Max context (2GB VRAM) | ∞ |
| VRAM per token | 0 bytes (fixed 96KB state) |

## Full documentation

See **[LAMBDA_ARCHITECTURE.md](LAMBDA_ARCHITECTURE.md)** for complete architecture description.
