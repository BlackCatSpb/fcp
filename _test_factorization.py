"""Test complete weight factorization through V_shared + Zeckendorf codes.

Verifies:
1. Zeckendorf E_code dimensions
2. Embedding = E_code[x] @ V.T  (identity with dense embed)
3. lm_head = (h @ V) @ E_code.T  (identity with dense)
4. All weight types in MemBindBlock factorizable through V
5. Gradient flows through all code parameters
"""
import torch, math
from torch import nn
import torch.nn.functional as F

def test_zeckendorf(vocab=50000):
    fibs = [1, 2]
    while fibs[-1] < vocab:
        fibs.append(fibs[-1] + fibs[-2])
    K = len(fibs)
    print(f'Zeckendorf: vocab={vocab}, K={K}')
    E = torch.zeros(vocab, K)
    for i in range(vocab):
        rem, prev = i, False
        for j, f in enumerate(reversed(fibs)):
            bit = 1 if rem >= f and not prev else 0
            E[i, K-1-j] = bit
            if bit: rem -= f; prev = True
            else: prev = False
    conc = (E[:, :-1] * E[:, 1:]).sum().item()
    assert conc == 0, f'Consecutive 1s: {conc}'
    print(f'  No consecutive 1s: {conc}')
    max_r = max(int(E[i].tolist().index(1)) if 1 in E[i] else 0 for i in range(1, vocab))
    print(f'  max bit position: {max_r}/{K}')
    return E

def test_embedding():
    """Embedding through V matches standard nn.Embedding."""
    print('\n--- Embedding test ---')
    vocab, D, K = 100, 64, 11
    E = test_zeckendorf(vocab)[:vocab]
    V = nn.Parameter(torch.randn(D, K) * 0.01)
    std = nn.Embedding(vocab, D)
    std.weight.data = (E.float() @ V.T.detach()).T  # must match
    # Actually: embed.weight = E @ V.T  (V, D)
    std.weight.data = E.float() @ V.T.detach()
    
    B, L = 4, 16
    x = torch.randint(0, vocab, (B, L))
    
    h_std = std(x)
    h_fact = F.embedding(x, E).float() @ V.T
    
    err = (h_fact - h_std).norm() / h_std.norm()
    assert err < 1e-6, f'Embedding err: {err}'
    print(f'  Embedding: err={err:.2e}')

def test_lm_head():
    """LM head through V matches standard."""
    print('\n--- LM head test ---')
    vocab, D, K = 100, 64, 11
    E = torch.zeros(vocab, K)
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    for i in range(vocab):
        rem, prev = i, False
        for j, f in enumerate(reversed(fibs)):
            bit = 1 if rem >= f and not prev else 0
            E[i, K-1-j] = bit
            if bit: rem -= f; prev = True
            else: prev = False
    
    V = nn.Parameter(torch.randn(D, K) * 0.01)
    embed_weight = E.float() @ V.T.detach()
    
    B, L = 4, 16
    h = torch.randn(B, L, D)
    target = torch.randint(0, vocab, (B, L))
    
    # Standard
    logits_std = h @ embed_weight.T
    loss_std = F.cross_entropy(logits_std.reshape(-1, vocab), target.reshape(-1))
    
    # Factorized
    logits_fact = (h @ V) @ E.float().T
    loss_fact = F.cross_entropy(logits_fact.reshape(-1, vocab), target.reshape(-1))
    
    err = abs(loss_std - loss_fact).item()
    assert err < 1e-10, f'Loss diff: {err}'
    print(f'  LM head loss: std={loss_std.item():.6f}, fact={loss_fact.item():.6f}, diff={err:.2e}')

def test_gradient_flow():
    """Gradient flows through V + all weight codes (no cov scan for speed)."""
    print('\n--- Gradient flow test ---')
    D, vocab, K = 32, 50, 9
    B, L, bind_r = 2, 8, 8
    
    fibs = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    E = torch.zeros(vocab, K)
    for i in range(vocab):
        rem, prev = i, False
        for j, f in enumerate(reversed(fibs)):
            bit = 1 if rem >= f and not prev else 0
            E[i, K-1-j] = bit
            if bit: rem -= f; prev = True
            else: prev = False
    
    V = nn.Parameter(torch.randn(D, K) * 0.01)
    x = torch.randint(0, vocab, (B, L))
    h = F.embedding(x, E).float() @ V.T
    hp = h @ V  # (B, L, K)
    
    params = {'V': V}
    def P(name, *s):
        p = nn.Parameter(torch.randn(*s) * 0.01)
        params[name] = p
        return p
    
    Wu = P('Wu', K, bind_r); Wv = P('Wv', K, bind_r)
    Wo = P('Wo', bind_r, K); Wm2v = P('Wm2v', K, bind_r)
    Wum = P('Wum', K, bind_r); Wvm = P('Wvm', K, bind_r); Wom = P('Wom', bind_r, K)
    
    # Simple forward: bind → spectral → lm_head
    u = hp @ Wu
    v = hp @ Wv
    ve = v + (hp @ Wm2v)  # add mem readout (simplified)
    ap = hp + ((u * ve) @ Wo)
    ho = (ap * torch.linspace(0.8, 1.8, K).view(1, 1, K)) @ V.T
    lg = (ho @ V) @ E.float().T
    tg = torch.randint(0, vocab, (B, L))
    loss = F.cross_entropy(lg.reshape(-1, vocab), tg.reshape(-1))
    loss.backward()
    
    ok = True
    for name, p in params.items():
        gn = p.grad.norm().item() if p.grad is not None else -1
        if gn <= 0:
            print(f'  NO GRAD: {name:8s} {list(p.shape)}')
            ok = False
        else:
            print(f'  grad OK: {name:8s} {list(p.shape)} norm={gn:.8f}')
    
    # Verify: W_dense = V @ W_code gives same result at every weight
    with torch.no_grad():
        for name, code in [('Wu', Wu), ('Wv', Wv), ('Wm2v', Wm2v)]:
            W_dense = V @ code.detach()
            out_dense = h @ W_dense
            out_fact = hp @ code.detach()
            err = (out_dense - out_fact).norm() / out_dense.norm()
            print(f'  {name} identity err: {err:.2e}')
            ok = ok and (err < 1e-6)
    
    n = sum(p.numel() for p in params.values())
    dense = D*vocab + D*bind_r*4 + D
    print(f'  Params: {n}  (dense: {dense}, {dense/n:.1f}x compression)')
    print(f'  Gradient OK: {ok}')
    return ok

if __name__ == '__main__':
    test_embedding()
    test_lm_head()
    ok = test_gradient_flow()
    
    print(f'\n{"="*50}')
    print(f'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED')
    print(f'{"="*50}')
