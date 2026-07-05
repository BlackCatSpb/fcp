"""Zeckendorf tree readout with learnable centroids."""

import torch
import torch.nn.functional as F


# ─── Fibonacci numbers ──────────────────────────────────────────────────

def fibonacci_bases(vocab_size: int) -> list[int]:
    """Generate Fibonacci numbers up to vocab_size (Zeckendorf bases).
    
    F[0] = 1, F[1] = 2, F[k] = F[k-1] + F[k-2]
    Returns list of all F[k] until the next would exceed vocab_size.
    With F = [1, 2, 3, 5, 8, ..., F_{K-1}], tokens 0..V-1 are covered
    iff F_K - 1 >= V-1  (by Zeckendorf theorem).
    """
    fibs = [1, 2]
    while fibs[-1] < vocab_size:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs  # [1, 2, 3, 5, ..., F_{K-1}]


def zeckendorf_code(token_id: int, fibs: list[int]) -> list[int]:
    """Compute Zeckendorf representation: unique binary, no consecutive 1s.
    
    Returns: [b_{K-1}, ... , b_0]  MSB first (largest Fibonacci first)
    """
    bits = []
    remaining = token_id
    prev = False
    for f in reversed(fibs):
        if remaining >= f and not prev:
            bits.append(1)
            remaining -= f
            prev = True
        else:
            bits.append(0)
            prev = False
    return bits  # MSB first


# ─── Zeckendorf Tree Readout ───────────────────────────────────────────

class ZeckendorfReadout(torch.nn.Module):
    """Zeckendorf tree readout with learnable centroids.
    
    For each token i with Zeckendorf code bits = [b_0,...,b_{K-1}]:
      P(i|h) = Π_k P(b_k | h, state_k)
    
    Centroids: c_{k, state, digit} ∈ ℝᴰ (3 per level, learned)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.vocab = cfg.vocab
        self.D = cfg.D
        
        # Fibonacci bases
        fibs = fibonacci_bases(cfg.vocab)
        self.K = len(fibs)
        self.register_buffer('fibs', torch.tensor(fibs, dtype=torch.long))
        
        # Precompute Zeckendorf codes for accessible tokens
        # tokens 0..min(V, F_K-1) have valid Zeckendorf codes
        self.max_representable = fibs[-1] + (fibs[-2] if len(fibs) > 1 else 1) - 1
        valid_vocab = min(self.vocab, self.max_representable + 1)
        
        print(f'  Zeckendorf: K={self.K} levels, '
              f'vocab={cfg.vocab}, max_repr={self.max_representable}, '
              f'valid={valid_vocab}')
        
        # Precompute codes: codes[token, k] = bit at level k (MSB first)
        codes = torch.zeros(valid_vocab, self.K, dtype=torch.long)
        for i in range(valid_vocab):
            bits = zeckendorf_code(i, fibs)
            for k, b in enumerate(bits):
                codes[i, k] = b
        self.register_buffer('codes', codes)  # (V', K) - only valid tokens
        
        # Learnable centroids: c[level, state, digit] = (D,)
        # state=0 (prev=0): can choose {0, 1}
        # state=1 (prev=1): forced 0
        # We store c[level, 0, 0], c[level, 0, 1], c[level, 1, 0]
        self.centroids = torch.nn.Parameter(
            torch.randn(self.K, 2, 2, self.D) * 0.1
        )
        # c[level, 1, 1] is invalid (no consecutive 1s), set to zero
        with torch.no_grad():
            self.centroids[:, 1, 1, :] = 0.0
    
    def log_probs_for_target(self, h: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
        """Efficient log P(target[i] | h[i]) — no full-V' computation.

        O(B · K) instead of O(B · V' · K).

        Args:
            h: (B, D) hidden state
            target: (B,) target token IDs

        Returns:
            log_probs: (B,) log-probabilities of each target token
        """
        B, D = h.shape
        K = self.K

        c = self.centroids.float()
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1)
        log_probs_k_s = F.log_softmax(logit, dim=-1)
        log_p_flat = log_probs_k_s.reshape(B, K, 4)

        codes = self.codes.to(h.device)
        prev_bits = torch.zeros_like(codes)
        if K > 1:
            prev_bits[:, 1:] = codes[:, :-1]
        combined_idx = prev_bits * 2 + codes

        Vp = codes.shape[0]
        target_clamped = target.clamp(0, Vp - 1)
        idx = combined_idx[target_clamped]  # (B, K)
        log_p = log_p_flat[torch.arange(B, device=h.device)[:, None],
                           torch.arange(K, device=h.device), idx]
        return log_p.sum(dim=-1)

    def forward_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """Compute log P(i|h) for all valid Zeckendorf tokens.

        Vectorized via prev_bit indexing: O(K · V') gather ops.

        Args:
            h: (B, D) hidden state

        Returns:
            log_probs: (B, V') log-probabilities
        """
        B, D = h.shape
        K = self.K

        c = self.centroids.float()
        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1)
        log_probs_k_s = F.log_softmax(logit, dim=-1)
        log_p_flat = log_probs_k_s.reshape(B, K, 4)

        Vp, _ = self.codes.shape
        codes = self.codes

        prev_bits = torch.zeros_like(codes)
        if K > 1:
            prev_bits[:, 1:] = codes[:, :-1]
        combined_idx = prev_bits * 2 + codes

        log_probs = torch.zeros(B, Vp, device=h.device)
        for k in range(K):
            log_probs += log_p_flat[:, k, combined_idx[:, k]]

        return log_probs
    
    def predict(self, h: torch.Tensor, greedy: bool = True,
                temperature: float = 1.0) -> torch.Tensor:
        """Generate token by traversing Zeckendorf tree.

        Args:
            h: (B, D) hidden state
            greedy: if True, pick max at each node; else sample with temp

        Returns:
            tokens: (B,) token IDs
        """
        B, D = h.shape
        device = h.device
        K = self.K
        c = self.centroids.float()

        h_exp = h.view(B, 1, 1, 1, D)
        c_exp = c.view(1, K, 2, 2, D)
        logit = (h_exp * c_exp).sum(dim=-1) / temperature
        probs = F.softmax(logit, dim=-1)

        tokens = torch.zeros(B, dtype=torch.long, device=device)
        fibs = self.fibs

        for b in range(B):
            state = 0
            token_id = 0
            for k in range(K):
                p1 = probs[b, k, state, 1]
                if state == 1:
                    bit = 0
                elif greedy:
                    bit = 1 if p1 > 0.5 else 0
                else:
                    bit = 1 if torch.rand(1, device=device).item() < p1.item() else 0
                if bit:
                    token_id += fibs[k].item()
                state = bit
            tokens[b] = min(token_id, self.vocab - 1)

        return tokens
    
    def compare_with_lm_head(self, h: torch.Tensor, W_embed: torch.Tensor,
                              top_k: int = 10) -> dict:
        """Compare Zeckendorf readout with standard lm_head (h @ W^T).

        Args:
            h: (B, D) hidden state
            W_embed: (V, D) embedding matrix

        Returns:
            dict with overlap, KL, etc.
        """
        B, D = h.shape
        V = W_embed.shape[0]

        h_f = h.float()
        W_f = W_embed.float()
        logits_lm = h_f @ W_f.T
        probs_lm = F.softmax(logits_lm / 1.0, dim=-1)
        
        # Zeckendorf (only valid tokens have codes)
        log_probs_zk = self.forward_log_probs(h)  # (B, V')
        V_zk = log_probs_zk.shape[1]
        probs_zk = torch.exp(log_probs_zk)
        probs_zk = probs_zk / probs_zk.sum(dim=-1, keepdim=True)
        
        # Only compare on valid tokens
        V_common = min(V, V_zk)
        
        # Top-k overlap
        top_lm = probs_lm[:, :V_common].topk(top_k).indices
        top_zk = probs_zk[:, :top_k].topk(top_k).indices  # probs_zk is V_zk-dim
        
        overlap = 0
        for b in range(B):
            overlap += len(set(top_lm[b].tolist()) & set(top_zk[b].tolist()))
        
        # KL on common tokens
        p_lm = probs_lm[:, :V_common].clamp(min=1e-30)
        p_zk = probs_zk[:, :V_common].clamp(min=1e-30)
        p_lm = p_lm / p_lm.sum(dim=-1, keepdim=True)
        p_zk = p_zk / p_zk.sum(dim=-1, keepdim=True)
        kl = (p_lm * (p_lm.log() - p_zk.log())).sum(dim=-1).mean().item()
        
        return {
            'top_k_overlap': overlap / B,
            'kl_div': kl,
            'V_common': V_common,
            'V_total': V,
        }
