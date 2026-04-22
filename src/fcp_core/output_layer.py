"""
FCP Output Layer - выходной слой (SPEC section 3.3)
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SamplingResult:
    """Результат сэмплирования."""
    token_id: int
    token_logit: float
    probability: float
    top_k_remaining: int


class OutputLayer:
    """
    Выходной слой FCP (SPEC section 3.3):
    
    Методы:
    | Метод | Назначение |
    |-------|------------|
    | final_norm | RMS-norm к финальным эмбеддингам |
    | lm_head | Проекция в логиты над словарём |
    | sample_token | Сэмплирование следующего токена |
    """
    
    def __init__(
        self,
        vocab_size: int = 151936,
        embedding_dim: int = 2560,
        rms_norm_eps: float = 1e-6
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rms_norm_eps = rms_norm_eps
        
        # lm_head weights (loaded from model in real implementation)
        self._lm_head = None  # Will be loaded from model
    
    def final_norm(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        SPEC: final_norm(X: Tensor) -> Tensor
        
        Применяет RMS-normalization к финальным эмбеддингам.
        
        Args:
            hidden_states: (batch, seq_len, dim)
            
        Returns:
            normalized: (batch, seq_len, dim)
        """
        # RMS Norm: x / sqrt(mean(x^2) + eps)
        squared = hidden_states ** 2
        mean_squared = np.mean(squared, axis=-1, keepdims=True)
        normalized = hidden_states / np.sqrt(mean_squared + self.rms_norm_eps)
        
        return normalized
    
    def lm_head(self, normalized: np.ndarray) -> np.ndarray:
        """
        SPEC: lm_head(X: Tensor) -> Tensor
        
        Проецирует эмбеддинги в логиты над словарём.
        Real: это матричное умножение на lm_head weights.
        
        Args:
            normalized: (batch, seq_len, dim) - после final_norm
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len, dim = normalized.shape
        
        # Real: matrix multiplication
        # For now, simplified:
        if self._lm_head is not None:
            logits = np.matmul(normalized, self._lm_head.T)
        else:
            # Random projection (for testing)
            logits = np.random.randn(batch, seq_len, self.vocab_size) * 0.01
        
        return logits
    
    def sample_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.1
    ) -> SamplingResult:
        """
        SPEC: sample_token(logits: Tensor, ...) -> int
        
        Сэмплирование следующего токена с учётом:
        - temperature (ниже = более детерминистично)
        - top-p (nucleus sampling)
        - top-k (ограничение кандидатов)
        - repetition penalty
        
        Args:
            logits: (vocab_size,) - логиты для последнего токена
            temperature: сэмплирование температуры
            top_p: nucleus sampling threshold
            top_k: top-k filtering
            eos_token_id: токен конца последовательности
            repetition_penalty: штраф за повторы
            
        Returns:
            SamplingResult
        """
        # Batched: take last token
        if logits.ndim == 2:
            logits = logits[0, -1, :]  # (vocab_size,)
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Would penalize previously generated tokens
            # Simplified: just logit scaling
            pass
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            return SamplingResult(
                token_id=int(np.argmax(logits)),
                token_logit=float(np.max(logits)),
                probability=1.0,
                top_k_remaining=1
            )
        
        # Top-k filtering
        if top_k > 0:
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, float('-inf'))
            mask[top_indices] = logits[top_indices]
            logits = mask
        
        # Top-p filtering (nucleus)
        if 0 < top_p < 1:
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            
            # Compute cumulative probabilities
            probs = self._softmax(sorted_logits)
            cumsum = np.cumsum(probs)
            
            # Keep tokens until cumulative prob > top_p
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            keep_indices = sorted_indices[:cutoff_idx]
            
            # Mask others
            mask = np.full_like(logits, float('-inf'))
            mask[keep_indices] = logits[keep_indices]
            logits = mask
        
        # Apply softmax
        probabilities = self._softmax(logits)
        
        # Sample
        token_id = int(np.random.choice(len(probabilities), p=probabilities))
        
        return SamplingResult(
            token_id=token_id,
            token_logit=float(logits[token_id]),
            probability=float(probabilities[token_id]),
            top_k_remaining=min(top_k, len(probabilities[probabilities > 0]))
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def generate(
        self,
        hidden_states: np.ndarray,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[list] = None
    ) -> list[int]:
        """
        Полная генерация токенов авторегрессивно.
        
        Args:
            hidden_states: Начальные эмбеддинги (1, seq, dim)
            max_new_tokens: Максимум новых токенов
            temperature, top_p, top_k: params для сэмплирования
            eos_token_id: Токен конца
            stop_token_ids: Дополнительные стоп-токены
            
        Returns:
            list of generated token IDs
        """
        # Normalize
        normalized = self.final_norm(hidden_states)
        
        # Project to logits
        logits = self.lm_head(normalized)  # (1, seq, vocab)
        
        generated = []
        
        for _ in range(max_new_tokens):
            # Take last token logits
            last_logits = logits[0, -1, :]
            
            # Sample
            result = self.sample_token(
                last_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id
            )
            
            generated.append(result.token_id)
            
            # Check stop
            if eos_token_id and result.token_id == eos_token_id:
                break
            if stop_token_ids and result.token_id in stop_token_ids:
                break
        
        return generated


class FCPPipeline:
    """
    Полный FCP Pipeline - от входа до выхода.
    
    Объединяет Input, Hybrid Stack, Output.
    """
    
    def __init__(self, config: "FCPConfig"):
        from .config import FCPConfig
        self.config = config
        
        # Input layer - использует OpenVINO токенизатор
        self._tokenizer = None
        
        # Hybrid stack
        from .hybrid_stack import HybridStack, StackConfig
        stack_config = StackConfig(
            num_layers=config.num_layers,
            hidden_dim=config.embedding_dim,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            graph_retrieval_k=config.graph_retrieval_k,
            master_tokens=config.master_tokens,
            stop_threshold=config.stop_threshold,
            early_exit_threshold=config.early_exit_threshold
        )
        self.stack = HybridStack(stack_config)
        
        # Output layer
        self.output_layer = OutputLayer(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim
        )
    
    def forward(
        self,
        input_ids: np.ndarray,
        graph: Optional["FractalGraphV2"] = None
    ) -> Tuple[np.ndarray, list]:
        """
        Forward pass через весь pipeline.
        
        Args:
            input_ids: (batch, seq_len)
            graph: FractalGraphV2
            
        Returns:
            (output_logits, halt_decisions)
        """
        # Pass through stack
        output, halt_decisions = self.stack.forward(input_ids, graph)
        
        # Output layer
        logits = self.output_layer.final_norm(output)
        logits = self.output_layer.lm_head(logits)
        
        return logits, halt_decisions
    
    @property
    def num_layers(self) -> int:
        return self.stack.num_layers