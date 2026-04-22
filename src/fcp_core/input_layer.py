"""
FCP Input Layer - токенизация, эмбеддинги, RoPE
"""
import math
import numpy as np
from typing import Optional
from dataclasses import dataclass


class InputLayer:
    """
    Входной слой FCP:
    - Токенизация (BPE)
    - Embeddings table
    - Rotary Positional Encoding (RoPE)
    """
    
    def __init__(
        self,
        tokenizer,  # OpenVINO Tokenizer
        embedding_dim: int = 2048,
        max_seq_len: int = 4096
    ):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Build embeddings table (будет загружена из модели)
        vocab_size = tokenizer.vocab_size()
        self.embeddings = None  # Loaded from model
        
        # Precompute RoPE cache
        self._rope_cache = self._build_rope_cache(max_seq_len, embedding_dim)
    
    def _build_rope_cache(self, max_seq_len: int, dim: int) -> np.ndarray:
        """Precompute Rotary Positional Embeddings."""
        # RoPE uses sin/cos at different frequencies
        half_dim = dim // 2
        freqs = np.exp(
            -np.log(1e9) * np.arange(half_dim) / half_dim
        ).reshape(1, 1, -1)
        
        positions = np.arange(max_seq_len).reshape(-1, 1, 1)
        angles = positions * freqs
        
        # Interleave sin and cos
        cache = np.zeros((max_seq_len, dim))
        cache[:, ::2] = np.sin(angles).squeeze()
        cache[:, 1::2] = np.cos(angles).squeeze()
        
        return cache
    
    def get_positional_embedding(self, position: int) -> np.ndarray:
        """Get RoPE embedding for position."""
        if position >= self.max_seq_len:
            raise ValueError(f"Position {position} >= max_seq_len {self.max_seq_len}")
        return self._rope_cache[position].copy()
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings with RoPE.
        
        Args:
            token_ids: (batch, seq_len) - token IDs
            
        Returns:
            embeddings: (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings from table
        embeddings = self._get_token_embeddings(token_ids)
        
        # Apply RoPE to embeddings
        embeddings = self._apply_rope(embeddings, token_ids)
        
        return embeddings
    
    def _get_token_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Get embeddings for token IDs."""
        # This is a placeholder - actual embeddings come from OpenVINO model
        batch_size, seq_len = token_ids.shape
        embeddings = np.random.randn(batch_size, seq_len, self.embedding_dim) * 0.1
        return embeddings
    
    def _apply_rope(self, embeddings: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        """Apply Rotary Positional Encoding."""
        batch_size, seq_len, dim = embeddings.shape
        
        # Get positions for each token
        positions = np.arange(seq_len)
        
        # Apply rotation
        half_dim = dim // 2
        x1 = embeddings[:, :, :half_dim]
        x2 = embeddings[:, :, half_dim:]
        
        rope = self._rope_cache[positions]  # (seq_len, dim)
        rope1 = rope[:, :half_dim]
        rope2 = rope[:, half_dim:]
        
        # Rotate
        x1_rot = x1 * rope1 - x2 * rope2
        x2_rot = x1 * rope2 + x2 * rope1
        
        embeddings = np.concatenate([x1_rot, x2_rot], axis=-1)
        
        return embeddings


class LayerState:
    """Состояние гибридного слоя для передачи между слоями."""
    
    def __init__(
        self,
        cluster_assignments: Optional[np.ndarray] = None,
        master_tokens: Optional[np.ndarray] = None,
        accumulated_confidence: Optional[np.ndarray] = None
    ):
        self.cluster_assignments = cluster_assignments  # (num_nodes,)
        self.master_tokens = master_tokens  # (num_master, dim)
        self.accumulated_confidence = accumulated_confidence  # (seq_len,)
    
    def is_empty(self) -> bool:
        return self.master_tokens is None


@dataclass
class GraphContext:
    """Контекст из графа знаний."""
    nodes: list  # Извлечённые узлы
    edges: list  # Извлечённые рёбра
    embeddings: Optional[np.ndarray] = None  # (num_nodes, dim)
    node_mask: Optional[np.ndarray] = None  # (num_nodes,)
    routing_decisions: Optional[np.ndarray] = None  # (num_nodes,) - needs LM


@dataclass  
class LayerOutput:
    """Выход гибридного слоя."""
    hidden_states: np.ndarray  # (batch, seq_len, dim)
    layer_state: LayerState
    stop_mask: np.ndarray  # (batch, seq_len) - True если токен остановлен
    layer_confidence: float  # 0-1 - готовность завершить генерацию