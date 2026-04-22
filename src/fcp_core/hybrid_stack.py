"""
FCP Hybrid Stack - модульный стек гибридных слоёв
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .types import LayerState, HaltDecision
from .hybrid_layer import FractalGatedHybridLayer
from .config import FCPConfig


@dataclass
class StackConfig:
    """Конфигурация стека."""
    num_layers: int = 36
    hidden_dim: int = 2560
    num_heads: int = 32
    max_seq_len: int = 262144
    graph_retrieval_k: int = 32
    master_tokens: int = 8
    gnn_iterations: int = 2
    stop_threshold: float = 0.85
    early_exit_threshold: float = 0.90


class HybridStack:
    """
    Модульный стек гибридных слоёв FCP.
    
    Особенности:
    - Динамическое количество слоёв
    - Состояние передаётся между слоями (stateful GNN)
    - Early exit на основе накопленной уверенности
    """
    
    def __init__(self, config: StackConfig):
        self.config = config
        
        # Создаём слои
        self._layers: List[FractalGatedHybridLayer] = []
        self._build_layers()
        
        # Глобальное состояние стека
        self._global_state = {
            "total_layers_processed": 0,
            "early_exits": 0,
            "avg_confidence": 0.0
        }
    
    def _build_layers(self):
        """Создать все слои стека."""
        self._layers = []
        
        for i in range(self.config.num_layers):
            layer = FractalGatedHybridLayer(
                layer_id=i,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_len,
                graph_retrieval_k=self.config.graph_retrieval_k,
                master_tokens=self.config.master_tokens,
                gnn_iterations=self.config.gnn_iterations,
                stop_threshold=self.config.stop_threshold
            )
            self._layers.append(layer)
    
    @property
    def num_layers(self) -> int:
        return len(self._layers)
    
    def add_layers(self, num_new: int) -> int:
        """
        Динамически добавить слои.
        
        Returns: новое количество слоёв
        """
        current = len(self._layers)
        
        for i in range(current, current + num_new):
            layer = FractalGatedHybridLayer(
                layer_id=i,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                max_seq_len=self.config.max_seq_len,
                graph_retrieval_k=self.config.graph_retrieval_k,
                master_tokens=self.config.master_tokens,
                gnn_iterations=self.config.gnn_iterations,
                stop_threshold=self.config.stop_threshold
            )
            self._layers.append(layer)
        
        return len(self._layers)
    
    def remove_layers(self, num_remove: int) -> int:
        """
        Убрать слои (минимум 1).
        
        Returns: новое количество слоёв
        """
        min_layers = 1
        new_count = max(min_layers, len(self._layers) - num_remove)
        
        if new_count < len(self._layers):
            self._layers = self._layers[:new_count]
        
        return len(self._layers)
    
    def forward(
        self,
        hidden_states: np.ndarray,
        graph: Optional["FractalGraphV2"],
        attention_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[HaltDecision]]:
        """
        Forward pass через весь стек.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            graph: FractalGraphV2 для retrieval
            attention_mask: (batch, seq_len)
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            halt_decisions: List[HaltDecision] по каждому слою
        """
        batch, seq_len, dim = hidden_states.shape
        
        # Инициализация маски внимания
        if attention_mask is None:
            attention_mask = np.ones((batch, seq_len), dtype=bool)
        
        # Начальное состояние
        layer_state = None
        
        # Результаты по слоям
        halt_decisions = []
        
        # Early exit threshold
        early_exit_thresh = self.config.early_exit_threshold
        accumulated_confidence = 0.0
        
        # Проход по всем слоям
        for idx, layer in enumerate(self._layers):
            # Проверка early exit
            if accumulated_confidence > early_exit_thresh:
                self._global_state["early_exits"] += 1
                break
            
            # Forward через слой
            output, layer_state, halt = layer.forward(
                hidden_states,
                graph,
                layer_state,
                attention_mask
            )
            
            hidden_states = output
            halt_decisions.append(halt)
            
            # Обновление уверенности
            accumulated_confidence = (
                accumulated_confidence * 0.7 + 
                halt.layer_confidence * 0.3
            )
            
            # Обновление глобального состояния
            self._global_state["total_layers_processed"] += 1
            self._global_state["avg_confidence"] = (
                self._global_state["avg_confidence"] * 0.9 + 
                halt.layer_confidence * 0.1
            )
        
        return hidden_states, halt_decisions
    
    def get_statistics(self) -> dict:
        """Получить статистику стека."""
        return {
            "num_layers": len(self._layers),
            "total_processed": self._global_state["total_layers_processed"],
            "early_exits": self._global_state["early_exits"],
            "avg_confidence": self._global_state["avg_confidence"],
            "layers_used_ratio": (
                self._global_state["total_layers_processed"] / 
                max(1, len(self._layers))
            )
        }
    
    def reset_statistics(self):
        """Сбросить статистику."""
        self._global_state = {
            "total_layers_processed": 0,
            "early_exits": 0,
            "avg_confidence": 0.0
        }


# Forward declaration for type hints
class FractalGraphV2:
    """Placeholder - будет импортировано"""
    def extract_subgraph(self, query, k):
        from .types import Subgraph
        return Subgraph([], np.array([]), [], [])