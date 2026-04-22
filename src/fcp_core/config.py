"""
FCP Configuration - Модульная конфигурация для FCP
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FCPConfig:
    """
    Конфигурация FCP - полностью модульная
    
    Слоёв может быть любое количество - FCP адаптируется под рабочую модель.
    """
    # Model architecture (определяются из модели)
    vocab_size: int = 151936
    embedding_dim: int = 2560       # hidden_size
    num_layers: int = 36           # из модели (Qwen3: 36)
    num_heads: int = 32
    kv_heads: int = 8              # GQA
    head_dim: int = 128           # embedding_dim / num_heads
    intermediate_size: int = 9728 # FFN size
    max_seq_len: int = 262144
    
    # Hybrid layer config
    graph_retrieval_k: int = 32
    master_tokens: int = 8
    gnn_iterations: int = 2
    
    # Activation gates
    stop_threshold: float = 0.85
    early_exit_threshold: float = 0.90
    
    # TCM
    tcm_max_segments: int = 1000
    tcm_top_k: int = 10
    
    # LoRA (future)
    lora_rank_base: int = 4
    lora_rank_domain: int = 8
    lora_rank_reasoning: int = 16
    
    # Paths
    model_path: str = "C:/Users/black/OneDrive/Desktop/Models/BF16.gguf"
    graph_db_path: str = "C:/Users/black/OneDrive/Desktop/FCP/data/graph.db"
    
    # OpenVINO
    device: str = "CPU"
    num_threads: int = 8
    performance_hint: str = "LATENCY"
    
    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.2
    
    # Modular: возможность добавлять слои динамически
    enable_dynamic_layers: bool = True
    dynamic_layer_threshold: int = 4   # минимум 4 слоя
    
    @classmethod
    def from_model(cls, model_path: str, **overrides) -> "FCPConfig":
        """Создать конфиг на основе модели (BF16)."""
        # Сurrent: Qwen3 4B base parameters
        config = cls(
            model_path=model_path,
            vocab_size=151936,
            embedding_dim=2560,
            num_layers=36,
            num_heads=32,
            kv_heads=8,
            head_dim=128,
            intermediate_size=9728,
            max_seq_len=262144,
            **overrides
        )
        return config
    
    @classmethod
    def minimal(cls, model_path: str = "") -> "FCPConfig":
        """Минимальная конфигурация для тестирования."""
        return cls(
            model_path=model_path,
            num_layers=4,  # Минимум для теста
            graph_retrieval_k=8,
            master_tokens=4,
            tcm_max_segments=100
        )
    
    def add_layers(self, num_new: int) -> int:
        """
        Динамически добавить слои.
        
        Returns: новое общее количество слоёв
        """
        if not self.enable_dynamic_layers:
            return self.num_layers
        
        self.num_layers += num_new
        return self.num_layers
    
    def remove_layers(self, num_remove: int) -> int:
        """
        Динамически убрать слои (не ниже минимума).
        
        Returns: новое общее количество слоёв
        """
        new_count = max(self.dynamic_layer_threshold, self.num_layers - num_remove)
        self.num_layers = new_count
        return self.num_layers
    
    def summary(self) -> str:
        """Строка-резюме конфигурации."""
        return (
            f"FCP: {self.num_layers} layers, "
            f"{self.embedding_dim}d, "
            f"{self.num_heads} heads, "
            f"vocab={self.vocab_size}"
        )