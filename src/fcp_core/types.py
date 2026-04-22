"""
FCP Types - типы данных для FCP
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class Subgraph:
    """Извлечённый подграф из FractalGraphV2."""
    node_ids: List[str]
    node_embeddings: np.ndarray  # (num_nodes, dim)
    edges: List[Tuple[str, str]]  # [(source, target), ...]
    edge_types: List[str]
    
    def __len__(self) -> int:
        return len(self.node_ids)
    
    @property
    def is_empty(self) -> bool:
        return len(self.node_ids) == 0


@dataclass
class MemorySegment:
    """Сегмент в TCM."""
    segment_id: str
    text: str
    embedding: np.ndarray
    timestamp: float
    time_encoding: np.ndarray  # (4,) - hour, day, week, month
    relevance: float = 0.5
    variance: float = 0.1
    consolidated: bool = False


@dataclass
class Concept:
    """Концепт для графа."""
    concept_id: str
    name: str
    embedding: np.ndarray
    domain: str = "general"
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    """Факт (триплет)."""
    fact_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.5
    source: str = "user"
    timestamp: float = 0.0


@dataclass
class Contradiction:
    """Обнаруженное противоречие."""
    contradiction_id: str
    fact_a_id: str
    fact_b_id: str
    similarity: float
    conflict_description: str
    resolution_suggestion: str = ""
    status: str = "detected"  # detected, resolved, ignored


@dataclass
class ResolutionResult:
    """Результат разрешения противоречия."""
    contradiction_id: str
    strategy_used: str
    chosen_fact_id: str
    explanation: str
    success: bool


@dataclass
class ComputeTopology:
    """Модель вычислительной топологии."""
    cpu_cores: int
    gpu_devices: List[str]
    memory_gb: float
    vector_extensions: List[str]
    cache_sizes: Dict[str, int]
    
    @property
    def has_gpu(self) -> bool:
        return len(self.gpu_devices) > 0


@dataclass
class RequestMetrics:
    """Метрики запроса."""
    latency_ms: float
    tokens_generated: int
    early_exit_layers: int
    layer_confidences: List[float]
    stopped_tokens_ratio: float


@dataclass
class LayerState:
    """Состояние гибридного слоя."""
    layer_id: int
    cluster_assignments: Optional[np.ndarray] = None
    master_tokens: Optional[np.ndarray] = None
    accumulated_confidence: float = 0.0
    kv_cache: Optional[Any] = None


@dataclass
class TransformerBlockOutput:
    """Выход трансформерного блока."""
    hidden_states: np.ndarray
    attention_weights: np.ndarray
    entropy: np.ndarray


@dataclass
class HaltDecision:
    """Решение об остановке токена."""
    stop_probabilities: np.ndarray  # (seq_len,)
    active_mask: np.ndarray  # (seq_len,) - True = still active
    layer_confidence: float
    should_early_exit: bool


@dataclass
class FusionOutput:
    """Выход слияния потоков."""
    fused_hidden: np.ndarray
    master_tokens_used: np.ndarray
    fusion_weights: np.ndarray


@dataclass
class ExecutionPlan:
    """План выполнения."""
    tasks: List[str]
    device_assignments: Dict[str, str]
    estimated_time_ms: float
    memory_estimate_gb: float


@dataclass
class CompiledKernel:
    """Скомпилированное ядро."""
    kernel_id: str
    source_ir: str
    native_code: bytes
    optimizations_applied: List[str]