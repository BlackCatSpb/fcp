"""
Fractal Graph V2 - Фрактальная иерархическая семантическая сеть

Архитектура:
- MySQL-подобная структура (реализовано на SQLite с расширениями)
- Векторные эмбеддинги для семантического поиска
- Фрактальные уровни (L0-LN) с семантическими группами
- Автоматическая кластеризация и детекция противоречий

Структура БД:
  nodes - атомарные единицы знаний
  edges - семантические связи
  semantic_groups - кластеры-образы верхнего уровня
  node_embeddings - векторы узлов
"""

import os
import json
import time
import uuid
import hashlib
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger("eva_ai.fractal_graph_v2")


class NodeType(Enum):
    """Типы узлов фрактального графа."""
    # Фрактальные уровни (L0-L3)
    ROOT = "root"           # L0 - корень графа
    CONCEPT = "concept"     # L1 - концепты
    FACT = "fact"           # L2 - факты
    DETAIL = "detail"       # L3 - детали
    
    # Атрибуты
    ATTRIBUTE = "attribute"
    CONTEXT = "context"
    EMOTION = "emotion"
    SENSORY = "sensory"
    
    # Объекты
    OBJECT = "object"
    ENTITY = "entity"
    
    # Связи
    CAUSAL = "causal"
    RELATION = "relation"
    
    # Модели (статичные)
    MODEL_A = "model_a"
    MODEL_B = "model_b"
    MODEL_C = "model_c"
    MODEL_ROOT = "model_root"      # L0 - GGUF модель метаданные
    
    # GGUF Shadow (гибридная интеграция)
    DOMAIN_PROFILE = "domain_profile"           # L1 - доменная специализация
    ACTIVATION_FINGERPRINT = "activation_fingerprint"  # L1 - профиль активаций
    ROUTING_RULE = "routing_rule"               # L2 - правило маршрутизации
    QUANTIZATION_PROFILE = "quantization_profile"  # L2 - профиль квантования
    LAYER_STATS = "layer_stats"                 # L3 - статистика слоёв
    PARAMETER_TUNING_RECORD = "parameter_tuning_record"  # L3 - запись настройки
    
    # Семантическая группа (образ)
    SEMANTIC_GROUP = "semantic_group"


class RelationType(Enum):
    """Типы связей между узлами."""
    ATTRIBUTE_OF = "attribute_of"
    CONTEXT_OF = "context_of"
    OBJECT_OF = "object_of"
    EMOTIONAL = "emotional"
    SENSORY = "sensory"
    CAUSAL = "causal"
    IS_A = "is_a"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"
    
    # GGUF Shadow связи
    ROUTES_TO = "routes_to"           # fingerprint -> routing_rule
    BELONGS_TO_DOMAIN = "belongs_to_domain"  # activation -> domain
    HAS_PROFILE = "has_profile"        # model_root -> quantization_profile
    HAS_STATS = "has_stats"           # model_root -> layer_stats
    APPLIES_TO = "applies_to"         # routing_rule -> query


class MemoryTier(Enum):
    """Уровни хранения памяти."""
    HOT = "hot"     # В RAM
    WARM = "warm"   # Сжатые в RAM
    COLD = "cold"   # На диске


@dataclass
class FractalNode:
    """Узел фрактальной памяти - атомарная единица знания."""
    id: str
    content: str                     # Текстовое содержание
    node_type: str                   # Тип узла
    level: int = 0                   # Фрактальный уровень (0 - самый глубокий)
    
    # Иерархия
    parent_group_id: Optional[str] = None  # Семантическая группа
    
    # Вектор (вычисляется отдельно)
    embedding: Optional[List[float]] = None
    
    # Уверенность (0-1), накапливается из подтверждений
    confidence: float = 0.5
    
    # Временной вес для Confidence Decay (P2)
    # w(t) = w₀ * e^(-λ*Δt), где λ зависит от домена
    temporal_weight: float = 1.0
    domain_lambda: float = 0.01  # λ для временного распада (по умолчанию 1%/день)
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Статистика
    access_count: int = 0
    version: int = 1
    
    # Флаги
    is_static: bool = False          # Статичные узлы (модели) не удаляются
    is_contradiction: bool = False   # Помечен как противоречие
    
    def get_effective_confidence(self) -> float:
        """
        Вычислить эффективную уверенность с учётом временного распада.
        Формула: w(t) = w₀ * e^(-λ*Δt)
        """
        import math
        delta_t = time.time() - self.last_accessed
        # Δt в днях (86400 секунд в дне)
        delta_days = delta_t / 86400
        decay_factor = math.exp(-self.domain_lambda * delta_days)
        return self.confidence * self.temporal_weight * decay_factor
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalNode':
        return cls(**data)


@dataclass
class FractalEdge:
    """Связь между узлами в графе."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    
    # Вес связи (частота подтверждения, семантическая близость)
    weight: float = 0.5
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Флаги
    contradiction_flag: bool = False  # Связь была предметом противоречия
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FractalEdge':
        return cls(**data)


@dataclass
class SemanticGroup:
    """Семантическая группа (образ) - кластер узлов верхнего уровня."""
    id: str
    name: str                        # Название образа (например, "снег")
    node_type: str = "semantic_group"
    level: int = 2                   # Уровень группы (обычно 2+)
    
    # Вектор группы (центроид или агрегация векторов членов)
    embedding: Optional[List[float]] = None
    
    # Статистика
    member_count: int = 0
    avg_confidence: float = 0.5
    
    # Иерархия
    parent_group_id: Optional[str] = None  # Группа более высокого уровня
    
    # Временные метки
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Кластеризация
    cluster_coherence: float = 0.0    # Связность кластера (0-1)
    needs_recluster: bool = False     # Флаг перекластеризации
    
    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticGroup':
        return cls(**data)


def create_node_id(content: str, node_type: str) -> str:
    """Создать ID узла на основе хеша контента."""
    hash_input = f"{content}:{node_type}"
    return f"node_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"


def create_edge_id(source_id: str, target_id: str, relation: str) -> str:
    """Создать ID связи."""
    hash_input = f"{source_id}:{target_id}:{relation}"
    return f"edge_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"


def create_group_id(name: str) -> str:
    """Создать ID группы."""
    hash_input = f"{name}:{time.time()}"
    return f"group_{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"