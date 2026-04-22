"""
FCP Temporal Context Memory - иерархическая память контекста (SPEC section 4)
"""
import time
import numpy as np
from typing import List, Optional, Dict, Union, Any
from dataclasses import dataclass, field

from fcp_core.types import MemorySegment, Fact


class TemporalContextMemory:
    """
    Temporal Context Memory (TCM) - SPEC section 4
    
    Методы:
    | Метод | Назначение |
    |-------|------------|
    | write | Добавляет сегмент с иерархическим временным кодированием |
    | retrieve | Извлекает k сегментов по мягкому адресованию |
    | update_async | Асинхронно обновляет параметры |
    | consolidate | Переносит стабильные сегменты в граф |
    | apply_temporal_decay | Применяет экспоненциальное затухание |
    """
    
    def __init__(
        self,
        max_segments: int = 1000,
        embedding_dim: int = 2048,
        time_scales: int = 4
    ):
        self.max_segments = max_segments
        self.embedding_dim = embedding_dim
        self.time_scales = time_scales
        
        self._segments: List[MemorySegment] = []
        self._segment_index: Dict[str, int] = {}
    
    # =========================================================================
    # SPEC METHOD: write
    # =========================================================================
    
    def write(
        self,
        text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        SPEC: write(text: str, embedding: Tensor, metadata: Dict) -> str
        
        Добавляет новый сегмент диалога в TCM с иерархическим временным кодированием.
        
        Returns:
            segment_id
        """
        now = time.time()
        
        # Compute time encoding
        time_encoding = self._encode_time(now)
        
        segment_id = f"seg_{len(self._segments):08d}"
        
        segment = MemorySegment(
            segment_id=segment_id,
            text=text,
            embedding=embedding,
            timestamp=now,
            time_encoding=time_encoding,
            relevance=metadata.get('relevance', 0.5) if metadata else 0.5,
            variance=metadata.get('variance', 0.1) if metadata else 0.1,
            consolidated=False
        )
        
        self._segments.append(segment)
        self._segment_index[segment_id] = len(self._segments) - 1
        
        # Evict oldest if needed
        if len(self._segments) > self.max_segments:
            evicted = self._segments.pop(0)
            del self._segment_index[evicted.segment_id]
            # Reindex
            self._segment_index = {
                s.segment_id: i for i, s in enumerate(self._segments)
            }
        
        return segment_id
    
    def _encode_time(self, timestamp: float) -> np.ndarray:
        """
        SPEC: иерархическое временное кодирование (час, день, неделя, месяц)
        
        Returns: (4,) - hour, day, week, month
        """
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        
        hour = dt.hour / 24.0
        day = dt.weekday() / 7.0
        week = (dt.day - 1) / 7.0 / 4.0  # approximate
        month = dt.month / 12.0
        
        return np.array([hour, day, week, month])
    
    # =========================================================================
    # SPEC METHOD: retrieve
    # =========================================================================
    
    def retrieve(
        self,
        query: Union[str, np.ndarray],
        k: int = 10
    ) -> List[MemorySegment]:
        """
        SPEC: retrieve(query: Union[str, Tensor], k: int) -> List[MemorySegment]
        
        Извлекает k наиболее релевантных сегментов по мягкому адресованию:
        - Семантическая близость
        - Временная близость  
        - Накопленная релевантность
        """
        if not self._segments:
            return []
        
        # Get query embedding
        if isinstance(query, str):
            # Would use model - simplified
            query_emb = self._segments[0].embedding  # Placeholder
        else:
            query_emb = query
        
        now = time.time()
        
        # Score each segment
        scores = []
        for segment in self._segments:
            # Semantic similarity
            sem_sim = self._cosine_similarity(query_emb, segment.embedding)
            
            # Time decay
            time_decay = self._time_decay(segment.timestamp, now)
            
            # Combined score
            score = (
                sem_sim * 0.5 + 
                time_decay * 0.3 + 
                segment.relevance * 0.2
            )
            scores.append(score)
        
        # Get top-k
        if k >= len(scores):
            return self._segments.copy()
        
        indices = np.argsort(scores)[-k:][::-1]
        return [self._segments[i] for i in indices]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return dot / (norm_a * norm_b)
    
    def _time_decay(self, timestamp: float, now: float) -> float:
        """
        SPEC: экспоненциальное затухание (half-life: 24 часа)
        
        Returns: 0-1, где 1 = недавнее
        """
        hours_old = (now - timestamp) / 3600
        decay = np.exp(-hours_old / 24)
        return max(0, min(1, decay))
    
    # =========================================================================
    # SPEC METHOD: update_async
    # =========================================================================
    
    def update_async(self, buffer: List[MemorySegment]) -> None:
        """
        SPEC: update_async(buffer: List[Segment]) -> None
        
        Асинхронно обновляет параметры TCM на основе контрастивной функции потерь.
        """
        # Would compute contrastive loss and update embeddings
        # Simplified placeholder
        pass
    
    # =========================================================================
    # SPEC METHOD: consolidate
    # =========================================================================
    
    def consolidate(
        self,
        graph: "FractalGraphV2"
    ) -> List[Fact]:
        """
        SPEC: consolidate() -> List[Fact]
        
        Переносит стабильные сегменты из TCM в долговременный граф,
        вызывая майнер концепций.
        
        Returns:
            List of Facts added to graph
        """
        if graph is None:
            return []
        
        now = time.time()
        added_facts = []
        
        for segment in self._segments:
            if segment.consolidated:
                continue
            
            # Check stability
            age_hours = (now - segment.timestamp) / 3600
            if age_hours < 24:  # min 24 hours
                continue
            
            stability = 1.0 - segment.variance
            if stability < 0.7:  # needs stability
                continue
            
            # Add to graph
            try:
                fact_id = graph.add_fact(
                    subject=segment.text[:50],
                    predicate="was_discussed",
                    object="in_context",
                    confidence=stability
                )
                added_facts.append(Fact(
                    fact_id=fact_id,
                    subject=segment.text[:50],
                    predicate="was_discussed",
                    object="in_context",
                    confidence=stability
                ))
                segment.consolidated = True
            except Exception:
                pass
        
        return added_facts
    
    # =========================================================================
    # SPEC METHOD: apply_temporal_decay
    # =========================================================================
    
    def apply_temporal_decay(self) -> None:
        """
        SPEC: apply_temporal_decay() -> None
        
        Применяет экспоненциальное затухание к весам сегментов.
        """
        now = time.time()
        
        for segment in self._segments:
            # Reduce relevance based on age
            decay = self._time_decay(segment.timestamp, now)
            segment.relevance *= decay
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_context_for_prompt(
        self,
        query_embedding: np.ndarray,
        max_chars: int = 2000
    ) -> str:
        """Get formatted context string for LLM."""
        segments = self.retrieve(query_embedding, k=5)
        
        if not segments:
            return ""
        
        parts = []
        total = 0
        for seg in segments:
            if total + len(seg.text) > max_chars:
                break
            parts.append(seg.text)
            total += len(seg.text)
        
        return "\n".join(parts)
    
    def __len__(self) -> int:
        return len(self._segments)
    
    def is_empty(self) -> bool:
        return len(self._segments) == 0


# Forward type
class FractalGraphV2:
    """Placeholder - import from EVA-Ai"""
    def add_fact(self, subject, predicate, obj, confidence):
        return f"fact_{subject}_{predicate}"