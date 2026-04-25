"""
Semantic Cache Evictor - Семантическое вытеснение кэша

Выбирает блоки для вытеснения на основе важности.
"""
import numpy as np
from typing import List, Optional, Tuple


class SemanticCacheEvictor:
    """
    Вытеснение на основе семантической важности.
    
    Оценивает важность токенов через связь с графом.
    """
    
    def __init__(
        self,
        gnn,
        graph
    ):
        self.gnn = gnn
        self.graph = graph
        self.node_importance: dict = {}
    
    def _token_importance(self, token_hidden: np.ndarray) -> float:
        """
        Оценить важность токена.
        
        Args:
            token_hidden: скрытое состояние токена
        
        Returns:
            важность (0-1)
        """
        if self.gnn is None:
            return 0.5
        
        # Найти ближайший узел в графе
        try:
            sub = self.gnn.retrieve_subgraph(token_hidden, k=1)
            
            if not sub.get("node_ids"):
                return 0.5
            
            node_id = sub["node_ids"][0]
            distance = sub.get("distances", [0.5])[0]
            
            # Temporal weight узла
            temporal_weight = self.graph.get_node(node_id).get("temporal_weight", 1.0) if self.graph else 1.0
            
            # Важность = 1 / (1 + distance) * temporal_weight
            importance = (1.0 / (1.0 + distance)) * temporal_weight
            
            return importance
            
        except Exception:
            return 0.5
    
    def select_blocks_to_evict(
        self,
        kv_blocks: List[any],
        num_to_free: int
    ) -> List[int]:
        """
        Выбрать блоки для вытеснения.
        
        Args:
            kv_blocks: блоки KV-кэша
            num_to_free: сколько освободить
        
        Returns:
            индексы блоков для вытеснения
        """
        if not kv_blocks:
            return []
        
        # Оценить важность каждого блока
        scores = []
        
        for i, block in enumerate(kv_blocks):
            # Получить last hidden state блока
            last_hidden = self._get_block_hidden(block)
            
            if last_hidden is not None:
                importance = self._token_importance(last_hidden)
            else:
                importance = 0.5
            
            scores.append((i, importance))
        
        # Сортировать по важности (меньшая важность = вытеснить)
        scores.sort(key=lambda x: x[1])
        
        # Вернуть индексы с наименьшей важностью
        return [idx for idx, _ in scores[:num_to_free]]
    
    def _get_block_hidden(self, block) -> Optional[np.ndarray]:
        """Получить hidden state из блока."""
        # Упрощённо - предполагаем что block имеет атрибут
        if hasattr(block, 'last_hidden_state'):
            return block.last_hidden_state
        elif hasattr(block, 'hidden'):
            return block.hidden
        else:
            return None
    
    def should_evict(
        self,
        current_size: int,
        max_size: int,
        threshold: float = 0.7
    ) -> bool:
        """Определить нужно ли вытеснение."""
        return current_size >= max_size * threshold


class CacheEvictionPolicy:
    """Политика вытеснения кэша."""
    
    def __init__(
        self,
        evictor: SemanticCacheEvictor,
        min_size: int = 10,
        max_size: int = 100
    ):
        self.evictor = evictor
        self.min_size = min_size
        self.max_size = max_size
    
    def evict(
        self,
        cache_blocks: List[any]
    ) -> List[any]:
        """
        Вытеснить блоки.
        
        Args:
            cache_blocks: текущие блоки кэша
        
        Returns:
            оставшиеся блоки
        """
        if len(cache_blocks) <= self.min_size:
            return cache_blocks
        
        num_to_free = len(cache_blocks) - self.min_size
        
        indices = self.evictor.select_blocks_to_evict(
            cache_blocks,
            num_to_free
        )
        
        # Удалить блоки
        result = [
            block for i, block in enumerate(cache_blocks)
            if i not in indices
        ]
        
        return result