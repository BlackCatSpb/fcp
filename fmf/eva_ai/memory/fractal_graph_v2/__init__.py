"""
Fractal Graph V2 - Основной API фрактального графа памяти
"""

import os
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import OrderedDict
from functools import wraps

from .types import FractalNode, FractalEdge, SemanticGroup, NodeType, RelationType
from .storage import FractalGraphV2, create_fractal_graph
from .embeddings import EmbeddingsManager, create_embeddings_manager
from .gguf_parser import parse_gguf_model, extract_to_graph
from .gguf_extractor import GGUFKnowledgeExtractor, create_extractor
from .gguf_shadow import GGUFShadowProfiler, create_gguf_shadow_profiler
from .hybrid_tokenizer import HybridTokenizer, create_hybrid_tokenizer
from .eva_generator import EVAGenerator, create_eva_generator, GenerationRequest, GenerationResult
from .semantic_context_cache import SemanticContextCache, create_semantic_context_cache
from .snapshot_manager import SnapshotManager, MemorySnapshot, create_snapshot_manager
from .virtual_token_handler import (
    VirtualTokenManager,
    VirtualTokenLogitsProcessor,
    StreamingVirtualTokenHandler,
    VirtualTokenInfo,
    create_virtual_token_manager
)
from .eva_container import EVAContainer, create_eva_container, load_eva_container
from .tokenizer import GraphTokenizer, create_graph_tokenizer

logger = logging.getLogger("eva_ai.fractal_graph_v2")

__all__ = [
    # Storage
    'FractalMemoryGraph',
    'FractalGraphV2',
    'create_fractal_memory_graph',
    
    # Types
    'FractalNode',
    'FractalEdge', 
    'SemanticGroup',
    'NodeType',
    'RelationType',
    
    # GGUF
    'parse_gguf_model',
    'extract_to_graph',
    'create_extractor',
    'GGUFKnowledgeExtractor',
    
    # GGUF Shadow (гибридная интеграция)
    'GGUFShadowProfiler',
    'create_gguf_shadow_profiler',
    
    # Hybrid Tokenizer (для EVA контейнера)
    'HybridTokenizer',
    'create_hybrid_tokenizer',
    
    # EVA Generator (единый генератор с виртуальными токенами)
    'EVAGenerator',
    'create_eva_generator',
    'GenerationRequest',
    'GenerationResult',
    
    # Semantic Context Cache (CPU-based semantic search)
    'SemanticContextCache',
    'create_semantic_context_cache',
    
    # Snapshot Manager (immutable memory snapshots for generation consistency)
    'SnapshotManager',
    'MemorySnapshot',
    'create_snapshot_manager',
    
    # Virtual Token Handler (LogitsProcessor + Streaming replacement)
    'VirtualTokenManager',
    'VirtualTokenLogitsProcessor',
    'StreamingVirtualTokenHandler',
    'VirtualTokenInfo',
    'create_virtual_token_manager',
    
    # EVA Container (unified .eva format)
    'EVAContainer',
    'create_eva_container',
    'load_eva_container',
    
    # Tokenizer
    'GraphTokenizer',
    'create_graph_tokenizer',
]


# ============================================================================
# Декоратор для тайминга производительности
# ============================================================================

def timed(logger_func=None, threshold_ms: float = 100.0):
    """Декоратор для измерения времени выполнения функции."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                log_func = logger_func or logger.debug
                if elapsed_ms > threshold_ms:
                    log_func(f"⏱️ {func.__name__}: {elapsed_ms:.1f}ms")
        return wrapper
    return decorator


# ============================================================================
# LRU Cache with TTL для оптимизации semantic_search
# ============================================================================

class LRUCacheWithTTL:
    """LRU кэш с time-to-live для семантического поиска."""
    
    def __init__(self, maxsize: int = 100, ttl_seconds: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                # TTL expired
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = (value, time.time())
            
            # Evict oldest if over limit
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl
            }


# ============================================================================
# Main API: FractalMemoryGraph
# ============================================================================

class FractalMemoryGraph:
    
    def __init__(
        self,
        storage_dir: str = None,
        embedding_model: str = None,  # Используем локальный путь из sentence_transformers_cache
        embedding_device: str = "cuda",
        embedding_dim: int = 768,
        event_bus = None
    ):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "fractal_graph_v2_data"
        )
        
        # Инициализация хранилища
        self.storage = create_fractal_graph(
            storage_dir=self.storage_dir,
            embedding_dim=embedding_dim
        )
        
        # Инициализация эмбеддингов
        self.embeddings = create_embeddings_manager(
            model_name=embedding_model,
            device=embedding_device
        )
        
        # LRU кэш для semantic_search (оптимизация производительности)
        self._search_cache = LRUCacheWithTTL(maxsize=100, ttl_seconds=300.0)
        
        self._background_thread = None
        self._running = False
        
        # EventBus интеграция (1.1.3)
        self._event_bus = event_bus
        self._subscription_ids = []
        
        logger.info(f"FractalMemoryGraph инициализирован: {self.storage_dir}")
        logger.info(f"Semantic search cache: maxsize={self._search_cache.maxsize}, ttl={self._search_cache.ttl}s")
        if self._event_bus:
            logger.info("EventBus интеграция активна")
    
    # === EventBus интеграция (1.1.3) ===
    
    def _publish_event(self, event_type: str, data: Dict):
        """Публикация события в EventBus."""
        if self._event_bus is None:
            return
        try:
            from eva_ai.core.event_bus import Event, EventPriority
            event = Event(
                event_type=event_type,
                source="fractal_graph_v2",
                data=data,
                priority=EventPriority.NORMAL
            )
            self._event_bus.publish(event)
        except Exception as e:
            logger.warning(f"Failed to publish event {event_type}: {e}")
    
    def start(self):
        """Подписаться на системные события."""
        if self._event_bus is None:
            return
        # Подписки на системные события
        self._running = True
        logger.info("FractalMemoryGraph подписан на EventBus")
    
    def stop(self):
        """Отписаться от событий."""
        self._running = False
        for sub_id in self._subscription_ids:
            try:
                self._event_bus.unsubscribe(sub_id)
            except:
                pass
        self._subscription_ids.clear()
        logger.info("FractalMemoryGraph отписан от EventBus")
    
    # === ОСНОВНЫЕ ОПЕРАЦИИ ===
    
    def add_node(
        self,
        content: str,
        node_type: str = "concept",
        level: int = 1,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
        auto_vectorize: bool = True,
        auto_cluster: bool = True,
        cluster_threshold: float = 0.6,
        is_static: bool = False,
        is_contradiction: bool = False
    ) -> FractalNode:
        """
        Добавить узел в граф.
        
        Args:
            content: Текстовое содержание
            node_type: Тип узла (concept, fact, detail, attribute и др.)
            level: Фрактальный уровень (0 - самый глубокий)
            confidence: Уверенность (0-1)
            metadata: Дополнительные метаданные
            auto_vectorize: Автоматически вычислить эмбеддинг
            auto_cluster: Автоматически присоединить к ближайшей группе
            cluster_threshold: Порог similarity для кластеризации
            is_static: Защищённый узел (не удаляется без force)
            is_contradiction: Узел-противоречие
            
        Returns:
            FractalNode
        """
        node = self.storage.add_node(
            content=content,
            node_type=node_type,
            level=level,
            confidence=confidence,
            metadata=metadata,
            auto_cluster=False,  # Кластеризация после векторизации
            is_static=is_static,
            is_contradiction=is_contradiction
        )
        
        if auto_vectorize:
            self._vectorize_single_node(node.id)
            
            # Инкрементальная кластеризация после векторизации
            if auto_cluster and node.embedding:
                best_group = self.storage._find_nearest_group(
                    node.embedding, level, cluster_threshold
                )
                if best_group:
                    node.parent_group_id = best_group
                    self.storage._save_node(node)
                    if best_group in self.storage.semantic_groups:
                        self.storage.semantic_groups[best_group].member_count += 1
        
        # Публикуем событие о добавлении узла (1.1.3)
        self._publish_event("memory.node_created", {
            "node_id": node.id,
            "node_type": node.node_type,
            "level": node.level,
            "content_preview": node.content[:100] if len(node.content) > 100 else node.content
        })
        
        return node
    
    def add_nodes_batch(
        self,
        contents: List[str],
        node_type: str = "concept",
        level: int = 1,
        confidence: float = 0.5,
        auto_vectorize: bool = True,
        auto_cluster: bool = True,
        cluster_threshold: float = 0.6,
        batch_size: int = 32
    ) -> List[FractalNode]:
        """
        Batch добавление узлов с оптимизированной векторизацией.
        
        Этот метод значительно быстрее чем add_node() в цикле,
        так как использует batch processing для эмбеддингов.
        
        Args:
            contents: Список текстовых содержаний
            node_type: Тип узла
            level: Фрактальный уровень
            confidence: Уверенность
            auto_vectorize: Автоматически вычислить эмбеддинги
            auto_cluster: Автоматически присоединить к группам
            cluster_threshold: Порог similarity для кластеризации
            batch_size: Размер батча для векторизации
            
        Returns:
            Список созданных узлов
        """
        start_time = time.time()
        
        # Создаем узлы без векторизации
        nodes = []
        contents_to_vectorize = []
        node_indices = []
        
        for i, content in enumerate(contents):
            node = self.storage.add_node(
                content=content,
                node_type=node_type,
                level=level,
                confidence=confidence,
                auto_cluster=False
            )
            nodes.append(node)
            
            if auto_vectorize and node.embedding is None:
                contents_to_vectorize.append(content)
                node_indices.append(i)
        
        # Batch векторизация
        if contents_to_vectorize and auto_vectorize:
            logger.info(f"Batch векторизация {len(contents_to_vectorize)} узлов (batch_size={batch_size})...")
            
            # Разбиваем на батчи
            all_embeddings = []
            for i in range(0, len(contents_to_vectorize), batch_size):
                batch = contents_to_vectorize[i:i + batch_size]
                batch_embeddings = self.embeddings.encode(batch, normalize=True, show_progress=False)
                if batch_embeddings is None:
                    logger.warning("Embedding model недоступен, пропускаем векторизацию")
                    break
                all_embeddings.extend(batch_embeddings)
            
            # Присваиваем эмбеддинги узлам
            for idx, emb in zip(node_indices, all_embeddings):
                if emb is not None:
                    nodes[idx].embedding = emb.tolist()
                    self.storage._save_node(nodes[idx])
        
        # Инкрементальная кластеризация
        if auto_cluster:
            for node in nodes:
                if node.embedding:
                    best_group = self.storage._find_nearest_group(
                        node.embedding, level, cluster_threshold
                    )
                    if best_group:
                        node.parent_group_id = best_group
                        self.storage._save_node(node)
                        if best_group in self.storage.semantic_groups:
                            self.storage.semantic_groups[best_group].member_count += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Batch добавление завершено: {len(nodes)} узлов за {elapsed:.2f}s")
        
        # Публикуем событие о batch добавлении (1.1.3)
        self._publish_event("memory.nodes_batch_created", {
            "count": len(nodes),
            "node_type": node_type,
            "level": level,
            "time_elapsed": elapsed
        })
        
        # Публикуем событие об изменении графа для GraphCurator
        self._publish_event("memory.graph_updated", {
            "change_type": "batch_add",
            "nodes_added": len(nodes),
            "skip_curation": False
        })
        
        return nodes
    
    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        node_type: Optional[str] = None,
        level: Optional[int] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
        is_static: Optional[bool] = None,
        is_contradiction: Optional[bool] = None,
        recompute_embedding: bool = True,
        recluster: bool = True
    ) -> Optional[FractalNode]:
        """
        Обновить существующий узел.
        
        Args:
            node_id: ID узла для обновления
            content: Новый текст (если изменился - пересчитается embedding)
            node_type: Новый тип узла
            level: Новый уровень
            confidence: Новая уверенность
            metadata: Новые метаданные (мержится с существующими)
            is_static: Изменить статус статичности
            is_contradiction: Изменить статус противоречия
            recompute_embedding: Пересчитать embedding при изменении content
            recluster: Ре-кластеризовать при изменении embedding
            
        Returns:
            Обновлённый узел или None если не найден
        """
        node = self.storage.get_node(node_id)
        if node is None:
            logger.warning(f"Узел не найден для обновления: {node_id}")
            return None
        
        old_content = node.content
        old_embedding = node.embedding.copy() if node.embedding else None
        old_group = node.parent_group_id
        
        # Обновляем поля
        if content is not None and content != node.content:
            node.content = content
            node.updated_at = time.time()
            node.version += 1
            
            # Обновляем word_index (удаляем старые слова)
            for word in old_content.lower().split():
                if len(word) > 2 and word in self.storage.word_index:
                    self.storage.word_index[word].discard(node_id)
                    if not self.storage.word_index[word]:
                        del self.storage.word_index[word]
            # Добавляем новые слова
            for word in content.lower().split():
                if len(word) > 2:
                    if word not in self.storage.word_index:
                        self.storage.word_index[word] = set()
                    self.storage.word_index[word].add(node_id)
        
        if node_type is not None:
            if node_type != node.node_type:
                # Обновляем nodes_by_type
                if node.node_type in self.storage.nodes_by_type:
                    if node_id in self.storage.nodes_by_type[node.node_type]:
                        self.storage.nodes_by_type[node.node_type].remove(node_id)
                if node_type not in self.storage.nodes_by_type:
                    self.storage.nodes_by_type[node_type] = []
                self.storage.nodes_by_type[node_type].append(node_id)
            node.node_type = node_type
        
        if level is not None and level != node.level:
            if node.level in self.storage.nodes_by_level:
                if node_id in self.storage.nodes_by_level[node.level]:
                    self.storage.nodes_by_level[node.level].remove(node_id)
            if level not in self.storage.nodes_by_level:
                self.storage.nodes_by_level[level] = []
            self.storage.nodes_by_level[level].append(node_id)
            node.level = level
        
        if confidence is not None:
            node.confidence = confidence
        
        if metadata is not None:
            node.metadata.update(metadata)
        
        if is_static is not None:
            node.is_static = is_static
        
        if is_contradiction is not None:
            node.is_contradiction = is_contradiction
        
        # Пересчёт embedding если content изменился
        embedding_changed = content is not None and content != old_content
        if recompute_embedding and embedding_changed:
            self._vectorize_single_node(node_id)
            node = self.storage.get_node(node_id)  # Получаем обновлённый узел
            if node and node.embedding:
                # Обновляем нормализованный embedding cache
                import numpy as np
                v = np.array(node.embedding)
                v = v / (np.linalg.norm(v) + 1e-8)
                self.storage._normalized_embeddings[node_id] = v
        
        # Ре-кластеризация если embedding изменился
        if recluster and node and node.embedding:
            new_embedding = node.embedding
            if old_group and old_group in self.storage.semantic_groups:
                # Уменьшаем счётчик старой группы
                self.storage.semantic_groups[old_group].member_count = max(
                    0, self.storage.semantic_groups[old_group].member_count - 1
                )
                # Удаляем из nodes_by_group старой группы
                if old_group in self.storage.nodes_by_group and node_id in self.storage.nodes_by_group[old_group]:
                    self.storage.nodes_by_group[old_group].remove(node_id)
            
            # Находим новую ближайшую группу
            best_group = self.storage._find_nearest_group(
                new_embedding, node.level, cluster_threshold=0.6
            )
            if best_group:
                node.parent_group_id = best_group
                if best_group in self.storage.semantic_groups:
                    self.storage.semantic_groups[best_group].member_count += 1
                if best_group not in self.storage.nodes_by_group:
                    self.storage.nodes_by_group[best_group] = []
                if node_id not in self.storage.nodes_by_group[best_group]:
                    self.storage.nodes_by_group[best_group].append(node_id)
        
        # Сохраняем изменения
        if node:
            node.updated_at = time.time()
            self.storage._save_node(node)
        
        # Инвалидируем кэш поиска
        self._search_cache.invalidate(node_id)
        
        # Публикуем событие (1.1.3)
        self._publish_event("memory.node_updated", {
            "node_id": node_id,
            "content_changed": embedding_changed,
            "reclustered": recluster and embedding_changed,
            "old_group": old_group,
            "new_group": node.parent_group_id if node else None,
            "version": node.version if node else None
        })
        
        logger.info(f"Узел обновлён: {node_id}, v{node.version if node else '?'}")
        return node
    
    def delete_node(
        self,
        node_id: str,
        force: bool = False
    ) -> bool:
        """
        Удалить узел из графа.
        
        Args:
            node_id: ID узла для удаления
            force: Принудительное удаление (игнорирует is_static)
            
        Returns:
            True если удалён, False если не найден или защищён
        """
        node = self.get_node(node_id)
        if node is None:
            logger.warning(f"Узел не найден для удаления: {node_id}")
            return False
        
        # Проверка защищённых узлов
        if node.is_static and not force:
            logger.warning(f"Узел защищён (is_static=True): {node_id}. Используйте force=True")
            self._publish_event("memory.node_delete_blocked", {
                "node_id": node_id,
                "reason": "protected_static"
            })
            return False
        
        # Запоминаем данные для события
        old_group = node.parent_group_id
        node_type = node.node_type
        content_preview = node.content[:100] if len(node.content) > 100 else node.content
        
        # Удаляем связи
        edges_removed = self.storage.remove_edges_for_node(node_id)
        
        # Удаляем из индексов
        self.storage.remove_node_from_indexes(node_id, node)
        
        # Удаляем из nodes dict
        del self.storage.nodes[node_id]
        
        # Удаляем из БД
        conn = self.storage._get_connection()
        conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        conn.commit()
        conn.close()
        
        # Удаляем embedding из кэша
        if node_id in self.storage._normalized_embeddings:
            del self.storage._normalized_embeddings[node_id]
        
        # Инвалидируем кэш поиска
        self._search_cache.invalidate(node_id)
        
        # Инвалидируем кэш кластеров
        if hasattr(self, '_clusters_cache'):
            self._clusters_cache = None
        
        # Публикуем событие (1.1.3)
        self._publish_event("memory.node_deleted", {
            "node_id": node_id,
            "node_type": node_type,
            "old_group": old_group,
            "edges_removed": edges_removed,
            "content_preview": content_preview
        })
        
        logger.info(f"Узел удалён: {node_id}, связей: {edges_removed}")
        return True
    
    def add_knowledge(
        self,
        subject: str,
        relation: str,
        object_: str,
        subject_level: int = 1,
        object_level: int = 1,
        confidence: float = 0.5
    ) -> Tuple[FractalNode, FractalNode, FractalEdge]:
        """
        Добавить знание в формате S-P-O (Subject-Predicate-Object).
        
        Args:
            subject: Субъект
            relation: Отношение (is_a, part_of, attribute_of и др.)
            object_: Объект
            subject_level: Уровень субъекта
            object_level: Уровень объекта
            
        Returns:
            (subject_node, object_node, edge)
        """
        # Добавляем субъект
        subject_node = self.add_node(
            content=subject,
            node_type="concept",
            level=subject_level,
            confidence=confidence
        )
        
        # Добавляем объект
        object_node = self.add_node(
            content=object_,
            node_type="concept",
            level=object_level,
            confidence=confidence
        )
        
        # Добавляем связь
        edge = self.storage.add_edge(
            source_id=subject_node.id,
            target_id=object_node.id,
            relation_type=relation,
            weight=confidence
        )
        
        return subject_node, object_node, edge
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.5
    ) -> Optional[FractalEdge]:
        """Добавить связь между узлами."""
        return self.storage.add_edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight
        )
    
    def create_group(
        self,
        name: str,
        member_ids: List[str],
        level: int = 2
    ) -> SemanticGroup:
        """Создать семантическую группу (образ)."""
        return self.storage.create_semantic_group(
            name=name,
            member_ids=member_ids,
            level=level
        )
    
    # === ПОИСК ===
    
    @timed(threshold_ms=50.0)
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_level: int = 2,
        min_similarity: float = 0.5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Семантический поиск по запросу с LRU кэшированием.
        
        Args:
            query: Текстовый запрос
            top_k: Количество результатов
            min_level: Минимальный уровень для поиска
            min_similarity: Минимальная схожесть (по умолчанию 0.5 - повышено)
            use_cache: Использовать кэш (оптимизация)
            
        Returns:
            List of {node, similarity, group}
        """
        # Проверяем кэш
        cache_key = f"{query}:{top_k}:{min_level}:{min_similarity}"
        if use_cache:
            cached_result = self._search_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Semantic search cache hit for query: {query[:50]}...")
                return cached_result
        
        # Векторизуем запрос
        query_emb = self.embeddings.encode_single(query, normalize=True)
        
        if query_emb is None:
            return []
        
        # Поиск в графе (берём больше для фильтрации)
        results = self.storage.semantic_search(
            query_embedding=query_emb.tolist(),
            top_k=min(top_k * 3, 30),  # Берём больше для фильтрации
            min_level=min_level
        )
        
        # Форматируем и фильтруем результаты
        formatted = []
        for node_id_or_group_id, similarity, group_id in results:
            # Фильтр по минимальной схожести
            if similarity < min_similarity:
                continue
                
            if node_id_or_group_id in self.storage.nodes:
                node = self.storage.nodes[node_id_or_group_id]
                formatted.append({
                    "type": "node",
                    "id": node.id,
                    "content": node.content,
                    "node_type": node.node_type,
                    "level": node.level,
                    "confidence": node.confidence,
                    "similarity": similarity,
                    "group_id": group_id
                })
            elif node_id_or_group_id in self.storage.semantic_groups:
                group = self.storage.semantic_groups[node_id_or_group_id]
                # Получаем членов группы
                members = self.storage.get_group_members(group.id)
                formatted.append({
                    "type": "group",
                    "id": group.id,
                    "name": group.name,
                    "member_count": len(members),
                    "avg_confidence": group.avg_confidence,
                    "similarity": similarity,
                    "members": [m.content[:50] for m in members[:5]]
                })
        
        # Сохраняем в кэш
        if use_cache:
            self._search_cache.put(cache_key, formatted)
        
        return formatted
    
    def semantic_search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        min_level: int = 2,
        min_similarity: float = 0.5,
        use_cache: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch семантический поиск для нескольких запросов.
        
        Этот метод эффективнее чем semantic_search() в цикле,
        так как использует batch encoding для запросов.
        
        Args:
            queries: Список текстовых запросов
            top_k: Количество результатов на запрос
            min_level: Минимальный уровень для поиска
            min_similarity: Минимальная схожесть
            use_cache: Использовать кэш
            
        Returns:
            Dict {query: results}
        """
        start_time = time.time()
        
        # Проверяем кэш для всех запросов
        results = {}
        queries_to_process = []
        
        if use_cache:
            for query in queries:
                cache_key = f"{query}:{top_k}:{min_level}:{min_similarity}"
                cached = self._search_cache.get(cache_key)
                if cached is not None:
                    results[query] = cached
                else:
                    queries_to_process.append(query)
        else:
            queries_to_process = queries
        
        if not queries_to_process:
            logger.debug(f"Batch search: все {len(queries)} запросов из кэша")
            return results
        
        # Batch encoding для оставшихся запросов
        logger.debug(f"Batch encoding для {len(queries_to_process)} запросов...")
        query_embeddings = self.embeddings.encode(queries_to_process, normalize=True)
        
        if query_embeddings is None:
            logger.warning("Embedding model недоступен, возвращаю пустые результаты")
            return results
        
        # Поиск для каждого запроса
        for query, query_emb in zip(queries_to_process, query_embeddings):
            # Поиск в графе
            search_results = self.storage.semantic_search(
                query_embedding=query_emb.tolist(),
                top_k=min(top_k * 3, 30),
                min_level=min_level
            )
            
            # Форматируем и фильтруем
            formatted = []
            for node_id_or_group_id, similarity, group_id in search_results:
                if similarity < min_similarity:
                    continue
                    
                if node_id_or_group_id in self.storage.nodes:
                    node = self.storage.nodes[node_id_or_group_id]
                    formatted.append({
                        "type": "node",
                        "id": node.id,
                        "content": node.content,
                        "node_type": node.node_type,
                        "level": node.level,
                        "confidence": node.confidence,
                        "similarity": similarity,
                        "group_id": group_id
                    })
                elif node_id_or_group_id in self.storage.semantic_groups:
                    group = self.storage.semantic_groups[node_id_or_group_id]
                    members = self.storage.get_group_members(group.id)
                    formatted.append({
                        "type": "group",
                        "id": group.id,
                        "name": group.name,
                        "member_count": len(members),
                        "avg_confidence": group.avg_confidence,
                        "similarity": similarity,
                        "members": [m.content[:50] for m in members[:5]]
                    })
            
            results[query] = formatted
            
            # Сохраняем в кэш
            if use_cache:
                cache_key = f"{query}:{top_k}:{min_level}:{min_similarity}"
                self._search_cache.put(cache_key, formatted)
        
        elapsed = time.time() - start_time
        logger.debug(f"Batch search завершен: {len(queries)} запросов за {elapsed:.3f}s")
        
        return results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[FractalNode]:
        """Поиск по ключевым словам."""
        node_ids = self.storage.keyword_search(query, top_k)
        return [self.storage.nodes[nid] for nid in node_ids if nid in self.storage.nodes]
    
    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Получить контекст узла (группа, связи, атрибуты)."""
        return self.storage.get_node_context(node_id)
    
    # === GGUF МОДЕЛИ ===
    
    def load_gguf_knowledge(self, model_path: str) -> Dict[str, Any]:
        """
        Извлечь знания из GGUF модели и добавить в граф.
        
        Args:
            model_path: Путь к GGUF файлу
            
        Returns:
            Результат добавления
        """
        return extract_to_graph(model_path, self.storage)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """Получить информацию о GGUF модели."""
        info = parse_gguf_model(model_path)
        return {
            "architecture": info.architecture,
            "model_type": info.model_type,
            "vocab_size": info.vocab_size,
            "hidden_size": info.hidden_size,
            "num_layers": info.num_layers,
            "num_attention_heads": info.num_attention_heads,
            "max_context": info.max_position_embeddings,
            "file_size": info.file_size
        }
    
    # === ВЕКТОРИЗАЦИЯ ===
    
    def _vectorize_single_node(self, node_id: str):
        """Векторизовать один узел."""
        if node_id not in self.storage.nodes:
            return
        
        node = self.storage.nodes[node_id]
        if node.embedding is not None:
            return  # Уже векторизован
        
        emb = self.embeddings.encode_single(node.content, normalize=True)
        if emb is not None:
            node.embedding = emb.tolist()
            self.storage._save_node(node)
    
    def vectorize_all(self, level_filter: int = None):
        """Векторизовать все узлы."""
        nodes_to_vectorize = []
        
        for node_id, node in self.storage.nodes.items():
            if node.embedding is None:
                if level_filter is None or node.level >= level_filter:
                    nodes_to_vectorize.append(node)
        
        if not nodes_to_vectorize:
            logger.info("Все узлы уже векторизованы")
            return
        
        logger.info(f"Векторизация {len(nodes_to_vectorize)} узлов...")
        
        texts = [node.content for node in nodes_to_vectorize]
        embeddings = self.embeddings.encode(texts, normalize=True)
        
        if embeddings is None:
            logger.warning("Embedding model недоступен, пропускаем векторизацию")
            return
        
        for node, emb in zip(nodes_to_vectorize, embeddings):
            if emb is not None:
                node.embedding = emb.tolist()
                self.storage._save_node(node)
        
        logger.info("Векторизация завершена")
    
    def vectorize_groups(self):
        """Векторизовать семантические группы (вычислить центроиды)."""
        for group_id, group in self.storage.semantic_groups.items():
            members = self.storage.get_group_members(group_id)
            
            embeddings = []
            for member in members:
                if member.embedding:
                    import numpy as np
                    embeddings.append(np.array(member.embedding))
            
            if embeddings:
                import numpy as np
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                
                group.embedding = centroid.tolist()
                group.avg_confidence = np.mean([m.confidence for m in members])
                self.storage._save_group(group)
        
        logger.info("Группы векторизованы")
    
    # === КЛАСТЕРИЗАЦИЯ ===
    
    def auto_cluster(
        self,
        level: int = 1,
        threshold: float = 0.5,
        method: str = "agglomerative"
    ):
        """
        Автоматическая кластеризация узлов уровня.
        
        Args:
            level: Уровень для кластеризации
            threshold: Порог similarity для объединения в группу
            method: Метод кластеризации (agglomerative, dbscan, simple)
        """
        clusters = self.storage.cluster_nodes(
            level=level,
            threshold=threshold,
            method=method
        )
        
        created_groups = 0
        for cluster_name, member_ids in clusters.items():
            if not member_ids:
                continue
            
            # Создаём группу
            group = self.storage.create_semantic_group(
                name=cluster_name,
                member_ids=member_ids,
                level=level + 1
            )
            
            # Векторизуем группу (центроид)
            members = [self.storage.nodes[mid] for mid in member_ids if mid in self.storage.nodes]
            if members:
                import numpy as np
                embeddings = [np.array(m.embedding) for m in members if m.embedding]
                if embeddings:
                    centroid = np.mean(embeddings, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    group.embedding = centroid.tolist()
                    self.storage._save_group(group)
            
            created_groups += 1
        
        logger.info(f"Создано {created_groups} групп методом {method}")
        
        return created_groups
    
    # === ПРОТИВОРЕЧИЯ ===
    
    def check_contradiction(self, content: str, group_id: str = None) -> Dict[str, Any]:
        """
        Проверить текст на противоречие с группой.
        
        Args:
            content: Текст для проверки
            group_id: ID группы (опционально)
            
        Returns:
            {is_contradiction, distance, suggestions}
        """
        # Векторизуем текст
        emb = self.embeddings.encode_single(content, normalize=True)
        
        if emb is None:
            return {"is_contradiction": False, "error": "no embedding"}
        
        # Если группа не указана - ищем ближайшую
        if group_id is None:
            results = self.storage.semantic_search(emb.tolist(), top_k=1, min_level=1)
            if results and results[0][2]:  # group_id
                group_id = results[0][2]
            # Или ищем ближайший узел
            elif results:
                node_id = results[0].get('id')
                if node_id:
                    node = self.storage.nodes.get(node_id)
                    group_id = node.parent_group_id if node else None
        
        if group_id:
            is_contr, distance = self.storage.detect_contradiction(
                emb.tolist(), group_id, threshold=0.7
            )
            return {
                "is_contradiction": is_contr,
                "distance": distance,
                "group_id": group_id
            }
        
        return {"is_contradiction": False, "reason": "no group found"}
    
    def resolve_contradiction(self, node_id: str, resolution: str = "remove"):
        """
        Разрешить противоречие.
        
        Args:
            node_id: ID противоречивого узла
            resolution: one of "remove", "merge", "keep"
        """
        if node_id not in self.storage.nodes:
            return {"error": "node not found"}
        
        node = self.storage.nodes[node_id]
        
        if resolution == "remove":
            self.storage.mark_contradiction(node_id, "removed by resolution")
            return {"status": "removed", "node_id": node_id}
        
        elif resolution == "keep":
            node.is_contradiction = False
            node.confidence = 1.0  # Подтверждено
            self.storage._save_node(node)
            return {"status": "confirmed", "node_id": node_id}
        
        return {"status": "unknown_resolution"}
    
    def self_dialogue(self, new_knowledge: str) -> Dict[str, Any]:
        """
        Самодиалог - автоматическая верификация нового знания.
        
        Согласно спецификации:
        1. Проверяем противоречие с существующими группами
        2. Ищем подтверждающие или опровергающие факты в графе
        3. Разрешаем противоречие на основе анализа
        
        Args:
            new_knowledge: Новое знание для проверки
            
        Returns:
            {confirmed, action, reasoning, new_nodes}
        """
        # 1. Проверяем на противоречие
        check_result = self.check_contradiction(new_knowledge)
        
        if not check_result.get("is_contradiction"):
            # Нет противоречия - просто добавляем знание
            return {
                "confirmed": True,
                "action": "add",
                "reasoning": "No contradiction detected",
                "new_nodes": []
            }
        
        # 2. Противоречие найдено - анализируем
        group_id = check_result.get("group_id")
        group = self.storage.semantic_groups.get(group_id)
        
        if not group:
            return {"confirmed": False, "action": "reject", "reasoning": "No group found"}
        
        # 3. Ищем связанные факты в графе
        related_facts = self._search_related_facts(new_knowledge, group_id)
        
        # 4. Анализ: есть ли подтверждающие факты?
        confirming_facts = [f for f in related_facts if f.get("similarity", 0) > 0.7]
        
        if len(confirming_facts) >= 2:
            # Много подтверждений - вероятно знание верное
            # Добавляем как новый контекстный узел
            new_node = self.add_node(
                content=new_knowledge,
                node_type="context",
                level=2,
                confidence=0.3,  # Низкая уверенность initially
                metadata={"source": "self_dialogue", "parent_group": group_id}
            )
            
            return {
                "confirmed": True,
                "action": "add_as_context",
                "reasoning": f"Found {len(confirming_facts)} confirming facts",
                "new_nodes": [new_node.id]
            }
        
        elif len(confirming_facts) == 1:
            # Один подтверждающий факт - неопределённо
            # Добавляем с низкой уверенностью
            new_node = self.add_node(
                content=new_knowledge,
                node_type="context",
                level=1,
                confidence=0.2,
                metadata={"source": "self_dialogue", "needs_verification": True}
            )
            
            return {
                "confirmed": False,
                "action": "add_uncertain",
                "reasoning": f"Only {len(confirming_facts)} confirming fact",
                "new_nodes": [new_node.id]
            }
        
        else:
            # Нет подтверждений - скорее всего ошибка
            # Помечаем как противоречие
            return {
                "confirmed": False,
                "action": "reject",
                "reasoning": "No confirming facts found",
                "new_nodes": []
            }
    
    def _search_related_facts(self, query: str, group_id: str, top_k: int = 10) -> List[Dict]:
        """Поиск связанных фактов в группе."""
        results = self.semantic_search(query, top_k=top_k, min_level=1)
        
        # Фильтруем только факты из той же группы
        related = []
        for r in results:
            node_id = r.get("id")
            if node_id and node_id in self.storage.nodes:
                node = self.storage.nodes[node_id]
                if node.parent_group_id == group_id:
                    related.append(r)
        
        return related
    
    # === СТАТИСТИКА ===
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику графа."""
        base_stats = self.storage.get_stats()
        
        # Дополнительная статистика
        base_stats["embedding_cache_size"] = self.embeddings.get_cache_size()
        
        # Распределение по уровням
        level_dist = {}
        for node_id, node in self.storage.nodes.items():
            level_dist[node.level] = level_dist.get(node.level, 0) + 1
        base_stats["nodes_by_level"] = level_dist
        
        return base_stats
    
    def get_node(self, node_id: str) -> Optional[FractalNode]:
        """Получить узел по ID."""
        return self.storage.nodes.get(node_id)
    
    def get_all_nodes(self, level: int = None, node_type: str = None) -> List[FractalNode]:
        """Получить все узлы с фильтрацией."""
        nodes = list(self.storage.nodes.values())
        
        if level is not None:
            nodes = [n for n in nodes if n.level == level]
        
        if node_type is not None:
            nodes = [n for n in nodes if n.node_type == node_type]
        
        return nodes
    
    def get_groups(self, level: int = None) -> List[SemanticGroup]:
        """Получить семантические группы."""
        groups = list(self.storage.semantic_groups.values())
        
        if level is not None:
            groups = [g for g in groups if g.level == level]
        
        return groups
    
    def get_nodes_list(self, limit: int = 200) -> List[FractalNode]:
        """Получить список узлов для API."""
        return list(self.storage.nodes.values())[:limit]
    
    def get_edges_list(self, limit: int = 500) -> List[FractalEdge]:
        """Получить список связей для API."""
        return list(self.storage.edges.values())[:limit]
    
    # === ИНТЕГРАЦИЯ СО СТАРОЙ СИСТЕМОЙ ===
    
    def save_experience(
        self,
        query: str,
        response: str,
        model_used: str,
        quality_score: float = 0.5
    ) -> str:
        """
        Сохранить опыт (query/response) в граф.
        Аналог UnifiedFractalMemory.save_experience()
        
        Args:
            query: Запрос пользователя
            response: Ответ системы
            model_used: Какая модель использовалась (model_a, model_b, model_c, web_ui)
            quality_score: Оценка качества (0-1)
            
        Returns:
            ID созданного query узла
        """
        # Создаём узел для query
        query_node = self.add_node(
            content=query[:500],  # Ограничиваем длину
            node_type="query",
            level=2,
            confidence=quality_score,
            metadata={
                "source": "experience",
                "model": model_used,
                "timestamp": time.time()
            },
            auto_vectorize=True
        )
        
        # Создаём узел для response
        if response and len(response) > 3:
            response_node = self.add_node(
                content=response[:1000],
                node_type="response",
                level=2,
                confidence=quality_score,
                metadata={
                    "source": "experience", 
                    "model": model_used,
                    "timestamp": time.time()
                },
                auto_vectorize=True
            )
            
            # Связываем query -> response
            self.storage.add_edge(
                source_id=query_node.id,
                target_id=response_node.id,
                relation_type="generated_by",
                weight=quality_score
            )
        
        return query_node.id
    
    def get_context_for_query(self, query: str, max_length: int = 512, min_similarity: float = 0.5) -> str:
        """
        Получить контекст для запроса.
        Аналог UnifiedFractalMemory.get_context_for_query()
        
        Args:
            query: Запрос пользователя
            max_length: Максимальная длина контекста
            min_similarity: Минимальная схожесть для включения в контекст
            
        Returns:
            Текстовый контекст из графа
        """
        # Семантический поиск с фильтрацией по схожести
        results = self.semantic_search(query, top_k=10, min_level=1, min_similarity=min_similarity)
        
        if not results:
            return ""
        
        # Фильтрация мусора и формирование контекста
        context_parts = []
        template_patterns = [
            'продолжим разговор', 'перспективы развития',
            '###', '##', 'особенности данного',
            'q:', 'a:', 'пример:'
        ]
        
        for r in results:
            content = r.get('content', '')
            if not content:
                continue
            
            # Проверка на мусор
            content_lower = content.lower()
            is_garbage = any(p in content_lower for p in template_patterns)
            if is_garbage:
                continue
            
            # Проверка минимальной длины
            if len(content) < 30:
                continue
            
            context_parts.append(content)
        
        context = "\n".join(context_parts)
        
        # Обрезаем по длине
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context
    
    def retrieve_knowledge(self, query: str, top_k: int = 5, min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """
        Извлечь знания по запросу.
        Аналог UnifiedFractalMemory.retrieve_knowledge()
        
        Args:
            query: Запрос
            top_k: Количество результатов
            min_similarity: Минимальная схожесть
            
        Returns:
            List of {node_id, content, similarity, level}
        """
        results = self.semantic_search(query, top_k=top_k, min_level=1, min_similarity=min_similarity)
        
        knowledge = []
        template_patterns = [
            'продолжим разговор', 'перспективы развития',
            '###', '##', 'особенности данного',
            'q:', 'a:', 'пример:'
        ]
        
        for r in results:
            content = r.get('content', '')
            if not content:
                continue
            
            content_lower = content.lower()
            if any(p in content_lower for p in template_patterns):
                continue
            if len(content) < 30:
                continue
            
            knowledge.append({
                "node_id": r.get("id"),
                "content": content,
                "similarity": r.get("similarity"),
                "level": r.get("level")
            })
        
        return knowledge
    
    # === УПРАВЛЕНИЕ LLAMA ИНСТАНСАМИ ===
    
    def register_model_instance(self, model_type: str, llama_instance):
        """
        Зарегистрировать Llama инстанс модели.
        Аналог UnifiedFractalMemory.register_model_instance()
        
        Args:
            model_type: Тип модели (model_a, model_b, model_c)
            llama_instance: Llama инстанс из llama_cpp
        """
        if not hasattr(self, '_model_instances'):
            self._model_instances: Dict[str, Any] = {}
        
        self._model_instances[model_type] = llama_instance
        logger.info(f"Зарегистрирован Llama инстанс: {model_type}")
    
    def get_model_instance(self, model_type: str):
        """Получить Llama инстанс модели."""
        if hasattr(self, '_model_instances'):
            return self._model_instances.get(model_type)
        return None
    
    def get_model_context(self, model_type: str) -> Dict[str, Any]:
        """
        Получить контекст для конкретной модели.
        Аналог UnifiedFractalMemory.get_model_context()
        """
        # Ищем узлы связанные с этой моделью
        model_nodes = self.get_all_nodes(node_type=model_type)
        
        return {
            "model_type": model_type,
            "nodes_count": len(model_nodes),
            "nodes": [
                {"id": n.id, "content": n.content[:50], "level": n.level}
                for n in model_nodes[:5]
            ]
        }
    
    def get_static_models(self) -> List[Dict[str, Any]]:
        """
        Получить информацию о статичных моделях.
        Аналог UnifiedFractalMemory.get_static_models()
        """
        models = []
        for model_type in ['model_a', 'model_b', 'model_c']:
            nodes = self.get_all_nodes(node_type=model_type)
            if nodes:
                models.append({
                    "type": model_type,
                    "content": nodes[0].content[:100],
                    "nodes_count": len(nodes)
                })
        
        return models
    
    # === УПРАВЛЕНИЕ КЭШЕМ (Оптимизация производительности) ===
    
    def get_search_cache_stats(self) -> Dict[str, Any]:
        """
        Получить статистику кэша семантического поиска.
        
        Returns:
            Статистика кэша (size, hits, misses, hit_rate)
        """
        return self._search_cache.stats()
    
    def clear_search_cache(self):
        """Очистить кэш семантического поиска."""
        self._search_cache.clear()
        logger.info("Semantic search cache cleared")
    
    def invalidate_cache_for_node(self, node_id: str):
        """
        Инвалидировать кэш при изменении узла.
        Простая реализация - очищает весь кэш.
        """
        self._search_cache.clear()
        self._clusters_cache = None
        logger.debug(f"Cache invalidated for node: {node_id}")
    
    def get_clusters(self, force_refresh: bool = False) -> Dict[str, List[str]]:
        """
        Получить кластеры узлов с кэшированием.
        Это оптимизация для ConceptMiner - избегаем O(n²) вычисления при каждом запросе.
        
        Args:
            force_refresh: Принудительно обновить кэш
            
        Returns:
            Dict[str, List[str]]: {cluster_name: [node_ids]}
        """
        if hasattr(self, '_clusters_cache') and self._clusters_cache is not None and not force_refresh:
            return self._clusters_cache
        
        import numpy as np
        
        clusters = {}
        
        if not hasattr(self.storage, 'nodes'):
            return clusters
        
        nodes_with_embeddings = []
        for node_id, node in self.storage.nodes.items():
            emb = getattr(node, 'embedding', None)
            if emb is not None:
                nodes_with_embeddings.append((node_id, np.array(emb)))
        
        if not nodes_with_embeddings:
            self._clusters_cache = clusters
            return clusters
        
        visited = set()
        cluster_id = 0
        
        for i, (node_id_i, emb_i) in enumerate(nodes_with_embeddings):
            if node_id_i in visited:
                continue
            cluster_nodes = [node_id_i]
            visited.add(node_id_i)
            
            for j, (node_id_j, emb_j) in enumerate(nodes_with_embeddings[i+1:], i+1):
                if node_id_j in visited:
                    continue
                
                norm_i = np.linalg.norm(emb_i)
                norm_j = np.linalg.norm(emb_j)
                if norm_i > 0 and norm_j > 0:
                    similarity = np.dot(emb_i, emb_j) / (norm_i * norm_j)
                    if similarity > 0.7:
                        cluster_nodes.append(node_id_j)
                        visited.add(node_id_j)
            
            if len(cluster_nodes) >= 3:
                clusters[f"auto_cluster_{cluster_id}"] = cluster_nodes
                cluster_id += 1
        
        self._clusters_cache = clusters
        logger.debug(f"Clusters cached: {len(clusters)} clusters, {sum(len(v) for v in clusters.values())} nodes")
        
        # Публикуем событие о завершении кластеризации (1.1.3)
        self._publish_event("memory.clustering_complete", {
            "cluster_count": len(clusters),
            "total_nodes": sum(len(v) for v in clusters.values()),
            "force_refresh": force_refresh
        })
        
        return clusters


def create_fractal_memory_graph(
    storage_dir: str = None,
    embedding_device: str = "cuda",
    event_bus = None
) -> FractalMemoryGraph:
    """Фабричная функция для создания фрактального графа памяти."""
    return FractalMemoryGraph(
        storage_dir=storage_dir,
        embedding_device=embedding_device,
        event_bus=event_bus
    )


# === ЭКСПОРТ ТИПОВ ===
__all__ = [
    # Main
    'FractalMemoryGraph',
    'create_fractal_memory_graph',
    
    # Types
    'FractalNode',
    'FractalEdge', 
    'SemanticGroup',
    'NodeType',
    'RelationType',
    
    # GGUF
    'parse_gguf_model',
    'extract_to_graph',
    'create_extractor',
    'GGUFKnowledgeExtractor',
    
    # Tokenizer
    'GraphTokenizer',
    'create_graph_tokenizer',
]