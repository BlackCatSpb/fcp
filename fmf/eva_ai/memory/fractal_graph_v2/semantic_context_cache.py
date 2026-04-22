"""
SemanticContextCache - Кэш необработанного контекста с семантическим поиском

Особенности:
- CPU-based (GPU уже занят эмбеддером)
- FAISS для быстрого семантического поиска
- LRU eviction для управления памятью
- Интеграция с HybridTokenCache
"""

import os
import time
import logging
import threading
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

logger = logging.getLogger("eva_ai.fractal_graph_v2.semantic_cache")


@dataclass
class ContextEntry:
    """Запись контекста."""
    text: str
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticContextCache:
    """
    Кэш необработанного контекста с семантическим поиском.
    
    Хранит тексты НЕ в графе, но релевантные для семантического поиска.
    Использует CPU-based эмбеддинги для экономии GPU памяти.
    """
    
    def __init__(
        self,
        max_contexts: int = 1000,
        embedding_dim: int = 768,
        use_faiss: bool = True,
        cache_dir: str = None
    ):
        """
        Args:
            max_contexts: Максимум контекстов в памяти
            embedding_dim: Размерность эмбеддингов (меньше = меньше памяти)
            use_faiss: Использовать FAISS для быстрого поиска
            cache_dir: Директория для дискового кэша
        """
        self.max_contexts = max_contexts
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "semantic_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.contexts: OrderedDict[str, ContextEntry] = OrderedDict()
        self.index = None
        self.index_to_id: List[str] = []
        
        if use_faiss:
            self._init_faiss()
        
        self._lock = threading.RLock()
        
        self.stats = {
            'adds': 0,
            'searches': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        logger.info(f"SemanticContextCache: max={max_contexts}, dim={embedding_dim}, faiss={use_faiss}")
    
    def _init_faiss(self):
        """Инициализация FAISS индекса."""
        try:
            import faiss
            self.faiss = faiss
            
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                min(100, self.max_contexts // 10),
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(np.random.randn(1000, self.embedding_dim).astype(np.float32))
            
            logger.info("FAISS index initialized")
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self.faiss = None
            self.use_faiss = False
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг текста (CPU-based)."""
        hash_key = hashlib.md5(text.encode()).digest()[:16]
        
        cache_path = os.path.join(self.cache_dir, f"{hash_key.hex()}.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path)
        
        embedding = self._compute_embedding(text)
        
        try:
            np.save(cache_path, embedding)
        except Exception:
            pass
        
        return embedding
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Вычислить эмбеддинг (CPU эмбеддер).
        
        Returns:
            np.ndarray или None если эмбеддинг не работает.
            Случайные векторы НЕ возвращаются - они ломают семантический поиск.
        """
        try:
            from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
            
            if not hasattr(self, '_embedder'):
                self._embedder = get_sentence_transformer(device='cpu')
            
            if self._embedder is None:
                return None
            
            emb = self._embedder.encode(text, convert_to_numpy=True)
            
            if len(emb) > self.embedding_dim:
                emb = emb[:self.embedding_dim]
            elif len(emb) < self.embedding_dim:
                emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
            
            return emb.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Embedding failed: {e} - context will not be indexed")
            return None
    
    def add(
        self, 
        text: str, 
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Добавить контекст в кэш.
        
        Returns:
            context_id или None если embedding недоступен.
        """
        with self._lock:
            context_id = hashlib.sha256(text.encode()).hexdigest()[:16]
            
            if context_id in self.contexts:
                self.contexts.move_to_end(context_id)
                entry = self.contexts[context_id]
                entry.access_count += 1
                entry.timestamp = time.time()
                return context_id
            
            embedding = self._get_embedding(text)
            
            # Если embedding не работает - не добавляем в индекс
            if embedding is None:
                logger.warning(f"Skipping index for context {context_id[:8]}... - embedding unavailable")
                return None
            
            entry = ContextEntry(
                text=text,
                embedding=embedding,
                timestamp=time.time(),
                session_id=session_id,
                metadata=metadata or {}
            )
            
            self.contexts[context_id] = entry
            self.contexts.move_to_end(context_id)
            
            if self.use_faiss and self.index and not self.index.is_trained:
                self.index.train(np.array([embedding]))
            
            if self.use_faiss and self.index:
                self.index.add(np.array([embedding]))
                self.index_to_id.append(context_id)
            
            while len(self.contexts) > self.max_contexts:
                evicted_id, evicted_entry = self.contexts.popitem(last=False)
                self.stats['evictions'] += 1
                
                if self.use_faiss and self.index and evicted_id in self.index_to_id:
                    idx = self.index_to_id.index(evicted_id)
                    self.index_to_id.pop(idx)
            
            self.stats['adds'] += 1
            
            return context_id
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.3,
        session_filter: str = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожего контекста.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов
            min_similarity: Минимальная схожесть
            session_filter: Фильтр по session_id
            
        Returns:
            List of {context_id, text, similarity, metadata}
        """
        self.stats['searches'] += 1
        
        query_emb = self._get_embedding(query)
        
        results = []
        
        if self.use_faiss and self.index and len(self.contexts) > 0:
            k = min(top_k * 2, len(self.contexts))
            
            try:
                query_emb_2d = query_emb.reshape(1, -1)
                if hasattr(self.index, 'nprobe'):
                    distances, indices = self.index.search(query_emb_2d, k)
                else:
                    distances, indices = self.index.search(query_emb_2d, k)
                
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self.index_to_id):
                        continue
                    
                    context_id = self.index_to_id[idx]
                    if context_id not in self.contexts:
                        continue
                    
                    entry = self.contexts[context_id]
                    
                    if session_filter and entry.session_id != session_filter:
                        continue
                    
                    similarity = float(dist)
                    if similarity >= min_similarity:
                        results.append({
                            'context_id': context_id,
                            'text': entry.text,
                            'similarity': similarity,
                            'timestamp': entry.timestamp,
                            'metadata': entry.metadata
                        })
                        self.stats['hits'] += 1
            except Exception as e:
                logger.debug(f"FAISS search failed: {e}")
                results = self._numpy_search(query_emb, top_k, min_similarity, session_filter)
        else:
            results = self._numpy_search(query_emb, top_k, min_similarity, session_filter)
        
        if not results:
            self.stats['misses'] += 1
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _numpy_search(
        self,
        query_emb: np.ndarray,
        top_k: int,
        min_similarity: float,
        session_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Поиск через numpy (fallback)."""
        results = []
        
        for context_id, entry in self.contexts.items():
            if session_filter and entry.session_id != session_filter:
                continue
            
            similarity = np.dot(query_emb, entry.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(entry.embedding) + 1e-8
            )
            
            if similarity >= min_similarity:
                results.append({
                    'context_id': context_id,
                    'text': entry.text,
                    'similarity': float(similarity),
                    'timestamp': entry.timestamp,
                    'metadata': entry.metadata
                })
        
        return results
    
    def get_hot_window(
        self, 
        window_size: int = 10,
        session_id: str = None
    ) -> List[str]:
        """
        Получить последние N контекстов (hot window).
        
        Args:
            window_size: Количество последних контекстов
            session_id: Фильтр по сессии
            
        Returns:
            List of context texts
        """
        with self._lock:
            contexts = []
            
            for context_id, entry in reversed(list(self.contexts.items())):
                if session_id and entry.session_id != session_id:
                    continue
                contexts.append(entry.text)
                if len(contexts) >= window_size:
                    break
            
            return contexts
    
    def get_session_contexts(self, session_id: str) -> List[str]:
        """Получить все контексты сессии."""
        with self._lock:
            return [
                entry.text 
                for entry in self.contexts.values() 
                if entry.session_id == session_id
            ]
    
    def clear_session(self, session_id: str):
        """Очистить контексты сессии."""
        with self._lock:
            to_remove = [
                cid for cid, entry in self.contexts.items()
                if entry.session_id == session_id
            ]
            for cid in to_remove:
                del self.contexts[cid]
            
            if self.use_faiss and self.index:
                self._rebuild_index()
    
    def _rebuild_index(self):
        """Перестроить FAISS индекс."""
        if not self.use_faiss or not self.index:
            return
        
        self.index.reset()
        self.index_to_id.clear()
        
        embeddings = []
        for context_id in self.contexts.keys():
            entry = self.contexts[context_id]
            embeddings.append(entry.embedding)
            self.index_to_id.append(context_id)
        
        if embeddings:
            self.index.add(np.array(embeddings))
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total if total > 0 else 0
        
        return {
            'total_contexts': len(self.contexts),
            'max_contexts': self.max_contexts,
            'embedding_dim': self.embedding_dim,
            'uses_faiss': self.use_faiss,
            'stats': self.stats,
            'hit_rate': hit_rate,
            'memory_estimate_mb': len(self.contexts) * self.embedding_dim * 4 / 1024 / 1024
        }
    
    def save(self, path: str = None) -> bool:
        """Сохранить кэш на диск."""
        import json
        
        path = path or os.path.join(self.cache_dir, "semantic_cache.json")
        
        try:
            data = {
                'max_contexts': self.max_contexts,
                'embedding_dim': self.embedding_dim,
                'contexts': [
                    {
                        'id': cid,
                        'text': entry.text,
                        'embedding': entry.embedding.tolist(),
                        'timestamp': entry.timestamp,
                        'session_id': entry.session_id,
                        'metadata': entry.metadata
                    }
                    for cid, entry in self.contexts.items()
                ],
                'stats': self.stats
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            
            logger.info(f"SemanticContextCache saved: {len(self.contexts)} contexts")
            return True
        except Exception as e:
            logger.error(f"Failed to save SemanticContextCache: {e}")
            return False
    
    def load(self, path: str = None) -> bool:
        """Загрузить кэш с диска."""
        import json
        
        path = path or os.path.join(self.cache_dir, "semantic_cache.json")
        
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.contexts.clear()
            
            for item in data.get('contexts', []):
                entry = ContextEntry(
                    text=item['text'],
                    embedding=np.array(item['embedding'], dtype=np.float32),
                    timestamp=item['timestamp'],
                    session_id=item.get('session_id'),
                    metadata=item.get('metadata', {})
                )
                self.contexts[item['id']] = entry
            
            self.stats = data.get('stats', self.stats)
            
            if self.use_faiss and self.index:
                self._rebuild_index()
            
            logger.info(f"SemanticContextCache loaded: {len(self.contexts)} contexts")
            return True
        except Exception as e:
            logger.error(f"Failed to load SemanticContextCache: {e}")
            return False


    def touch(
        self, 
        context_id: str, 
        importance: float = 1.0,
        access_bonus: float = 0.5
    ) -> bool:
        """
        Обновляет вес контекста при обращении (умная эвикция).
        
        Args:
            context_id: ID контекста.
            importance: Важность (добавляется к весу).
            access_bonus: Бонус за обращение.
            
        Returns:
            True если обновлено, False если не найден.
        """
        with self._lock:
            if context_id not in self.contexts:
                return False
            
            entry = self.contexts[context_id]
            entry.access_count += 1
            entry.timestamp = time.time()
            
            current_weight = entry.metadata.get('_importance_weight', 1.0)
            entry.metadata['_importance_weight'] = current_weight + importance + access_bonus
            
            self.contexts.move_to_end(context_id)
            return True

    def smart_evict(self) -> Optional[str]:
        """
        Умная эвикция - вытесняет элемент с наименьшим весом важности.
        
        Returns:
            ID вытесненного элемента или None.
        """
        if not self.contexts:
            return None
        
        min_weight = float('inf')
        min_id = None
        
        for cid, entry in self.contexts.items():
            weight = entry.metadata.get('_importance_weight', 1.0)
            recency_bonus = (time.time() - entry.timestamp) / 3600
            total_weight = weight + recency_bonus
            
            if total_weight < min_weight:
                min_weight = total_weight
                min_id = cid
        
        if min_id:
            del self.contexts[min_id]
            self.stats['evictions'] += 1
            
            if self.use_faiss and self.index and min_id in self.index_to_id:
                idx = self.index_to_id.index(min_id)
                self.index_to_id.pop(idx)
            
            return min_id
        
        return None

    def add_with_importance(
        self, 
        text: str, 
        importance: float = 1.0,
        session_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Добавить контекст с указанием важности.
        
        Args:
            text: Текст контекста.
            importance: Важность (чем выше, тем дольше живет в кэше).
            session_id: ID сессии.
            metadata: Дополнительные метаданные.
            
        Returns:
            context_id или None.
        """
        meta = metadata.copy() if metadata else {}
        meta['_importance_weight'] = importance
        
        return self.add(text, session_id=session_id, metadata=meta)

    def get_weighted_contexts(
        self, 
        min_weight: float = 1.0,
        limit: int = 20
    ) -> List[str]:
        """
        Получить контексты с весом выше порога.
        
        Args:
            min_weight: Минимальный вес.
            limit: Максимум результатов.
            
        Returns:
            List текстов контекстов.
        """
        with self._lock:
            results = []
            for entry in self.contexts.values():
                weight = entry.metadata.get('_importance_weight', 1.0)
                if weight >= min_weight:
                    results.append(entry.text)
                    if len(results) >= limit:
                        break
            return results


def create_semantic_context_cache(
    max_contexts: int = 1000,
    cache_dir: str = None
) -> SemanticContextCache:
    """Фабричная функция."""
    return SemanticContextCache(
        max_contexts=max_contexts,
        cache_dir=cache_dir
    )