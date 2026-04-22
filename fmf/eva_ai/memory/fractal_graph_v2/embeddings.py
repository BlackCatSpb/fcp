"""
Embeddings Manager - Векторизация узлов графа через multilingual-e5-base

Обеспечивает:
- Векторизацию текстовых данных
- Семантический поиск
- Кластеризацию по эмбеддингам
"""

import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("eva_ai.fractal_graph_v2.embeddings")

# Устанавливаем HF_HOME на локальный кеш если не установлен
_HF_CACHE_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core', 'hf_cache')
if not os.environ.get('HF_HOME'):
    os.environ['HF_HOME'] = _HF_CACHE_DEFAULT
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# SINGLETON - единственный экземпляр на всю систему
_embeddings_manager_instance: Optional['EmbeddingsManager'] = None
_embeddings_manager_lock = threading.Lock()


class EmbeddingsManager:
    """
    Менеджер эмбеддингов для фрактального графа.
    
    Использует sentence-transformers через singleton кеш.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        cache_dir: str = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели через singleton кеш."""
        try:
            from eva_ai.mlearning.sentence_transformers_cache import get_sentence_transformer
            
            logger.info(f"Загрузка модели эмбеддингов через singleton кеш на {self.device}")
            
            self.model = get_sentence_transformer(device=self.device)
            if self.model:
                self.tokenizer = getattr(self.model, 'tokenizer', None)
                emb_dim = getattr(self.model, 'get_sentence_embedding_dimension', lambda: 768)()
                logger.info(f"Модель эмбеддингов загружена: {emb_dim}d")
            else:
                logger.warning("SentenceTransformer не доступен")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            self.model = None
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Векторизация текстов.
        
        Args:
            texts: Список текстов для векторизации
            normalize: Нормализовать векторы (для косинусного сходства)
            show_progress: Показывать прогресс
            
        Returns:
            np.ndarray формы (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Проверяем кэш
        uncached_texts = []
        cached_indices = []
        cached_embeddings = []
        
        with self._cache_lock:
            for i, text in enumerate(texts):
                text_hash = str(hash(text))
                if text_hash in self._embedding_cache:
                    cached_indices.append(i)
                    cached_embeddings.append(self._embedding_cache[text_hash])
                else:
                    uncached_texts.append((i, text))
        
        if not uncached_texts:
            # Все из кэша
            result = np.array(cached_embeddings)
            return self._normalize(result) if normalize else result
        
        # Векторизуемuncached
        texts_to_encode = [text for _, text in uncached_texts]
        
        if self.model is not None:
            try:
                embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=self.batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                
                # Сохраняем в кэш
                with self._cache_lock:
                    for (orig_idx, text), emb in zip(uncached_texts, embeddings):
                        text_hash = str(hash(text))
                        self._embedding_cache[text_hash] = emb.tolist()
                
                # Объединяем с кэшированными
                result = np.zeros((len(texts), embeddings.shape[1]))
                for i, emb in zip([i for i, _ in uncached_texts], embeddings):
                    result[i] = emb
                for i, emb in zip(cached_indices, cached_embeddings):
                    result[i] = emb
                
                return result
                
            except Exception as e:
                logger.warning(f"Ошибка векторизации через model: {e}")
        
        # Fallback - возвращаем None вместо случайных векторов
        # Случайные векторы ломают семантический поиск
        logger.warning("Embedding model недоступен, возвращаю None")
        return None
    
    def encode_single(
        self,
        text: str,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """Векторизация одного текста."""
        result = self.encode([text], normalize=normalize)
        return result[0] if len(result) > 0 else None
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Нормализация векторов (L2)."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    
    def _random_embeddings(self, n: int, normalize: bool = True) -> np.ndarray:
        """Генерация случайных эмбеддингов (fallback)."""
        dim = 768  # multilingual-e5-base dimension
        
        if self.model is not None:
            dim = self.model.get_sentence_embedding_dimension()
        
        embeddings = np.random.randn(n, dim).astype(np.float32)
        
        if normalize:
            embeddings = self._normalize(embeddings)
        
        return embeddings
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Вычисление косинусного сходства между двумя векторами."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Нормализация
        e1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        e2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        return float(np.dot(e1, e2))
    
    def compute_similarities_batch(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """Вычисление косинусного сходства запроса с кандидатами."""
        if candidate_embeddings.shape[0] == 0:
            return np.array([])
        
        # Нормализация
        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidates = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        return np.dot(candidates, query)
    
    def find_similar(
        self,
        query_text: str,
        texts: List[str],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Найти наиболее похожие тексты на запрос.
        
        Returns:
            List of (index, similarity_score)
        """
        # Векторизуем запрос и тексты
        all_texts = [query_text] + texts
        embeddings = self.encode(all_texts, normalize=True)
        
        query_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        # Вычисляем сходства
        similarities = self.compute_similarities_batch(query_emb, candidate_embs)
        
        # Сортируем по убыванию
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append((idx, score))
        
        return results
    
    def clear_cache(self):
        """Очистка кэша эмбеддингов."""
        with self._cache_lock:
            self._embedding_cache.clear()
        logger.info("Кэш эмбеддингов очищен")
    
    def get_cache_size(self) -> int:
        """Получить размер кэша."""
        with self._cache_lock:
            return len(self._embedding_cache)


def create_embeddings_manager(
    model_name: str = None,
    device: str = "cuda"
) -> EmbeddingsManager:
    """Фабричная функция - возвращает singleton экземпляр EmbeddingsManager."""
    global _embeddings_manager_instance
    
    with _embeddings_manager_lock:
        if _embeddings_manager_instance is None:
            logger.info(f"Создание синглтона EmbeddingsManager (device={device})")
            _embeddings_manager_instance = EmbeddingsManager(model_name=model_name, device=device)
        else:
            logger.info("Повторное использование синглтона EmbeddingsManager")
        return _embeddings_manager_instance


def get_embeddings_manager() -> Optional['EmbeddingsManager']:
    """Получить текущий синглтон экземпляр (None если ещё не создан)."""
    return _embeddings_manager_instance