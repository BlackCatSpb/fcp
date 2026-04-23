"""
FractalGraphSearch - Семантический поиск по графу с использованием HNSW

Реализует:
- HNSW-индекс для быстрого поиска ближайших соседей
- Прямая загрузка узлов из SQLite БД
- SentenceTransformer для эмбеддингов запросов
"""

import os
import sqlite3
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("fcp.graph_search")

try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False
    logger.warning("hnswlib не установлен, используется fallback поиск")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers не установлен")


@dataclass
class SearchResult:
    """Результат поиска по графу."""
    node_id: str
    content: str
    score: float
    node_type: str


class SQLiteGraphLoader:
    """Загрузчик узлов графа напрямую из SQLite БД."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_nodes(self, node_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Загрузка всех узлов из БД."""
        if not os.path.exists(self.db_path):
            logger.warning(f"БД не найдена: {self.db_path}")
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        nodes = {}
        if 'nodes' in tables:
            cursor.execute("PRAGMA table_info(nodes)")
            columns = {row[1] for row in cursor.fetchall()}
            
            id_col = 'id' if 'id' in columns else 'node_id'
            content_col = 'content' if 'content' in columns else 'text'
            type_col = 'node_type' if 'node_type' in columns else 'type'
            
            query = f"SELECT {id_col}, {content_col}, {type_col} FROM nodes"
            params = []
            if node_types:
                placeholders = ','.join(['?' for _ in node_types])
                query += f" WHERE {type_col} IN ({placeholders})"
                params = node_types
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                node_id, content, node_type = row[0], row[1], row[2]
                nodes[node_id] = {
                    'id': node_id,
                    'content': content,
                    'node_type': node_type
                }
        
        conn.close()
        logger.info(f"Загружено {len(nodes)} узлов")
        return nodes


class FractalGraphSearch:
    """ Семантический поиск по графу с использованием HNSW."""
    
    def __init__(
        self,
        graph=None,
        db_path: str = None,
        embedding_dim: int = 384,
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 10000,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.graph = graph
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.max_elements = max_elements
        
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer(encoder_name)
                self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
                logger.info(f"SentenceTransformer загружен: {encoder_name}, dim={self.embedding_dim}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить SentenceTransformer: {e}")
        
        self.hnsw_index = None
        self.node_ids: List[str] = []
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.is_built = False
        
        if HNSWLIB_AVAILABLE:
            self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
            self.hnsw_index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            logger.info(f"HNSW-индекс инициализирован: dim={embedding_dim}, M={M}")
    
    def _encode_text(self, text: str) -> np.ndarray:
        if self.encoder is not None:
            return self.encoder.encode(text, normalize_embeddings=True)
        else:
            emb = np.random.randn(self.embedding_dim).astype(np.float32)
            emb /= np.linalg.norm(emb)
            return emb
    
    def build_index(self, node_types: Optional[List[str]] = None) -> int:
        if node_types is None:
            node_types = ['concept', 'fact', 'entity']
        
        self.node_ids = []
        self.node_embeddings = {}
        
        if self.graph is not None:
            for node_id, node in self.graph.nodes.items():
                if node.node_type not in node_types:
                    continue
                emb = self._encode_text(node.content)
                self.node_embeddings[node_id] = emb
                self.node_ids.append(node_id)
        elif self.db_path:
            loader = SQLiteGraphLoader(self.db_path)
            nodes = loader.load_nodes(node_types=node_types)
            for node_id, node_data in nodes.items():
                content = node_data.get('content', node_data.get('text', ''))
                if content:
                    emb = self._encode_text(content)
                    self.node_embeddings[node_id] = emb
                    self.node_ids.append(node_id)
        else:
            logger.warning("Не указан graph или db_path")
            return 0
        
        if not self.node_ids:
            logger.warning("Нет узлов для индексации")
            return 0
        
        if HNSWLIB_AVAILABLE and self.hnsw_index is not None:
            if len(self.node_ids) > 0:
                for i, node_id in enumerate(self.node_ids):
                    self.hnsw_index.add_items(self.node_embeddings[node_id].astype(np.float32), i)
                self.hnsw_index.set_ef(100)
                logger.info(f"HNSW-индекс построен: {len(self.node_ids)} узлов")
            else:
                logger.info("HNSW: нет узлов для индексации")
        else:
            logger.info(f"Fallback: {len(self.node_ids)} узлов загружено в память")
        
        self.is_built = True
        return len(self.node_ids)
    
    def search(self, query: str, k: int = 5, min_score: float = 0.3, node_types: Optional[List[str]] = None) -> List[SearchResult]:
        if not self.is_built:
            logger.warning("Индекс не построен, выполняю build_index()")
            self.build_index(node_types=node_types)
        
        query_embedding = self._encode_text(query)
        
        results = []
        
        if HNSWLIB_AVAILABLE and self.hnsw_index is not None:
            labels, scores = self.hnsw_index.knn_query(query_embedding.astype(np.float32), k=k)
            for idx, score in zip(labels[0], scores[0]):
                if idx < len(self.node_ids) and score >= min_score:
                    node_id = self.node_ids[idx]
                    content = node_id if self.graph is None else getattr(self.graph.nodes.get(node_id), 'content', node_id)
                    node_type = "unknown" if self.graph is None else getattr(self.graph.nodes.get(node_id), 'node_type', "unknown")
                    results.append(SearchResult(node_id=node_id, content=content, score=float(score), node_type=node_type))
        else:
            results = self._fallback_search(query_embedding, k, min_score, node_types)
        
        return results[:k]
    
    def _fallback_search(self, query_embedding: np.ndarray, k: int, min_score: float, node_types: Optional[List[str]]) -> List[SearchResult]:
        results = []
        for node_id, emb in self.node_embeddings.items():
            score = np.dot(query_embedding, emb)
            if score >= min_score:
                content = node_id if self.graph is None else getattr(self.graph.nodes.get(node_id), 'content', node_id)
                node_type = "unknown" if self.graph is None else getattr(self.graph.nodes.get(node_id), 'node_type', "unknown")
                results.append(SearchResult(node_id=node_id, content=content, score=float(score), node_type=node_type))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def get_subgraph(self, query: str, k: int = 5, max_hops: int = 2) -> Tuple[List[str], List[Tuple[str, str]]]:
        search_results = self.search(query, k=k)
        seed_ids = [r.node_id for r in search_results]
        return seed_ids, []
    
    def add_node(self, node_id: str, content: str, node_type: str = "entity"):
        if node_id not in self.node_embeddings:
            emb = self._encode_text(content)
            self.node_embeddings[node_id] = emb
            if len(self.node_ids) < self.max_elements and HNSWLIB_AVAILABLE:
                idx = len(self.node_ids)
                self.node_ids.append(node_id)
                self.hnsw_index.add_items(emb.astype(np.float32), idx)
            elif node_id not in self.node_ids:
                self.node_ids.append(node_id)
    
    def update_index(self):
        self.build_index()


class GraphVectorExtractor:
    """Извлечение графового вектора для инъекции в LLM."""
    
    def __init__(self, graph_search: FractalGraphSearch, output_dim: int = 2560):
        self.graph_search = graph_search
        self.output_dim = output_dim
    
    def extract(self, query: str, k: int = 10, aggregation: str = "weighted") -> np.ndarray:
        results = self.graph_search.search(query, k=k)
        
        if not results:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        embeddings = []
        weights = []
        
        for r in results:
            if r.node_id in self.graph_search.node_embeddings:
                emb = self.graph_search.node_embeddings[r.node_id]
                embeddings.append(emb)
                weights.append(r.score)
        
        if not embeddings:
            return np.zeros(self.output_dim, dtype=np.float32)
        
        embeddings = np.stack(embeddings)
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        if aggregation == "weighted":
            aggregated = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        else:
            aggregated = embeddings.mean(axis=0)
        
        if len(aggregated) < self.output_dim:
            padding = np.zeros(self.output_dim - len(aggregated), dtype=np.float32)
            aggregated = np.concatenate([aggregated, padding])
        elif len(aggregated) > self.output_dim:
            aggregated = aggregated[:self.output_dim]
        
        return aggregated.astype(np.float32)


def create_graph_search(
    graph=None,
    db_path: str = None,
    embedding_dim: int = 384,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FractalGraphSearch:
    return FractalGraphSearch(
        graph=graph,
        db_path=db_path,
        embedding_dim=embedding_dim,
        encoder_name=encoder_name
    )