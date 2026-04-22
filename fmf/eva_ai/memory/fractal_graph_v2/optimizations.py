"""
Optimizations for Fractal Graph V2

Based on recommendations:
1. HNSW vector index for ANN search
2. Hybrid contradiction detection with NLI
3. Incremental clustering
4. Path caching
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger("eva_ai.fractal_graph_v2.optimizations")


class HNSWIndex:
    """
    HNSW-index for ANN search.
    Hierarchical Navigable Small World - O(log n) search instead of O(n)
    """
    
    def __init__(
        self,
        ef_construction: int = 200,
        ef_search: int = 50,
        m: int = 16,
        dim: int = 768
    ):
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m = m
        self.dim = dim
        
        self._index = None
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._vectors = []
        
        self._init_index()
    
    def _init_index(self):
        """Initialize HNSW index using nmslib or faiss."""
        try:
            import nmslib
            self._index = nmslib.init(
                method='hnsw',
                space='cosinesimil',
                ef_construction=self.ef_construction,
                ef=self.ef_search,
                m=self.m
            )
            logger.info("HNSW index initialized (nmslib)")
        except ImportError:
            try:
                import faiss
                index = faiss.IndexHNSWFlat(self.dim, self.m)
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = self.ef_search
                self._faiss_index = index
                logger.info("HNSW index initialized (faiss)")
            except ImportError:
                logger.warning("Neither nmslib nor faiss available, using fallback")
                self._index = None
    
    def add_items(self, ids: List[str], vectors: List[List[float]]):
        """Add vectors to index."""
        if self._faiss_index is not None:
            vecs = np.array(vectors, dtype=np.float32)
            self._faiss_index.add(vecs)
            for i, id_ in enumerate(ids):
                self._id_to_idx[id_] = i
                self._idx_to_id[i] = id_
            logger.info(f"Added {len(ids)} vectors to HNSW index")
        elif self._index is not None:
            for i, (id_, vec) in enumerate(zip(ids, vectors)):
                self._index.addPoint(vec, i)
                self._id_to_idx[id_] = i
                self._idx_to_id[i] = id_
        else:
            self._vectors = vectors
            self._id_to_idx = {id_: i for i, id_ in enumerate(ids)}
            self._idx_to_id = {i: id_ for i, id_ in enumerate(ids)}
    
    def search(self, query_vec: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        query = np.array([query_vec], dtype=np.float32)
        
        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(query, k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0:
                    id_ = self._idx_to_id.get(idx)
                    similarity = 1.0 / (1.0 + dist)
                    results.append((id_, similarity))
            return results
        
        elif self._index is not None:
            results = self._index.knnQuery(query, k=k)
            return [(self._idx_to_id.get(i), 1.0/(1.0+d)) for i, d in zip(results[0][0], results[1][0])]
        
        else:
            # Fallback - cosine similarity
            q = np.array(query_vec)
            q = q / (np.linalg.norm(q) + 1e-8)
            
            best = []
            for i, vec in enumerate(self._vectors):
                v = np.array(vec)
                v = v / (np.linalg.norm(v) + 1e-8)
                sim = float(np.dot(q, v))
                best.append((self._idx_to_id[i], sim))
            
            best.sort(key=lambda x: x[1], reverse=True)
            return best[:k]


class NLIContradictionDetector:
    """
    Lightweight NLI model for contradiction detection.
    Uses small model (deberta-v3-xsmall-mnli) for classification.
    """
    
    def __init__(self, model_name: str = "MoritzLaworski/deberta-v3-xsmall-mnli"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load lightweight NLI model."""
        try:
            from transformers import pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
            )
            logger.info(f"NLI model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load NLI model: {e}")
            self.classifier = None
    
    def check_contradiction(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Check if two texts contradict each other.
        
        Returns:
            {has_contradiction: bool, contradiction_score: float, label: str}
        """
        if self.classifier is None:
            return {"has_contradiction": False, "error": "model not loaded"}
        
        try:
            result = self.classifier(f"{text1} [SEP] {text2}")
            
            label = result[0]['label']
            score = result[0]['score']
            
            has_contradiction = label in ['contradiction']
            
            return {
                "has_contradiction": has_contradiction,
                "contradiction_score": score if has_contradiction else 1.0 - score,
                "label": label,
                "confidence": score
            }
        except Exception as e:
            logger.warning(f"NLI check error: {e}")
            return {"has_contradiction": False, "error": str(e)}


class IncrementalClustering:
    """
    Incremental clustering for dynamic graph updates.
    Triggers local reclustering when Δ > 15% changes.
    """
    
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self._last_clusters: Dict[str, List[str]] = {}
    
    def should_recluster(
        self,
        node_id: str,
        new_neighbors: List[str],
        level: int
    ) -> bool:
        """Check if local reclustering is needed."""
        key = f"level_{level}"
        
        if key not in self._last_clusters:
            self._last_clusters[key] = []
            return True
        
        old_set = set(self._last_clusters[key])
        new_set = set(new_neighbors)
        
        # Calculate change ratio
        union = old_set | new_set
        if not union:
            return False
        
        changed = (old_set - new_set) | (new_set - old_set)
        change_ratio = len(changed) / len(union)
        
        return change_ratio > self.threshold
    
    def update_cluster(self, level: int, node_ids: List[str]):
        """Update cluster state after reclustering."""
        key = f"level_{level}"
        self._last_clusters[key] = node_ids.copy()


class PathCache:
    """
    Cache for frequently requested hierarchical paths.
    TTL based on confidence and access patterns.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_count: Dict[str, int] = defaultdict(int)
    
    def get(self, path_key: str) -> Optional[Any]:
        """Get cached path."""
        if path_key in self._cache:
            entry = self._cache[path_key]
            
            # Check TTL based on confidence
            confidence = entry.get('confidence', 0.5)
            ttl = 3600 * confidence  # Higher confidence = longer TTL
            
            if time.time() - entry['timestamp'] < ttl:
                self._access_count[path_key] += 1
                return entry['data']
            
            # Expired
            del self._cache[path_key]
        
        return None
    
    def put(self, path_key: str, data: Any, confidence: float = 0.5):
        """Cache path data."""
        # Evict if full
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        self._cache[path_key] = {
            'data': data,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self._access_count[path_key] = 1
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Sort by access count
        sorted_items = sorted(
            self._access_count.items(),
            key=lambda x: x[1]
        )
        
        # Remove bottom 10%
        to_remove = max(1, len(sorted_items) // 10)
        for key, _ in sorted_items[:to_remove]:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]


def create_hnsw_index(dim: int = 768) -> HNSWIndex:
    """Factory for HNSW index."""
    return HNSWIndex(dim=dim)


def create_contradiction_detector() -> NLIContradictionDetector:
    """Factory for NLI detector."""
    return NLIContradictionDetector()


def create_path_cache(max_size: int = 1000) -> PathCache:
    """Factory for path cache."""
    return PathCache(max_size=max_size)


# === Integration with FractalGraphV2 ===

class FractalGraphOptimized:
    """
    Extended FractalGraph with all optimizations.
    """
    
    def __init__(self, base_graph):
        self.base = base_graph
        
        # HNSW index for fast ANN search
        self.hnsw_index = create_hnsw_index(dim=768)
        
        # NLI for contradiction detection
        self.contradiction_detector = create_contradiction_detector()
        
        # Incremental clustering
        self.clustering = IncrementalClustering(threshold=0.15)
        
        # Path cache
        self.path_cache = create_path_cache(max_size=1000)
        
        # Build HNSW index from existing nodes
        self._rebuild_hnsw_index()
    
    def _rebuild_hnsw_index(self):
        """Rebuild HNSW index from graph nodes."""
        nodes_with_embeddings = [
            (nid, node.embedding)
            for nid, node in self.base.storage.nodes.items()
            if node.embedding is not None
        ]
        
        if nodes_with_embeddings:
            ids = [n[0] for n in nodes_with_embeddings]
            vecs = [n[1] for n in nodes_with_embeddings]
            self.hnsw_index.add_items(ids, vecs)
            logger.info(f"HNSW index built with {len(ids)} vectors")
    
    def hnsw_search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Fast HNSW search."""
        return self.hnsw_index.search(query_embedding, top_k)
    
    def check_contradiction_nli(self, text1: str, text2: str) -> Dict[str, Any]:
        """Check contradiction using NLI model."""
        return self.contradiction_detector.check_contradiction(text1, text2)
    
    def get_cached_path(self, query: str) -> Optional[Any]:
        """Get cached path for query."""
        return self.path_cache.get(query)
    
    def cache_path(self, query: str, path_data: Any, confidence: float = 0.5):
        """Cache path for query."""
        self.path_cache.put(query, path_data, confidence)


import time