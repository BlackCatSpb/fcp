"""
FractalGraphV2 - Основное хранилище графа памяти

Реализует:
- SQLite с векторными индексами
- Фрактальные уровни и семантические группы
- ANN-поиск (косинусное расстояние)
- Кластеризация узлов
- Детекция противоречий
"""

import os
import sqlite3
import json
import time
import uuid
import hashlib
import logging
import threading
import math
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from .types import (
    FractalNode, FractalEdge, SemanticGroup,
    NodeType, RelationType, MemoryTier,
    create_node_id, create_edge_id, create_group_id
)

logger = logging.getLogger("eva_ai.fractal_graph_v2")


class FractalGraphV2:
    """
    Фрактальная иерархическая семантическая сеть V2.
    
    Особенности:
    - Фрактальные уровни (L0-L3)
    - Семантические группы с векторными центроидами
    - ANN-поиск по косинусному расстоянию
    - Автоматическая кластеризация
    - Детекция противоречий
    """
    
    def __init__(
        self,
        storage_dir: str = None,
        embedding_dim: int = 768,  # intfloat/multilingual-e5-base
        vector_index_type: str = "simple"  # simple, hnsw, ivf
    ):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "fractal_graph_v2_data"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.vector_index_type = vector_index_type
        
        # Пути к файлам
        self.db_path = os.path.join(self.storage_dir, "fractal_graph.db")
        
        # Инициализация БД
        self._init_database()
        
        # Загрузка данных в память
        self.nodes: Dict[str, FractalNode] = {}
        self.edges: Dict[str, FractalEdge] = {}
        self.semantic_groups: Dict[str, SemanticGroup] = {}
        
        self._load_data()
        
        # Индексы для быстрого поиска
        self._build_indexes()
        
        logger.info(f"FractalGraphV2 инициализирован: {len(self.nodes)} узлов, {len(self.semantic_groups)} групп")
    
    def _get_connection(self):
        """Получить соединение с БД."""
        return sqlite3.connect(self.db_path)
    
    def _init_database(self):
        """Инициализация SQLite БД с таблицами для фрактального графа."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # WAL mode для конкурентного доступа
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        
        # Таблица узлов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                level INTEGER DEFAULT 0,
                parent_group_id TEXT,
                embedding BLOB,
                confidence REAL DEFAULT 0.5,
                created_at REAL,
                updated_at REAL,
                last_accessed REAL,
                metadata TEXT,
                access_count INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                is_static INTEGER DEFAULT 0,
                is_contradiction INTEGER DEFAULT 0
            )
        """)
        
        # Таблица связей
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at REAL,
                updated_at REAL,
                contradiction_flag INTEGER DEFAULT 0,
                metadata TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        """)
        
        # Таблица семантических групп
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT DEFAULT 'semantic_group',
                level INTEGER DEFAULT 2,
                embedding BLOB,
                member_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.5,
                parent_group_id TEXT,
                created_at REAL,
                updated_at REAL,
                cluster_coherence REAL DEFAULT 0.0,
                needs_recluster INTEGER DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Индексы
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_parent_group ON nodes(parent_group_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_groups_level ON semantic_groups(level)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"БД инициализирована: {self.db_path}")
    
    def _load_data(self):
        """Загрузка данных из БД в память."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Загрузка узлов
        cursor = conn.execute("SELECT * FROM nodes")
        for row in cursor:
            node = FractalNode(
                id=row['id'],
                content=row['content'],
                node_type=row['node_type'],
                level=row['level'],
                parent_group_id=row['parent_group_id'],
                embedding=self._deserialize_embedding(row['embedding']),
                confidence=row['confidence'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                last_accessed=row['last_accessed'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                access_count=row['access_count'],
                version=row['version'],
                is_static=bool(row['is_static']),
                is_contradiction=bool(row['is_contradiction'])
            )
            self.nodes[node.id] = node
        
        # Загрузка связей
        cursor = conn.execute("SELECT * FROM edges")
        for row in cursor:
            edge = FractalEdge(
                id=row['id'],
                source_id=row['source_id'],
                target_id=row['target_id'],
                relation_type=row['relation_type'],
                weight=row['weight'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                contradiction_flag=bool(row['contradiction_flag']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.edges[edge.id] = edge
        
        # Загрузка групп
        cursor = conn.execute("SELECT * FROM semantic_groups")
        for row in cursor:
            group = SemanticGroup(
                id=row['id'],
                name=row['name'],
                node_type=row['node_type'],
                level=row['level'],
                embedding=self._deserialize_embedding(row['embedding']),
                member_count=row['member_count'],
                avg_confidence=row['avg_confidence'],
                parent_group_id=row['parent_group_id'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                cluster_coherence=row['cluster_coherence'],
                needs_recluster=bool(row['needs_recluster']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            self.semantic_groups[group.id] = group
        
        conn.close()
        
        logger.info(f"Загружено: {len(self.nodes)} узлов, {len(self.edges)} связей, {len(self.semantic_groups)} групп")
    
    def _build_indexes(self):
        """Построение индексов для быстрого поиска."""
        # Индексы для поиска
        self.nodes_by_type: Dict[str, List[str]] = defaultdict(list)
        self.nodes_by_level: Dict[int, List[str]] = defaultdict(list)
        self.nodes_by_group: Dict[str, List[str]] = defaultdict(list)
        self.groups_by_level: Dict[int, List[str]] = defaultdict(list)
        
        # Word index для простого поиска
        self.word_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Кэш нормализованных векторов для семантического поиска
        self._normalized_embeddings: Dict[str, np.ndarray] = {}
        self._group_embeddings: Dict[str, np.ndarray] = {}
        
        for node_id, node in self.nodes.items():
            # Индексы
            self.nodes_by_type[node.node_type].append(node_id)
            self.nodes_by_level[node.level].append(node_id)
            if node.parent_group_id:
                self.nodes_by_group[node.parent_group_id].append(node_id)
            
            # Word index
            words = node.content.lower().split()
            for word in words:
                if len(word) > 2:
                    self.word_index[word].add(node_id)
            
            # Нормализованные вектора
            if node.embedding:
                v = np.array(node.embedding)
                v = v / (np.linalg.norm(v) + 1e-8)
                self._normalized_embeddings[node_id] = v
        
        # Группы
        for group_id, group in self.semantic_groups.items():
            if group.embedding:
                gv = np.array(group.embedding)
                gv = gv / (np.linalg.norm(gv) + 1e-8)
                self._group_embeddings[group_id] = gv
        
        logger.info(f"Индексы построены: vectors={len(self._normalized_embeddings)}, groups={len(self._group_embeddings)}")
    
    # === ОСНОВНЫЕ ОПЕРАЦИИ ===
    
    def add_node(
        self,
        content: str,
        node_type: str = "concept",
        level: int = 1,
        embedding: Optional[List[float]] = None,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
        parent_group_id: Optional[str] = None,
        auto_cluster: bool = True,
        cluster_threshold: float = 0.6,
        is_static: bool = False,
        is_contradiction: bool = False
    ) -> FractalNode:
        """Добавить узел в граф с опциональной инкрементальной кластеризацией."""
        node_id = create_node_id(content, node_type)
        
        # Проверяем, не существует ли уже такой узел
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        now = time.time()
        node = FractalNode(
            id=node_id,
            content=content,
            node_type=node_type,
            level=level,
            parent_group_id=parent_group_id,
            embedding=embedding,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            last_accessed=now,
            metadata=metadata or {},
            access_count=0,
            version=1,
            is_static=is_static,
            is_contradiction=is_contradiction
        )
        
        # Инкрементальная кластеризация
        if auto_cluster and embedding and level > 0:
            best_group = self._find_nearest_group(embedding, level, cluster_threshold)
            if best_group:
                node.parent_group_id = best_group
                if best_group in self.semantic_groups:
                    self.semantic_groups[best_group].member_count += 1
                    self.semantic_groups[best_group].updated_at = now
        
        self.nodes[node_id] = node
        
        # Сохранение в БД
        self._save_node(node)
        
        # Обновление индексов
        self.nodes_by_type[node_type].append(node_id)
        self.nodes_by_level[level].append(node_id)
        if node.parent_group_id:
            self.nodes_by_group[node.parent_group_id].append(node_id)
        
        # Обновление поискового индекса
        words = content.lower().split()
        for word in words:
            if len(word) > 2:
                self.word_index[word].add(node_id)
        
        # Обновление кэша нормализованных векторов
        if node.embedding:
            v = np.array(node.embedding)
            v = v / (np.linalg.norm(v) + 1e-8)
            self._normalized_embeddings[node_id] = v
        
        logger.debug(f"Добавлен узел: {node_id} ({node_type}), group: {node.parent_group_id}")
        
        return node
    
    def _find_nearest_group(
        self,
        embedding: List[float],
        level: int,
        threshold: float = 0.6
    ) -> Optional[str]:
        """Найти ближайшую группу или связанный узел для нового узла."""
        if not embedding:
            return None
        
        node_vec = np.array(embedding)
        node_vec = node_vec / (np.linalg.norm(node_vec) + 1e-8)
        
        # Сначала ищем среди существующих групп
        best_group = None
        best_sim = threshold
        
        for group_id, group in self.semantic_groups.items():
            if group.level != level + 1:
                continue
            if not group.embedding:
                continue
            
            group_vec = np.array(group.embedding)
            group_vec = group_vec / (np.linalg.norm(group_vec) + 1e-8)
            
            sim = float(np.dot(node_vec, group_vec))
            
            if sim > best_sim:
                best_sim = sim
                best_group = group_id
        
        if best_group:
            return best_group
        
        # Если нет группы - ищем ближайший связанный узел того же уровня
        best_node = None
        best_node_sim = threshold
        
        for node_id, node in self.nodes.items():
            if node.level != level or not node.embedding:
                continue
            
            node_vec2 = np.array(node.embedding)
            node_vec2 = node_vec2 / (np.linalg.norm(node_vec2) + 1e-8)
            
            sim = float(np.dot(node_vec, node_vec2))
            
            # Проверяем есть ли связь между узлами
            if sim > best_node_sim:
                # Ищем общую группу
                if node.parent_group_id:
                    best_node = node.parent_group_id
                    break
                # Или ближайший узел
                best_node_sim = sim
                best_node = node_id
        
        return best_node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.5,
        metadata: Optional[Dict] = None
    ) -> Optional[FractalEdge]:
        """Добавить связь между узлами."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Узлы не найдены: {source_id} -> {target_id}")
            return None
        
        edge_id = create_edge_id(source_id, target_id, relation_type)
        
        if edge_id in self.edges:
            return self.edges[edge_id]
        
        now = time.time()
        edge = FractalEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        self.edges[edge_id] = edge
        self._save_edge(edge)
        
        logger.debug(f"Добавлена связь: {source_id} -> {target_id} ({relation_type})")
        
        return edge
    
    def remove_edges_for_node(self, node_id: str) -> int:
        """Удалить все связи связанные с узлом. Возвращает количество удалённых."""
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge.source_id == node_id or edge.target_id == node_id
        ]
        
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
            self._delete_edge_from_db(edge_id)
        
        logger.debug(f"Удалено связей для узла {node_id}: {len(edges_to_remove)}")
        return len(edges_to_remove)
    
    def _delete_edge_from_db(self, edge_id: str):
        """Удаление связи из БД."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        conn.commit()
        conn.close()
    
    def remove_node_from_indexes(self, node_id: str, node: 'FractalNode'):
        """Удалить узел из всех поисковых индексов."""
        # Удаление из nodes_by_type
        if node.node_type in self.nodes_by_type:
            if node_id in self.nodes_by_type[node.node_type]:
                self.nodes_by_type[node.node_type].remove(node_id)
        
        # Удаление из nodes_by_level
        if node.level in self.nodes_by_level:
            if node_id in self.nodes_by_level[node.level]:
                self.nodes_by_level[node.level].remove(node_id)
        
        # Удаление из nodes_by_group
        if node.parent_group_id in self.nodes_by_group:
            if node_id in self.nodes_by_group[node.parent_group_id]:
                self.nodes_by_group[node.parent_group_id].remove(node_id)
            # Уменьшаем счётчик группы
            if node.parent_group_id in self.semantic_groups:
                self.semantic_groups[node.parent_group_id].member_count = max(
                    0, self.semantic_groups[node.parent_group_id].member_count - 1
                )
        
        # Удаление из word_index
        for word in node.content.lower().split():
            if len(word) > 2 and word in self.word_index:
                self.word_index[word].discard(node_id)
                if not self.word_index[word]:
                    del self.word_index[word]
        
        # Удаление из normalized embeddings cache
        if node_id in self._normalized_embeddings:
            del self._normalized_embeddings[node_id]
        
        logger.debug(f"Узел {node_id} удалён из индексов")
    
    def create_semantic_group(
        self,
        name: str,
        member_ids: List[str],
        embedding: Optional[List[float]] = None,
        level: int = 2
    ) -> SemanticGroup:
        """Создать семантическую группу (образ) из списка узлов."""
        group_id = create_group_id(name)
        
        # Вычисляем центроид если не передан
        if embedding is None and member_ids:
            embeddings = []
            for mid in member_ids:
                if mid in self.nodes and self.nodes[mid].embedding:
                    embeddings.append(np.array(self.nodes[mid].embedding))
            if embeddings:
                embedding = np.mean(embeddings, axis=0).tolist()
        
        now = time.time()
        avg_conf = 0.0
        if member_ids:
            confidences = [self.nodes[mid].confidence for mid in member_ids if mid in self.nodes]
            avg_conf = np.mean(confidences) if confidences else 0.5
        
        group = SemanticGroup(
            id=group_id,
            name=name,
            node_type="semantic_group",
            level=level,
            embedding=embedding,
            member_count=len(member_ids),
            avg_confidence=avg_conf,
            created_at=now,
            updated_at=now,
            cluster_coherence=0.8,  # Начальное значение
            needs_recluster=False,
            metadata={}
        )
        
        # Обновляем parent_group_id у членов группы
        for member_id in member_ids:
            if member_id in self.nodes:
                self.nodes[member_id].parent_group_id = group_id
                self._save_node(self.nodes[member_id])
        
        self.semantic_groups[group_id] = group
        self._save_group(group)
        
        # Обновляем индексы
        self.groups_by_level[level].append(group_id)
        self.nodes_by_group[group_id] = member_ids.copy()
        
        logger.info(f"Создана семантическая группа: {name} ({len(member_ids)} членов)")
        
        return group
    
    # === ПОИСК ===
    
    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_level: int = 1,
        use_groups: bool = True
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Семантический поиск по косинусному расстоянию.
        
        Оптимизации:
        - Кэширование векторизованных узлов
        - Предварительная нормализация векторов
        
        Returns:
            List of (node_id, similarity, group_id)
        """
        if not query_embedding:
            return []
        
        # Ensure query_embedding is a proper list of floats
        if isinstance(query_embedding, str):
            return []
        
        try:
            query_vec = np.array(query_embedding, dtype=np.float64)
        except (ValueError, TypeError):
            return []
        
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        results = []
        
        # Кэшируем нормализованные вектора если нужно
        if not hasattr(self, '_normalized_embeddings'):
            self._normalized_embeddings = {}
            for node_id, node in self.nodes.items():
                if node.embedding:
                    v = np.array(node.embedding)
                    v = v / (np.linalg.norm(v) + 1e-8)
                    self._normalized_embeddings[node_id] = v
        
        # Ищем в группах
        if use_groups and self.semantic_groups:
            for group_id, group in self.semantic_groups.items():
                if group.level >= min_level and group.embedding:
                    # Проверяем кэш
                    if not hasattr(self, '_group_embeddings'):
                        self._group_embeddings = {}
                    
                    if group_id not in self._group_embeddings:
                        gv = np.array(group.embedding)
                        gv = gv / (np.linalg.norm(gv) + 1e-8)
                        self._group_embeddings[group_id] = gv
                    
                    group_vec = self._group_embeddings[group_id]
                    similarity = float(np.dot(query_vec, group_vec))
                    
                    if similarity > 0.5:
                        results.append((group_id, similarity, group.parent_group_id))
        
        # Ищем в узлах
        for node_id, node in self.nodes.items():
            if node.level >= min_level and node.embedding:
                if node_id in self._normalized_embeddings:
                    node_vec = self._normalized_embeddings[node_id]
                    similarity = float(np.dot(query_vec, node_vec))
                    
                    if similarity > 0.5:
                        results.append((node_id, similarity, node.parent_group_id))
        
        # Сортируем
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[str]:
        """Поиск по ключевым словам."""
        query_words = query.lower().split()
        
        scores: Dict[str, float] = defaultdict(float)
        
        for word in query_words:
            if word in self.word_index:
                for node_id in self.word_index[word]:
                    scores[node_id] += 1
        
        # Сортируем по убыванию
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [node_id for node_id, _ in sorted_ids[:top_k]]
    
    def get_group_members(self, group_id: str) -> List[FractalNode]:
        """Получить все узлы группы."""
        if group_id not in self.semantic_groups:
            return []
        
        member_ids = self.nodes_by_group.get(group_id, [])
        return [self.nodes[mid] for mid in member_ids if mid in self.nodes]
    
    def get_node_context(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Получить контекст узла (связанные узлы, группа, атрибуты)."""
        if node_id not in self.nodes:
            return {}
        
        node = self.nodes[node_id]
        
        context = {
            "node": node.to_dict(),
            "group": None,
            "attributes": [],
            "related": []
        }
        
        # Группа
        if node.parent_group_id and node.parent_group_id in self.semantic_groups:
            context["group"] = self.semantic_groups[node.parent_group_id].to_dict()
        
        # Связанные узлы
        for edge_id, edge in self.edges.items():
            if edge.source_id == node_id:
                if edge.target_id in self.nodes:
                    context["related"].append({
                        "node": self.nodes[edge.target_id].to_dict(),
                        "relation": edge.relation_type,
                        "weight": edge.weight
                    })
            elif edge.target_id == node_id:
                if edge.source_id in self.nodes:
                    context["related"].append({
                        "node": self.nodes[edge.source_id].to_dict(),
                        "relation": edge.relation_type,
                        "weight": edge.weight
                    })
        
        return context
    
    # === КЛАСТЕРИЗАЦИЯ ===
    
    def cluster_nodes(
        self,
        level: int = 1,
        threshold: float = 0.5,
        method: str = "agglomerative"  # agglomerative, dbscan, simple
    ) -> Dict[str, List[str]]:
        """
        Кластеризация узлов одного уровня по косинусному расстоянию.
        
        Реализации:
        - agglomerative: иерархическая кластеризация с полным связыванием
        - dbscan: плотностная кластеризация
        - simple: простая (присоединение к ближайшей группе)
        
        Returns:
            Dict[group_name, List[node_ids]]
        """
        # Получаем узлы нужного уровня с эмбеддингами
        level_nodes = [
            (nid, node) for nid, node in self.nodes.items()
            if node.level == level and node.embedding is not None
        ]
        
        if not level_nodes:
            return {}
        
        if method == "agglomerative":
            return self._agglomerative_clustering(level_nodes, threshold)
        elif method == "dbscan":
            return self._dbscan_clustering(level_nodes, threshold)
        else:
            return self._simple_clustering(level_nodes, threshold)
    
    def _agglomerative_clustering(
        self,
        level_nodes: List[Tuple[str, FractalNode]],
        threshold: float
    ) -> Dict[str, List[str]]:
        """Иерархическая агломеративная кластеризация."""
        if len(level_nodes) <= 1:
            if level_nodes:
                nid, node = level_nodes[0]
                return {node.content if node.content else nid: [nid]}
            return {}
        
        # Создаём mapping: node_id -> index
        node_id_to_idx = {nid: i for i, (nid, _) in enumerate(level_nodes)}
        
        # Вычисляем векторы
        vectors = []
        for _, node in level_nodes:
            v = np.array(node.embedding)
            v = v / (np.linalg.norm(v) + 1e-8)
            vectors.append(v)
        
        # Инициализация: каждый узел - отдельный кластер (ключ = node_id)
        clusters = {nid: [nid] for nid, _ in level_nodes}
        cluster_centroids = {nid: vectors[i].copy() for i, (nid, _) in enumerate(level_nodes)}
        
        # Иерархическое объединение
        while len(clusters) > 1:
            best_merge = None
            best_sim = -1
            
            # Ищем пару кластеров с максимальной близостью
            cluster_keys = list(clusters.keys())
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    cid1, cid2 = cluster_keys[i], cluster_keys[j]
                    
                    # similarity между центроидами
                    sim = float(np.dot(cluster_centroids[cid1], cluster_centroids[cid2]))
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_merge = (cid1, cid2)
            
            if best_merge is None or best_sim < threshold:
                break
            
            cid1, cid2 = best_merge
            
            # Новый центроид = усреднение
            new_centroid = (cluster_centroids[cid1] + cluster_centroids[cid2]) / 2
            new_centroid = new_centroid / (np.linalg.norm(new_centroid) + 1e-8)
            
            # Объединяем кластеры (новый ключ = cid1)
            clusters[cid1].extend(clusters[cid2])
            cluster_centroids[cid1] = new_centroid
            
            del clusters[cid2]
            del cluster_centroids[cid2]
        
        # Формируем результат
        result = {}
        for cluster_key, node_ids in clusters.items():
            if node_ids:
                first_node = self.nodes.get(node_ids[0])
                name = first_node.content if first_node else f"cluster_{cluster_key[:8]}"
                result[name] = node_ids
        
        return result
    
    def _dbscan_clustering(
        self,
        level_nodes: List[Tuple[str, FractalNode]],
        min_samples: int = 2
    ) -> Dict[str, List[str]]:
        """Упрощённый DBSCAN-подобный алгоритм."""
        if len(level_nodes) < 2:
            if level_nodes:
                return {f"cluster_{level_nodes[0][0]}": [level_nodes[0][0]]}
            return {}
        
        # Вычисляем попарные расстояния
        n = len(level_nodes)
        vectors = []
        for _, node in level_nodes:
            v = np.array(node.embedding)
            v = v / (np.linalg.norm(v) + 1e-8)
            vectors.append(v)
        
        # Матрица попарных similarity
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(vectors[i], vectors[j]))
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        # Порог для связи (0.5 = cos_distance 0.5)
        eps = 0.5
        visited = set()
        clusters = []
        
        def get_neighbors(idx: int) -> List[int]:
            return [j for j in range(n) if sim_matrix[idx, j] >= eps and j != idx]
        
        for i in range(n):
            if i in visited:
                continue
            
            cluster = [i]
            queue = [i]
            visited.add(i)
            
            while queue:
                curr = queue.pop(0)
                neighbors = get_neighbors(curr)
                
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        cluster.append(nb)
                        queue.append(nb)
            
            if len(cluster) >= min_samples:
                clusters.append(cluster)
        
        # Формируем результат
        result = {}
        for cid, members in enumerate(clusters):
            node_ids = [level_nodes[i][0] for i in members]
            if node_ids:
                first_node = self.nodes.get(node_ids[0])
                name = first_node.content if first_node else f"cluster_{cid}"
                result[name] = node_ids
        
        return result
    
    def _simple_clustering(
        self,
        level_nodes: List[Tuple[str, FractalNode]],
        threshold: float,
        level: int = 1
    ) -> Dict[str, List[str]]:
        """Простая кластеризация - присоединение к ближайшей группе."""
        clusters = {}
        
        for node_id, node in level_nodes:
            if node.embedding is None:
                continue
            
            node_vec = np.array(node.embedding)
            node_vec = node_vec / (np.linalg.norm(node_vec) + 1e-8)
            
            # Ищем ближайшую существующую группу
            best_group = None
            best_sim = 0
            
            for group_id, group in self.semantic_groups.items():
                if group.level != level:
                    continue
                if group.embedding is None:
                    continue
                
                group_vec = np.array(group.embedding)
                group_vec = group_vec / (np.linalg.norm(group_vec) + 1e-8)
                
                sim = float(np.dot(node_vec, group_vec))
                
                if sim > best_sim:
                    best_sim = sim
                    best_group = group_id
            
            if best_sim > threshold:
                if best_group not in clusters:
                    clusters[best_group] = []
                clusters[best_group].append(node_id)
            else:
                new_group_name = f"cluster_{node_id[:12]}"
                clusters[new_group_name] = [node_id]
        
        return clusters
    
    # === ПРОТИВОРЕЧИЯ ===
    
    def detect_contradiction(
        self,
        new_embedding: List[float],
        group_id: Optional[str] = None,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        Детекция противоречия: проверяем косинусное расстояние до центроида группы.
        
        Returns:
            (is_contradiction, distance)
        """
        if not new_embedding or not group_id:
            return False, 0.0
        
        if group_id not in self.semantic_groups:
            return False, 0.0
        
        group = self.semantic_groups[group_id]
        if not group.embedding:
            return False, 0.0
        
        new_vec = np.array(new_embedding)
        new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-8)
        
        group_vec = np.array(group.embedding)
        group_vec = group_vec / (np.linalg.norm(group_vec) + 1e-8)
        
        # Косинусное сходство -> расстояние
        similarity = float(np.dot(new_vec, group_vec))
        distance = 1.0 - similarity
        
        is_contradiction = distance > (1.0 - threshold)
        
        return is_contradiction, distance
    
    def mark_contradiction(self, node_id: str, note: str = ""):
        """Пометить узел как противоречие."""
        if node_id in self.nodes:
            self.nodes[node_id].is_contradiction = True
            self.nodes[node_id].metadata['contradiction_note'] = note
            self._save_node(self.nodes[node_id])
            
            # Находим связи этого узла и помечаем их
            for edge_id, edge in self.edges.items():
                if edge.source_id == node_id or edge.target_id == node_id:
                    edge.contradiction_flag = True
                    self._save_edge(edge)
    
    # === ВСПОМОГАТЕЛЬНЫЕ ===
    
    def _serialize_embedding(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """Сериализация эмбеддинга в blob."""
        if embedding is None:
            return None
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, blob: Optional[bytes]) -> Optional[List[float]]:
        """Десериализация эмбеддинга из blob."""
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32).tolist()
    
    def _save_node(self, node: FractalNode):
        """Сохранение узла в БД."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO nodes (
                id, content, node_type, level, parent_group_id, embedding,
                confidence, created_at, updated_at, last_accessed, metadata,
                access_count, version, is_static, is_contradiction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.id, node.content, node.node_type, node.level, node.parent_group_id,
            self._serialize_embedding(node.embedding), node.confidence,
            node.created_at, node.updated_at, node.last_accessed,
            json.dumps(node.metadata), node.access_count, node.version,
            int(node.is_static), int(node.is_contradiction)
        ))
        conn.commit()
        conn.close()
    
    def _save_edge(self, edge: FractalEdge):
        """Сохранение связи в БД."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO edges (
                id, source_id, target_id, relation_type, weight,
                created_at, updated_at, contradiction_flag, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            edge.id, edge.source_id, edge.target_id, edge.relation_type, edge.weight,
            edge.created_at, edge.updated_at, int(edge.contradiction_flag),
            json.dumps(edge.metadata)
        ))
        conn.commit()
        conn.close()
    
    def _save_group(self, group: SemanticGroup):
        """Сохранение группы в БД."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO semantic_groups (
                id, name, node_type, level, embedding, member_count,
                avg_confidence, parent_group_id, created_at, updated_at,
                cluster_coherence, needs_recluster, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            group.id, group.name, group.node_type, group.level,
            self._serialize_embedding(group.embedding), group.member_count,
            group.avg_confidence, group.parent_group_id, group.created_at,
            group.updated_at, group.cluster_coherence, int(group.needs_recluster),
            json.dumps(group.metadata)
        ))
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику графа."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_groups": len(self.semantic_groups),
            "nodes_by_type": {k: len(v) for k, v in self.nodes_by_type.items()},
            "nodes_by_level": {k: len(v) for k, v in self.nodes_by_level.items()},
            "groups_by_level": {k: len(v) for k, v in self.groups_by_level.items()},
            "nodes_with_embeddings": sum(1 for n in self.nodes.values() if n.embedding),
            "groups_with_embeddings": sum(1 for g in self.semantic_groups.values() if g.embedding),
            "contradictions": sum(1 for n in self.nodes.values() if n.is_contradiction)
        }
    
    # === EVA CONTAINER: Graph Serialization ===
    
    def save_to_blob(self, compression: str = "zstd") -> bytes:
        """
        Сериализовать граф в сжатый BLOB для хранения в .eva файле.
        
        Args:
            compression: Тип сжатия ('zstd', 'gzip', 'none')
            
        Returns:
            Сжатый бинарный блоб с графом
        """
        import gzip
        try:
            import zstandard as zstd
            HAS_ZSTD = True
        except ImportError:
            HAS_ZSTD = False
        
        data = {
            "version": 2,
            "graph_version": "fractal_graph_v2",
            "created_at": time.time(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "group_count": len(self.semantic_groups),
            "nodes": {},
            "edges": {},
            "semantic_groups": {}
        }
        
        for node_id, node in self.nodes.items():
            node_dict = {
                "id": node.id,
                "content": node.content,
                "node_type": node.node_type,
                "level": node.level,
                "parent_group_id": node.parent_group_id,
                "embedding": node.embedding,
                "confidence": node.confidence,
                "temporal_weight": node.temporal_weight,
                "domain_lambda": node.domain_lambda,
                "created_at": node.created_at,
                "updated_at": node.updated_at,
                "last_accessed": node.last_accessed,
                "metadata": node.metadata,
                "access_count": node.access_count,
                "version": node.version,
                "is_static": node.is_static,
                "is_contradiction": node.is_contradiction
            }
            data["nodes"][node_id] = node_dict
        
        for edge_id, edge in self.edges.items():
            edge_dict = {
                "id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation_type": edge.relation_type,
                "weight": edge.weight,
                "created_at": edge.created_at,
                "updated_at": edge.updated_at,
                "contradiction_flag": edge.contradiction_flag,
                "metadata": edge.metadata
            }
            data["edges"][edge_id] = edge_dict
        
        for group_id, group in self.semantic_groups.items():
            group_dict = {
                "id": group.id,
                "name": group.name,
                "node_type": group.node_type,
                "level": group.level,
                "embedding": group.embedding,
                "member_count": group.member_count,
                "avg_confidence": group.avg_confidence,
                "parent_group_id": group.parent_group_id,
                "created_at": group.created_at,
                "updated_at": group.updated_at,
                "cluster_coherence": group.cluster_coherence,
                "needs_recluster": group.needs_recluster,
                "metadata": group.metadata
            }
            data["semantic_groups"][group_id] = group_dict
        
        json_data = json.dumps(data, ensure_ascii=False)
        json_bytes = json_data.encode('utf-8')
        
        if compression == "zstd" and HAS_ZSTD:
            compressed = zstd.compress(json_bytes, level=3)
        elif compression == "gzip":
            compressed = gzip.compress(json_bytes, compresslevel=6)
        else:
            compressed = json_bytes
        
        return compressed
    
    def load_from_blob(self, blob: bytes, compression: str = "zstd") -> bool:
        """
        Загрузить граф из сжатого BLOB.
        
        Args:
            blob: Сжатый бинарный блоб
            compression: Тип сжатия ('zstd', 'gzip', 'none')
            
        Returns:
            True если успешно, False иначе
        """
        import gzip
        try:
            import zstandard as zstd
            HAS_ZSTD = True
        except ImportError:
            HAS_ZSTD = False
        
        try:
            if compression == "zstd" and HAS_ZSTD:
                json_bytes = zstd.decompress(blob)
            elif compression == "gzip":
                json_bytes = gzip.decompress(blob)
            else:
                json_bytes = blob
            
            data = json.loads(json_bytes.decode('utf-8'))
            
            if data.get("graph_version") != "fractal_graph_v2":
                logger.warning(f"Unknown graph version: {data.get('graph_version')}")
                return False
            
            self.nodes.clear()
            self.edges.clear()
            self.semantic_groups.clear()
            
            for node_id, node_dict in data.get("nodes", {}).items():
                node = FractalNode(
                    id=node_dict["id"],
                    content=node_dict["content"],
                    node_type=node_dict["node_type"],
                    level=node_dict.get("level", 0),
                    parent_group_id=node_dict.get("parent_group_id"),
                    embedding=node_dict.get("embedding"),
                    confidence=node_dict.get("confidence", 0.5),
                    temporal_weight=node_dict.get("temporal_weight", 1.0),
                    domain_lambda=node_dict.get("domain_lambda", 0.01),
                    created_at=node_dict.get("created_at", time.time()),
                    updated_at=node_dict.get("updated_at", time.time()),
                    last_accessed=node_dict.get("last_accessed", time.time()),
                    metadata=node_dict.get("metadata", {}),
                    access_count=node_dict.get("access_count", 0),
                    version=node_dict.get("version", 1),
                    is_static=node_dict.get("is_static", False),
                    is_contradiction=node_dict.get("is_contradiction", False)
                )
                self.nodes[node_id] = node
            
            for edge_id, edge_dict in data.get("edges", {}).items():
                edge = FractalEdge(
                    id=edge_dict["id"],
                    source_id=edge_dict["source_id"],
                    target_id=edge_dict["target_id"],
                    relation_type=edge_dict["relation_type"],
                    weight=edge_dict.get("weight", 0.5),
                    created_at=edge_dict.get("created_at", time.time()),
                    updated_at=edge_dict.get("updated_at", time.time()),
                    contradiction_flag=edge_dict.get("contradiction_flag", False),
                    metadata=edge_dict.get("metadata", {})
                )
                self.edges[edge_id] = edge
            
            for group_id, group_dict in data.get("semantic_groups", {}).items():
                group = SemanticGroup(
                    id=group_dict["id"],
                    name=group_dict["name"],
                    node_type=group_dict.get("node_type", "semantic_group"),
                    level=group_dict.get("level", 2),
                    embedding=group_dict.get("embedding"),
                    member_count=group_dict.get("member_count", 0),
                    avg_confidence=group_dict.get("avg_confidence", 0.5),
                    parent_group_id=group_dict.get("parent_group_id"),
                    created_at=group_dict.get("created_at", time.time()),
                    updated_at=group_dict.get("updated_at", time.time()),
                    cluster_coherence=group_dict.get("cluster_coherence", 0.0),
                    needs_recluster=group_dict.get("needs_recluster", False),
                    metadata=group_dict.get("metadata", {})
                )
                self.semantic_groups[group_id] = group
            
            self._build_indexes()
            
            logger.info(f"Loaded graph from blob: {len(self.nodes)} nodes, {len(self.edges)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load graph from blob: {e}")
            return False
    
    def save_to_file(self, path: str, compression: str = "zstd") -> bool:
        """
        Сохранить граф в файл.
        
        Args:
            path: Путь к файлу
            compression: Тип сжатия
            
        Returns:
            True если успешно
        """
        try:
            blob = self.save_to_blob(compression)
            
            header = {
                "format": "eva_graph_v2",
                "version": 2,
                "compression": compression,
                "size": len(blob),
                "checksum": hashlib.sha256(blob).hexdigest(),
                "created_at": time.time()
            }
            
            with open(path, 'wb') as f:
                header_bytes = json.dumps(header).encode('utf-8')
                header_len = len(header_bytes)
                f.write(header_len.to_bytes(4, 'little'))
                f.write(header_bytes)
                f.write(blob)
            
            logger.info(f"Saved graph to {path}: {len(blob)} bytes")
            return True
        except Exception as e:
            logger.error(f"Failed to save graph to file: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, path: str, storage_dir: str = None) -> Optional['FractalGraphV2']:
        """
        Загрузить граф из файла.
        
        Args:
            path: Путь к файлу
            storage_dir: Директория для хранения
            
        Returns:
            FractalGraphV2 instance или None
        """
        try:
            with open(path, 'rb') as f:
                header_len = int.from_bytes(f.read(4), 'little')
                header_bytes = f.read(header_len)
                header = json.loads(header_bytes.decode('utf-8'))
                
                if header.get("format") != "eva_graph_v2":
                    logger.error(f"Unknown file format: {header.get('format')}")
                    return None
                
                compression = header.get("compression", "zstd")
                blob = f.read()
                
                expected_checksum = header.get("checksum")
                actual_checksum = hashlib.sha256(blob).hexdigest()
                if expected_checksum and expected_checksum != actual_checksum:
                    logger.error(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")
                    return None
            
            graph = cls(storage_dir=storage_dir)
            if graph.load_from_blob(blob, compression):
                logger.info(f"Loaded graph from {path}")
                return graph
            return None
        except Exception as e:
            logger.error(f"Failed to load graph from file: {e}")
            return None


def create_fractal_graph(
    storage_dir: str = None,
    embedding_dim: int = 768
) -> FractalGraphV2:
    """Фабричная функция для создания графа."""
    return FractalGraphV2(
        storage_dir=storage_dir,
        embedding_dim=embedding_dim
    )


# === УПРАВЛЕНИЕ LLAMA ИНСТАНСАМИ ===

def register_model_instance(self, model_type: str, llama_instance):
    """
    Зарегистрировать Llama инстанс модели.
    
    Args:
        model_type: Тип модели (model_a, model_b, model_c)
        llama_instance: Llama инстанс из llama_cpp
    """
    if not hasattr(self, '_model_instances'):
        self._model_instances = {}
    
    self._model_instances[model_type] = llama_instance
    logger.info(f"Зарегистрирован Llama инстанс: {model_type}")

def get_model_instance(self, model_type: str):
    """Получить Llama инстанс модели."""
    if hasattr(self, '_model_instances'):
        return self._model_instances.get(model_type)
    return None

# Добавляем методы к классу
FractalGraphV2.register_model_instance = register_model_instance
FractalGraphV2.get_model_instance = get_model_instance