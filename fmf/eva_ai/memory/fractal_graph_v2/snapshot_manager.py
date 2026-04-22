"""
SnapshotManager - Неизменяемые снимки памяти для консистентности генерации

Обеспечивает:
- Неизменяемый контекст на время генерации ответа
- Защиту от race conditions при параллельных обновлениях графа
- Консистентность данных между сессиями
"""

import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from copy import deepcopy

logger = logging.getLogger("eva_ai.fractal_graph_v2.snapshot")


@dataclass
class MemorySnapshot:
    """Неизменяемый снимок части графа, релевантной текущему запросу."""
    snapshot_id: str
    created_at: float
    node_contents: Dict[str, str]
    node_confidences: Dict[str, float]
    node_metadata: Dict[str, Dict[str, Any]]
    dialogue_context: str
    session_id: str
    ttl_seconds: float = 300.0
    
    def is_expired(self) -> bool:
        """Проверяет, истёк ли снимок."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def get_content(self, node_id: str) -> Optional[str]:
        """Получить содержимое узла."""
        return self.node_contents.get(node_id)
    
    def get_confidence(self, node_id: str) -> float:
        """Получить уверенность узла."""
        return self.node_confidences.get(node_id, 0.0)
    
    def get_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Получить метаданные узла."""
        return self.node_metadata.get(node_id)
    
    def get_all_nodes(self) -> List[str]:
        """Получить список всех узлов в снимке."""
        return list(self.node_contents.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "node_count": len(self.node_contents),
            "session_id": self.session_id,
            "is_expired": self.is_expired()
        }


class SnapshotManager:
    """
    Управляет созданием и хранением снимков для сессий генерации.
    
    Особенности:
    - Thread-safe создание снимков
    - Автоматическая очистка устаревших снимков
    - Защита от race conditions
    """
    
    def __init__(
        self,
        fractal_graph=None,
        ttl_seconds: float = 300.0,
        max_active_snapshots: int = 20
    ):
        self.fractal_graph = fractal_graph
        self.ttl_seconds = ttl_seconds
        self.max_active_snapshots = max_active_snapshots
        
        self._lock = threading.RLock()
        self._active_snapshots: Dict[str, MemorySnapshot] = {}
        self._session_snapshots: Dict[str, str] = {}
        
        self._cleanup_thread = None
        self._running = False
        
        logger.info(f"SnapshotManager initialized: ttl={ttl_seconds}s, max={max_active_snapshots}")
    
    def start(self):
        """Запустить фоновую очистку."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("SnapshotManager cleanup thread started")
    
    def stop(self):
        """Остановить фоновую очистку."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=2.0)
        logger.info("SnapshotManager stopped")
    
    def _cleanup_loop(self):
        """Фоновый цикл очистки."""
        while self._running:
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.error(f"Snapshot cleanup error: {e}")
            time.sleep(30)
    
    def create_snapshot(
        self,
        session_id: str,
        node_ids: List[str],
        dialogue_context: str = ""
    ) -> MemorySnapshot:
        """
        Создаёт неизменяемый снимок для указанной сессии.
        
        Args:
            session_id: Идентификатор сессии
            node_ids: Список ID узлов для включения в снимок
            dialogue_context: Контекст диалога
            
        Returns:
            MemorySnapshot - неизменяемый снимок
        """
        with self._lock:
            snapshot_id = self._generate_snapshot_id(session_id)
            
            node_contents = {}
            node_confidences = {}
            node_metadata = {}
            
            if self.fractal_graph:
                for node_id in node_ids:
                    node = self._get_node(node_id)
                    if node:
                        node_contents[node_id] = getattr(node, 'content', '')
                        node_confidences[node_id] = getattr(node, 'confidence', 0.5)
                        node_metadata[node_id] = getattr(node, 'metadata', {})
            
            snapshot = MemorySnapshot(
                snapshot_id=snapshot_id,
                created_at=time.time(),
                node_contents=node_contents,
                node_confidences=node_confidences,
                node_metadata=node_metadata,
                dialogue_context=dialogue_context,
                session_id=session_id,
                ttl_seconds=self.ttl_seconds
            )
            
            self._active_snapshots[snapshot_id] = snapshot
            self._session_snapshots[session_id] = snapshot_id
            
            self._evict_if_needed()
            
            logger.debug(f"Created snapshot {snapshot_id[:8]} with {len(node_ids)} nodes for session {session_id}")
            return snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[MemorySnapshot]:
        """
        Получить снимок по ID.
        
        Returns:
            MemorySnapshot или None если не найден или истёк
        """
        with self._lock:
            snapshot = self._active_snapshots.get(snapshot_id)
            
            if snapshot and snapshot.is_expired():
                del self._active_snapshots[snapshot_id]
                if snapshot.session_id in self._session_snapshots:
                    del self._session_snapshots[snapshot.session_id]
                return None
            
            return snapshot
    
    def get_session_snapshot(self, session_id: str) -> Optional[MemorySnapshot]:
        """Получить последний снимок для сессии."""
        with self._lock:
            snapshot_id = self._session_snapshots.get(session_id)
            if snapshot_id:
                return self.get_snapshot(snapshot_id)
            return None
    
    def cleanup_expired(self) -> int:
        """
        Удаляет устаревшие снимки.
        
        Returns:
            Количество удалённых снимков
        """
        with self._lock:
            expired = [
                sid for sid, snap in self._active_snapshots.items()
                if snap.is_expired()
            ]
            
            for sid in expired:
                snap = self._active_snapshots[sid]
                if snap.session_id in self._session_snapshots:
                    del self._session_snapshots[snap.session_id]
                del self._active_snapshots[sid]
            
            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired snapshots")
            
            return len(expired)
    
    def _evict_if_needed(self):
        """Вытеснение старых снимков при превышении лимита."""
        if len(self._active_snapshots) > self.max_active_snapshots:
            sorted_snapshots = sorted(
                self._active_snapshots.items(),
                key=lambda x: x[1].created_at
            )
            
            to_remove = len(self._active_snapshots) - self.max_active_snapshots
            for i in range(to_remove):
                sid, snap = sorted_snapshots[i]
                if snap.session_id in self._session_snapshots:
                    del self._session_snapshots[snap.session_id]
                del self._active_snapshots[sid]
            
            logger.debug(f"Evicted {to_remove} old snapshots")
    
    def _generate_snapshot_id(self, session_id: str) -> str:
        """Генерирует уникальный ID снимка."""
        data = f"{session_id}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _get_node(self, node_id: str):
        """Получает узел из графа (с защитой от ошибок)."""
        if not self.fractal_graph:
            return None
        
        try:
            if hasattr(self.fractal_graph, 'get_node'):
                return self.fractal_graph.get_node(node_id)
            elif hasattr(self.fractal_graph, 'storage') and hasattr(self.fractal_graph.storage, 'nodes'):
                return self.fractal_graph.storage.nodes.get(node_id)
        except Exception as e:
            logger.warning(f"Failed to get node {node_id}: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику менеджера."""
        with self._lock:
            active_count = len(self._active_snapshots)
            expired_count = sum(1 for s in self._active_snapshots.values() if s.is_expired())
            
            return {
                "active_snapshots": active_count,
                "max_snapshots": self.max_active_snapshots,
                "expired_count": expired_count,
                "session_count": len(self._session_snapshots),
                "ttl_seconds": self.ttl_seconds,
                "cleanup_running": self._running
            }


def create_snapshot_manager(
    fractal_graph=None,
    ttl_seconds: float = 300.0,
    max_active_snapshots: int = 20
) -> SnapshotManager:
    """Фабричная функция для создания SnapshotManager."""
    return SnapshotManager(
        fractal_graph=fractal_graph,
        ttl_seconds=ttl_seconds,
        max_active_snapshots=max_active_snapshots
    )
