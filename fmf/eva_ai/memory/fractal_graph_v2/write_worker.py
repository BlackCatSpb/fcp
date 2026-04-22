"""
KnowledgeWriteWorker - Фоновый воркер для асинхронной записи узлов в граф знаний.

Предотвращает блокировку основного потока при векторизации и сохранении.
"""

import logging
import queue
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("eva_ai.fractal_graph_v2.write_worker")


class KnowledgeWriteWorker:
    """
    Фоновый воркер для асинхронной записи узлов в граф знаний.
    Предотвращает блокировку основного потока при векторизации и сохранении.
    """
    
    def __init__(self, fractal_graph_instance=None):
        self.fg = fractal_graph_instance
        self.task_queue = queue.Queue()
        self.running = True
        
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info("KnowledgeWriteWorker started")

    def enqueue_write(
        self, 
        content: str, 
        node_type: str, 
        metadata: Optional[Dict[str, Any]] = None,
        auto_vectorize: bool = True
    ):
        """
        Поставляет задачу записи в очередь.
        
        Args:
            content: Содержание узла.
            node_type: Тип узла.
            metadata: Метаданные узла.
            auto_vectorize: Автоматическая векторизация в фоне.
        """
        self.task_queue.put({
            'action': 'add_node',
            'content': content,
            'type': node_type,
            'meta': metadata or {},
            'auto_vectorize': auto_vectorize
        })

    def enqueue_update(
        self, 
        node_id: str, 
        updates: Dict[str, Any]
    ):
        """
        Поставляет задачу обновления узла.
        
        Args:
            node_id: ID узла для обновления.
            updates: Словарь обновлений.
        """
        self.task_queue.put({
            'action': 'update_node',
            'node_id': node_id,
            'updates': updates
        })

    def enqueue_delete(self, node_id: str):
        """
        Поставляет задачу удаления узла.
        
        Args:
            node_id: ID узла для удаления.
        """
        self.task_queue.put({
            'action': 'delete_node',
            'node_id': node_id
        })

    def _worker_loop(self):
        """Основной цикл обработки очереди."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                
                if task['action'] == 'add_node':
                    self._process_add_node(task)
                elif task['action'] == 'update_node':
                    self._process_update_node(task)
                elif task['action'] == 'delete_node':
                    self._process_delete_node(task)
                
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Write worker error: {e}")

    def _process_add_node(self, task: Dict[str, Any]):
        """Обработка добавления узла."""
        if not self.fg:
            logger.warning("Fractal graph not available")
            return
            
        try:
            self.fg.add_node(
                content=task['content'],
                node_type=task['type'],
                metadata=task.get('meta', {}),
                auto_vectorize=task.get('auto_vectorize', True)
            )
            logger.debug(f"Added node: {task['type']}")
        except Exception as e:
            logger.error(f"Error adding node: {e}")

    def _process_update_node(self, task: Dict[str, Any]):
        """Обработка обновления узла."""
        if not self.fg:
            return
            
        try:
            node_id = task.get('node_id')
            updates = task.get('updates', {})
            
            if hasattr(self.fg, 'update_node'):
                self.fg.update_node(node_id, **updates)
            elif hasattr(self.fg, 'update_node_metadata'):
                self.fg.update_node_metadata(node_id, updates)
                
            logger.debug(f"Updated node: {node_id}")
        except Exception as e:
            logger.error(f"Error updating node: {e}")

    def _process_delete_node(self, task: Dict[str, Any]):
        """Обработка удаления узла."""
        if not self.fg:
            return
            
        try:
            node_id = task.get('node_id')
            if hasattr(self.fg, 'delete_node'):
                self.fg.delete_node(node_id)
            elif hasattr(self.fg, 'remove_node'):
                self.fg.remove_node(node_id)
                
            logger.debug(f"Deleted node: {node_id}")
        except Exception as e:
            logger.error(f"Error deleting node: {e}")

    def stop(self):
        """Останавливает воркер."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info("KnowledgeWriteWorker stopped")

    def get_queue_size(self) -> int:
        """Возвращает размер очереди задач."""
        return self.task_queue.qsize()

    def is_running(self) -> bool:
        """Проверяет, работает ли воркер."""
        return self.running and self.thread.is_alive()