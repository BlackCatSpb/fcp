"""
GraphCurator - Курация графа знаний

Фоновый процесс для:
- Дедупликации узлов
- Временного распада
- Поиска противоречий
- Активного обучения через уточняющие вопросы
"""
import time
from typing import List, Dict, Optional, Callable
from threading import Thread, Event


class GraphCurator:
    """
    Куратор графа знаний.
    
    Запускается в фоне и выполняет:
    1. detect_contradictions() - поиск противоречий
    2. resolve() - разрешение противоречий
    3. prune() - дедупликация
    4. decay() - временной распад
    5. ClarificationGenerator - активное обучение
    """
    
    def __init__(
        self,
        graph,
        contradiction_detector=None,
        clarification_generator=None
    ):
        self.graph = graph
        self.contradiction_detector = contradiction_detector
        self.clarification_generator = clarification_generator
        
        # Параметры
        self.interval = 300  # секунд между циклами
        self.enabled = False
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        
        # Статистика
        self.cycles_run = 0
        self.contradictions_found = 0
        self.duplicates_removed = 0
        self.nodes_decayed = 0
    
    def start(self, interval: int = 300):
        """Запустить фоновый процесс."""
        self.interval = interval
        self.enabled = True
        self._stop_event.clear()
        
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        print(f"GraphCurator started (interval={interval}s)")
    
    def stop(self):
        """Остановить фоновый процесс."""
        self.enabled = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        print("GraphCurator stopped")
    
    def _run_loop(self):
        """Основной цикл."""
        while not self._stop_event.is_set():
            try:
                self.run_cycle()
                self.cycles_run += 1
            except Exception as e:
                print(f"Curator cycle error: {e}")
            
            # Ждать интервал или остановку
            self._stop_event.wait(self.interval)
    
    def run_cycle(self):
        """Выполнить один цикл курации."""
        # 1. Detect contradictions
        contradictions = self.detect_contradictions()
        
        # 2. Resolve
        for contra in contradictions:
            self.resolve(contra)
        
        # 3. Prune duplicates
        self.prune()
        
        # 4. Temporal decay
        self.decay()
        
        return {
            "contradictions": len(contradictions),
            "duplicates": self.duplicates_removed,
            "decayed": self.nodes_decayed
        }
    
    def detect_contradictions(self) -> List[Dict]:
        """Найти противоречия в графе."""
        if self.contradiction_detector:
            return self.contradiction_detector.detect_all()
        
        # Fallback - простой поиск
        return self._simple_contradiction_detection()
    
    def _simple_contradiction_detection(self) -> List[Dict]:
        """Упрощённый поиск противоречий."""
        contradictions = []
        
        if not self.graph:
            return contradictions
        
        # Получаем все факты
        try:
            nodes = self.graph.get_nodes_by_type("fact")
            
            # Простой парный поиск
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i+1:]:
                    # Проверить на противоречие
                    if self._is_contradiction(node_a, node_b):
                        contradictions.append({
                            "node_a": node_a,
                            "node_b": node_b,
                            "type": "semantic"
                        })
        except Exception:
            pass
        
        self.contradictions_found = len(contradictions)
        return contradictions
    
    def _is_contradiction(self, node_a, node_b) -> bool:
        """Проверить являются ли узлы противоречивыми."""
        # Простая эвристика - схожие названия но разный контент
        if hasattr(node_a, "content") and hasattr(node_b, "content"):
            content_a = str(node_a.content)
            content_b = str(node_b.content)
            
            # Если названия похожи
            if len(set(content_a.split()) & set(content_b.split())) > 2:
                # Но контент разный - возможное противоречие
                return content_a[:50] != content_b[:50]
        
        return False
    
    def resolve(self, contradiction: Dict):
        """Разрешить противоречие."""
        node_a = contradiction.get("node_a")
        node_b = contradiction.get("node_b")
        
        if not node_a or not node_b:
            return
        
        # Использовать clarification generator
        if self.clarification_generator and hasattr(self.clarification_generator, 'generate'):
            question = self.clarification_generator.generate(
                node_a.content if hasattr(node_a, 'content') else str(node_a),
                node_b.content if hasattr(node_b, 'content') else str(node_b)
            )
            
            # Сохранить вопрос для пользователя
            # (в реальной системе - показать пользователю)
        
        # Или просто пометить как конфликт
        self.graph.add_edge(node_a.id, node_b.id, "contradicts")
    
    def prune(self) -> int:
        """Удалить дубликаты."""
        self.duplicates_removed = 0
        
        if not self.graph:
            return 0
        
        try:
            # Найти дубликаты
            duplicates = self._find_duplicates()
            
            for dup in duplicates:
                # Удалить дубликат
                self._remove_node(dup)
                self.duplicates_removed += 1
        except Exception:
            pass
        
        return self.duplicates_removed
    
    def _find_duplicates(self) -> List:
        """Найти дубликаты."""
        duplicates = []
        
        try:
            nodes = self.graph.get_all_nodes()
            
            # Group by content hash
            content_groups = {}
            for node in nodes:
                content = str(getattr(node, "content", ""))
                if content:
                    key = hash(content[:100])  # First 100 chars
                    if key not in content_groups:
                        content_groups[key] = []
                    content_groups[key].append(node)
            
            # Оставить только первый
            for nodes in content_groups.values():
                if len(nodes) > 1:
                    duplicates.extend(nodes[1:])
        except Exception:
            pass
        
        return duplicates
    
    def _remove_node(self, node):
        """Удалить узел."""
        try:
            if hasattr(self.graph, 'remove_node'):
                self.graph.remove_node(node.id)
            elif hasattr(self.graph, 'delete_node'):
                self.graph.delete_node(node.id)
        except Exception:
            pass
    
    def decay(self) -> int:
        """Применить временной распад."""
        self.nodes_decayed = 0
        
        if not self.graph:
            return 0
        
        try:
            # Получить все узлы
            nodes = self.graph.get_all_nodes()
            
            now = time.time()
            decay_rate = 0.01  # 1% в день
            
            for node in nodes:
                # Проверить temporal_weight
                if hasattr(node, "last_access"):
                    age = now - node.last_access
                    
                    # older = decayed more
                    days_old = age / 86400
                    weight = max(0.1, 1.0 - decay_rate * days_old)
                    
                    node.temporal_weight = weight
                    self.nodes_decayed += 1
        except Exception:
            pass
        
        return self.nodes_decayed
    
    def get_statistics(self) -> Dict:
        """Получить статистику."""
        return {
            "cycles_run": self.cycles_run,
            "contradictions_found": self.contradictions_found,
            "duplicates_removed": self.duplicates_removed,
            "nodes_decayed": self.nodes_decayed,
            "enabled": self.enabled
        }


class ContradictionDetector:
    """
    Детектор противоречий в графе.
    
    Анализирует факты и ищет логические противоречия.
    """
    
    def __init__(self, graph, threshold: float = 0.65):
        self.graph = graph
        self.threshold = threshold
    
    def detect_all(self) -> List[Dict]:
        """Найти все противоречия."""
        contradictions = []
        
        try:
            nodes = self.graph.get_nodes_by_type("fact")
            
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i+1:]:
                    if self._check_contradiction(node_a, node_b):
                        contradictions.append({
                            "node_a": node_a,
                            "node_b": node_b,
                            "type": "logical"
                        })
        except Exception:
            pass
        
        return contradictions
    
    def detect_multi(self, responses: List[str]) -> Optional[Dict]:
        """Проверить несколько ответов на противоречие."""
        if len(responses) < 2:
            return None
        
        # Простой анализ - есть ли идентичные ответы
        unique = set(responses)
        
        if len(unique) == len(responses):
            # Все разные - возможно противоречие
            return {
                "type": "multi_expert",
                "responses": responses,
                "conflict": True
            }
        
        return None
    
    def _check_contradiction(self, node_a, node_b) -> bool:
        """Проверить противоречие между узлами."""
        # Используем embeddings для similarity
        # Если sim > 0.75 но утверждения противоположные
        
        content_a = str(getattr(node_a, "content", ""))
        content_b = str(getattr(node_b, "content", ""))
        
        # Ключевые слова-маркеры противоречий
        positive_markers = ["да", "да, это верно", "конечно"]
        negative_markers = ["нет", "это неверно", "неправильно"]
        
        has_positive = any(m in content_a.lower() for m in positive_markers)
        has_negative = any(m in content_a.lower() for m in negative_markers)
        
        if has_positive:
            return has_negative in [m in content_b.lower() for m in negative_markers]
        
        return False
    
    def has_pending(self) -> bool:
        """Есть ли неразрешённые противоречия."""
        try:
            contra = self.detect_all()
            return len(contra) > 0
        except Exception:
            return False