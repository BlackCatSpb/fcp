"""
LearningGraphManager - Управление сигналами обратной связи

Собирает статистику по слоям и доменам.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class LearningSignal:
    """Сигнал обратной связи для обучения."""
    query: str
    domain: str
    layer_id: int
    success: bool
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    response_quality: float = 0.0
    user_feedback: Optional[int] = None  # 1=good, -1=bad


@dataclass
class LayerSensitivity:
    """Чувствительность слоя к домену."""
    domain: str
    layer_id: int
    success_rate: float = 0.0
    num_queries: int = 0
    avg_confidence: float = 0.0
    needs_retraining: bool = False


class LearningGraphManager:
    """
    Менеджер сигналов обратной связи.
    
    Отслеживает:
    - Успешность по каждому слою и домену
    - Когда нужен переобучение
    - Статистику точности
    """
    
    def __init__(self, num_layers: int = 32):
        self.num_layers = num_layers
        self.signals: List[LearningSignal] = []
        self.layer_sensitivity: Dict[str, Dict[int, LayerSensitivity]] = {}
        
        # Домены
        self.domains = ["facts", "reasoning", "creative", "memory", "general"]
        
        # Инициализировать sensitivity матрицу
        for domain in self.domains:
            self.layer_sensitivity[domain] = {}
            for layer_id in range(num_layers):
                self.layer_sensitivity[domain][layer_id] = LayerSensitivity(
                    domain=domain,
                    layer_id=layer_id
                )
    
    def add_signal(
        self,
        query: str,
        domain: str,
        layer_id: int,
        success: bool,
        confidence: float,
        response_quality: float = 0.0,
        user_feedback: Optional[int] = None
    ):
        """Добавить сигнал."""
        signal = LearningSignal(
            query=query,
            domain=domain,
            layer_id=layer_id,
            success=success,
            confidence=confidence,
            response_quality=response_quality,
            user_feedback=user_feedback
        )
        
        self.signals.append(signal)
        
        # Обновить sensitivity
        self._update_sensitivity(signal)
        
        # Проверить нужно ли переобучение
        self._check_retraining_needed(domain, layer_id)
    
    def _update_sensitivity(self, signal: LearningSignal):
        """Обновить статистику чувствительности."""
        domain = signal.domain
        layer_id = signal.layer_id
        
        if domain not in self.layer_sensitivity:
            self.layer_sensitivity[domain] = {}
        
        if layer_id not in self.layer_sensitivity[domain]:
            self.layer_sensitivity[domain][layer_id] = LayerSensitivity(
                domain=domain,
                layer_id=layer_id
            )
        
        sens = self.layer_sensitivity[domain][layer_id]
        
        # Update running stats
        n = sens.num_queries + 1
        
        if signal.success:
            sens.success_rate = (sens.success_rate * (n - 1) + 1) / n
        else:
            sens.success_rate = sens.success_rate * (n - 1) / n
        
        sens.avg_confidence = (sens.avg_confidence * (n - 1) + signal.confidence) / n
        sens.num_queries = n
    
    def _check_retraining_needed(self, domain: str, layer_id: int):
        """Проверить нужно ли переобучение."""
        sens = self.layer_sensitivity.get(domain, {}).get(layer_id)
        
        if sens and sens.num_queries >= 10:
            # Если успешность < 60% - нужно переобучение
            if sens.success_rate < 0.6:
                sens.needs_retraining = True
    
    def get_layer_for_domain(self, domain: str) -> List[int]:
        """Получить список слоёв для домена, отсортированных по успешности."""
        if domain not in self.layer_sensitivity:
            return list(range(self.num_layers))
        
        layers = []
        for layer_id, sens in self.layer_sensitivity[domain].items():
            layers.append((layer_id, sens.success_rate))
        
        # Sort by success rate
        layers.sort(key=lambda x: x[1], reverse=True)
        
        return [l[0] for l in layers]
    
    def get_layers_needing_retraining(self, domain: str) -> List[int]:
        """Получить слои, которым нужно переобучение."""
        if domain not in self.layer_sensitivity:
            return []
        
        return [
            layer_id for layer_id, sens in self.layer_sensitivity[domain].items()
            if sens.needs_retraining
        ]
    
    def get_statistics(self, domain: str) -> Dict:
        """Получить статистику по домену."""
        if domain not in self.layer_sensitivity:
            return {}
        
        layers = self.layer_sensitivity[domain]
        
        total_queries = sum(s.num_queries for s in layers.values())
        avg_success = np.mean([s.success_rate for s in layers.values() if s.num_queries > 0])
        
        return {
            "domain": domain,
            "total_queries": total_queries,
            "avg_success_rate": avg_success,
            "layers_needing_retraining": len(self.get_layers_needing_retraining(domain))
        }
    
    def clear_signals(self):
        """Очистить старые сигналы."""
        self.signals.clear()


class LearningOrchestrator:
    """
    Оркестратор обучения.
    
    На основе данных LearningGraphManager принимает решения о переобучении.
    """
    
    def __init__(
        self,
        learning_manager: LearningGraphManager,
        lora_manager
    ):
        self.learning_manager = learning_manager
        self.lora_manager = lora_manager
        
        # Пороги
        self.retrain_threshold = 0.6
        self.min_queries_for_decision = 10
    
    def should_retrain(self, domain: str) -> bool:
        """Определить нужно ли переобучение."""
        stats = self.learning_manager.get_statistics(domain)
        
        if stats["total_queries"] < self.min_queries_for_decision:
            return False
        
        return stats["avg_success_rate"] < self.retrain_threshold
    
    def get_retrain_plan(self, domain: str) -> Dict:
        """Получить план переобучения."""
        layers = self.learning_manager.get_layers_needing_retraining(domain)
        
        # Определить rank - чем ниже успешность, тем больше rank
        rank_map = {}
        for layer_id in layers:
            sens = self.learning_manager.layer_sensitivity[domain][layer_id]
            if sens.success_rate < 0.4:
                rank_map[layer_id] = 16
            elif sens.success_rate < 0.5:
                rank_map[layer_id] = 8
            else:
                rank_map[layer_id] = 4
        
        return {
            "domain": domain,
            "layers": layers,
            "ranks": rank_map,
            "reason": f"success_rate < {self.retrain_threshold}"
        }
    
    def execute_retrain(self, domain: str) -> bool:
        """Выполнить переобучение."""
        if not self.should_retrain(domain):
            return False
        
        plan = self.get_retrain_plan(domain)
        
        # Запустить переобучение через lora_manager
        for layer_id, rank in plan["ranks"].items():
            # Logic here
            pass
        
        return True