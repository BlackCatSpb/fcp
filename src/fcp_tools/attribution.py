"""
Attribution Report - Отчёт об атрибуции

Отслеживает какие источники использовались.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AttributionRecord:
    """Запись атрибуции."""
    layer: int
    attention_weights: Optional[List[float]] = None
    graph_nodes: List[str] = field(default_factory=list)
    lora_importance: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AttributionReport:
    """
    Отчёт об атрибуции.
    
    Отслеживает какие компоненты использовались при генерации.
    """
    
    def __init__(self):
        self.records: List[AttributionRecord] = []
    
    def track(
        self,
        layer_id: int,
        attention_weights: Optional[List[float]] = None,
        graph_nodes: Optional[List[str]] = None,
        lora_importances: Optional[List[float]] = None
    ):
        """
        Записать данные атрибуции.
        
        Args:
            layer_id: ID слоя
            attention_weights: веса внимания
            graph_nodes: использованные узлы графа
            lora_importances: важность LoRA адаптеров
        """
        record = AttributionRecord(
            layer=layer_id,
            attention_weights=attention_weights,
            graph_nodes=graph_nodes or [],
            lora_importance=lora_importances[0] if lora_importances else None
        )
        
        self.records.append(record)
    
    def explain(self) -> str:
        """
        Сгенерировать объяснение.
        
        Returns:
            текстовое объяснение
        """
        if not self.records:
            return "Нет данных для объяснения."
        
        # Простое объяснение
        layers = [r.layer for r in self.records]
        nodes_count = sum(len(r.graph_nodes) for r in self.records)
        
        return (
            f"Активировано слоёв: {len(layers)}. "
            f"Использовано узлов графа: {nodes_count}."
        )
    
    def get_sources(self) -> Dict[str, Any]:
        """Получить все источники."""
        graph_nodes = []
        for r in self.records:
            graph_nodes.extend(r.graph_nodes)
        
        return {
            "layers": [r.layer for r in self.records],
            "graph_nodes": list(set(graph_nodes)),
            "num_records": len(self.records)
        }
    
    def clear(self):
        """Очистить записи."""
        self.records.clear()


class AttributionTracker:
    """Трекер атрибуции в реальном времени."""
    
    def __init__(self):
        self.current_attention = None
        self.current_nodes = []
        self.current_lora = None
    
    def start_layer(self, layer_id: int):
        """Начать слой."""
        self.current_attention = {"layer": layer_id, "weights": []}
    
    def add_attention(self, weight: float):
        """Добавить вес внимания."""
        if self.current_attention:
            self.current_attention["weights"].append(weight)
    
    def add_graph_node(self, node_id: str):
        """Добавить узел графа."""
        if node_id not in self.current_nodes:
            self.current_nodes.append(node_id)
    
    def finalize(self, report: AttributionReport):
        """Завершить и записать."""
        if self.current_attention:
            report.track(
                layer_id=self.current_attention["layer"],
                attention_weights=self.current_attention["weights"],
                graph_nodes=self.current_nodes,
                lora_importances=[self.current_lora] if self.current_lora else None
            )
        
        self.current_attention = None
        self.current_nodes = []
        self.current_lora = None