"""
Performance Analyzer для FMF - простой анализ производительности
Адаптировано из EVA-Ai
"""
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("fmf.performance_analyzer")


class AnalysisType(Enum):
    PERFORMANCE = "performance"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    OPPORTUNITY = "opportunity"


class OpportunityPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LearningOpportunity:
    """Возможность для обучения."""
    opportunity_id: str
    description: str
    priority: OpportunityPriority
    estimated_impact: float
    timestamp: float


class FMFPerformanceAnalyzer:
    """
    Простой анализатор производительности для FMF.
    
    Отслеживает:
    - Время генерации
    - Успешность извлечения концептов
    - Противоречия
    - Возможности для оптимизации
    """
    
    def __init__(self):
        self._start_time = time.time()
        self._generation_times: List[float] = []
        self._concept_extractions = 0
        self._contradictions_generated = 0
        self._errors = 0
        self._requests = 0
    
    def record_generation(self, latency_ms: float):
        """Записывает время генерации."""
        self._generation_times.append(latency_ms)
        self._requests += 1
        
        # Ограничиваем размер истории
        if len(self._generation_times) > 100:
            self._generation_times = self._generation_times[-100:]
    
    def record_concept_extraction(self, count: int):
        """Записывает извлечение концептов."""
        self._concept_extractions += count
    
    def record_contradiction(self, count: int = 1):
        """Записывает генерацию противоречий."""
        self._contradictions_generated += count
    
    def record_error(self):
        """Записывает ошибку."""
        self._errors += 1
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Анализирует производительность."""
        uptime = time.time() - self._start_time
        
        avg_latency = 0
        if self._generation_times:
            avg_latency = sum(self._generation_times) / len(self._generation_times)
        
        p50 = 0
        p95 = 0
        if self._generation_times:
            sorted_times = sorted(self._generation_times)
            p50_idx = int(len(sorted_times) * 0.5)
            p95_idx = int(len(sorted_times) * 0.95)
            p50 = sorted_times[p50_idx] if p50_idx < len(sorted_times) else 0
            p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
        
        # Оцениваем возможности
        opportunities = self._find_opportunities(avg_latency)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self._requests,
            "total_errors": self._errors,
            "latency": {
                "average_ms": avg_latency,
                "p50_ms": p50,
                "p95_ms": p95,
                "min_ms": min(self._generation_times) if self._generation_times else 0,
                "max_ms": max(self._generation_times) if self._generation_times else 0,
            },
            "concepts_extracted": self._concept_extractions,
            "contradictions_generated": self._contradictions_generated,
            "opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "description": opp.description,
                    "priority": opp.priority.value,
                    "estimated_impact": opp.estimated_impact,
                    "timestamp": opp.timestamp
                }
                for opp in opportunities
            ],
            "health_score": self._calculate_health_score(avg_latency),
        }
    
    def _find_opportunities(self, avg_latency: float) -> List[LearningOpportunity]:
        """Находит возможности для оптимизации."""
        opportunities = []
        
        # Высокая задержка
        if avg_latency > 30000:
            opportunities.append(LearningOpportunity(
                opportunity_id="high_latency",
                description=f"Высокая задержка генерации: {avg_latency:.0f}ms",
                priority=OpportunityPriority.HIGH,
                estimated_impact=0.8,
                timestamp=time.time()
            ))
        
        # Много ошибок
        error_rate = self._errors / max(1, self._requests)
        if error_rate > 0.1:
            opportunities.append(LearningOpportunity(
                opportunity_id="high_error_rate",
                description=f"Высокий процент ошибок: {error_rate*100:.1f}%",
                priority=OpportunityPriority.CRITICAL,
                estimated_impact=0.9,
                timestamp=time.time()
            ))
        
        # Мало концептов извлечено
        if self._requests > 10 and self._concept_extractions < self._requests * 0.5:
            opportunities.append(LearningOpportunity(
                opportunity_id="low_concept_extraction",
                description="Низкий процент извлечения концептов",
                priority=OpportunityPriority.MEDIUM,
                estimated_impact=0.5,
                timestamp=time.time()
            ))
        
        return opportunities
    
    def _calculate_health_score(self, avg_latency: float) -> float:
        """Рассчитывает.score здоровья системы."""
        score = 1.0
        
        # Штраф за задержку
        if avg_latency > 60000:
            score -= 0.3
        elif avg_latency > 30000:
            score -= 0.15
        elif avg_latency > 15000:
            score -= 0.05
        
        # Штраф за ошибки
        error_rate = self._errors / max(1, self._requests)
        score -= error_rate * 0.5
        
        return max(0.0, min(1.0, score))
    
    def format_report(self) -> str:
        """Форматирует отчёт."""
        stats = self.analyze_performance()
        
        return f"""=== FMF Performance Report ===
Uptime: {stats['uptime_seconds']:.0f}s
Requests: {stats['total_requests']}
Errors: {stats['total_errors']}

Latency:
  Avg: {stats['latency']['average_ms']:.0f}ms
  P50: {stats['latency']['p50_ms']:.0f}ms
  P95: {stats['latency']['p95_ms']:.0f}ms

Concepts: {stats['concepts_extracted']}
Contradictions: {stats['contradictions_generated']}
Health: {stats['health_score']:.2f}

Opportunities: {len(stats['opportunities'])}"""