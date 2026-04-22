"""
Health Monitor для FMF - мониторинг состояния системы
Адаптировано из EVA-Ai
"""
import time
import logging
import psutil
from typing import Dict, Any, List

logger = logging.getLogger("fmf.health_monitor")


class FMFHealthMonitor:
    """
    Простой мониторинг здоровья для FMF.
    
    Отслеживает:
    - CPU использование
    - RAM использование  
    - Состояние генератора
    - Ошибки
    """
    
    def __init__(self):
        self._start_time = time.time()
        self._error_count = 0
        self._warning_count = 0
        self._last_check = time.time()
    
    def analyze_health(self) -> Dict[str, Any]:
        """Анализирует здоровье системы."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
        except Exception:
            cpu_percent = 0
            memory = None
        
        # Рассчитываем score
        health_score = 1.0
        
        # CPU штраф
        if cpu_percent > 90:
            health_score -= 0.3
        elif cpu_percent > 70:
            health_score -= 0.15
        
        # Memory штраф
        if memory:
            if memory.percent > 90:
                health_score -= 0.2
            elif memory.percent > 70:
                health_score -= 0.1
        
        # Error штраф
        if self._error_count > 10:
            health_score -= 0.2
        elif self._error_count > 5:
            health_score -= 0.1
        
        health_score = max(0.0, min(1.0, health_score))
        
        # Определяем статус
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "degraded"
        else:
            status = "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "uptime": time.time() - self._start_time,
            "cpu_percent": cpu_percent,
            "memory": {
                "total_mb": memory.total / (1024*1024) if memory else 0,
                "available_mb": memory.available / (1024*1024) if memory else 0,
                "percent": memory.percent if memory else 0,
            } if memory else {},
            "errors": self._error_count,
            "warnings": self._warning_count,
            "timestamp": time.time()
        }
    
    def record_error(self):
        """Записывает ошибку."""
        self._error_count += 1
    
    def record_warning(self):
        """Записывает предупреждение."""
        self._warning_count += 1
    
    def reset_errors(self):
        """Сбрасывает счётчики ошибок."""
        self._error_count = 0
        self._warning_count = 0
    
    def get_summary(self) -> str:
        """Возвращает краткое резюме."""
        health = self.analyze_health()
        
        return f"""=== FMF Health ===
Status: {health['status'].upper()}
Score: {health['health_score']:.2f}
CPU: {health['cpu_percent']:.1f}%
RAM: {health['memory'].get('percent', 0):.1f}%
Errors: {health['errors']}
Warnings: {health['warnings']}
Uptime: {health['uptime']:.0f}s"""