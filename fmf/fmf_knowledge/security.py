"""
Security Framework для FMF - базовая безопасность
Адаптировано из EVA-Ai
"""
import time
import logging
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger("fmf.security")


@dataclass
class SecurityEvent:
    """Событие безопасности."""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    timestamp: datetime
    details: Dict


class RateLimiter:
    """Ограничитель частоты запросов."""
    
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Проверяет, разрешен ли запрос."""
        current_time = time.time()
        window_start = current_time - 60
        
        with self.lock:
            # Очистка старых запросов
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if ts > window_start
            ]
            
            # Burst limit
            recent = [ts for ts in self.requests[identifier] if ts > current_time - 1]
            if len(recent) >= self.burst_limit:
                return False
            
            # Общий лимит
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False
            
            self.requests[identifier].append(current_time)
            return True
    
    def get_remaining(self, identifier: str) -> int:
        """Оставшиеся запросы."""
        current_time = time.time()
        with self.lock:
            recent = [ts for ts in self.requests[identifier] if ts > current_time - 60]
            return max(0, self.requests_per_minute - len(recent))


class InputValidator:
    """Валидация ввода."""
    
    def __init__(self):
        self.max_input_length = 10000
        self.blocked_patterns = [
            '<script>',
            'javascript:',
            'onerror=',
            'onclick=',
            '{{',
            '{%',
        ]
    
    def validate(self, text: str) -> bool:
        """Валидирует ввод."""
        if not text:
            return False
        
        if len(text) > self.max_input_length:
            logger.warning(f"Input too long: {len(text)}")
            return False
        
        # Проверка на опасные паттерны
        text_lower = text.lower()
        for pattern in self.blocked_patterns:
            if pattern.lower() in text_lower:
                logger.warning(f"Blocked pattern detected: {pattern}")
                return False
        
        return True
    
    def sanitize(self, text: str) -> str:
        """Санитизирует ввод."""
        if not text:
            return ""
        
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        
        return text[:self.max_input_length]


class FMSSecurityFramework:
    """
    Базовый Security Framework для FMF.
    
    Включает:
    - Rate Limiting
    - Input Validation
    - Security Event Logging
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            requests_per_minute=60,
            burst_limit=10
        )
        self.validator = InputValidator()
        self._events: List[SecurityEvent] = []
        self._max_events = 100
    
    def check_request(self, identifier: str = "default") -> bool:
        """Проверяет запрос."""
        return self.rate_limiter.is_allowed(identifier)
    
    def validate_input(self, text: str) -> bool:
        """Валидирует ввод."""
        return self.validator.validate(text)
    
    def sanitize_input(self, text: str) -> str:
        """Санитизирует ввод."""
        return self.validator.sanitize(text)
    
    def log_event(self, event_type: str, user_id: Optional[str] = None, details: Dict = None):
        """Логирует событие."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address="local",
            timestamp=datetime.now(),
            details=details or {}
        )
        
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
    
    def get_stats(self) -> Dict:
        """Возвращает статистику."""
        return {
            "rate_limit_remaining": self.rate_limiter.get_remaining("default"),
            "events_count": len(self._events),
        }