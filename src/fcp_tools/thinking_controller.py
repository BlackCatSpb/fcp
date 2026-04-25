"""
Thinking Controller - Управление режимом мышления Qwen3

Контролирует включение thinking mode.
"""
from typing import Optional


class ThinkingController:
    """
    Управление режимом мышления.
    
    Определяет когда включать thinking mode для задач.
    """
    
    def __init__(
        self,
        contradiction_detector,
        routing_engine,
        tokenizer
    ):
        self.contradictions = contradiction_detector
        self.routing = routing_engine
        self.tokenizer = tokenizer
    
    def should_enable_thinking(self, prompt: str) -> bool:
        """
        Определить нужно ли мышление.
        
        Args:
            prompt: пользовательский запрос
        
        Returns:
            True если нужно мышление
        """
        # Если есть неразрешённые противоречия - включить
        if hasattr(self.contradictions, 'has_pending'):
            if self.contradictions.has_pending():
                return True
        
        # Если routing рекомендует
        if self.routing:
            config = self.routing.get_generation_config(prompt)
            return getattr(config, 'enable_thinking', False)
        
        # Эвристика: сложные запросы
        complex_keywords = [
            "почему", "как", "объясни", "докажи",
            "проанализируй", "сравни", "оцени"
        ]
        
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in complex_keywords)
    
    def build_chat_prompt(
        self,
        user_msg: str,
        enable_thinking: bool = False
    ) -> str:
        """
        Построить промпт с учётом thinking mode.
        
        Args:
            user_msg: сообщение пользователя
            enable_thinking: включить мышление
        
        Returns:
            форматированный промпт
        """
        messages = [{"role": "user", "content": user_msg}]
        
        # Использовать chat template с enable_thinking если поддерживается
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=enable_thinking,
                add_generation_prompt=True
            )
        
        # Fallback
        return user_msg


class GenerationConfig:
    """Конфигурация генерации."""
    
    def __init__(
        self,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool = False,
        **kwargs
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_thinking = enable_thinking
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self) -> dict:
        return {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'enable_thinking': self.enable_thinking
        }


class SimpleRoutingEngine:
    """Простой routing engine (fallback)."""
    
    def __init__(self):
        self._configs = {}
    
    def get_generation_config(self, prompt: str) -> GenerationConfig:
        """Получить конфиг для промпта."""
        # Простой keyword-based routing
        prompt_lower = prompt.lower()
        
        if any(kw in prompt_lower for kw in ["факт", "что такое", "кто такой"]):
            return GenerationConfig(max_tokens=256, temperature=0.3)
        
        if any(kw in prompt_lower for kw in ["творчеств", "напиши", "сочини"]):
            return GenerationConfig(max_tokens=512, temperature=0.9)
        
        # По умолчанию
        return GenerationConfig() 
    
    def register_config(self, prompt_pattern: str, config: GenerationConfig):
        """Зарегистрировать конфиг."""
        self._configs[prompt_pattern] = config