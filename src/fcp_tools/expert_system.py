"""
Expert System - Мультиагентное обсуждение

Несколько экспертов обсуждают задачу.
"""
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import time


class Expert:
    """Эксперт - агент с LoRA адаптером."""
    
    def __init__(
        self,
        name: str,
        generate_fn: Callable[[str], str],
        adapter_name: Optional[str] = None
    ):
        self.name = name
        self.generate_fn = generate_fn
        self.adapter_name = adapter_name
    
    def generate(self, prompt: str) -> str:
        """Сгенерировать ответ."""
        return self.generate_fn(prompt)


class ExpertSystem:
    """
    Мультиагентная система обсуждения.
    
    Несколько экспертов генерируют ответы параллельно,
    затем voting или synthesis.
    """
    
    def __init__(
        self,
        experts: List[Expert],
        critic: Optional[object] = None
    ):
        self.experts = experts
        self.critic = critic  # ContradictionDetector
    
    def discuss(self, prompt: str) -> str:
        """
        Запустить обсуждение.
        
        Args:
            prompt: запрос
        
        Returns:
            согласованный ответ
        """
        if not self.experts:
            return "No experts configured"
        
        # Параллельная генерация
        responses = self._parallel_generate(prompt)
        
        if len(responses) == 1:
            return responses[0]
        
        # Проверить на противоречия
        conflicts = self._detect_conflicts(responses)
        
        if conflicts:
            return self._resolve(responses, conflicts)
        
        # Простой voting
        return self._vote(responses)
    
    def _parallel_generate(self, prompt: str) -> List[str]:
        """Параллельная генерация всеми экспертами."""
        responses = []
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(exp.generate, prompt) for exp in self.experts]
            
            for f in futures:
                try:
                    result = f.result(timeout=30)
                    responses.append(result)
                except Exception as e:
                    print(f"Expert error: {e}")
        
        return responses
    
    def _detect_conflicts(self, responses: List[str]) -> Optional[Dict]:
        """Обнаружить противоречия между ответами."""
        if not self.critic:
            return None
        
        if hasattr(self.critic, 'detect_multi'):
            return self.critic.detect_multi(responses)
        
        return None
    
    def _resolve(
        self,
        responses: List[str],
        conflicts: Dict
    ) -> str:
        """Разрешить противоречия."""
        # Simple voting - most common
        return max(set(responses), key=responses.count)
    
    def _vote(self, responses: List[str]) -> str:
        """Голосование."""
        # Count occurrences
        counts = {}
        for r in responses:
            # Normalize
            norm = r.strip()
            counts[norm] = counts.get(norm, 0) + 1
        
        # Return most common
        return max(counts.keys(), key=lambda k: counts[k])
    
    def add_expert(self, expert: Expert):
        """Добавить эксперта."""
        self.experts.append(expert)
    
    def remove_expert(self, name: str):
        """Удалить эксперта."""
        self.experts = [e for e in self.experts if e.name != name]


class SimpleExpert(Expert):
    """Простой эксперт без LoRA."""
    
    def __init__(self, name: str, pipeline):
        super().__init__(name, None)
        self.pipeline = pipeline
    
    def generate(self, prompt: str) -> str:
        return self.pipeline.generate(prompt)