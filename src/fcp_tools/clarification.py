"""
Clarification Generator - генерация уточняющих вопросов

Задаёт вопросы при неопределённости.
"""
from typing import List, Optional


class ClarificationGenerator:
    """
    Генератор уточняющих вопросов.
    
    Генерирует вопросы для уточнения фактов.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def generate(
        self,
        fact_a: str,
        fact_b: str
    ) -> str:
        """
        Сгенерировать уточняющий вопрос.
        
        Args:
            fact_a: первый факт
            fact_b: второй факт (противоречие)
        
        Returns:
            вопрос
        """
        prompt = (
            f"Обнаружено противоречие:\n"
            f"1. {fact_a}\n"
            f"2. {fact_b}\n\n"
            f"Какое утверждение верно? "
            f"Ответь номером 1 или 2."
        )
        
        if self.pipeline:
            return self.pipeline.generate(prompt, max_new_tokens=10)
        
        return "Какой факт вы имеете в виду?"
    
    def generate_clarification(
        self,
        ambiguous_term: str,
        context: str
    ) -> str:
        """Генерировать уточнение для неоднозначного термина."""
        prompt = (
            f"Термин '{ambiguous_term}' в контексте '{context}' неоднозначен.\n"
            f"Уточните, пожалуйста, что именно вы имеете в виду?"
        )
        
        return prompt


def handle_clarification_response(
    user_response: str,
    contradiction: dict,
    graph
) -> str:
    """
    Обработать ответ пользователя на уточнение.
    
    Args:
        user_response: ответ пользователя
        contradiction: данные противоречия
        graph: граф знаний
    
    Returns:
        результат обработки
    """
    response = user_response.strip()
    
    if "1" in response:
        choice = 1
    elif "2" in response:
        choice = 2
    else:
        return "Пожалуйста, ответьте 1 или 2."
    
    # Разрешить противоречие в графе
    if hasattr(graph, 'resolve_contradiction'):
        graph.resolve_contradiction(contradiction, choice=choice)
        return "Противоречие разрешено."
    
    return "Противоречие обработано."