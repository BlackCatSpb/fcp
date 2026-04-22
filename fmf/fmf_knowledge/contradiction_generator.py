"""
ContradictionGenerator для FMF - генерация противоречий
Адаптировано из EVA-Ai с упрощениями для FMF
"""
import random
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("fmf.contradiction_generator")


@dataclass
class FMFContradiction:
    """Противоречие с двумя точками зрения."""
    concept: str
    viewpoint_a: str
    viewpoint_b: str
    divergence: float
    reasoning_a: str
    reasoning_b: str
    resolution: Optional[str] = None


class FMFContradictionGenerator:
    """
    Генерирует противоречия для концептов через шаблоны.
    """
    
    def __init__(self, graph=None):
        self._graph = graph
        self._templates = self._load_templates()
    
    def _load_templates(self) -> dict:
        return {
            'general': [
                ('{concept} является положительным явлением',
                 '{concept} несёт негативные последствия'),
                ('{concept} приносит пользу обществу',
                 '{concept} создаёт проблемы для общества'),
                ('{concept} следует развивать',
                 '{concept} требует ограничений'),
                ('{concept} важен для прогресса',
                 '{concept} тормозит развитие'),
            ],
            'technology': [
                ('{concept} делает жизнь лучше',
                 '{concept} создаёт новые проблемы'),
                ('{concept} повышает эффективность',
                 '{concept} снижает качество'),
                ('{concept} автоматизирует рутину',
                 '{concept} лишает работы'),
                ('{concept} доступен каждому',
                 '{concept} увеличивает неравенство'),
            ],
            'science': [
                ('{concept} объясняет мир',
                 '{concept} ограничивает понимание'),
                ('{concept} даёт объективные знания',
                 '{concept} зависит от интерпретации'),
                ('{concept} доказано наукой',
                 '{concept} требует дальнейшего изучения'),
            ],
            'philosophy': [
                ('{concept} имеет объективную природу',
                 '{concept} является субъективным конструктом'),
                ('{concept} универсален',
                 '{concept} культурно-специфичен'),
                ('{concept} можно понять разумом',
                 '{concept} выходит за рамки рационального'),
            ]
        }
    
    def generate(self, concept_name: str, domain: str = 'general') -> Optional[FMFContradiction]:
        """
        Генерирует противоречие для концепта.
        
        Args:
            concept_name: Имя концепта
            domain: Домен для шаблонов
        """
        templates = self._templates.get(domain, self._templates['general'])
        
        template_pair = random.choice(templates)
        viewpoint_a = template_pair[0].format(concept=concept_name)
        viewpoint_b = template_pair[1].format(concept=concept_name)
        
        reasoning_a = self._generate_reasoning(concept_name, 'positive')
        reasoning_b = self._generate_reasoning(concept_name, 'negative')
        
        divergence = random.uniform(0.6, 0.95)
        
        contradiction = FMFContradiction(
            concept=concept_name,
            viewpoint_a=viewpoint_a,
            viewpoint_b=viewpoint_b,
            divergence=divergence,
            reasoning_a=reasoning_a,
            reasoning_b=reasoning_b
        )
        
        logger.info(f"Сгенерировано противоречие для '{concept_name}' (divergence: {divergence:.2f})")
        return contradiction
    
    def _generate_reasoning(self, concept: str, perspective: str) -> str:
        """Генерирует обоснование."""
        reasoning_templates = {
            'positive': [
                f'Потому что {concept} создаёт возможности для развития',
                f'Это подтверждается положительным опытом применения {concept}',
                f'{concept} показало эффективность в различных областях',
            ],
            'negative': [
                f'Однако существуют риски связанные с {concept}',
                f'Практика показывает негативные последствия {concept}',
                f'{concept} может привести к нежелательным результатам',
            ]
        }
        
        templates = reasoning_templates.get(perspective, reasoning_templates['positive'])
        return random.choice(templates)
    
    def format_for_dialog(self, contradiction: FMFContradiction) -> str:
        """Форматирует противоречие для самодиалога."""
        return f"""Противоречие: {contradiction.concept}

Точка зрения А: {contradiction.viewpoint_a}
Обоснование: {contradiction.reasoning_a}

Точка зрения Б: {contradiction.viewpoint_b}
Обоснование: {contradiction.reasoning_b}

Уровень расхождения: {contradiction.divergence:.2f}

Как разрешить это противоречие?"""