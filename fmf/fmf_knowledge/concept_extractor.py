"""
ConceptExtractor для FMF - извлечение концептов из запросов и ответов
Адаптировано из EVA-Ai с упрощениями для FMF
Включает NLI анализ для проверки противоречий
"""
import re
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("fmf.concept_extractor")


@dataclass
class Concept:
    """Представление концепта."""
    name: str
    description: str
    domain: str
    confidence: float
    related_terms: List[str]


class NLIRelation:
    """NLI отношение между утверждениями."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class FMFConceptExtractor:
    """
    Извлекает концепты из текста для FMF.
    
    Флоу:
    1. Извлечь ключевые термины из текста
    2. Создать Concept с фактами
    3. Вернуть для обработки
    """
    
    def __init__(self, graph=None):
        self._graph = graph
        self._stop_words = self._load_stop_words()
        self._known_concepts: Set[str] = set()
    
    def _load_stop_words(self) -> Set[str]:
        return {
            'это', 'что', 'как', 'где', 'когда', 'почему', 'потому', 'для',
            'от', 'до', 'при', 'над', 'под', 'между', 'который', 'которая',
            'которое', 'свой', 'своя', 'своё', 'быть', 'был', 'была', 'было',
            'были', 'есть', 'will', 'are', 'was', 'were', 'have', 'has',
            'the', 'a', 'an', 'is', 'been', 'being', 'and', 'or', 'but',
            'чем', 'такой', 'такая', 'такое', 'такие', 'все', 'весь',
            'можно', 'нужно', 'надо', 'должен', 'должна', 'должно'
        }
    
    def extract_concepts(self, query: str, response: str = "") -> List[Concept]:
        """Извлекает концепты из текста."""
        full_text = f"{query} {response}".lower()
        
        terms = self._extract_terms(full_text)
        new_terms = [t for t in terms if t not in self._known_concepts]
        
        concepts = []
        for term in new_terms[:5]:
            concept = self._create_concept(term, query)
            if concept:
                concepts.append(concept)
                self._known_concepts.add(term)
        
        logger.info(f"Извлечено {len(concepts)} концептов")
        return concepts
    
    def _extract_terms(self, text: str) -> List[str]:
        """Извлекает ключевые термины."""
        words = re.findall(r'\b[а-яёa-z]{4,}\b', text.lower())
        
        freq = {}
        for word in words:
            if word not in self._stop_words:
                freq[word] = freq.get(word, 0) + 1
        
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:15]]
    
    def _create_concept(self, term: str, context: str) -> Optional[Concept]:
        """Создаёт концепт."""
        domain = self._detect_domain(context)
        
        related = [w for w in self._extract_terms(context.lower()) 
                  if w != term and w not in self._stop_words][:5]
        
        return Concept(
            name=term,
            description=f"Понятие: {term}",
            domain=domain,
            confidence=0.8,
            related_terms=related
        )
    
    def _detect_domain(self, text: str) -> str:
        """Определяет домен текста."""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['программирование', 'код', 'алгоритм', 'данные']):
            return 'technology'
        if any(w in text_lower for w in ['наука', 'исследование', 'теория']):
            return 'science'
        if any(w in text_lower for w in ['философия', 'смысл', 'существование']):
            return 'philosophy'
        return 'general'
    
    def save_to_graph(self, concept: Concept) -> bool:
        """Сохраняет концепт в граф."""
        if not self._graph:
            return False
        try:
            self._graph.add_unique_concept(concept.name, metadata={
                'domain': concept.domain,
                'confidence': concept.confidence,
                'description': concept.description
            })
            return True
        except Exception as e:
            logger.warning(f"Failed to save concept: {e}")
            return False
    
    # === NLI Analysis (from EVA-Ai) ===
    
    def analyze_nli(self, premise: str, hypothesis: str) -> str:
        """
        Анализирует NLI отношение между premise и hypothesis.
        
        Returns:
            - entailment: hypothesis следует из premise
            - contradiction: hypothesis противоречит premise
            - neutral: hypothesis не связано с premise
        """
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # 1. Проверка на противоречие через ключевые слова
        contradiction_indicators = [
            ('не позволяет', 'позволяет'),
            ('невозможно', 'возможно'),
            ('нельзя', 'можно'),
            ('опасно', 'безопасно'),
            ('вредно', 'полезно'),
            ('не развивается', 'развивается'),
            ('не помогает', 'помогает'),
            ('не используется', 'используется'),
            ('не является', 'является'),
            ('безопасно', 'опасно'),
            ('безопасно', 'угроза'),
            ('полезно', 'вредно'),
            ('полезно', 'угроза'),
        ]
        
        for neg, pos in contradiction_indicators:
            # Проверяем есть ли противоречие
            has_neg_in_premise = neg in premise_lower
            has_pos_in_hypothesis = pos in hypothesis_lower
            has_pos_in_premise = pos in premise_lower
            has_neg_in_hypothesis = neg in hypothesis_lower
            
            if has_neg_in_premise and has_pos_in_hypothesis:
                return NLIRelation.CONTRADICTION
            if has_pos_in_premise and has_neg_in_hypothesis:
                return NLIRelation.CONTRADICTION
        
        # 2. Проверка на entailment через причинно-следственную связь
        entailment_indicators = [
            'поэтому', 'значит', 'следовательно', 'таким образом', 
            'приводит к', 'потому что', 'ведь'
        ]
        for ind in entailment_indicators:
            if ind in premise_lower:
                return NLIRelation.ENTAILMENT
        
        # 3. Проверка на связь через общие слова (без стоп-слов)
        stop_words = {'это', 'что', 'как', 'и', 'а', 'в', 'на', 'по', 'для', 'с'}
        
        premise_words = set(premise_lower.split()) - stop_words
        hypothesis_words = set(hypothesis_lower.split()) - stop_words
        
        common = premise_words & hypothesis_words
        
        # Если есть существенные общие слова - entailment
        if len(common) >= 2:
            return NLIRelation.ENTAILMENT
        
        # Иначе - neutral
        return NLIRelation.NEUTRAL
    
    def find_contradictions(self, statements: List[str]) -> List[Tuple[int, int, str]]:
        """
        Находит пары противоречивых утверждений.
        
        Returns:
            Список (index_a, index_b, relation) противоречивых пар
        """
        contradictions = []
        
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                # Проверяем в обе стороны
                relation_ab = self.analyze_nli(statements[i], statements[j])
                relation_ba = self.analyze_nli(statements[j], statements[i])
                
                if relation_ab == NLIRelation.CONTRADICTION:
                    contradictions.append((i, j, relation_ab))
                if relation_ba == NLIRelation.CONTRADICTION:
                    contradictions.append((j, i, relation_ba))
        
        return contradictions
    
    def validate_fact(self, fact: str, context: str) -> Tuple[bool, str]:
        """
        Проверяет факт на противоречие с контекстом.
        
        Returns:
            (is_valid, relation) - валидность и тип отношения
        """
        relation = self.analyze_nli(context, fact)
        
        if relation == NLIRelation.CONTRADICTION:
            return False, relation
        return True, relation