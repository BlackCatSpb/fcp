"""
Curiosity Engine для FMF - генерация вопросов для самодиалога
Адаптировано из EVA-Ai
"""
import re
import logging
import time
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("fmf.curiosity_engine")


class CuriosityType(Enum):
    ENTITY_EXPLORATION = "entity_exploration"
    KNOWLEDGE_GAP = "knowledge_gap"
    PATTERN_DISCOVERY = "pattern_discovery"
    TOPIC_EXPANSION = "topic_expansion"
    UNCERTAINTY = "uncertainty"


@dataclass
class CuriosityTrigger:
    """Триггер любопытства."""
    trigger_id: str
    trigger_type: CuriosityType
    topic: str
    confidence: float
    source_text: str
    timestamp: float
    related_entities: List[str] = field(default_factory=list)
    learning_questions: List[str] = field(default_factory=list)


class FMFCuriosityEngine:
    """
    Engine для самодиалога и самообучения.
    
    Генерирует вопросы для самодиалога на основе:
    - Неизвестных сущностей
    - Пробелов в знаниях
    - Паттернов для исследования
    - Неопределённости
    """
    
    def __init__(self, graph=None):
        self._graph = graph
        self._stop_words = {
            'это', 'что', 'как', 'и', 'а', 'в', 'на', 'по', 'для', 'с',
            'кто', 'что', 'какой', 'какая', 'какое', 'почему', 'потому'
        }
        
        self.explored_entities = set()
        self.explored_topics = set()
        
        self.trigger_patterns = [
            (r'\bкто\s+это\b', CuriosityType.ENTITY_EXPLORATION),
            (r'\bчто\s+такое\b', CuriosityType.ENTITY_EXPLORATION),
            (r'\bпочему\b', CuriosityType.KNOWLEDGE_GAP),
            (r'\bкак\s+(работает|работать)\b', CuriosityType.TOPIC_EXPANSION),
            (r'\bне\s+знаю\b', CuriosityType.UNCERTAINTY),
            (r'\buncertain\b', CuriosityType.UNCERTAINTY),
        ]
    
    def detect_triggers(self, text: str) -> List[CuriosityTrigger]:
        """Обнаруживает триггеры любопытства в тексте."""
        triggers = []
        
        for pattern, trigger_type in self.trigger_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                trigger = self._create_trigger(
                    trigger_type=trigger_type,
                    topic=matches[0] if isinstance(matches[0], str) else pattern,
                    source_text=text,
                    confidence=0.7
                )
                if trigger:
                    triggers.append(trigger)
        
        # Дополнительная проверка на сущности
        triggers.extend(self._detect_unknown_entities(text))
        
        logger.info(f"Detected {len(triggers)} curiosity triggers")
        return triggers
    
    def _create_trigger(
        self, 
        trigger_type: CuriosityType, 
        topic: str, 
        source_text: str,
        confidence: float
    ) -> Optional[CuriosityTrigger]:
        """Создаёт триггер."""
        if topic in self.explored_topics:
            return None
        
        self.explored_topics.add(topic)
        
        learning_questions = self._generate_questions(trigger_type, topic)
        
        return CuriosityTrigger(
            trigger_id=f"{trigger_type.value}_{topic}_{int(time.time())}",
            trigger_type=trigger_type,
            topic=topic,
            confidence=confidence,
            source_text=source_text[:100],
            timestamp=time.time(),
            related_entities=self._extract_entities(source_text),
            learning_questions=learning_questions
        )
    
    def _detect_unknown_entities(self, text: str) -> List[CuriosityTrigger]:
        """Обнаруживает неизвестные сущности."""
        triggers = []
        
        # Простое извлечение именованных сущностей (капитализированные слова)
        words = re.findall(r'[A-ZА-Я][a-zа-яё]+', text)
        
        for word in words:
            if word.lower() in self._stop_words:
                continue
            if word not in self.explored_entities:
                # Считаем что это новая сущность
                trigger = self._create_trigger(
                    trigger_type=CuriosityType.ENTITY_EXPLORATION,
                    topic=word,
                    source_text=text,
                    confidence=0.5
                )
                if trigger:
                    triggers.append(trigger)
        
        return triggers
    
    def _extract_entities(self, text: str) -> List[str]:
        """Извлекает сущности из текста."""
        words = re.findall(r'\b[А-Яа-яё]{3,}\b', text)
        entities = [w for w in words if w not in self._stop_words][:5]
        return entities
    
    def _generate_questions(self, trigger_type: CuriosityType, topic: str) -> List[str]:
        """Генерирует вопросы для самодиалога."""
        base_questions = {
            CuriosityType.ENTITY_EXPLORATION: [
                f"Что я знаю о {topic}?",
                f"Как {topic} связано с другими концептами?",
            ],
            CuriosityType.KNOWLEDGE_GAP: [
                f"Почему я не знаю ответа на вопрос о {topic}?",
                f"Какие есть источники информации о {topic}?",
            ],
            CuriosityType.TOPIC_EXPANSION: [
                f"Как работает {topic}?",
                f"Какие примеры применения {topic}?",
            ],
            CuriosityType.UNCERTAINTY: [
                f"Как проверить информацию о {topic}?",
                f"Где найти достоверную информацию о {topic}?",
            ],
            CuriosityType.PATTERN_DISCOVERY: [
                f"Какие паттерны я вижу в информации о {topic}?",
            ],
        }
        
        return base_questions.get(trigger_type, [f"Что нового о {topic}?"])
    
    def format_for_dialog(self, trigger: CuriosityTrigger) -> str:
        """Форматирует триггер для самодиалога."""
        questions = "\n".join(f"  - {q}" for q in trigger.learning_questions)
        
        return f"""Триггер любопытства: {trigger.trigger_type.value}

Тема: {trigger.topic}
Контекст: {trigger.source_text}
Вопросы для изучения:
{questions}

Исследую эту тему..."""