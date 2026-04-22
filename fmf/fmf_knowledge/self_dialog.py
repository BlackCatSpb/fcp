"""
Self-Dialog для FMF - самодиалог для обучения
Адаптировано из EVA-Ai
"""
import time
import logging
import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("fmf.self_dialog")


class DialogRole(Enum):
    """Роли участников."""
    ASSISTANT = "assistant"
    CRITIC = "critic"
    LEARNER = "learner"
    TEACHER = "teacher"
    OBSERVER = "observer"


class LearningType(Enum):
    """Типы обучения."""
    EXPANSION = "expansion"
    REFINEMENT = "refinement"
    UPDATING = "updating"
    INTEGRATION = "integration"


@dataclass
class DialogTurn:
    """Ход в диалоге."""
    role: DialogRole
    content: str
    timestamp: float
    quality_score: float = 0.0


@dataclass
class SelfDialog:
    """Самодиалог."""
    id: str
    topic: str
    turns: List[DialogTurn]
    start_time: float
    end_time: Optional[float] = None
    outcome: Optional[str] = None


class FMFSelfDialog:
    """
    Система самодиалога для FMF.
    
    Генерирует диалог между ролями:
    - ASSISTANT: представляет концепт
    - CRITIC: ищет противоречия
    - LEARNER: предлагает направления
    - TEACHER: даёт рекомендации
    """
    
    def __init__(self, generator=None):
        self._generator = generator
        self._active_dialogs: Dict[str, SelfDialog] = {}
        self._dialog_history: List[SelfDialog] = []
        self._max_history = 20
    
    def create_dialog(self, topic: str) -> SelfDialog:
        """Создаёт новый самодиалог."""
        dialog_id = f"dialog_{uuid.uuid4().hex[:8]}"
        
        dialog = SelfDialog(
            id=dialog_id,
            topic=topic,
            turns=[],
            start_time=time.time()
        )
        
        self._active_dialogs[dialog_id] = dialog
        logger.info(f"Created dialog for: {topic}")
        
        return dialog
    
    def add_turn(self, dialog_id: str, role: DialogRole, content: str) -> bool:
        """Добавляет ход в диалог."""
        if dialog_id not in self._active_dialogs:
            return False
        
        turn = DialogTurn(
            role=role,
            content=content,
            timestamp=time.time()
        )
        
        self._active_dialogs[dialog_id].turns.append(turn)
        return True
    
    def close_dialog(self, dialog_id: str, outcome: str = None) -> Optional[SelfDialog]:
        """Закрывает диалог."""
        if dialog_id not in self._active_dialogs:
            return None
        
        dialog = self._active_dialogs[dialog_id]
        dialog.end_time = time.time()
        dialog.outcome = outcome
        
        # Добавляем в историю
        self._dialog_history.append(dialog)
        if len(self._dialog_history) > self._max_history:
            self._dialog_history = self._dialog_history[-self._max_history:]
        
        del self._active_dialogs[dialog_id]
        
        logger.info(f"Closed dialog: {dialog.topic} ({len(dialog.turns)} turns)")
        
        return dialog
    
    def get_active_dialogs(self) -> List[SelfDialog]:
        """Возвращает активные диалоги."""
        return list(self._active_dialogs.values())
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Возвращает историю диалогов."""
        return [
            {
                "id": d.id,
                "topic": d.topic,
                "turns_count": len(d.turns),
                "outcome": d.outcome,
                "duration": d.end_time - d.start_time if d.end_time else 0
            }
            for d in self._dialog_history[-limit:]
        ]
    
    def format_dialog_for_model(self, topic: str, concept_data: Dict = None) -> str:
        """Форматирует диалог для генерации модели."""
        concept_info = ""
        if concept_data:
            concept_info = f"""
Концепт: {concept_data.get('name', 'Unknown')}
Домен: {concept_data.get('domain', 'general')}
Связанные термины: {', '.join(concept_data.get('related_terms', []))}
"""
        
        return f"""{concept_info}
Тема для самодиалога: {topic}

Участники диалога:
1. ASSISTANT - представляет информацию о теме
2. CRITIC - анализирует и ищет противоречия
3. LEARNER - предлагает направления для изучения
4. TEACHER - даёт рекомендации по улучшению понимания

Начни самодиалог..."""