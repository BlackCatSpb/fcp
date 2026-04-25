"""
Scenario TCM - Эпизодическая память

Сохраняет цепочки диалогов как сценарии.
"""
import numpy as np
from typing import List, Dict, Optional, Any
from uuid import uuid4
import time


class ScenarioTCM:
    """
    Эпизодическая память для сценариев.
    
    Сохраняет цепочки диалогов.
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.current_chain: List[Dict] = []
    
    def add_turn(
        self,
        role: str,
        text: str,
        embedding: np.ndarray
    ):
        """
        Добавить ход диалога.
        
        Args:
            role: "user" или "assistant"
            text: текст
            embedding: эмбеддинг
        """
        turn = {
            "role": role,
            "text": text,
            "emb": embedding,
            "timestamp": time.time()
        }
        
        self.current_chain.append(turn)
        
        # Проверить конец сценария
        if self._is_end(text):
            self._save_chain()
    
    def _is_end(self, text: str) -> bool:
        """Определить конец сценария."""
        end_keywords = ["спасибо", "пока", "новый вопрос", "до свидания"]
        text_lower = text.lower()
        return any(kw in text_lower for kw in end_keywords)
    
    def _save_chain(self):
        """Сохранить цепочку в граф."""
        if not self.current_chain:
            return
        
        prev_id = None
        
        for turn in self.current_chain:
            node_id = self.graph.add_node(
                content=turn["text"],
                node_type="scenario_turn",
                embedding=turn["emb"].tobytes()
            )
            
            if prev_id:
                self.graph.add_edge(prev_id, node_id, "next_turn")
            
            prev_id = node_id
        
        # Очистить текущую цепочку
        self.current_chain.clear()
    
    def get_current_chain(self) -> List[Dict]:
        """Получить текущую цепочку."""
        return self.current_chain.copy()
    
    def clear(self):
        """Очистить текущую цепочку."""
        self.current_chain.clear()
    
    def get_recent_scenarios(self, k: int = 5) -> List[str]:
        """Получить недавние сценарии (тексты)."""
        # Упрощённо - возвращаем последние тексты
        recent = self.current_chain[-k:] if self.current_chain else []
        return [t["text"] for t in recent]


class ScenarioMemory:
    """Память сценариев с поиском."""
    
    def __init__(self):
        self.scenarios: List[Dict] = []
    
    def add_scenario(self, chain: List[Dict]):
        """Добавить сценарий."""
        self.scenarios.append({
            "id": str(uuid4()),
            "chain": chain,
            "timestamp": time.time()
        })
    
    def search(self, query_emb: np.ndarray, k: int = 3) -> List[Dict]:
        """Поиск похожих сценариев."""
        results = []
        
        for scenario in self.scenarios:
            # Mean pool эмбеддингов цепочки
            embs = [t["emb"] for t in scenario["chain"] if "emb" in t]
            if not embs:
                continue
            
            mean_emb = np.mean(embs, axis=0)
            
            # Cosine similarity
            sim = np.dot(query_emb, mean_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(mean_emb) + 1e-8)
            
            results.append((sim, scenario))
        
        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [r[1] for r in results[:k]]