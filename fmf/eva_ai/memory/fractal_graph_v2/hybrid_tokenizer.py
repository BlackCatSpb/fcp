"""
HybridTokenizer - Гибридный токенизатор для EVA

Объединяет:
- BPE токенизатор GGUF модели (subword tokens)
- Aho-Corasick: поиск узлов графа в тексте
- Виртуальные токены для сущностей из графа

Виртуальные токены: ID от 100000 до 200000 (зарезервировано)
"""

import re
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("eva_ai.fractal_graph_v2.hybrid_tokenizer")

VIRTUAL_TOKEN_START = 100000
VIRTUAL_TOKEN_END = 200000
VIRTUAL_TOKEN_RANGE = VIRTUAL_TOKEN_END - VIRTUAL_TOKEN_START


@dataclass
class Token:
    """Токен (BPE или виртуальный)."""
    text: str
    token_id: int
    is_virtual: bool = False
    node_id: Optional[str] = None
    virtual_id: Optional[int] = None


class AhoCorasick:
    """
    Реализация Aho-Corasick для поиска подстрок.
    """
    
    def __init__(self):
        self.next = [{}]
        self.fail = [0]
        self.output = [[]]
        self.values = [None]
    
    def add(self, pattern: str, value: Any):
        """Добавить паттерн в автомат."""
        node = 0
        for char in pattern:
            if char not in self.next[node]:
                new_node = len(self.next)
                self.next[node][char] = new_node
                self.next.append({})
                self.fail.append(0)
                self.output.append([])
                self.values.append(None)
            node = self.next[node][char]
        
        self.values[node] = value
    
    def build(self):
        """Построить автомат (вычисление fail ссылок)."""
        queue = []
        
        for char, next_node in self.next[0].items():
            self.fail[next_node] = 0
            queue.append(next_node)
        
        while queue:
            current = queue.pop(0)
            
            for char, next_node in self.next[current].items():
                fail_node = self.fail[current]
                while fail_node != 0 and char not in self.next[fail_node]:
                    fail_node = self.fail[fail_node]
                
                if char in self.next[fail_node]:
                    self.fail[next_node] = self.next[fail_node][char]
                else:
                    self.fail[next_node] = 0
                
                if self.values[self.fail[next_node]]:
                    self.output[next_node].append(self.values[self.fail[next_node]])
                
                queue.append(next_node)
    
    def search(self, text: str) -> List[Tuple[int, int, Any]]:
        """
        Найти все вхождения паттернов.
        
        Returns:
            List of (start, end, value)
        """
        results = []
        node = 0
        
        for i, char in enumerate(text):
            while node != 0 and char not in self.next[node]:
                node = self.fail[node]
            
            if char in self.next[node]:
                node = self.next[node][char]
            else:
                node = 0
            
            if self.values[node]:
                pattern = self.values[node]['content'] if isinstance(self.values[node], dict) else str(self.values[node])
                start = i - len(pattern) + 1
                results.append((start, i + 1, self.values[node]))
            
            for output in self.output[node]:
                pattern = output['content'] if isinstance(output, dict) else str(output)
                start = i - len(pattern) + 1
                results.append((start, i + 1, output))
        
        return results


class HybridTokenizer:
    """
    Гибридный токенизатор.
    
    Работает в два этапа:
    1. Aho-Corasick: поиск узлов графа в тексте
    2. BPE токенизация: стандартный GGUF токенизатор
    """
    
    def __init__(
        self, 
        fractal_graph,
        base_tokenizer=None,
        virtual_token_start: int = VIRTUAL_TOKEN_START
    ):
        """
        Args:
            fractal_graph: FractalGraphV2 instance
            base_tokenizer: Базовый BPE токенизатор (опционально)
            virtual_token_start: Начало диапазона виртуальных токенов
        """
        self.graph = fractal_graph
        self.base_tokenizer = base_tokenizer
        self.virtual_token_start = virtual_token_start
        
        self.automaton = None
        self.node_to_virtual = {}
        self.virtual_to_node = {}
        
        self._build_entity_index()
    
    def _build_entity_index(self):
        """Построить Aho-Corasick автомат по узлам графа."""
        self.automaton = AhoCorasick()
        self.node_to_virtual = {}
        
        nodes = getattr(self.graph, 'nodes', {})
        
        for node_id, node in nodes.items():
            content = getattr(node, 'content', '')
            if not content or len(content) < 3:
                continue
            
            virtual_id = self._get_virtual_id(node_id)
            
            self.automaton.add(content.lower(), {
                'node_id': node_id,
                'virtual_id': virtual_id,
                'content': content
            })
            
            self.node_to_virtual[node_id] = virtual_id
        
        self.automaton.build()
        
        self.virtual_to_node = {v: k for k, v in self.node_to_virtual.items()}
        
        logger.info(f"Built Aho-Corasick: {len(self.node_to_virtual)} patterns")
    
    def _get_virtual_id(self, node_id: str) -> int:
        """Получить виртуальный ID для узла."""
        hash_val = int(hashlib.md5(node_id.encode()).hexdigest()[:8], 16)
        return self.virtual_token_start + (hash_val % VIRTUAL_TOKEN_RANGE)
    
    def encode(
        self, 
        text: str, 
        return_virtual_only: bool = False
    ) -> List[Token]:
        """
        Токенизировать текст.
        
        Args:
            text: Входной текст
            return_virtual_only: Если True, возвращать только виртуальные токены
            
        Returns:
            List of Token
        """
        tokens = []
        
        matches = self.automaton.search(text.lower())
        
        if not matches:
            if return_virtual_only:
                return []
            return self._encode_bpe(text)
        
        covered = []
        
        for start, end, match_data in matches:
            if start < 0 or end > len(text):
                continue
            
            overlap = any(s < end and e > start for s, e in covered)
            if overlap:
                continue
            
            covered.append((start, end))
        
        covered.sort()
        
        last_pos = 0
        for start, end, match_data in matches:
            if start < 0 or end > len(text):
                continue
            
            if any(s < end and e > start for s, e in covered if (s, e) != (start, end)):
                continue
            
            if start > last_pos:
                before_text = text[last_pos:start]
                if not return_virtual_only:
                    tokens.extend(self._encode_bpe(before_text))
            
            token = Token(
                text=match_data['content'],
                token_id=match_data['virtual_id'],
                is_virtual=True,
                node_id=match_data['node_id'],
                virtual_id=match_data['virtual_id']
            )
            tokens.append(token)
            last_pos = end
        
        if last_pos < len(text) and not return_virtual_only:
            tokens.extend(self._encode_bpe(text[last_pos:]))
        
        return tokens
    
    def _encode_bpe(self, text: str) -> List[Token]:
        """BPE токенизация через базовый токенизатор."""
        if self.base_tokenizer:
            try:
                token_ids = self.base_tokenizer.encode(text)
                token_texts = self.base_tokenizer.tokenizer.convert_ids_to_tokens(token_ids) if hasattr(self.base_tokenizer.tokenizer, 'convert_ids_to_tokens') else [str(t) for t in token_ids]
                
                tokens = []
                for tid, ttext in zip(token_ids, token_texts):
                    tokens.append(Token(
                        text=str(ttext),
                        token_id=tid,
                        is_virtual=False
                    ))
                return tokens
            except Exception as e:
                logger.debug(f"BPE tokenization failed: {e}")
        
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            tokens.append(Token(
                text=word,
                token_id=i + 1000,
                is_virtual=False
            ))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Декодировать токены обратно в текст."""
        parts = []
        
        for tid in token_ids:
            if VIRTUAL_TOKEN_START <= tid < VIRTUAL_TOKEN_END:
                node_id = self.virtual_to_node.get(tid)
                if node_id and node_id in self.graph.nodes:
                    content = self.graph.nodes[node_id].content
                    parts.append(f"[{content}]")
                else:
                    parts.append(f"<virtual_{tid}>")
            else:
                if self.base_tokenizer:
                    try:
                        parts.append(self.base_tokenizer.decode([tid]))
                    except:
                        parts.append(str(tid))
                else:
                    parts.append(str(tid))
        
        return ' '.join(parts)
    
    def get_virtual_token_info(self, token_id: int) -> Optional[Dict[str, Any]]:
        """Получить информацию о виртуальном токене."""
        if not (VIRTUAL_TOKEN_START <= token_id < VIRTUAL_TOKEN_END):
            return None
        
        node_id = self.virtual_to_node.get(token_id)
        if not node_id or node_id not in self.graph.nodes:
            return None
        
        node = self.graph.nodes[node_id]
        return {
            'node_id': node_id,
            'content': node.content,
            'node_type': node.node_type,
            'level': node.level,
            'confidence': node.confidence
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Извлечь все сущности из текста."""
        tokens = self.encode(text, return_virtual_only=True)
        
        entities = []
        for token in tokens:
            if token.is_virtual and token.node_id:
                node = self.graph.nodes.get(token.node_id)
                if node:
                    entities.append({
                        'node_id': token.node_id,
                        'content': node.content,
                        'type': node.node_type,
                        'level': node.level,
                        'confidence': node.confidence
                    })
        
        return entities
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику токенизатора."""
        return {
            'virtual_token_range': f"{VIRTUAL_TOKEN_START}-{VIRTUAL_TOKEN_END}",
            'total_virtual_tokens': len(self.node_to_virtual),
            'has_base_tokenizer': self.base_tokenizer is not None
        }


def create_hybrid_tokenizer(fractal_graph, base_tokenizer=None) -> HybridTokenizer:
    """Фабричная функция."""
    return HybridTokenizer(fractal_graph, base_tokenizer)