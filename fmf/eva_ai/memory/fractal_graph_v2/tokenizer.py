"""
Graph Tokenizer - Токенизатор учитывающий архитектуру графа памяти

Особенности:
- Работает с фрактальными уровнями графа
- Семантическая токенизация на основе узлов и групп
- Контекстное извлечение из графа для генерации
"""

import os
import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("eva_ai.fractal_graph_v2.tokenizer")


@dataclass
class Token:
    """Токен из графа памяти."""
    text: str
    token_type: str           # word, subword, node_ref, group_ref, special
    node_id: Optional[str] = None
    group_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    confidence: float = 1.0


class GraphTokenizer:
    """
    Токенизатор для работы с фрактальным графом памяти.
    
    Особенности:
    - Разбивает текст на токены
    - Сопоставляет токены с узлами графа
    - Добавляет ссылки на семантические группы
    - Поддерживает специальные токены для модели
    
    Пример:
        tokenizer = GraphTokenizer(graph)
        
        tokens = tokenizer.tokenize("Снег - это белые осадки")
        
        # Результат:
        # [Token("Снег", node_ref, node_id=...), 
        #  Token("-", special),
        #  Token("это", word),
        #  Token("белые", attribute),
        #  Token("осадки", node_ref, ...)]
    """
    
    # Специальные токены
    SPECIAL_TOKENS = {
        'PAD': '[PAD]',
        'UNK': '[UNK]',
        'BOS': '[BOS]',
        'EOS': '[EOS]',
        'SEP': '[SEP]',
        'MASK': '[MASK]',
        'NODE_REF': '[NODE]',
        'GROUP_REF': '[GROUP]',
    }
    
    def __init__(self, graph, embeddings_manager=None):
        """
        Инициализация токенизатора.
        
        Args:
            graph: FractalMemoryGraph экземпляр
            embeddings_manager: EmbeddingsManager (опционально)
        """
        self.graph = graph
        self.embeddings = embeddings_manager
        
        # Словарь для быстрого сопоставления слов с узлами
        self._word_to_nodes: Dict[str, List[str]] = {}
        
        # Кэш токенизации (LRU)
        self._tokenize_cache: Dict[str, List[Token]] = {}
        self._max_cache_size = 2000
        
        # Кэш для контекста генерации
        self._context_cache: Dict[str, Tuple[str, List[str]]] = {}
        self._max_context_cache = 500
        
        # Предвычисленные n-граммы для быстрого поиска
        self._bigram_index: Dict[str, List[str]] = {}
        self._trigram_index: Dict[str, List[str]] = {}
        
        self._build_word_index()
        self._build_ngram_indexes()
    
    def _build_ngram_indexes(self):
        """Построение n-gram индексов для быстрого поиска."""
        nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
        for node_id, node in nodes.items():
            content = node.content.lower()
            words = content.split()
            
            # Биграммы
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in self._bigram_index:
                    self._bigram_index[bigram] = []
                self._bigram_index[bigram].append(node_id)
            
            # Триграммы
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if trigram not in self._trigram_index:
                    self._trigram_index[trigram] = []
                self._trigram_index[trigram].append(node_id)
        
        logger.info(f"N-gram indexes: bigrams={len(self._bigram_index)}, trigrams={len(self._trigram_index)}")
    
    def _build_word_index(self):
        """Построение индекса слов -> узлы для быстрого поиска."""
        # FractalGraphV2 имеет .nodes напрямую, FractalMemoryGraph имеет .storage.nodes
        nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
        for node_id, node in nodes.items():
            words = self._tokenize_text(node.content)
            for word in words:
                if len(word) > 2:
                    if word not in self._word_to_nodes:
                        self._word_to_nodes[word] = []
                    self._word_to_nodes[word].append(node_id)
        
        logger.info(f"Word index построен: {len(self._word_to_nodes)} уникальных слов")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Простая токенизация текста (разбиение по пробелам и punctuation)."""
        # Упрощённая токенизация
        text = text.lower().strip()
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return [t for t in tokens if t.strip()]
    
    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[Token]:
        """
        Токенизировать текст с учётом графа памяти.
        
        Args:
            text: Текст для токенизации
            add_special_tokens: Добавить специальные токены (BOS/EOS)
            
        Returns:
            List[Token]
        """
        # Проверяем кэш
        if text in self._tokenize_cache:
            return self._tokenize_cache[text]
        
        tokens = []
        
        # Добавляем BOS
        if add_special_tokens:
            tokens.append(Token(
                text=self.SPECIAL_TOKENS['BOS'],
                token_type='special'
            ))
        
        # Разбиваем текст на слова
        words = self._tokenize_text(text)
        
        current_word = ""
        i = 0
        while i < len(text):
            char = text[i]
            current_word += char
            
            # Проверяем, является ли текущее начало слова словом из индекса
            word_lower = current_word.lower().strip()
            
            # Ищем самое длинное совпадение
            matched = False
            for word in self._word_to_nodes:
                if word_lower.startswith(word):
                    # Нашли совпадение
                    node_ids = self._word_to_nodes[word]
                    
                    # Выбираем узел с наивысшей уверенностью
                    best_node = None
                    best_conf = 0
                    nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
                    for nid in node_ids:
                        if nid in nodes:
                            node = self.graph.storage.nodes[nid]
                            if node.confidence > best_conf:
                                best_conf = node.confidence
                                best_node = node
                    
                    if best_node:
                        # Добавляем узел как токен
                        tokens.append(Token(
                            text=word,
                            token_type='node_ref',
                            node_id=best_node.id,
                            group_id=best_node.parent_group_id,
                            embedding=best_node.embedding,
                            confidence=best_node.confidence
                        ))
                        
                        # Очищаем текущее слово
                        current_word = ""
                        matched = True
                        break
            
            if not matched and (char in ' \t\n' or i == len(text) - 1):
                # Конец слова - добавляем как обычный токен
                word = current_word.lower().strip()
                if word and len(word) > 0:
                    # Проверяем, есть ли это слово в индексе
                    if word in self._word_to_nodes:
                        node_ids = self._word_to_nodes[word]
                        nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
                        if node_ids and node_ids[0] in nodes:
                            node = nodes[node_ids[0]]
                            tokens.append(Token(
                                text=word,
                                token_type='node_ref',
                                node_id=node.id,
                                confidence=node.confidence
                            ))
                    else:
                        # Обычное слово
                        tokens.append(Token(
                            text=word,
                            token_type='word'
                        ))
                current_word = ""
            
            i += 1
        
        # Добавляем EOS
        if add_special_tokens:
            tokens.append(Token(
                text=self.SPECIAL_TOKENS['EOS'],
                token_type='special'
            ))
        
        # Обновляем кэш
        if len(self._tokenize_cache) >= self._max_cache_size:
            # Удаляем старые записи
            oldest_key = next(iter(self._tokenize_cache))
            del self._tokenize_cache[oldest_key]
        
        self._tokenize_cache[text] = tokens
        
        return tokens
    
    def decode(self, token_ids: List[int], id_to_token: Dict[int, str]) -> str:
        """Декодировать token IDs обратно в текст."""
        return ' '.join([id_to_token.get(tid, '[UNK]') for tid in token_ids])
    
    def get_context_for_generation(
        self,
        query: str,
        max_context_length: int = 512
    ) -> Tuple[str, List[str]]:
        """
        Получить контекст для генерации из графа памяти.
        
        Args:
            query: Запрос пользователя
            max_context_length: Максимальная длина контекста
            
        Returns:
            (context_string, list_of_node_ids)
        """
        # Проверяем кэш
        cache_key = f"{query}:{max_context_length}"
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]
        
        # Семантический поиск - используем min_level=1 для поиска во всех уровнях
        results = self.graph.semantic_search(query, top_k=10, min_level=1)
        
        context_parts = []
        node_ids = []
        
        for result in results:
            if result.get('type') == 'node':
                content = result.get('content', '')
                context_parts.append(content)
                node_ids.append(result.get('id', ''))
            elif result.get('type') == 'group':
                name = result.get('name', '')
                context_parts.append(name)
        
        # Формируем контекст
        context = "\n".join(context_parts)
        
        # Обрезаем по длине
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Обновляем кэш
        if len(self._context_cache) >= self._max_context_cache:
            oldest = next(iter(self._context_cache))
            del self._context_cache[oldest]
        
        self._context_cache[cache_key] = (context, node_ids)
        
        return context, node_ids
    
    def enhance_prompt(
        self,
        base_prompt: str,
        max_context_nodes: int = 10
    ) -> str:
        """
        Улучшить промпт, добавив контекст из графа памяти.
        
        Args:
            base_prompt: Базовый промпт
            max_context_nodes: Максимальное количество узлов контекста
            
        Returns:
            Улучшенный промпт с контекстом
        """
        # Извлекаем ключевые слова из промпта
        keywords = self._tokenize_text(base_prompt)[:5]
        
        # Ищем релевантные узлы
        relevant_nodes = []
        nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
        
        for kw in keywords:
            if kw in self._word_to_nodes:
                node_ids = self._word_to_nodes[kw]
                for nid in node_ids:
                    if nid in nodes:
                        node = self.graph.storage.nodes[nid]
                        if node.level >= 1:  # Только факты и выше
                            relevant_nodes.append(node)
        
        # Удаляем дубликаты и ограничиваем
        seen = set()
        unique_nodes = []
        for node in relevant_nodes:
            if node.id not in seen:
                seen.add(node.id)
                unique_nodes.append(node)
                if len(unique_nodes) >= max_context_nodes:
                    break
        
        if not unique_nodes:
            return base_prompt
        
        # Формируем контекст
        context_str = "Контекст из памяти:\n"
        for node in unique_nodes:
            context_str += f"- {node.content}\n"
        
        # Объединяем с промптом
        enhanced = f"{context_str}\nЗапрос: {base_prompt}"
        
        return enhanced
    
    def get_vocab_size(self) -> int:
        """Получить размер словаря (узлы + специальные токены)."""
        nodes = getattr(self.graph, 'nodes', None) or getattr(self.graph.storage, 'nodes', {})
        return len(nodes) + len(self.SPECIAL_TOKENS)
    
    def get_token_id(self, token: Token, token_to_id: Dict[str, int]) -> int:
        """Получить ID токена."""
        if token.token_type == 'special':
            return token_to_id.get(token.text, token_to_id.get('[UNK]', 0))
        
        if token.node_id:
            return token_to_id.get(f'[NODE_{token.node_id[:8]}]', token_to_id.get('[UNK]', 0))
        
        return token_to_id.get(token.text, token_to_id.get('[UNK]', 0))


def create_graph_tokenizer(graph, embeddings_manager=None) -> GraphTokenizer:
    """Фабричная функция."""
    return GraphTokenizer(graph=graph, embeddings_manager=embeddings_manager)