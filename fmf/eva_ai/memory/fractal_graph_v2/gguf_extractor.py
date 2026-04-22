"""
GGUF Knowledge Extractor - Извлечение знаний из GGUF через инференс

Методы:
1. Извлечение vocab с русскими токенами
2. Генерация знаний через промты на русском
3. Конвертация ответов в узлы графа
4. Инкрементальное обучение графа
"""

import os
import json
import time
import logging
import threading
import re
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("eva_ai.fractal_graph_v2.gguf_extractor")


@dataclass
class KnowledgeEntry:
    """Запись знания для сохранения в граф."""
    subject: str
    predicate: str          # Отношение (is_a, has_property, etc.)
    object: str
    confidence: float
    source_prompt: str       # Промт, которым получено знание
    model_response: str      # Полный ответ модели
    extraction_method: str   # how (explicit, implicit, generated)


class GGUFKnowledgeExtractor:
    """
    Извлекатель знаний из GGUF моделей.
    
    Работает через llama.cpp:
    1. Загружает GGUF модель
    2. Генерирует ответы на русские промты
    3. Парсит ответы в S-P-O triples
    4. Сохраняет в граф памяти
    """
    
    def __init__(
        self,
        model_path: str,
        graph,
        n_ctx: int = 2048,
        n_threads: int = 8
    ):
        self.model_path = model_path
        self.graph = graph
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        
        self.llama = None
        self._vocab_ru = {}  # Русские токены из vocab
        
        # Шаблоны для извлечения S-P-O
        self.spo_patterns = [
            # Явные паттерны
            r'(\w+)\s+-\s+(\w+)',  # "снег - осадки"
            r'(\w+)\s+это\s+(\w+)',  # "снег это осадки"
            r'(\w+)\s+является\s+(\w+)',
            r'(\w+)\s+относится\s+к\s+(\w+)',
            r'(\w+)\s+относится\s+к\s+категории\s+(\w+)',
            # Атрибуты
            r'(\w+)\s+имеет\s+свойство\s+(\w+)',
            r'(\w+)\s+обладает\s+(\w+)',
            # Причинно-следственные
            r'(\w+)\s+вызывает\s+(\w+)',
            r'(\w+)\s+приводит\s+к\s+(\w+)',
            r'(\w+)\s+влияет\s+на\s+(\w+)',
            # Части
            r'(\w+)\s+состоит\s+из\s+(\w+)',
            r'(\w+)\s+включает\s+в\s+себя\s+(\w+)',
            # Контекст
            r'(\w+)\s+связан\s+с\s+(\w+)',
            r'(\w+)\s+ассоциируется\s+с\s+(\w+)',
        ]
    
    def load_model(self):
        """Загрузить GGUF модель через llama.cpp."""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Загрузка GGUF модели: {self.model_path}")
            
            self.llama = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            
            logger.info("Модель загружена")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def extract_vocab_ru(self) -> Dict[str, int]:
        """
        Извлечь русские токены из vocab модели.
        
        Токенизатор GGUF содержит словарь всех токенов.
        Фильтруем по русским символам (а-я, А-Я).
        """
        if not self.llama:
            logger.warning("Модель не загружена")
            return {}
        
        if self._vocab_ru:
            return self._vocab_ru
        
        try:
            # Пробуем получить vocab из токенизатора
            if hasattr(self.llama, 'tokenizer'):
                tokenizer = self.llama.tokenizer
                
                if hasattr(tokenizer, 'vocab'):
                    vocab = tokenizer.vocab
                    
                    for token_str, token_id in vocab.items():
                        # Проверяем, содержит ли токен русские буквы
                        if re.search(r'[а-яА-ЯёЁ]', token_str):
                            # Фильтруем слишком длинные/короткие
                            if 2 <= len(token_str) <= 50:
                                self._vocab_ru[token_str] = token_id
                
                logger.info(f"Найдено {len(self._vocab_ru)} русских токенов")
            
            # Fallback - пробуем через токенизацию известных слов
            if not self._vocab_ru:
                test_words = [
                    'что', 'это', 'как', 'почему', 'когда', 'где', 'кто',
                    'снег', 'вода', 'огонь', 'земля', 'воздух', 'солнце', 'луна',
                    'человек', 'животное', 'растение', 'машина', 'дом', 'город',
                    'рука', 'глаз', 'голова', 'сердце', 'мозг',
                    'холодный', 'теплый', 'большой', 'маленький', 'быстрый', 'медленный',
                    'идти', 'бежать', 'летать', 'плавать', 'думать', 'говорить', 'слушать'
                ]
                
                for word in test_words:
                    try:
                        tokens = self.llama.tokenize(word.encode())
                        if tokens:
                            self._vocab_ru[word] = tokens[0]
                    except:
                        pass
                
                logger.info(f"Найдено {len(self._vocab_ru)} русских слов через токенизацию")
                
        except Exception as e:
            logger.error(f"Ошибка извлечения vocab: {e}")
        
        return self._vocab_ru
    
    def generate_knowledge(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> List[str]:
        """
        Сгенерировать ответы на список промтов.
        
        Args:
            prompts: Список промтов на русском языке
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (ниже = точнее)
            
        Returns:
            List of generated responses
        """
        if not self.llama:
            if not self.load_model():
                return []
        
        responses = []
        
        for prompt in prompts:
            try:
                output = self.llama(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=['\n\n', '---', '===']
                )
                
                response = output['choices'][0]['text'].strip()
                responses.append(response)
                
                logger.debug(f"Промт: {prompt[:50]}...")
                logger.debug(f"Ответ: {response[:100]}...")
                
            except Exception as e:
                logger.warning(f"Ошибка генерации: {e}")
                responses.append("")
        
        return responses
    
    def extract_spo(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Извлечь S-P-O тройки из текста.
        
        Returns:
            List of (subject, predicate, object)
        """
        triples = []
        
        for pattern in self.spo_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    # Определяем predicate по паттерну
                    predicate = self._infer_predicate(pattern)
                    
                    if len(match) == 2:
                        subject, obj = match
                    else:
                        subject, obj = match[0], match[-1]
                    
                    # Фильтруем слишком короткие/длинные
                    if 2 <= len(subject) <= 30 and 2 <= len(obj) <= 30:
                        triples.append((subject.strip(), predicate, obj.strip()))
        
        # Убираем дубликаты
        seen = set()
        unique_triples = []
        for triple in triples:
            key = triple[0].lower() + '|' + triple[1].lower() + '|' + triple[2].lower()
            if key not in seen:
                seen.add(key)
                unique_triples.append(triple)
        
        return unique_triples
    
    def _infer_predicate(self, pattern: str) -> str:
        """Определить тип отношения по паттерну."""
        if 'это' in pattern or 'является' in pattern:
            return 'is_a'
        elif 'состоит' in pattern or 'включает' in pattern:
            return 'part_of'
        elif 'имеет' in pattern or 'обладает' in pattern:
            return 'has_property'
        elif 'вызывает' in pattern or 'приводит' in pattern:
            return 'causes'
        elif 'связан' in pattern or 'ассоциируется' in pattern:
            return 'related_to'
        else:
            return 'related_to'
    
    def extract_knowledge_from_prompts(
        self,
        domain_prompts: Dict[str, List[str]],
        save_to_graph: bool = True
    ) -> List[KnowledgeEntry]:
        """
        Извлечь знания из набора промтов для разных доменов.
        
        Args:
            domain_prompts: Dict[domain, List[prompts]]
            save_to_graph: Сохранять ли в граф
            
        Returns:
            List of KnowledgeEntry
        """
        all_entries = []
        
        for domain, prompts in domain_prompts.items():
            logger.info(f"Обработка домена: {domain}")
            
            # Генерируем ответы
            responses = self.generate_knowledge(prompts)
            
            for prompt, response in zip(prompts, responses):
                if not response:
                    continue
                
                # Извлекаем S-P-O
                triples = self.extract_spo(response)
                
                for subject, predicate, obj in triples:
                    entry = KnowledgeEntry(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.7,  # Базовая уверенность
                        source_prompt=prompt,
                        model_response=response,
                        extraction_method='explicit' if triples else 'generated'
                    )
                    
                    all_entries.append(entry)
                    
                    # Сохраняем в граф
                    if save_to_graph:
                        try:
                            self.graph.add_knowledge(
                                subject=subject,
                                relation=predicate,
                                object_=obj,
                                subject_level=1,
                                object_level=1,
                                confidence=0.7
                            )
                        except Exception as e:
                            logger.debug(f"Не удалось сохранить: {e}")
        
        logger.info(f"Извлечено {len(all_entries)} знаний")
        
        return all_entries
    
    def build_concept_hierarchy(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Построить иерархию концепта через модель.
        
        Args:
            concept: Название концепта
            depth: Глубина спуска в иерархию
            
        Returns:
            Dict с иерархией и связями
        """
        hierarchy = {
            'concept': concept,
            'level': 0,
            'definitions': [],
            'attributes': [],
            'relations': [],
            'children': []
        }
        
        # 1. Определение
        def_prompt = f"Дай краткое определение понятия '{concept}'. Отвечай одним предложением."
        def_resp = self.generate_knowledge([def_prompt])
        if def_resp:
            hierarchy['definitions'].append(def_resp[0])
        
        # 2. Атрибуты
        attr_prompt = f"Перечисли основные атрибуты и свойства понятия '{concept}'. Формат: одно слово или короткая фраза."
        attr_resp = self.generate_knowledge([attr_prompt])
        if attr_resp:
            # Парсим атрибуты
            attrs = [a.strip() for a in attr_resp[0].split('\n') if a.strip()]
            hierarchy['attributes'] = attrs[:10]
        
        # 3. Связи
        rel_prompt = f"С какими понятиями связано '{concept}'? Перечисли 5-7 связанных понятий."
        rel_resp = self.generate_knowledge([rel_prompt])
        if rel_resp:
            rels = [r.strip() for r in rel_resp[0].split('\n') if r.strip()]
            hierarchy['relations'] = rels[:7]
        
        # 4. Рекурсивно для связей (если depth > 0)
        if depth > 0:
            for rel_concept in hierarchy['relations'][:3]:
                child = self.build_concept_hierarchy(rel_concept, depth - 1)
                child['level'] = hierarchy['level'] + 1
                hierarchy['children'].append(child)
        
        return hierarchy
    
    def save_to_graph_hierarchy(
        self,
        concept: str,
        hierarchy: Dict[str, Any],
        parent_node_id: str = None
    ):
        """Сохранить иерархию концепта в граф."""
        
        # Сохраняем сам концепт
        node = self.graph.add_node(
            content=hierarchy['concept'],
            node_type='concept',
            level=hierarchy['level'],
            confidence=0.8,
            metadata={
                'definitions': hierarchy.get('definitions', []),
                'attributes': hierarchy.get('attributes', []),
                'source': 'gguf_extraction'
            }
        )
        
        current_node_id = node.id
        
        # Сохраняем атрибуты как узлы уровня level+1
        for attr in hierarchy.get('attributes', []):
            try:
                self.graph.add_knowledge(
                    subject=hierarchy['concept'],
                    relation='has_property',
                    object_=attr,
                    subject_level=hierarchy['level'],
                    object_level=hierarchy['level'] + 1,
                    confidence=0.7
                )
            except:
                pass
        
        # Сохраняем связи
        for rel in hierarchy.get('relations', []):
            try:
                self.graph.add_knowledge(
                    subject=hierarchy['concept'],
                    relation='related_to',
                    object_=rel,
                    subject_level=hierarchy['level'],
                    object_level=hierarchy['level'],
                    confidence=0.6
                )
            except:
                pass
        
        # Рекурсивно для детей
        for child in hierarchy.get('children', []):
            self.save_to_graph_hierarchy(child['concept'], child, current_node_id)
    
    def create_russian_corpus_training(
        self,
        min_entities: int = 100,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Создать корпус русскоязычных знаний для обучения графа.
        
        Args:
            min_entities: Минимальное количество сущностей
            max_depth: Максимальная глубина иерархии
            
        Returns:
            Результат с статистикой
        """
        # Базовые концепты для русскоязычного корпуса
        base_concepts = [
            # Природа
            'снег', 'дождь', 'ветер', 'солнце', 'луна', 'звезда', 'небо', 'земля',
            'вода', 'огонь', 'воздух', 'камень', 'гора', 'река', 'море', 'озеро',
            'лес', 'дерево', 'трава', 'цветок', 'плод', 'семя',
            # Животные
            'собака', 'кошка', 'птица', 'рыба', 'лошадь', 'корова', 'свинья', 'овца',
            'медведь', 'волк', 'лиса', 'заяц', 'белка', 'еж',
            'человек', 'мужчина', 'женщина', 'ребенок', 'семья',
            # Время и пространство
            'время', 'пространство', 'движение', 'покой', 'начало', 'конец',
            'прошлое', 'настоящее', 'будущее',
            # Абстракции
            'мысль', 'чувство', 'эмоция', 'память', 'воображение', 'сознание',
            'знание', 'правда', 'ложь', 'добро', 'зло', 'красота',
            # Технологии
            'компьютер', 'телефон', 'интернет', 'программа', 'алгоритм',
            'данные', 'информация', 'знание', 'обучение'
        ]
        
        total_extracted = 0
        total_nodes = 0
        
        logger.info(f"Начало создания корпуса из {len(base_concepts)} концептов")
        
        for concept in base_concepts:
            try:
                logger.debug(f"Обработка: {concept}")
                
                # Строим иерархию
                hierarchy = self.build_concept_hierarchy(concept, depth=max_depth)
                
                # Сохраняем в граф
                self.save_to_graph_hierarchy(concept, hierarchy)
                
                # Считаем
                total_extracted += len(hierarchy.get('definitions', []))
                total_extracted += len(hierarchy.get('attributes', []))
                total_extracted += len(hierarchy.get('relations', []))
                total_nodes += 1
                
            except Exception as e:
                logger.warning(f"Ошибка для {concept}: {e}")
        
        # Векторизуем всё
        logger.info("Векторизация узлов...")
        self.graph.vectorize_all()
        self.graph.vectorize_groups()
        
        result = {
            'concepts_processed': total_nodes,
            'total_knowledge_extracted': total_extracted,
            'graph_stats': self.graph.get_stats()
        }
        
        logger.info(f"Корпус создан: {result}")
        
        return result


def create_extractor(model_path: str, graph, **kwargs) -> GGUFKnowledgeExtractor:
    """Фабричная функция."""
    return GGUFKnowledgeExtractor(model_path=model_path, graph=graph, **kwargs)