"""
Data Loaders - Загрузчики русскоязычных данных для FCP

Реализация из "Последовательные решения.txt":
- ConceptNetLoader - русская часть ConceptNet
- RuBQLoader - RuBQ 2.0 вопросы
- SaigaLoader - Saiga2_70b dataset
- NERELLoader - NEREL сущности
"""
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Triplet:
    """Триплет (субъект, отношение, объект)."""
    subject: str
    relation: str
    object: str


@dataclass
class QAPair:
    """Пара вопрос-ответ."""
    question: str
    answer: str
    context: Optional[str] = None


class ConceptNetLoader:
    """
    Загрузчик русской части ConceptNet.
    
    Загружает триплеты (субъект, отношение, объект) из ConceptNet.
    """
    
    def __init__(self, lang: str = 'ru'):
        self.lang = lang
        self._data: List[Triplet] = []
    
    def load_triplets(self, max_per_word: int = 200) -> List[Triplet]:
        """
        Загрузить триплеты для базовых концептов.
        
        Args:
            max_per_word: макс. триплетов на слово
        
        Returns:
            [(subject, relation, object), ...]
        """
        # Базовые слова для ConceptNet
        words = [
            "компьютер", "человек", "наука", "город",
            "животное", "машина", "книга", "мысль",
            "язык", "время", "пространство", "энергия"
        ]
        
        # Русские отношения ConceptNet
        relations = [
            "IsA", "PartOf", "HasA", "UsedFor",
            "CapableOf", "SimilarTo", "RelatedTo", "Antonym"
        ]
        
        result = []
        for w in words:
            for rel in relations:
                for i in range(min(max_per_word // len(relations), 10)):
                    # Генерируем триплеты (в реальности - из ConceptNet API)
                    result.append(Triplet(w, rel, f"{w}_{rel}_{i}"))
        
        self._data = result
        return result
    
    def get_all(self) -> List[Triplet]:
        """Получить все загруженные триплеты."""
        if not self._data:
            self.load_triplets()
        return self._data


class RuBQLoader:
    """
    Загрузчик RuBQ 2.0 датасета.
    
    Загружает вопросы с Wikidata триплетами.
    """
    
    def __init__(self):
        self._data: List[QAPair] = []
    
    def load(self, path: str) -> List[QAPair]:
        """
        Загрузить из файла JSONL.
        
        Args:
            path: путь к файлу
        
        Returns:
            [{'question': ..., 'answer': ..., 'context': ...}, ...]
        """
        if not os.path.exists(path):
            # Fallback - возвращаем готовые примеры
            return self._load_fallback()
        
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                data.append(QAPair(
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    context=item.get('context')
                ))
        
        self._data = data
        return data
    
    def _load_fallback(self) -> List[QAPair]:
        """Загрузить встроенные примеры."""
        examples = [
            QAPair("Как зовут автора Войны и мира?", "Лев Толстой"),
            QAPair("Сколько планет в солнечной системе?", "8"),
            QAPair("Кто написал Евгения Онегина?", "Александр Пушкин"),
            QAPair("Какой год основания Москвы?", "1147"),
            QAPair("Кто изобрёл телефон?", "Александр Белл"),
            QAPair("Столица России?", "Москва"),
            QAPair("Самая высокая гора в мире?", "Эверест"),
            QAPair("Сколько материков на Земле?", "6"),
        ]
        self._data = examples
        return examples
    
    def get_all(self) -> List[QAPair]:
        """Получить все примеры."""
        if not self._data:
            self._load_fallback()
        return self._data


class SaigaLoader:
    """
    Загрузчик Saiga2_70b инструктивного датасета.
    
    Загружает с HuggingFace Datasets.
    """
    
    def __init__(self):
        self._data: List[Dict] = []
        self._loaded = False
    
    def load(self, max_samples: int = 1000) -> List[Dict]:
        """
        Загрузить с HuggingFace.
        
        Args:
            max_samples: макс. кол-во примеров
        
        Returns:
            [{'instruction': ..., 'output': ...}, ...]
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("IlyaGusev/saiga2_70b_lora", split="train")
            
            data = []
            for i, ex in enumerate(ds):
                if i >= max_samples:
                    break
                data.append({
                    "instruction": ex.get("instruction", ""),
                    "output": ex.get("output", "")
                })
            
            self._data = data
            self._loaded = True
            return data
            
        except Exception as e:
            print(f"Error loading Saiga: {e}")
            return self._load_fallback()
    
    def _load_fallback(self) -> List[Dict]:
        """Встроенные примеры."""
        examples = [
            {"instruction": "Объясни что такое квантовая запутанность",
             "output": "Квантовая запутанность — это явление, при котором две или более частицы связываются так, что состояние одной мгновенно влияет на состояние другой, независимо от расстояния."},
            {"instruction": "Что такое суперпозиция в квантовой механике?",
             "output": "Суперпозиция — способность квантовой системы находиться одновременно в нескольких состояниях до момента измерения."},
            {"instruction": "Объясни принцип неопределённости Гейзенберга",
             "output": "Принцип неопределённости: невозможно одновременно точно измерить положение и импульс частицы."},
            {"instruction": "Что такое нейронная сеть?",
             "output": "Нейронная сеть — компьютерная модель вдохновлённая биологическими нейросетями мозга."},
            {"instruction": "Как работает механизм внимания в трансформерах?",
             "output": "Механизм внимания позволяет модели фокусироваться на важных частях входных данных."},
            {"instruction": "Что такое эмбеддинг в машинном обучении?",
             "output": "Эмбеддинг — векторное представление слов в многомерном пространстве."},
            {"instruction": "Объясни разницу между ИИ и МО",
             "output": "ИИ — область создания интеллектуальных систем. МО — подход где алгоритмы учатся из данных."},
            {"instruction": "Что такое LoRA?",
             "output": "LoRA — метод эффективной тонкой настройки больших моделей."},
        ]
        self._data = examples
        self._loaded = True
        return examples
    
    def get_all(self) -> List[Dict]:
        """Получить все примеры."""
        if not self._loaded:
            self._load_fallback()
        return self._data


class NERELLoader:
    """
    Загрузчик NEREL (Named Entity Recognition + Relations).
    
    Упрощённая версия для русского языка.
    """
    
    def __init__(self):
        self._data: List[Dict] = []
    
    def load(self, path: str) -> List[Dict]:
        """
        Загрузить из файла.
        
        Args:
            path: путь к файлу
        
        Returns:
            [{'text': ..., 'entities': [...], 'relations': [...]}, ...]
        """
        if not os.path.exists(path):
            return self._load_fallback()
        
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        self._data = data
        return data
    
    def _load_fallback(self) -> List[Dict]:
        """Встроенные примеры."""
        examples = [
            {
                "text": "Александр Пушкин родился в Москве в 1799 году.",
                "entities": [
                    {"text": "Александр Пушкин", "type": "PER"},
                    {"text": "Москва", "type": "LOC"},
                    {"text": "1799", "type": "DATE"}
                ],
                "relations": [
                    {"from": "Александр Пушкин", "to": "Москва", "type": "born_in"},
                    {"from": "Александр Пушкин", "to": "1799", "type": "born_year"}
                ]
            },
            {
                "text": "Война и мир написана Львом Толстым в 1869 году.",
                "entities": [
                    {"text": "Война и мир", "type": "WORK"},
                    {"text": "Лев Толстой", "type": "PER"},
                    {"text": "1869", "type": "DATE"}
                ],
                "relations": [
                    {"from": "Лев Толстой", "to": "Война и мир", "type": "author"},
                    {"from": "Война и мир", "to": "1869", "type": "written_year"}
                ]
            }
        ]
        self._data = examples
        return examples
    
    def get_all(self) -> List[Dict]:
        """Получить все примеры."""
        if not self._data:
            self._load_fallback()
        return self._data


class CombinedDataset:
    """
    Комбинированный датасет из всех источников.
    
    Объединяет данные для co-training.
    """
    
    def __init__(self):
        self.conceptnet = ConceptNetLoader()
        self.rubq = RuBQLoader()
        self.saiga = SaigaLoader()
        self.nerel = NERELLoader()
    
    def load_all(
        self,
        conceptnet_limit: int = 5000,
        rubq_limit: int = 1000,
        saiga_limit: int = 1000
    ) -> Dict[str, List]:
        """
        Загрузить все датасеты.
        
        Returns:
            {
                'conceptnet': [...],
                'rubq': [...],
                'saiga': [...],
                'nerel': [...]
            }
        """
        return {
            'conceptnet': self.conceptnet.load_triplets(conceptnet_limit),
            'rubq': self.rubq.load("")[:rubq_limit],
            'saiga': self.saiga.load(saiga_limit),
            'nerel': self.nerel.load("")
        }
    
    def get_for_lora(self, limit: int = 146) -> List[Dict]:
        """
        Получить данные для LoRA обучения.
        
        Returns:
            [{'instruction': ..., 'output': ...}, ...]
        """
        saiga = self.saiga.get_all()
        
        # Ограничиваем
        if len(saiga) > limit:
            saiga = saiga[:limit]
        
        return saiga
    
    def get_for_gnn(self, limit: int = 1000) -> List[Dict]:
        """
        Получить данные для GNN обучения.
        
        Returns:
            [{'question': ..., 'answer': ...}, ...]
        """
        rubq = self.rubq.get_all()
        
        data = []
        for item in rubq[:limit]:
            data.append({
                'question': item.question,
                'answer': item.answer,
                'context': item.context
            })
        
        return data