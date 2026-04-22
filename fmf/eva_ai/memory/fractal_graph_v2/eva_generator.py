"""
EVAGenerator - Единый генератор для EVA с гибридными токенами

Интегрирует:
- HybridTokenizer (BPE + виртуальные токены)
- RecursiveModelPipeline (модели A/B/C)
- GGUFShadowProfiler (маршрутизация)
- SemanticContextCache (необработанный контекст)
- Quality checks (оценка качества)

Принцип работы:
1. Входной текст → HybridTokenizer → BPE + виртуальные токены
2. SemanticContextCache → семантический поиск по необработанному контексту
3. Виртуальные токены → маршрутизация через GGUFShadowProfiler
4. Генерация через RecursiveModelPipeline
5. Постобработка: замена виртуальных токенов на контент узлов
"""

import os
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .hybrid_tokenizer import HybridTokenizer, Token
from .gguf_shadow import GGUFShadowProfiler
from .semantic_context_cache import SemanticContextCache
from .prompt_templates import SYSTEM_PROMPTS, FEW_SHOT_EXAMPLES, REASONING_CHAIN_PROMPT

logger = logging.getLogger("eva_ai.fractal_graph_v2.eva_generator")

QUERY_TYPE_KEYWORDS = {
    'кратко': ['кратко', 'вкратце', 'суть', 'что такое', 'кто такой', 'дай определение', 'назови', 'перечисли', 'brief'],
    'подробно': ['подробно', 'детально', 'развернуто', 'расскажи', 'объясни', 'опиши', 'проанализируй', 'объясни разницу', 'difference'],
    'technical': ['технич', 'как работает', 'механизм', 'algorithm', 'how does'],
    'comparison': ['сравни', 'различия', 'vs', 'versus', 'difference', 'compared']
}

GENERATION_PARAMS = {
    'кратко': {
        'max_tokens': 48,
        'temperature': 0.1,
        'use_chain': False
    },
    'подробно': {
        'max_tokens': 256,
        'temperature': 0.4,
        'use_chain': False
    },
    'technical': {
        'max_tokens': 192,
        'temperature': 0.3,
        'use_chain': False
    },
    'comparison': {
        'max_tokens': 320,
        'temperature': 0.4,
        'use_chain': False
    },
    'default': {
        'max_tokens': 128,
        'temperature': 0.3,
        'use_chain': False
    }
}


@dataclass
class GenerationRequest:
    """Запрос на генерацию."""
    text: str
    query_type: str = 'подробно'
    conversation_history: List[Dict] = None
    max_tokens: int = 512
    temperature: float = 0.5
    session_id: str = None


@dataclass
class GenerationResult:
    """Результат генерации."""
    response: str
    confidence: float
    quality_score: float
    virtual_tokens_used: List[str]
    reasoning_steps: List[Dict]
    processing_time: float
    query_type: str


class EVAGenerator:
    """
    Единый генератор EVA с поддержкой виртуальных токенов.
    
    Использует:
    - HybridTokenizer для токенизации с виртуальными токенами
    - SemanticContextCache для семантического поиска по необработанному контексту
    - RecursiveModelPipeline для генерации (опционально)
    - GGUFShadowProfiler для маршрутизации
    """
    
    def __init__(
        self,
        fractal_graph,
        model_pipeline=None,
        gguf_shadow=None,
        base_tokenizer=None,
        semantic_cache: SemanticContextCache = None,
        max_semantic_contexts: int = 500,
        llama_a=None,
        llama_b=None,
        n_ctx: int = 2048,
        n_threads: int = None  # По умолчанию: все ядра CPU
    ):
        """
        Args:
            fractal_graph: FractalGraphV2 instance
            model_pipeline: RecursiveModelPipeline (опционально)
            gguf_shadow: GGUFShadowProfiler (опционально)
            base_tokenizer: Базовый BPE токенизатор (опционально)
            semantic_cache: SemanticContextCache (опционально)
            max_semantic_contexts: Максимум контекстов в SemanticContextCache
            llama_a: Llama модель для Model A (опционально)
            llama_b: Llama модель для Model B (опционально)
            n_ctx: Размер контекста
            n_threads: Количество потоков
        """
        self.graph = fractal_graph
        self.pipeline = model_pipeline
        self.gguf_shadow = gguf_shadow
        self.llama_a = llama_a
        self.llama_b = llama_b
        self.n_ctx = n_ctx
        self.n_threads = n_threads or os.cpu_count() or 4  # Все ядра CPU
        
        self.tokenizer = HybridTokenizer(
            fractal_graph=fractal_graph,
            base_tokenizer=base_tokenizer
        )
        
        self.semantic_cache = semantic_cache
        if self.semantic_cache is None:
            self.semantic_cache = SemanticContextCache(
                max_contexts=max_semantic_contexts,
                embedding_dim=384,
                use_faiss=True
            )
        
        self.total_generations = 0
        self.total_quality_checks = 0
        
        logger.info(f"EVAGenerator инициализирован: llama_a={llama_a is not None}, llama_b={llama_b is not None}")
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Генерация ответа.
        
        Args:
            request: GenerationRequest с текстом и параметрами
            
        Returns:
            GenerationResult
        """
        start_time = time.time()
        self.total_generations += 1
        
        reasoning_steps = []
        
        reasoning_steps.append({
            'step': 1,
            'phase': 'tokenization',
            'action': 'Токенизация с виртуальными токенами'
        })
        
        tokens = self.tokenizer.encode(request.text)
        virtual_tokens = [t for t in tokens if t.is_virtual]
        virtual_node_ids = [t.node_id for t in virtual_tokens if t.node_id]
        
        reasoning_steps.append({
            'step': 2,
            'phase': 'entity_extraction',
            'action': f'Извлечено {len(virtual_node_ids)} сущностей из графа'
        })
        
        reasoning_steps.append({
            'step': 3,
            'phase': 'semantic_search',
            'action': 'Поиск в SemanticContextCache'
        })
        
        semantic_results = self.semantic_cache.search(
            request.text, 
            top_k=10,  # Динамически определяется в cache.search
            min_similarity=0.4,
            session_filter=getattr(request, 'session_id', None)
        )
        
        reasoning_steps.append({
            'step': 3.5,
            'phase': 'semantic_results',
            'action': f'Найдено {len(semantic_results)} релевантных контекстов'
        })
        
        query_type = getattr(request, 'query_type', None)
        if query_type is None or query_type == 'default':
            query_type = self._determine_query_type(request.text)
        reasoning_steps.append({
            'step': 4,
            'phase': 'query_type_detection',
            'action': f'Тип запроса: {query_type}'
        })
        
        params = self._get_generation_params(query_type)
        
        routing_config = None
        if self.gguf_shadow and virtual_node_ids:
            routing_config = self._get_routing_for_entities(virtual_node_ids)
            if routing_config:
                params = self._merge_params(params, routing_config)
                reasoning_steps.append({
                    'step': 5,
                    'phase': 'routing_applied',
                    'action': 'Применены параметры маршрутизации из графа'
                })
        
        prompt = self._build_prompt(
            request.text, 
            request.conversation_history, 
            virtual_node_ids,
            semantic_results,
            query_type=query_type
        )
        reasoning_steps.append({
            'step': 6,
            'phase': 'prompt_building',
            'action': f'Промпт сформирован ({len(prompt)} символов)'
        })
        
        response = self._generate_response(prompt, params)
        reasoning_steps.append({
            'step': 6,
            'phase': 'generation',
            'action': f'Сгенерировано {len(response)} символов'
        })
        
        quality = self._check_quality(response, request.text)
        self.total_quality_checks += 1
        reasoning_steps.append({
            'step': 7,
            'phase': 'quality_check',
            'action': f'Качество: {quality["score"]:.2f}'
        })
        
        if not quality.get('is_gibberish', False):
            response = self._sanitize_response(response, query_type)
        
        response = self._postprocess_virtual_tokens(response)
        reasoning_steps.append({
            'step': 8,
            'phase': 'postprocessing',
            'action': 'Заменены виртуальные токены на контент'
        })
        
        return GenerationResult(
            response=response,
            confidence=quality.get('score', 0.7),
            quality_score=quality.get('score', 0.7),
            virtual_tokens_used=virtual_node_ids,
            reasoning_steps=reasoning_steps,
            processing_time=time.time() - start_time,
            query_type=query_type
        )
    
    def _determine_query_type(self, text: str) -> str:
        """Определить тип запроса."""
        text_lower = text.lower()
        
        for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return qtype
        
        return 'default'
    
    def _get_generation_params(self, query_type: str) -> Dict[str, Any]:
        """Получить параметры генерации."""
        return GENERATION_PARAMS.get(query_type, GENERATION_PARAMS['подробно'])
    
    def _get_routing_for_entities(self, entity_ids: List[str]) -> Optional[Dict[str, Any]]:
        """Получить параметры маршрутизации для сущностей."""
        if not self.gguf_shadow:
            return None
        
        try:
            for entity_id in entity_ids:
                if entity_id in self.graph.nodes:
                    node = self.graph.nodes[entity_id]
                    metadata = getattr(node, 'metadata', {})
                    
                    routing = metadata.get('action', {})
                    if routing:
                        return routing.get('parameters')
            
            return None
        except Exception as e:
            logger.debug(f"Routing error: {e}")
            return None
    
    def _merge_params(self, base_params: Dict, routing_params: Dict) -> Dict:
        """Слить параметры базовые и маршрутизации."""
        merged = {}
        for model_key, model_params in base_params.items():
            if isinstance(model_params, dict):
                merged[model_key] = {**model_params}
                if isinstance(routing_params, dict):
                    for k, v in routing_params.items():
                        if k in ['temperature', 'max_tokens', 'top_p', 'repeat_penalty']:
                            merged[model_key][k] = v
        return merged
    
    def _build_prompt(
        self, 
        text: str, 
        history: Optional[List[Dict]],
        entity_ids: List[str],
        semantic_results: List[Dict] = None,
        query_type: str = "default"
    ) -> str:
        """Построить промпт с контекстом используя шаблоны."""
        semantic_results = semantic_results or []
        
        graph_context = self._get_graph_context(entity_ids)
        semantic_context = self._format_semantic_results(semantic_results)
        conversation_history = self._format_history(history)
        entity_context = self._get_entity_context(entity_ids)
        
        template = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["default"])
        
        try:
            prompt = template.format(
                graph_context=graph_context or "Нет данных из графа",
                semantic_context=semantic_context or "Нет дополнительных контекстов",
                conversation_history=conversation_history or "История пуста",
                entity_context=entity_context or "Нет данных об сущностях",
                query=text
            )
        except KeyError:
            prompt = self._build_simple_prompt(
                text, history, entity_ids, semantic_results
            )
        
        return prompt
    
    def _build_simple_prompt(
        self, 
        text: str, 
        history: Optional[List[Dict]],
        entity_ids: List[str],
        semantic_results: List[Dict] = None
    ) -> str:
        """Построить простой промпт без шаблонов."""
        parts = []
        semantic_results = semantic_results or []
        
        if semantic_results:
            semantic_contexts = [r['text'] for r in semantic_results[:3]]
            parts.append(f"Релевантные контексты: {' | '.join(semantic_contexts)}")
        
        if entity_ids:
            entity_contents = []
            for eid in entity_ids[:5]:
                if eid in self.graph.nodes:
                    node = self.graph.nodes[eid]
                    content = getattr(node, 'content', '')
                    if content:
                        entity_contents.append(content)
            
            if entity_contents:
                parts.append(f"Контекст из графа: {', '.join(entity_contents[:3])}")
        
        if history:
            recent = history[-5:]
            history_parts = []
            for msg in recent:
                role = 'Пользователь' if msg.get('role') == 'user' else 'Ассистент'
                content = msg.get('content', '')[:200]
                if content:
                    history_parts.append(f"{role}: {content}")
            
            if history_parts:
                parts.append(f"История: {' | '.join(history_parts)}")
        
        parts.append(f"Вопрос: {text}")
        
        return "\n\n".join(parts)
    
    def _get_graph_context(self, entity_ids: List[str]) -> str:
        """Получить контекст из графа для промпта."""
        if not entity_ids:
            return ""
        
        contexts = []
        for eid in entity_ids[:5]:
            if eid in self.graph.nodes:
                node = self.graph.nodes[eid]
                content = getattr(node, 'content', '')
                node_type = getattr(node, 'node_type', 'unknown')
                if content:
                    contexts.append(f"[{node_type}] {content}")
        
        return "\n".join(contexts)
    
    def _get_entity_context(self, entity_ids: List[str]) -> str:
        """Получить контекст сущностей."""
        return self._get_graph_context(entity_ids)
    
    def _format_semantic_results(self, results: List[Dict]) -> str:
        """Форматировать семантические результаты."""
        if not results:
            return ""
        
        formatted = []
        for r in results[:3]:
            text = r.get('text', '')
            score = r.get('similarity', 0)
            if text:
                formatted.append(f"({score:.2f}) {text[:150]}")
        
        return "\n".join(formatted)
    
    def _format_history(self, history: Optional[List[Dict]]) -> str:
        """Форматировать историю разговора."""
        if not history:
            return ""
        
        formatted = []
        for msg in history[-5:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]
            if content:
                role_str = 'Пользователь' if role == 'user' else 'EVA'
                formatted.append(f"{role_str}: {content}")
        
        return "\n".join(formatted)
    
    def _generate_response(self, prompt: str, params: Dict) -> str:
        """Сгенерировать ответ."""
        if self.pipeline:
            try:
                result = self.pipeline.process_query(
                    query=prompt,
                    gen_params=params
                )
                return result.get('model_b_result', {}).get('natural_response', '') or result.get('response', '')
            except Exception as e:
                logger.warning(f"Pipeline generation failed: {e}")
        
        if self.llama_a or self.llama_b:
            return self._generate_with_llama(prompt, params)
        
        return self._generate_fallback_response(prompt)
    
    def _generate_with_llama(self, prompt: str, params: Dict) -> str:
        """Генерация с использованием llama_cpp моделей."""
        try:
            use_chain = params.get('use_chain', True)
            max_tokens = params.get('max_tokens', 256)
            temperature = params.get('temperature', 0.3)
            
            if self.llama_a:
                response_a = self._call_llama(
                    self.llama_a,
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                return self._generate_fallback_response(prompt)
            
            if use_chain and self.llama_b and response_a:
                full_prompt = f"Вопрос: {prompt}\n\nОтвет: {response_a}\n\nРасширь и дополни ответ, сохраняя структуру:"
                response_b = self._call_llama(
                    self.llama_b,
                    full_prompt,
                    max_tokens=max(max_tokens, 256),
                    temperature=temperature
                )
                return response_b if response_b else response_a
            
            return response_a if response_a else "Модель не готова к генерации."
            
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _call_llama(self, llama_model, prompt: str, max_tokens: int = 256, temperature: float = 0.5) -> str:
        """Вызов llama_cpp модели."""
        try:
            output = llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "User:", "user:", "Human:", "human:"],
                echo=False
            )
            
            if isinstance(output, dict):
                return output.get('choices', [{}])[0].get('text', '')
            return str(output)
            
        except Exception as e:
            logger.error(f"Error calling llama model: {e}")
            return ""
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Fallback генерация без модели."""
        if 'контекст из графа' in prompt.lower():
            parts = prompt.split('\n\n')
            for part in parts:
                if part.startswith('Контекст из графа:'):
                    context = part.replace('Контекст из графа:', '').strip()
                    if context:
                        return f"Основываясь на имеющихся данных: {context}"
        
        return "Информация обрабатывается. Пожалуйста, уточните вопрос."
    
    def _check_quality(self, response: str, query: str) -> Dict[str, Any]:
        """Проверить качество ответа."""
        if not response or len(response.strip()) < 5:
            return {'score': 0.1, 'is_gibberish': True, 'reasons': ['Пустой ответ']}
        
        is_gibberish = False
        reasons = []
        score = 0.8
        
        words = response.split()
        if len(words) > 10:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.2:
                is_gibberish = True
                reasons.append('Сильные повторения')
                score = 0.3
        
        vowels_cyrillic = set('аеёиоуыэюя')
        vowels_english = set('aeiou')
        all_vowels = vowels_cyrillic | vowels_english
        vowel_count = sum(1 for c in response.lower() if c in all_vowels)
        if vowel_count < len(response) * 0.1:
            is_gibberish = True
            reasons.append('Подозрительно мало гласных')
            score = 0.2
        
        if not is_gibberish:
            if len(response) < 30:
                score = 0.5
                reasons.append('Слишком короткий')
            elif len(response) > 50:
                score = 0.9
                reasons.append('Хороший объём')
            else:
                reasons.append('OK')
        
        return {'score': score, 'is_gibberish': is_gibberish, 'reasons': reasons}
    
    def _sanitize_response(self, response: str, query_type: str = "default") -> str:
        """Очистить ответ от артефактов и ограничить длину."""
        filler_prefixes = [
            'хорошо,', 'давайте', 'начнём', 'итак,', 'что ж,', 'конечно,',
            'ok,', 'okay,', 'well,', 'sure,', 'of course,', 'разумеется,',
            'отлично,', 'понял,', 'понятно,'
        ]
        
        lines = response.split('\n')
        if lines:
            first_line_lower = lines[0].lower().strip()
            for prefix in filler_prefixes:
                if first_line_lower.startswith(prefix):
                    lines[0] = lines[0][len(prefix):].strip()
                    break
        
        response = '\n'.join(line for line in lines if line.strip())
        
        sentences = response.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        seen = {}
        for i, sent in enumerate(sentences):
            sent_norm = ' '.join(sent.lower().split())
            if sent_norm in seen and i - seen[sent_norm] >= 2:
                sentences = sentences[:i]
                break
            else:
                seen[sent_norm] = i
        
        result = '. '.join(sentences)
        
        if query_type == 'кратко':
            result = self._truncate_to_sentences(result, max_sentences=2)
        
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _truncate_to_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Обрезать текст до max_sentences предложений."""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()][:max_sentences]
        return '. '.join(sentences)
    
    def _postprocess_virtual_tokens(self, response: str) -> str:
        """Заменить виртуальные токены на контент узлов."""
        virtual_pattern = r'<virtual_(\d+)>'
        
        def replace_virtual(match):
            token_id = int(match.group(1))
            info = self.tokenizer.get_virtual_token_info(token_id)
            if info:
                return f"[{info['content']}]"
            return match.group(0)
        
        return re.sub(virtual_pattern, replace_virtual, response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            'total_generations': self.total_generations,
            'total_quality_checks': self.total_quality_checks,
            'virtual_token_range': self.tokenizer.get_stats(),
            'pipeline_available': self.pipeline is not None,
            'gguf_shadow_available': self.gguf_shadow is not None
        }


def create_eva_generator(
    fractal_graph,
    model_pipeline=None,
    gguf_shadow=None,
    base_tokenizer=None
) -> EVAGenerator:
    """Фабричная функция."""
    return EVAGenerator(
        fractal_graph=fractal_graph,
        model_pipeline=model_pipeline,
        gguf_shadow=gguf_shadow,
        base_tokenizer=base_tokenizer
    )