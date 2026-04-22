"""
Virtual Token System - Гибридный токенизатор с LogitsProcessor и Streaming

Архитектура:
1. VirtualTokenLogitsProcessor - управляет вероятностями виртуальных токенов
2. StreamingVirtualTokenHandler - заменяет <|node_xxx|> на контент из памяти
3. Интеграция с SnapshotManager для консистентности

Использование:
- Модель генерирует <|node_123|> как специальный токен
- LogitsProcessor повышает вероятность виртуальных токенов
- Streaming handler заменяет токены на реальный контент из памяти
"""

import logging
from typing import List, Dict, Optional, Any, Generator, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.fractal_graph_v2.virtual_tokens")


@dataclass
class VirtualTokenInfo:
    """Информация о виртуальном токене."""
    node_id: str
    token_ids: List[int]
    full_token: str
    content: str
    confidence: float


class VirtualTokenLogitsProcessor:
    """
    LogitsProcessor для управления вероятностями виртуальных токенов.
    
    Принцип работы:
    - Когда модель хочет сгенерировать виртуальный токен,
      повышаем вероятность его появления
    - Блокируем токены, которые уже были использованы (для избежания повторов)
    """
    
    def __init__(
        self,
        virtual_token_infos: List[VirtualTokenInfo],
        boost_factor: float = 50.0,
        block_used_tokens: bool = True
    ):
        """
        Args:
            virtual_token_infos: Список виртуальных токенов для усиления
            boost_factor: Множитель для logit виртуальных токенов
            block_used_tokens: Блокировать уже использованные токены
        """
        self.virtual_token_infos = virtual_token_infos
        self.boost_factor = boost_factor
        self.block_used_tokens = block_used_tokens
        
        self._token_id_to_info: Dict[int, VirtualTokenInfo] = {}
        self._used_tokens: Set[int] = set()
        
        for info in virtual_token_infos:
            for token_id in info.token_ids:
                self._token_id_to_info[token_id] = info
        
        logger.info(f"VirtualTokenLogitsProcessor: {len(virtual_token_infos)} tokens, "
                   f"{len(self._token_id_to_info)} token_ids")
    
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        """
        Обработка logits перед сэмплированием.
        
        Args:
            input_ids: Последние сгенерированные токены
            scores: Logits для каждого токена в словаре
            
        Returns:
            Модифицированные scores
        """
        last_token = input_ids[-1] if input_ids else None
        
        for token_id, info in self._token_id_to_info.items():
            if token_id < len(scores):
                if self.block_used_tokens and token_id in self._used_tokens:
                    scores[token_id] = float('-inf')
                else:
                    scores[token_id] += self.boost_factor
        
        return scores
    
    def mark_token_used(self, token_id: int):
        """Отметить токен как использованный."""
        self._used_tokens.add(token_id)
        if token_id in self._token_id_to_info:
            logger.debug(f"Token used: {self._token_id_to_info[token_id].full_token}")
    
    def reset_used_tokens(self):
        """Сбросить использованные токены."""
        self._used_tokens.clear()


class StreamingVirtualTokenHandler:
    """
    Обработчик потока генерации для замены виртуальных токенов.
    
    Принцип работы:
    - Накапливает текст из потока
    - При обнаружении полного виртуального токена <|node_xxx|>
      заменяет его на реальный контент из снимка памяти
    """
    
    def __init__(
        self,
        snapshot_or_contents: Any,
        virtual_prefix: str = "<|node_",
        virtual_suffix: str = "|>",
        max_content_length: int = 2000,
        raise_on_missing: bool = False
    ):
        """
        Args:
            snapshot_or_contents: SnapshotManager или Dict[str, str] с node_id -> content
            virtual_prefix: Префикс виртуального токена
            virtual_suffix: Суффикс виртуального токена
            max_content_length: Максимальная длина подставляемого контента
            raise_on_missing: Выбрасывать исключение при отсутствии узла
        """
        self.virtual_prefix = virtual_prefix
        self.virtual_suffix = virtual_suffix
        self.max_content_length = max_content_length
        self.raise_on_missing = raise_on_missing
        
        if hasattr(snapshot_or_contents, 'get_content'):
            self._snapshot = snapshot_or_contents
            self._contents_dict: Optional[Dict[str, str]] = None
        else:
            self._snapshot = None
            self._contents_dict = snapshot_or_contents
        
        self._accumulated = ""
        self._replaced_count = 0
        self._total_chars_replaced = 0
        
        self._injection_depth = 0
        self._max_injection_depth = 3
        
        logger.info("StreamingVirtualTokenHandler initialized")
    
    def process_stream(
        self, 
        stream: Generator[Dict, None, None]
    ) -> Generator[str, None, None]:
        """
        Обрабатывает поток генерации, заменяя виртуальные токены.
        
        Args:
            stream: Поток от LLM (create_completion с stream=True)
            
        Yields:
            Обработанные чанки текста
        """
        self._accumulated = ""
        
        for chunk in stream:
            text_chunk = self._extract_text(chunk)
            if not text_chunk:
                continue
            
            self._accumulated += text_chunk
            
            while self._virtual_prefix in self._accumulated:
                result = self._process_accumulated()
                if result:
                    yield result
                else:
                    break
            
            if self._accumulated:
                yield self._accumulated
                self._accumulated = ""
        
        if self._accumulated.strip():
            yield self._accumulated
    
    def _extract_text(self, chunk: Dict) -> str:
        """Извлекает текст из чанка."""
        try:
            if 'choices' in chunk:
                return chunk['choices'][0].get('text', '')
            return str(chunk)
        except Exception:
            return str(chunk)
    
    def _process_accumulated(self) -> Optional[str]:
        """Обрабатывает накопленный текст, заменяя виртуальные токены."""
        start_idx = self._accumulated.find(self.virtual_prefix)
        
        if start_idx == -1:
            return None
        
        end_idx = self._accumulated.find(
            self.virtual_suffix, 
            start_idx + len(self.virtual_prefix)
        )
        
        if end_idx == -1:
            return None
        
        before = self._accumulated[:start_idx]
        virtual_token = self._accumulated[start_idx:end_idx + len(self.virtual_suffix)]
        after = self._accumulated[end_idx + len(self.virtual_suffix):]
        
        node_id = self._extract_node_id(virtual_token)
        if not node_id:
            logger.warning(f"Invalid virtual token: {virtual_token}")
            self._accumulated = before + after
            return before
        
        content = self._get_content(node_id)
        if content is None:
            if self.raise_on_missing:
                raise ValueError(f"Node {node_id} not found in snapshot")
            logger.warning(f"Node {node_id} not found, skipping")
            self._accumulated = before + after
            return before
        
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        self._accumulated = before + content + after
        self._replaced_count += 1
        self._total_chars_replaced += len(virtual_token) - len(content)
        
        return before + content
    
    def _extract_node_id(self, token: str) -> Optional[str]:
        """Извлекает ID узла из виртуального токена."""
        if not token.startswith(self.virtual_prefix):
            return None
        if not token.endswith(self.virtual_suffix):
            return None
        
        return token[len(self.virtual_prefix):-len(self.virtual_suffix)]
    
    def _get_content(self, node_id: str) -> Optional[str]:
        """Получает контент узла из снимка или словаря."""
        if self._snapshot:
            return self._snapshot.get_content(node_id)
        return self._contents_dict.get(node_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику обработки."""
        return {
            "replaced_count": self._replaced_count,
            "total_chars_replaced": self._total_chars_replaced,
            "current_accumulated_len": len(self._accumulated)
        }


class VirtualTokenManager:
    """
    Менеджер виртуальных токенов - координирует работу LogitsProcessor и StreamingHandler.
    """
    
    def __init__(
        self,
        snapshot_or_contents: Any,
        llama_model: Any,
        virtual_prefix: str = "<|node_",
        virtual_suffix: str = "|>",
        max_tokens: int = 256
    ):
        """
        Args:
            snapshot_or_contents: SnapshotManager или Dict с контентом
            llama_model: Llama модель для токенизации
            virtual_prefix: Префикс виртуального токена
            virtual_suffix: Суффикс виртуального токена
            max_tokens: Максимальное количество токенов для инъекции
        """
        self.llama_model = llama_model
        self.virtual_prefix = virtual_prefix
        self.virtual_suffix = virtual_suffix
        
        self._logits_processor: Optional[VirtualTokenLogitsProcessor] = None
        self._streaming_handler: Optional[StreamingVirtualTokenHandler] = None
        self._setup_components(snapshot_or_contents, max_tokens)
        
        logger.info("VirtualTokenManager initialized")
    
    def _setup_components(self, snapshot_or_contents: Any, max_tokens: int):
        """Настраивает компоненты."""
        contents = self._extract_contents(snapshot_or_contents)
        
        if not contents:
            logger.warning("No contents for virtual tokens")
            return
        
        token_infos = []
        for node_id, content in contents.items():
            if not content:
                continue
            
            truncated = content[:max_tokens * 4]
            full_token = f"{self.virtual_prefix}{node_id}{self.virtual_suffix}"
            
            try:
                token_ids = self.llama_model.tokenize(
                    full_token.encode('utf-8'),
                    add_bos=False
                )
                
                token_infos.append(VirtualTokenInfo(
                    node_id=node_id,
                    token_ids=token_ids,
                    full_token=full_token,
                    content=truncated,
                    confidence=0.7
                ))
            except Exception as e:
                logger.warning(f"Failed to tokenize {node_id}: {e}")
        
        if token_infos:
            self._logits_processor = VirtualTokenLogitsProcessor(token_infos)
            self._streaming_handler = StreamingVirtualTokenHandler(snapshot_or_contents)
    
    def _extract_contents(self, snapshot_or_contents: Any) -> Dict[str, str]:
        """Извлекает содержимое из снимка или словаря."""
        if isinstance(snapshot_or_contents, dict):
            return snapshot_or_contents
        
        if hasattr(snapshot_or_contents, 'node_contents'):
            return snapshot_or_contents.node_contents
        
        if hasattr(snapshot_or_contents, 'get_content'):
            result = {}
            for node_id in getattr(snapshot_or_contents, 'node_contents', {}).keys():
                content = snapshot_or_contents.get_content(node_id)
                if content:
                    result[node_id] = content
            return result
        
        return {}
    
    def get_logits_processor(self) -> Optional[VirtualTokenLogitsProcessor]:
        """Получить LogitsProcessor для передачи в create_completion."""
        return self._logits_processor
    
    def create_streaming_handler(self, snapshot_or_contents: Any) -> Optional[StreamingVirtualTokenHandler]:
        """Создать новый StreamingHandler."""
        if not self._streaming_handler:
            return None
        
        return StreamingVirtualTokenHandler(
            snapshot_or_contents=snapshot_or_contents,
            virtual_prefix=self.virtual_prefix,
            virtual_suffix=self.virtual_suffix
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику."""
        stats = {
            "has_logits_processor": self._logits_processor is not None,
            "has_streaming_handler": self._streaming_handler is not None
        }
        
        if self._streaming_handler:
            stats["streaming_stats"] = self._streaming_handler.get_stats()
        
        return stats


def create_virtual_token_manager(
    snapshot_or_contents: Any,
    llama_model: Any,
    config: Optional[Dict[str, Any]] = None
) -> VirtualTokenManager:
    """
    Фабричная функция для создания VirtualTokenManager.
    
    Args:
        snapshot_or_contents: Snapshot или словарь с контентом
        llama_model: Llama модель
        config: Дополнительная конфигурация
        
    Returns:
        VirtualTokenManager instance
    """
    config = config or {}
    
    return VirtualTokenManager(
        snapshot_or_contents=snapshot_or_contents,
        llama_model=llama_model,
        virtual_prefix=config.get('virtual_prefix', '<|node_'),
        virtual_suffix=config.get('virtual_suffix', '|>'),
        max_tokens=config.get('max_injected_tokens', 256)
    )
