"""
TokenStreamingAPI - API для работы с токенами OpenVINO моделей.

Позволяет:
1. Получать raw token IDs от модели
2. Продолжать генерацию с переданными токенами
3. Передавать рассуждения между моделями

Использование:
    api = TokenStreamingAPI(pipeline)
    tokens = api.generate_tokens(prompt, config)
    
    # Продолжить с токенами
    tokens_b = api.generate_continuing(prompt, tokens_a, config)
"""

import logging
from typing import List, Optional, Callable, Generator, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger("eva_ai.token_streaming")

try:
    import openvino_genai as ov_genai
except ImportError:
    ov_genai = None


@dataclass
class TokenResult:
    """Результат генерации токенов"""
    tokens: List[int]
    text: str
    generation_time: float
    tokens_count: int


class TokenStreamingAPI:
    """
    API для получения и передачи токенов между моделями.
    
    Позволяет получить raw token IDs вместо декодированного текста,
    что даёт возможность передать их в другую модель без потерь при конвертации.
    """
    
    def __init__(self, pipeline, tokenizer=None):
        """
        Args:
            pipeline: OpenVINO pipeline
            tokenizer: Токенизатор (опционально, получим из pipeline)
        """
        self._pipeline = pipeline
        self._tokenizer = tokenizer
        
        if not tokenizer and pipeline:
            try:
                self._tokenizer = pipeline.get_tokenizer()
            except Exception as e:
                logger.warning(f"Could not get tokenizer: {e}")
    
    @property
    def tokenizer(self):
        """Получить токенизатор"""
        if not self._tokenizer and self._pipeline:
            try:
                self._tokenizer = self._pipeline.get_tokenizer()
            except:
                pass
        return self._tokenizer
    
    def get_tokens_from_text(self, text: str) -> List[int]:
        """Преобразовать текст в токены"""
        if not self.tokenizer:
            logger.error("No tokenizer available")
            return []
        
        try:
            tokens = self.tokenizer.encode(text)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return []
    
    def get_text_from_tokens(self, tokens: List[int]) -> str:
        """Преобразовать токены в текст"""
        if not self.tokenizer:
            logger.error("No tokenizer available")
            return ""
        
        try:
            text = self.tokenizer.decode(tokens)
            return text
        except Exception as e:
            logger.error(f"Detokenization error: {e}")
            return ""
    
    def generate_with_token_tracking(
        self,
        prompt: str,
        config: Optional[ov_genai.GenerationConfig] = None,
        stream_callback: Optional[Callable[[str, List[int]], bool]] = None
    ) -> TokenResult:
        """
        Генерирует и отслеживает токены.
        
        Args:
            prompt: Промпт
            config: Конфиг генерации
            stream_callback: Callback(text, tokens) - вызывается для каждого чанка
            
        Returns:
            TokenResult с токенами и текстом
        """
        if not self._pipeline:
            logger.error("No pipeline available")
            return TokenResult(tokens=[], text="", generation_time=0.0, tokens_count=0)
        
        import time
        start_time = time.time()
        
        all_tokens = []
        full_text = []
        
        def token_callback(text: str) -> bool:
            """Callback от OpenVINO - получает текст"""
            full_text.append(text)
            
            # Преобразуем текст в токены
            tokens = self.get_tokens_from_text(text)
            all_tokens.extend(tokens)
            
            # Вызываем пользовательский callback если есть
            if stream_callback:
                return stream_callback(text, tokens)
            return False
        
        try:
            if config is None:
                config = ov_genai.GenerationConfig()
                config.max_new_tokens = 512
            
            self._pipeline.generate(prompt, config, streamer=token_callback)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
        
        generation_time = time.time() - start_time
        final_text = ''.join(full_text)
        
        return TokenResult(
            tokens=all_tokens,
            text=final_text,
            generation_time=generation_time,
            tokens_count=len(all_tokens)
        )
    
    def generate_continuing(
        self,
        prompt: str,
        prefix_tokens: List[int],
        prefix_reasoning: str = None,
        config: Optional[ov_genai.GenerationConfig] = None
    ) -> TokenResult:
        """
        Продолжает генерацию с токенами от предыдущей модели.
        
        Это ключевой метод для передачи состояния между моделями.
        
        Args:
            prompt: Базовый промпт
            prefix_tokens: Токены от Model A
            prefix_reasoning: Рассуждения Model A (опционально)
            config: Конфиг генерации
            
        Returns:
            TokenResult от Model B
        """
        # Преобразуем токены в текст
        prefix_text = self.get_text_from_tokens(prefix_tokens)
        
        # Формируем расширенный промпт
        if prefix_reasoning:
            extended_prompt = (
                f"{prompt}\n\n"
                f"Краткий ответ: {prefix_text}\n\n"
                f"Рассуждения: {prefix_reasoning}\n\n"
                f"Дай развёрнутый и проверенный ответ:"
            )
        else:
            extended_prompt = (
                f"{prompt}\n\n"
                f"Краткий ответ: {prefix_text}\n\n"
                f"Дай развёрнутый ответ:"
            )
        
        return self.generate_with_token_tracking(extended_prompt, config)
    
    def validate_and_expand(
        self,
        prompt: str,
        tokens_a: List[int],
        reasoning_a: str,
        config: Optional[ov_genai.GenerationConfig] = None
    ) -> TokenResult:
        """
        Верифицирует и расширяет ответ Model A.
        
        Model B получает:
        1. Токены от A (точное состояние)
        2. Рассуждения A (почему такой ответ)
        
        И проверяет/расширяет ответ.
        
        Args:
            prompt: Оригинальный промпт
            tokens_a: Токены от Model A
            reasoning_a: Рассуждения Model A
            config: Конфиг генерации
            
        Returns:
            Верифицированный TokenResult
        """
        return self.generate_continuing(
            prompt=prompt,
            prefix_tokens=tokens_a,
            prefix_reasoning=reasoning_a,
            config=config
        )


class DualModelStreaming:
    """
    Класс для управления двумя моделями с передачей токенов.
    
    Обеспечивает:
    1. Последовательную генерацию A → B
    2. Передачу токенов (без потерь)
    3. Передачу рассуждений для валидации
    """
    
    def __init__(self, pipeline_a, pipeline_b, tokenizer_a=None, tokenizer_b=None):
        """
        Args:
            pipeline_a: Pipeline Model A
            pipeline_b: Pipeline Model B
            tokenizer_a: Токенизатор Model A
            tokenizer_b: Токенизатор Model B
        """
        self.api_a = TokenStreamingAPI(pipeline_a, tokenizer_a)
        self.api_b = TokenStreamingAPI(pipeline_b, tokenizer_b)
        
        self._reasoning_a = ""
        self._tokens_a = []
        self._text_a = ""
    
    def generate(
        self,
        prompt: str,
        use_validation: bool = True,
        config_a: Optional[ov_genai.GenerationConfig] = None,
        config_b: Optional[ov_genai.GenerationConfig] = None
    ) -> Tuple[TokenResult, TokenResult]:
        """
        Генерирует двумя моделями последовательно.
        
        Args:
            prompt: Промпт
            use_validation: Передавать рассуждения в Model B
            config_a: Конфиг для Model A
            config_b: Конфиг для Model B
            
        Returns:
            (result_a, result_b)
        """
        # Model A - быстрый ответ
        result_a = self.api_a.generate_with_token_tracking(prompt, config_a)
        
        self._tokens_a = result_a.tokens
        self._text_a = result_a.text
        self._reasoning_a = ""  #提取 из result_a если нужно
        
        # Model B - c токенами от A (+ рассуждения если нужно)
        if use_validation and self._reasoning_a:
            result_b = self.api_b.generate_continuing(
                prompt=prompt,
                prefix_tokens=self._tokens_a,
                prefix_reasoning=self._reasoning_a,
                config=config_b
            )
        else:
            result_b = self.api_b.generate_continuing(
                prompt=prompt,
                prefix_tokens=self._tokens_a,
                prefix_reasoning=None,
                config=config_b
            )
        
        return result_a, result_b
    
    @property
    def reasoning_a(self) -> str:
        """Получить рассуждения Model A"""
        return self._reasoning_a
    
    @property
    def tokens_a(self) -> List[int]:
        """Получить токены Model A"""
        return self._tokens_a
    
    @property
    def text_a(self) -> str:
        """Получить текст Model A"""
        return self._text_a


def create_token_api(pipeline, tokenizer=None) -> TokenStreamingAPI:
    """Factory function для создания TokenStreamingAPI"""
    return TokenStreamingAPI(pipeline, tokenizer)


def create_dual_streaming(
    pipeline_a,
    pipeline_b,
    tokenizer_a=None,
    tokenizer_b=None
) -> DualModelStreaming:
    """Factory function для создания DualModelStreaming"""
    return DualModelStreaming(pipeline_a, pipeline_b, tokenizer_a, tokenizer_b)