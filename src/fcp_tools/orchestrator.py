"""
Tool Orchestrator - Toolformer интеграция

Инструменты: Calculator, WebSearch и др.
"""
import json
import re
import subprocess
from typing import Dict, Any, Callable, Optional


class Tool:
    """Базовый класс инструмента."""
    
    def execute(self, params: Dict) -> str:
        raise NotImplementedError


class CalculatorTool(Tool):
    """Калькулятор - вычисляет выражения."""
    
    def execute(self, params: Dict) -> str:
        expr = params.get("expression", "")
        
        # Безопасное вычисление
        allowed_chars = set("0123456789+-*/.() ")
        if all(c in allowed_chars for c in expr):
            try:
                # Используем eval в безопасном контексте
                result = eval(expr, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        return "Invalid expression"


class WebSearchTool(Tool):
    """Веб-поиск (заглушка - нужен API ключ)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def execute(self, params: Dict) -> str:
        query = params.get("query", "")
        
        if not self.api_key:
            # Fallback - возвращаем симуляцию
            return f"Search results for '{query}': [simulated results]"
        
        # Реальная реализация с API
        # import requests
        # response = requests.get(f"https://api.search.com/search?q={query}&key={self.api_key}")
        # return response.json()
        
        return f"Results for: {query}"


class DateTimeTool(Tool):
    """Получить текущее время/дату."""
    
    def execute(self, params: Dict) -> str:
        from datetime import datetime
        dt = datetime.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class WeatherTool(Tool):
    """Прогноз погоды (заглушка)."""
    
    def execute(self, params: Dict) -> str:
        location = params.get("location", "Moscow")
        return f"Weather in {location}: +15°C, partly cloudy"


class TranslatorTool(Tool):
    """Переводчик (заглушка)."""
    
    def execute(self, params: Dict) -> str:
        text = params.get("text", "")
        target = params.get("target", "en")
        
        return f"[Translated to {target}]: {text}"


class ToolOrchestrator:
    """
    Оркестратор инструментов для Toolformer.
    
    Обрабатывает ответы модели и выполняет инструменты.
    """
    
    def __init__(self, graph=None):
        self.graph = graph
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Зарегистрировать инструменты по умолчанию."""
        self.register_tool("calculator", CalculatorTool())
        self.register_tool("datetime", DateTimeTool())
        # Эти требуют API ключей - только fallback
        # self.register_tool("web_search", WebSearchTool())
        # self.register_tool("weather", WeatherTool())
    
    def register_tool(self, name: str, tool: Tool):
        """Зарегистрировать инструмент."""
        self.tools[name] = tool
    
    def process_response(self, text: str) -> str:
        """
        Обработать ответ модели и выполнить инструменты.
        
        Ищет <|tool_call|> токены и выполняет инструменты.
        
        Args:
            text: ответ модели
        
        Returns:
            обработанный текст
        """
        if not self._has_tool_call(text):
            return text
        
        # Найти все tool_calls
        tool_calls = self._extract_tool_calls(text)
        
        for call in tool_calls:
            tool_name = call.get("tool", "")
            params = call.get("params", {})
            
            if tool_name in self.tools:
                result = self.tools[tool_name].execute(params)
                
                # Заменить вызов на результат
                text = text.replace(call.get("raw", ""), str(result))
                
                # Сохранить в граф если есть
                if self.graph:
                    self.graph.add_fact("tool_result", str(result))
        
        return text
    
    def _has_tool_call(self, text: str) -> bool:
        """Проверить есть ли tool call."""
        return "<|tool_call|>" in text or "{" in text and "tool" in text
    
    def _extract_tool_calls(self, text: str) -> list:
        """Извлечь tool calls из текста."""
        calls = []
        
        # Паттерн для JSON в tool_call
        pattern = r'<\|tool_call\|>(.*?)</\|tool_end\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                call = json.loads(match)
                call["raw"] = f"<|tool_call|>{match}</|tool_end|>"
                calls.append(call)
            except json.JSONDecodeError:
                pass
        
        # Альтернативный паттерн
        if not calls:
            pattern = r'\{[^}]*"tool"[^}]*\}'
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    call = json.loads(match)
                    call["raw"] = match
                    calls.append(call)
                except json.JSONDecodeError:
                    pass
        
        return calls
    
    def call_tool(self, tool_name: str, params: Dict) -> str:
        """Напрямую вызвать инструмент."""
        if tool_name not in self.tools:
            return f"Tool not found: {tool_name}"
        
        return self.tools[tool_name].execute(params)


class ToolResultCache:
    """Кэш результатов инструментов."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
    
    def get(self, key: str) -> Optional[str]:
        """Получить результат."""
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """Сохранить результат."""
        if len(self.cache) >= self.max_size:
            # Удалить первый
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = value
    
    def clear(self):
        """Очистить кэш."""
        self.cache.clear()