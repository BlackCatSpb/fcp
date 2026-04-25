"""
Tool Orchestrator - Toolformer интеграция (РЕАЛЬНЫЕ инструменты)

Инструменты: Calculator, WebSearch, DateTime, Weather, Translator
"""
import json
import re
import os
import subprocess
from typing import Dict, Any, Callable, Optional
import urllib.request
import urllib.parse
import urllib.error

# Import HTTP library for real requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


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
                result = eval(expr, {"__builtins__": {}}, {})
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        return "Invalid expression"


class WebSearchTool(Tool):
    """
    Веб-поиск через Exa API или DuckDuckGo.
    
    Поддерживает: Exa (рекомендуется), DuckDuckGo (бесплатный)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        self.use_exa = bool(self.api_key and HAS_REQUESTS)
    
    def execute(self, params: Dict) -> str:
        query = params.get("query", "")
        
        if not query:
            return "No query provided"
        
        # Try Exa first (recommended)
        if self.use_exa:
            return self._search_exa(query)
        
        # Fallback to DuckDuckGo
        return self._search_duckduckgo(query)
    
    def _search_exa(self, query: str) -> str:
        """Поиск через Exa API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "query": query,
                "num_results": 5,
                "type": "auto"
            }
            
            response = requests.post(
                "https://api.exa.ai/search",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                snippets = []
                for r in results.get("results", [])[:5]:
                    title = r.get("title", "No title")
                    url = r.get("url", "")
                    snippets.append(f"- {title} ({url})")
                
                return "Search results:\n" + "\n".join(snippets)
            else:
                return f"Exa error: {response.status_code}"
                
        except Exception as e:
            return f"Exa search error: {e}"
    
    def _search_duckduckgo(self, query: str) -> str:
        """Поиск через DuckDuckGo (бесплатный)."""
        try:
            # Using HTML parsing of DuckDuckGo
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")
            
            # Parse results
            results = []
            # Simple regex for titles and URLs
            pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)<'
            matches = re.findall(pattern, html)
            
            for url, title in matches[:5]:
                title = re.sub(r'<[^>]+>', '', title)
                results.append(f"- {title.strip()} ({url})")
            
            if results:
                return "Search results:\n" + "\n".join(results)
            else:
                return self._search_google(query)
                
        except Exception as e:
            # Try Google as last resort
            return self._search_google(query)
    
    def _search_google(self, query: str) -> str:
        """Простой Google-like поиск."""
        try:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read().decode("utf-8")
            
            # Extract snippets
            pattern = r'<span class="fXvOXb">([^<]+)</span>'
            matches = re.findall(pattern, html)[:3]
            
            if matches:
                return "Search results:\n" + "\n".join(f"- {m}" for m in matches)
            
            return "No results found"
            
        except Exception as e:
            return f"Search unavailable: {e}"


class DateTimeTool(Tool):
    """Получить текущее время/дату."""
    
    def execute(self, params: Dict) -> str:
        from datetime import datetime, timezone
        
        # Moscow time
        moscow_tz = None
        try:
            import pytz
            moscow_tz = pytz.timezone('Europe/Moscow')
        except ImportError:
            pass
        
        dt = datetime.now()
        
        # Format
        format_type = params.get("format", "full")
        
        if format_type == "date":
            return dt.strftime("%Y-%m-%d")
        elif format_type == "time":
            return dt.strftime("%H:%M:%S")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")


class WeatherTool(Tool):
    """
    Прогноз погоды через Open-Meteo API (бесплатный).
    """
    
    def __init__(self):
        self.default_lat = 55.7558  # Moscow
        self.default_lon = 37.6173
    
    def execute(self, params: Dict) -> str:
        location = params.get("location", "")
        
        # Parse coordinates or use default
        lat, lon = self._parse_location(location)
        
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get("current_weather", {})
                
                temp = current.get("temperature", "N/A")
                wind = current.get("windspeed", "N/A")
                code = current.get("weathercode", 0)
                
                description = self._weather_code_to_desc(code)
                
                return f"Weather: {description}, {temp}°C, wind: {wind} km/h"
            
            return f"Weather unavailable"
            
        except Exception as e:
            return f"Weather error: {e}"
    
    def _parse_location(self, location: str) -> tuple:
        """Parse location to lat/lon."""
        # Known cities
        cities = {
            "moscow": (55.7558, 37.6173),
            "spb": (59.9343, 30.3351),
            "london": (51.5074, -0.1278),
            "new york": (40.7128, -74.0060),
            "tokyo": (35.6762, 139.6503),
        }
        
        if location.lower() in cities:
            return cities[location.lower()]
        
        return self.default_lat, self.default_lon
    
    def _weather_code_to_desc(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            95: "Thunderstorm",
        }
        return codes.get(code, f"Weather code {code}")


class TranslatorTool(Tool):
    """
    Переводчик через MyMemory API (бесплатный, 5000 слов/день).
    """
    
    def __init__(self):
        self.default_target = "en"
        self.default_source = "auto"
    
    def execute(self, params: Dict) -> str:
        text = params.get("text", "")
        target = params.get("target", self.default_target)
        source = params.get("source", self.default_source)
        
        if not text:
            return "No text provided"
        
        try:
            # Use MyMemory API
            langpair = f"{source}|{target}"
            url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair={urllib.parse.quote(langpair)}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                response_data = data.get("responseData", {})
                
                translated = response_data.get("translatedText")
                
                if translated:
                    return translated
                
                return f"Translation failed: {response_data.get('match', 'unknown')}"
            
            return f"Translation error: {response.status_code}"
            
        except Exception as e:
            return f"Translation error: {e}"


class CalculatorAdvancedTool(Tool):
    """Продвинутый калькулятор с математическими функциями."""
    
    def execute(self, params: Dict) -> str:
        expr = params.get("expression", "")
        
        # Расширенные математические функции
        math_globals = {
            "__builtins__": {},
            "sin": lambda x: __import__("math").sin(x),
            "cos": lambda x: __import__("math").cos(x),
            "tan": lambda x: __import__("math").tan(x),
            "sqrt": lambda x: __import__("math").sqrt(x),
            "pow": lambda x, y: __import__("math").pow(x, y),
            "log": lambda x: __import__("math").log(x),
            "log10": lambda x: __import__("math").log10(x),
            "pi": __import__("math").pi,
            "e": __import__("math").e,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
        }
        
        try:
            result = eval(expr, math_globals, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class ToolOrchestrator:
    """
    Оркестратор инструментов для Toolformer.
    
    РЕАЛЬНЫЕ реализации всех инструментов!
    """
    
    def __init__(self, graph=None):
        self.graph = graph
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Зарегистрировать инструменты."""
        self.register_tool("calculator", CalculatorTool())
        self.register_tool("calculator_advanced", CalculatorAdvancedTool())
        self.register_tool("datetime", DateTimeTool())
        self.register_tool("weather", WeatherTool())
        self.register_tool("translator", TranslatorTool())
        
        # Web search - conditional
        web_search = WebSearchTool()
        # Register but might be limited
        self.register_tool("web_search", web_search)
    
    def register_tool(self, name: str, tool: Tool):
        """Зарегистрировать инструмент."""
        self.tools[name] = tool
    
    def process_response(self, text: str) -> str:
        """Обработать ответ модели и выполнить инструменты."""
        if not self._has_tool_call(text):
            return text
        
        tool_calls = self._extract_tool_calls(text)
        
        for call in tool_calls:
            tool_name = call.get("tool", "")
            params = call.get("params", {})
            
            if tool_name in self.tools:
                result = self.tools[tool_name].execute(params)
                
                text = text.replace(call.get("raw", ""), str(result))
                
                if self.graph:
                    self.graph.add_fact("tool_result", str(result))
        
        return text
    
    def _has_tool_call(self, text: str) -> bool:
        """Проверить есть ли tool call."""
        return "<|tool_call|>" in text or ("tool" in text and "{\n" in text)
    
    def _extract_tool_calls(self, text: str) -> list:
        """Извлечь tool calls из текста."""
        calls = []
        
        pattern = r'<\|tool_call\|>(.*?)</\|tool_end\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                call = json.loads(match)
                call["raw"] = f"<|tool_call|>{match}</|tool_end|>"
                calls.append(call)
            except json.JSONDecodeError:
                pass
        
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
    
    def list_tools(self) -> list:
        """Список доступных инструментов."""
        return list(self.tools.keys())


class ToolResultCache:
    """Кэш результатов инструментов."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
    
    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        if len(self.cache) >= self.max_size:
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()