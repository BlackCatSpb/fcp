"""
Web Search для FMF - упрощённый веб-поиск
Адаптировано из EVA-Ai
"""
import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any

logger = logging.getLogger("fmf.web_search")


def load_config() -> Dict:
    """Загружает конфигурацию."""
    possible_paths = [
        os.path.join(os.getcwd(), 'brain_config.json'),
        'C:\\Users\\black\\OneDrive\\Desktop\\EVA-Ai\\brain_config.json',
    ]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    return {}


def tavily_search(query: str, api_key: str = None, max_results: int = 5) -> Dict[str, Any]:
    """Поиск через Tavily API."""
    if not api_key:
        config = load_config()
        api_key = config.get('web_search', {}).get('tavily_api_key') or os.environ.get('TAVILY_API_KEY')
    
    if not api_key:
        return {"error": "API key не найден", "results": []}
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {"query": query, "max_results": max_results}
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}", "results": []}
            
    except requests.exceptions.Timeout:
        return {"error": "API timeout", "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}


def wikipedia_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Поиск в Wikipedia через их API."""
    try:
        url = "https://ru.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "pageid": item.get("pageid", 0),
                "url": f"https://ru.wikipedia.org/wiki?curid={item.get('pageid', 0)}",
                "source": "wikipedia"
            })
        
        return results
    
    except Exception as e:
        logger.warning(f"Wikipedia search error: {e}")
        return []


class FMFWebSearch:
    """
    Упрощённый поисковый движок для FMF.
    
    Использует:
    - Tavily API (основной)
    - Wikipedia (fallback)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir or os.path.join(os.getcwd(), '.web_cache')
        os.makedirs(self._cache_dir, exist_ok=True)
        
        self._cache: Dict[str, Dict] = {}
        self._stats = {
            "total_queries": 0,
            "successful": 0,
            "failed": 0
        }
    
    def search(self, query: str, max_results: int = 5, use_cache: bool = True) -> Dict[str, Any]:
        """
        Выполняет поиск.
        
        Args:
            query: Поисковый запрос
            max_results: Максимум результатов
            use_cache: Использовать кэш
            
        Returns:
            Dict с результатами
        """
        self._stats["total_queries"] += 1
        
        # Проверяем кэш
        if use_cache and query in self._cache:
            logger.info(f"Using cached result for: {query}")
            return self._cache[query]
        
        # Пробуем Tavily
        result = tavily_search(query, max_results=max_results)
        
        if "error" not in result and result.get("results"):
            self._stats["successful"] += 1
            
            formatted = {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "content": r.get("content", r.get("snippet", "")),
                        "url": r.get("url", ""),
                        "source": "tavily"
                    }
                    for r in result.get("results", [])[:max_results]
                ]
            }
            
            if use_cache:
                self._cache[query] = formatted
            
            return formatted
        
        # Fallback на Wikipedia
        self._stats["failed"] += 1
        wiki_results = wikipedia_search(query, max_results)
        
        if wiki_results:
            return {
                "query": query,
                "results": wiki_results
            }
        
        return {"error": "No results found", "results": []}
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику."""
        return self._stats
    
    def clear_cache(self):
        """Очищает кэш."""
        self._cache.clear()