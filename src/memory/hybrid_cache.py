"""Hybrid Token Cache for FCP - Simplified from EVA."""
import os
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any
from collections import OrderedDict
import numpy as np

import logging
logger = logging.getLogger("fcp.cache")


class HybridTokenCache:
    """
    Гибридный кэш токенов: RAM → Disk
    Упрощённая версия для FCP.
    """
    
    def __init__(
        self,
        max_memory_tokens: int = 50000,
        disk_cache_dir: str = "cache/disk",
        hot_threshold: int = 5
    ):
        self.max_memory_tokens = max_memory_tokens
        self.disk_cache_dir = disk_cache_dir
        self.hot_threshold = hot_threshold
        
        # RAM tier: LRU cache
        self.ram_cache: OrderedDict = OrderedDict()
        self.ram_lock = threading.RLock()
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "disk_reads": 0,
            "disk_writes": 0,
            "evictions": 0
        }
        
        # Create disk cache dir
        os.makedirs(disk_cache_dir, exist_ok=True)
        
        logger.info(f"HybridCache: RAM={max_memory_tokens}, disk={disk_cache_dir}")
    
    def _make_key(self, text: str) -> str:
        """Create cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_ram(self, key: str) -> Optional[str]:
        """Get from RAM cache."""
        with self.ram_lock:
            if key in self.ram_cache:
                # Move to end (most recent)
                self.ram_cache.move_to_end(key)
                return self.ram_cache[key]
        return None
    
    def _put_ram(self, key: str, value: str):
        """Put to RAM cache with eviction."""
        with self.ram_lock:
            # Evict if full
            while len(self.ram_cache) >= self.max_memory_tokens:
                self.ram_cache.popitem(last=False)
                self.stats["evictions"] += 1
            
            self.ram_cache[key] = value
            self.ram_cache.move_to_end(key)
    
    def _get_disk(self, key: str) -> Optional[str]:
        """Get from disk cache."""
        path = os.path.join(self.disk_cache_dir, f"{key}.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats["disk_reads"] += 1
                    return data.get("text")
            except:
                pass
        return None
    
    def _put_disk(self, key: str, value: str):
        """Put to disk cache."""
        path = os.path.join(self.disk_cache_dir, f"{key}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({"text": value, "time": time.time()}, f)
            self.stats["disk_writes"] += 1
        except:
            pass
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response for query."""
        key = self._make_key(query)
        
        # Try RAM first
        result = self._get_ram(key)
        if result:
            self.stats["hits"] += 1
            logger.info(f"Cache HIT (RAM): {query[:30]}...")
            return result
        
        # Try disk
        result = self._get_disk(key)
        if result:
            self.stats["hits"] += 1
            # Promote to RAM
            self._put_ram(key, result)
            logger.info(f"Cache HIT (disk): {query[:30]}...")
            return result
        
        self.stats["misses"] += 1
        logger.info(f"Cache MISS: {query[:30]}...")
        return None
    
    def put(self, query: str, response: str):
        """Cache response for query."""
        key = self._make_key(query)
        
        # Put to RAM
        self._put_ram(key, response)
        
        # Put to disk (async in real system, sync here)
        self._put_disk(key, response)
        
        logger.info(f"Cache PUT: {query[:30]}... -> {response[:30]}...")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "disk_reads": self.stats["disk_reads"],
            "disk_writes": self.stats["disk_writes"],
            "evictions": self.stats["evictions"],
            "ram_size": len(self.ram_cache)
        }
    
    def clear(self):
        """Clear cache."""
        with self.ram_lock:
            self.ram_cache.clear()
        logger.info("Cache cleared")


def test_cache():
    """Test hybrid cache."""
    print("=" * 60)
    print("Hybrid Cache Test")
    print("=" * 60)
    
    cache = HybridTokenCache(max_memory_tokens=10, disk_cache_dir="C:/Users/black/OneDrive/Desktop/FCP/cache/disk")
    
    # Test get/miss
    result = cache.get("Привет")
    print(f"Get 'Привет': {result}")
    
    # Test put
    cache.put("Привет", "Привет! Я - нейросеть.")
    print(f"Put 'Привет'")
    
    # Test hit
    result = cache.get("Привет")
    print(f"Get 'Привет' again: {result}")
    
    # Stats
    print(f"\nStats: {cache.get_stats()}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_cache()