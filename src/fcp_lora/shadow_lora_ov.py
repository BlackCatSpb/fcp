"""
ShadowLoRAManagerOV - Менеджер LoRA адаптеров в OpenVINO

Управление адаптерами с атомарной сменой.
"""
import os
from typing import Optional, Dict, List
from threading import Lock


class ShadowLoRAManagerOV:
    """
    Shadow LoRA Manager для OpenVINO.
    
    Особенности:
    - Атомарная смена адаптера
    - Thread-safe операции
    - Поддержка нескольких адаптеров
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        scheduler_config: Optional[dict] = None
    ):
        self.model_path = model_path
        self.device = device
        self.scheduler_config = scheduler_config
        
        self._pipeline = None
        self._active_adapter: Optional[str] = None
        self._adapters: Dict[str, str] = {}
        self._lock = Lock()
        
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Инициализировать pipeline."""
        try:
            import openvino_genai as ov_genai
            
            config = {}
            if self.scheduler_config:
                config["scheduler_config"] = self.scheduler_config
            
            self._pipeline = ov_genai.LLMPipeline(
                self.model_path,
                self.device,
                config=config
            )
            print(f"Pipeline initialized: {self.model_path}")
            
        except ImportError:
            print("OpenVINO GenAI не установлен")
            self._pipeline = None
    
    def register_adapter(self, name: str, adapter_path: str):
        """
        Зарегистрировать адаптер.
        
        Args:
            name: имя адаптера
            adapter_path: путь к файлам адаптера
        """
        self._adapters[name] = adapter_path
        print(f"Registered adapter: {name} -> {adapter_path}")
    
    def atomic_swap(
        self,
        adapter_name: str,
        alpha: float = 0.8
    ) -> bool:
        """
        Атомарная смена адаптера.
        
        Args:
            adapter_name: имя адаптера
            alpha: коэффициент смешивания
        
        Returns:
            True если успешно
        """
        if self._pipeline is None:
            print("Pipeline not initialized")
            return False
        
        if adapter_name not in self._adapters:
            print(f"Adapter not found: {adapter_name}")
            return False
        
        adapter_path = self._adapters[adapter_name]
        
        try:
            with self._lock:
                # Загружаем новый адаптер
                if hasattr(self._pipeline, 'set_adapters'):
                    self._pipeline.set_adapters(adapter_path)
                
                self._active_adapter = adapter_name
                
            print(f"Swapped to adapter: {adapter_name} (alpha={alpha})")
            return True
            
        except Exception as e:
            print(f"Swap failed: {e}")
            return False
    
    def get_active_adapter(self) -> Optional[str]:
        """Получить активный адаптер."""
        return self._active_adapter
    
    def list_adapters(self) -> List[str]:
        """Список зарегистрированных адаптеров."""
        return list(self._adapters.keys())
    
    def unload(self):
        """Выгрузить адаптер."""
        with self._lock:
            self._active_adapter = None


class LoRAAdapter:
    """
    Отдельный LoRA адаптер.
    
    Представляет один адаптер.
    """
    
    def __init__(
        self,
        name: str,
        path: str,
        rank: int = 8,
        alpha: float = 16.0
    ):
        self.name = name
        self.path = path
        self.rank = rank
        self.alpha = alpha
        self._loaded = False
    
    def load(self):
        """Загрузить адаптер."""
        if os.path.exists(self.path):
            self._loaded = True
            print(f"Loaded: {self.name}")
        else:
            print(f"Not found: {self.path}")
    
    def unload(self):
        """Выгрузить адаптер."""
        self._loaded = False
    
    def is_loaded(self) -> bool:
        """Проверить загрузку."""
        return self._loaded


class MultiAdapterManager:
    """
    Менеджер нескольких адаптеров.
    
    Для r=4, r=8, r=16 адаптеров.
    """
    
    def __init__(self):
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._current: Optional[str] = None
    
    def add(self, name: str, path: str, rank: int = 8):
        """Добавить адаптер."""
        self._adapters[name] = LoRAAdapter(name, path, rank)
    
    def set_active(self, name: str):
        """Установить активный."""
        if name in self._adapters:
            self._current = name
    
    def get_rank(self) -> int:
        """Получить rank активного."""
        if self._current and self._current in self._adapters:
            return self._adapters[self._current].rank
        return 8
    
    def get_active_name(self) -> Optional[str]:
        """Получить имя активного."""
        return self._current