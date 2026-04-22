"""
EVAContainer - Единый формат файла EVA

Структура .eva файла:
```
[Header - 256 bytes]
├── magic: "EVA2" (4 bytes)
├── version: uint16
├── flags: uint16
├── graph_offset: uint64 (offset to graph section)
├── graph_size: uint64 (size of graph section)
├── model_offset: uint64 (offset to model section, 0 if external)
├── model_size: uint64
├── metadata_offset: uint64
├── metadata_size: uint64
└── reserved: padding to 256 bytes

[Metadata Section] (JSON)
├── tokenizer.hybrid.enabled: bool
├── tokenizer.virtual_token_range: [start, end]
├── tokenizer.node_to_virtual: {node_id: virtual_id}
├── graph.version: str
├── graph.checksum: sha256
└── creation_timestamp: float

[Graph Section] (zstd compressed)
├── nodes: [...]
├── edges: [...]
└── semantic_groups: [...]

[Model Section] (optional - external GGUF)
└── path to GGUF file or embedded GGUF data
```

Usage:
```python
container = EVAContainer.create(
    model_path="qwen-2.5-3b.gguf",
    fractal_graph=fg,
    virtual_token_mapping=tokenizer.node_to_virtual
)
container.save("brain.eva")

# Load
container = EVAContainer.load("brain.eva")
model = container.get_model()
graph = container.get_graph()
```
"""

import os
import json
import struct
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .storage import FractalGraphV2

logger = logging.getLogger("eva_ai.fractal_graph_v2.eva_container")

EVA_MAGIC = b'EVA2'
EVA_VERSION = 2
EVA_HEADER_SIZE = 256

METADATA_KEYS = {
    'tokenizer.hybrid.enabled',
    'tokenizer.virtual_token_range',
    'tokenizer.node_to_virtual',
    'tokenizer.virtual_start',
    'tokenizer.virtual_end',
    'graph.version',
    'graph.checksum',
    'graph.node_count',
    'graph.edge_count',
    'model.path',
    'model.type',
    'model.architecture',
    'creation_timestamp',
    'author',
    'description'
}


@dataclass
class ContainerHeader:
    """Заголовок .eva файла."""
    magic: bytes = EVA_MAGIC
    version: int = EVA_VERSION
    flags: int = 0
    graph_offset: int = 0
    graph_size: int = 0
    model_offset: int = 0
    model_size: int = 0
    metadata_offset: int = 0
    metadata_size: int = 0
    embedding_offset: int = 0
    embedding_size: int = 0
    
    def to_bytes(self) -> bytes:
        """Сериализовать в байты."""
        data = struct.pack(
            '<4s H H Q Q Q Q Q Q Q Q',
            self.magic,
            self.version,
            self.flags,
            self.graph_offset,
            self.graph_size,
            self.model_offset,
            self.model_size,
            self.metadata_offset,
            self.metadata_size,
            self.embedding_offset,
            self.embedding_size
        )
        
        if len(data) < EVA_HEADER_SIZE:
            padding = b'\x00' * (EVA_HEADER_SIZE - len(data))
            return data + padding
        return data[:EVA_HEADER_SIZE]
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ContainerHeader':
        """Десериализовать из байтов."""
        if len(data) < EVA_HEADER_SIZE:
            data = data + b'\x00' * (EVA_HEADER_SIZE - len(data))
        
        part1 = struct.unpack('<4s H H Q Q Q Q Q Q Q Q', data[:72])
        
        return cls(
            magic=part1[0],
            version=part1[1],
            flags=part1[2],
            graph_offset=part1[3],
            graph_size=part1[4],
            model_offset=part1[5],
            model_size=part1[6],
            metadata_offset=part1[7],
            metadata_size=part1[8],
            embedding_offset=part1[9],
            embedding_size=part1[10]
        )


class EVAContainer:
    """
    Единый контейнер EVA.
    
    Объединяет:
    - GGUF модель (或其路径)
    - FractalGraphV2 (сериализованный граф)
    - Виртуальные токены (маппинг)
    - Метаданные
    """
    
    FLAG_EXTERNAL_MODEL = 0x0001
    FLAG_EMBEDDED_MODEL = 0x0002
    FLAG_COMPRESSED = 0x0004
    FLAG_SIGNED = 0x0008
    
    def __init__(
        self,
        fractal_graph=None,
        model_path: str = None,
        model_data: bytes = None,
        metadata: Dict = None,
        virtual_token_mapping: Dict[str, int] = None,
        embeddings_cache: Dict[str, List[float]] = None
    ):
        self.fractal_graph = fractal_graph
        self.model_path = model_path
        self.model_data = model_data
        self.metadata = metadata or {}
        self.virtual_token_mapping = virtual_token_mapping or {}
        self.embeddings_cache = embeddings_cache or {}
        
        self.header = ContainerHeader()
    
    @classmethod
    def create(
        cls,
        fractal_graph,
        model_path: str = None,
        model_data: bytes = None,
        metadata: Dict = None,
        virtual_token_mapping: Dict[str, int] = None,
        embeddings_cache: Dict[str, List[float]] = None,
        compression: str = 'zstd'
    ) -> 'EVAContainer':
        """Создать контейнер."""
        container = cls(
            fractal_graph=fractal_graph,
            model_path=model_path,
            model_data=model_data,
            metadata=metadata,
            virtual_token_mapping=virtual_token_mapping,
            embeddings_cache=embeddings_cache
        )
        
        if model_path and os.path.exists(model_path):
            container._load_model_info()
        
        if virtual_token_mapping:
            container.metadata['tokenizer.virtual_token_range'] = [
                min(virtual_token_mapping.values()) if virtual_token_mapping else 100000,
                max(virtual_token_mapping.values()) if virtual_token_mapping else 200000
            ]
        
        container.metadata['creation_timestamp'] = time.time()
        container.metadata['graph.version'] = 'fractal_graph_v2'
        
        return container
    
    def _load_model_info(self):
        """Загрузить информацию о модели."""
        if self.model_path and os.path.exists(self.model_path):
            self.metadata['model.path'] = self.model_path
            self.metadata['model.type'] = 'gguf'
            self.metadata['model.size'] = os.path.getsize(self.model_path)
    
    def save(self, path: str) -> bool:
        """
        Сохранить контейнер в файл.
        
        Args:
            path: Путь к файлу .eva
            
        Returns:
            True если успешно
        """
        try:
            with open(path, 'wb') as f:
                header = ContainerHeader()
                
                graph_data = self._serialize_graph()
                metadata_data = self._serialize_metadata()
                
                if self.model_data:
                    model_data = self.model_data
                    flags = self.FLAG_EMBEDDED_MODEL | self.FLAG_COMPRESSED
                elif self.model_path and os.path.exists(self.model_path):
                    model_data = None
                    flags = self.FLAG_EXTERNAL_MODEL
                else:
                    model_data = None
                    flags = 0
                
                embeddings_data = self._serialize_embeddings()
                
                metadata_offset = EVA_HEADER_SIZE
                graph_offset = metadata_offset + len(metadata_data)
                model_offset = graph_offset + len(graph_data) if model_data else 0
                embeddings_offset = model_offset + len(model_data) if model_data else graph_offset + len(graph_data)
                
                header = ContainerHeader(
                    version=EVA_VERSION,
                    flags=flags,
                    graph_offset=graph_offset,
                    graph_size=len(graph_data),
                    model_offset=model_offset,
                    model_size=len(model_data) if model_data else 0,
                    metadata_offset=metadata_offset,
                    metadata_size=len(metadata_data),
                    embedding_offset=embeddings_offset,
                    embedding_size=len(embeddings_data)
                )
                
                f.write(header.to_bytes())
                f.write(metadata_data)
                f.write(graph_data)
                
                if model_data:
                    f.write(model_data)
                
                if embeddings_data:
                    f.write(embeddings_data)
            
            checksum = self._calculate_checksum(path)
            logger.info(f"EVA container saved: {path}, size={os.path.getsize(path)/1024/1024:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save EVA container: {e}")
            return False
    
    @classmethod
    def load(cls, path: str) -> Optional['EVAContainer']:
        """
        Загрузить контейнер из файла.
        
        Args:
            path: Путь к файлу .eva
            
        Returns:
            EVAContainer или None
        """
        try:
            with open(path, 'rb') as f:
                header_data = f.read(EVA_HEADER_SIZE)
                header = ContainerHeader.from_bytes(header_data)
                
                if header.magic != EVA_MAGIC:
                    logger.error(f"Invalid EVA file: {path}")
                    return None
                
                f.seek(header.metadata_offset)
                metadata_data = f.read(header.metadata_size)
                metadata = cls._deserialize_metadata(metadata_data)
                
                f.seek(header.graph_offset)
                graph_data = f.read(header.graph_size)
                fractal_graph = cls._deserialize_graph(graph_data, metadata.get('graph.version'))
                
                model_data = None
                if header.model_size > 0:
                    f.seek(header.model_offset)
                    model_data = f.read(header.model_size)
                
                embeddings_cache = {}
                if header.embedding_size > 0:
                    f.seek(header.embedding_offset)
                    embeddings_data = f.read(header.embedding_size)
                    embeddings_cache = cls._deserialize_embeddings(embeddings_data)
                
                virtual_token_mapping = metadata.get('tokenizer.node_to_virtual', {})
                
                container = cls(
                    fractal_graph=fractal_graph,
                    model_data=model_data,
                    metadata=metadata,
                    virtual_token_mapping=virtual_token_mapping,
                    embeddings_cache=embeddings_cache
                )
                container.header = header
                
                logger.info(f"EVA container loaded: {path}")
                return container
                
        except Exception as e:
            logger.error(f"Failed to load EVA container: {e}")
            return None
    
    def _serialize_graph(self) -> bytes:
        """Сериализовать граф."""
        if hasattr(self.fractal_graph, 'save_to_blob'):
            import gzip
            import zstandard as zstd
            try:
                blob = self.fractal_graph.save_to_blob(compression='zstd')
                return blob
            except:
                blob = self.fractal_graph.save_to_blob(compression='none')
                return gzip.compress(blob)
        
        import json
        data = {'nodes': {}, 'edges': {}, 'version': 'unknown'}
        
        if hasattr(self.fractal_graph, 'nodes'):
            for nid, node in self.fractal_graph.nodes.items():
                data['nodes'][nid] = {
                    'content': getattr(node, 'content', ''),
                    'type': getattr(node, 'node_type', 'unknown'),
                    'level': getattr(node, 'level', 0)
                }
        
        return json.dumps(data, ensure_ascii=False).encode('utf-8')
    
    @staticmethod
    def _deserialize_graph(graph_blob: bytes, version: str = None) -> Any:
        """Десериализовать граф."""
        if graph_blob is None:
            logger.error("graph_blob is None in _deserialize_graph")
            return None
        
        graph = FractalGraphV2()
        if graph.load_from_blob(graph_blob, compression='zstd'):
            return graph
        
        if graph.load_from_blob(graph_blob, compression='none'):
            return graph
        
        return None
    
    def _serialize_metadata(self) -> bytes:
        """Сериализовать метаданные."""
        data = {
            k: v for k, v in self.metadata.items()
            if k in METADATA_KEYS or k.startswith('eva.')
        }
        data['tokenizer.node_to_virtual'] = self.virtual_token_mapping
        data['graph.checksum'] = hashlib.sha256(
            self.fractal_graph.save_to_blob() if hasattr(self.fractal_graph, 'save_to_blob') else b''
        ).hexdigest()
        
        return json.dumps(data, ensure_ascii=False).encode('utf-8')
    
    @staticmethod
    def _deserialize_metadata(data: bytes) -> Dict:
        """Десериализовать метаданные."""
        return json.loads(data.decode('utf-8'))
    
    def _serialize_embeddings(self) -> bytes:
        """Сериализовать кэш эмбеддингов."""
        if not self.embeddings_cache:
            return b''
        
        import json
        return json.dumps(self.embeddings_cache, ensure_ascii=False).encode('utf-8')
    
    @staticmethod
    def _deserialize_embeddings(data: bytes) -> Dict[str, List[float]]:
        """Десериализовать кэш эмбеддингов."""
        if not data:
            return {}
        try:
            return json.loads(data.decode('utf-8'))
        except:
            return {}
    
    def _calculate_checksum(self, path: str) -> str:
        """Вычислить checksum файла."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def get_graph(self):
        """Получить граф из контейнера."""
        return self.fractal_graph
    
    def get_model_path(self) -> Optional[str]:
        """Получить путь к модели."""
        if self.model_path:
            return self.model_path
        return self.metadata.get('model.path')
    
    def get_model_data(self) -> Optional[bytes]:
        """Получить данные модели (если встроены)."""
        return self.model_data
    
    def get_virtual_token_mapping(self) -> Dict[str, int]:
        """Получить маппинг виртуальных токенов."""
        return self.virtual_token_mapping or self.metadata.get('tokenizer.node_to_virtual', {})
    
    def get_info(self) -> Dict[str, Any]:
        """Получить информацию о контейнере."""
        return {
            'version': self.header.version,
            'flags': self.header.flags,
            'graph_size': self.header.graph_size,
            'model_size': self.header.model_size,
            'metadata_size': self.header.metadata_size,
            'metadata': self.metadata,
            'virtual_token_count': len(self.virtual_token_mapping),
            'embeddings_count': len(self.embeddings_cache)
        }


def create_eva_container(
    fractal_graph,
    model_path: str = None,
    virtual_token_mapping: Dict[str, int] = None,
    metadata: Dict = None
) -> EVAContainer:
    """Фабричная функция."""
    return EVAContainer.create(
        fractal_graph=fractal_graph,
        model_path=model_path,
        metadata=metadata,
        virtual_token_mapping=virtual_token_mapping
    )


def load_eva_container(path: str) -> Optional[EVAContainer]:
    """Фабричная функция для загрузки."""
    return EVAContainer.load(path)