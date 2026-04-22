"""
GGUF Model Parser - Парсинг GGUF моделей для извлечения знаний в граф памяти

Извлекает из GGUF:
1. Конфигурацию модели (архитектура, слои, размерности)
2. Архитектурные паттерны (attention heads, FFN, RoPE)
3. Информацию о квантизации
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("eva_ai.fractal_graph_v2.gguf_parser")


@dataclass
class GGUFModelInfo:
    """Информация о GGUF модели."""
    model_path: str
    model_type: str = ""                    # qwen3, qwen2, llama, etc.
    architecture: str = ""                   # Архитектура модели
    vocab_size: int = 0                      # Размер словаря
    hidden_size: int = 0                      # Размер скрытого слоя (embedding_length)
    num_layers: int = 0                     # Количество слоёв (block_count)
    num_attention_heads: int = 0            # Количество attention голов
    num_key_value_heads: int = 0            # KV heads (GQA)
    max_position_embeddings: int = 0        # Максимальная длина контекста
    intermediate_size: int = 0               # Размер FFN (feed_forward_length)
    rope_theta: float = 0                    # RoPE base frequency
    rms_norm_eps: float = 0                 # RMS Norm epsilon
    attention_key_length: int = 0            # Key length per head
    attention_value_length: int = 0           # Value length per head
    
    # Токенизатор
    tokenizer_type: str = ""
    bos_token_id: int = 0
    eos_token_id: int = 0
    
    # Метаданные
    quantization_version: str = ""
    quantization_type: str = ""
    file_size: int = 0


class GGUFModelParser:
    """
    Парсер GGUF моделей с использованием gguf библиотеки.
    
    Извлекает метаданные напрямую из GGUF файла без загрузки модели.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    
    def _bytes_to_str(self, data) -> str:
        """Конвертировать bytes/numpy array в строку."""
        if isinstance(data, bytes):
            return data.decode('utf-8', errors='replace').strip()
        elif isinstance(data, bytearray):
            return bytes(data).decode('utf-8', errors='replace').strip()
        
        dtype = getattr(data, 'dtype', None)
        if dtype is not None:
            dtype_str = str(dtype)
            
            if 'uint8' in dtype_str or 'int8' in dtype_str:
                if hasattr(data, 'tolist'):
                    byte_list = data.tolist()
                elif hasattr(data, '__iter__'):
                    byte_list = list(data)
                else:
                    return str(data)
                
                if len(byte_list) > 0:
                    try:
                        result_bytes = bytes(b for b in byte_list if isinstance(b, int))
                        return result_bytes.decode('utf-8', errors='replace').strip()
                    except:
                        pass
                    
                    result = []
                    for b in byte_list:
                        if isinstance(b, (int, np.integer)):
                            if 32 <= b <= 126:
                                result.append(chr(b))
                            elif b == 0:
                                break
                        else:
                            result.append(str(b))
                    return ''.join(result).strip()
        
        if isinstance(data, (list, tuple)):
            result = []
            for b in data:
                if isinstance(b, int):
                    if 32 <= b <= 126:
                        result.append(chr(b))
                    elif b == 0:
                        break
                    else:
                        result.append(f'<{b}>')
                elif isinstance(b, bytes):
                    result.append(b.decode('utf-8', errors='replace').strip())
                else:
                    result.append(str(b))
            return ''.join(result).strip()
        
        return str(data)
    
    def _get_field_value(self, field) -> Any:
        """Получить значение поля из GGUF."""
        if not hasattr(field, 'parts') or len(field.parts) == 0:
            return None
        value = field.parts[-1]
        
        if value is None:
            return None
        
        dtype_name = getattr(value, 'dtype', None)
        
        if dtype_name is not None:
            dtype_str = str(dtype_name)
            
            if 'uint8' in dtype_str or 'int8' in dtype_str:
                arr_len = getattr(value, 'shape', ())[0] if hasattr(value, 'shape') else 0
                if arr_len > 1:
                    return bytes(value)
                elif arr_len == 1:
                    try:
                        return chr(int(value.flat[0]))
                    except:
                        return int(value.flat[0])
            
            if hasattr(value, 'flat') and hasattr(value.flat, '__iter__'):
                try:
                    scalar = value.flat[0]
                    
                    if 'float' in dtype_str:
                        return float(scalar)
                    elif 'int' in dtype_str or 'uint' in dtype_str:
                        return int(scalar)
                    else:
                        return scalar
                except (IndexError, ValueError):
                    pass
            
            if hasattr(value, 'shape') and len(getattr(value, 'shape', [])) == 1:
                arr_len = value.shape[0] if hasattr(value, 'shape') else 0
                
                if arr_len == 1:
                    try:
                        return int(value.flat[0]) if 'int' in dtype_str or 'uint' in dtype_str else float(value.flat[0])
                    except:
                        pass
                
                if arr_len > 1 and arr_len <= 100:
                    try:
                        if 'uint8' in dtype_str or 'int8' in dtype_str:
                            return bytes(value)
                        else:
                            return [int(v) if 'int' in dtype_str else float(v) for v in value]
                    except:
                        pass
        
        if isinstance(value, (bytes, bytearray)):
            if len(value) <= 8:
                try:
                    import struct
                    if len(value) == 4:
                        return struct.unpack('I', value)[0]
                    elif len(value) == 8:
                        return struct.unpack('Q', value)[0]
                    elif len(value) == 2:
                        return struct.unpack('H', value)[0]
                except:
                    pass
            try:
                decoded = value.decode('utf-8').strip()
                if len(decoded) <= 100:
                    return decoded
            except:
                pass
            return value.hex()
        
        return value
    
    def parse(self) -> GGUFModelInfo:
        """Парсить GGUF файл и извлечь информацию."""
        logger.info(f"Парсинг GGUF: {self.model_path}")
        
        info = GGUFModelInfo(model_path=self.model_path)
        info.file_size = self.file_size
        
        try:
            from gguf import GGUFReader
            reader = GGUFReader(self.model_path, 'r')
            
            # Парсим метаданные
            for key, field in reader.fields.items():
                value = self._get_field_value(field)
                if value is None:
                    continue
                
                # Определяем архитектуру
                if key == 'general.architecture':
                    info.architecture = self._bytes_to_str(value)
                    info.model_type = info.architecture
                
                # Модель
                elif key == 'general.name':
                    info.model_type = self._bytes_to_str(value)
                
                # Архитектурные параметры Qwen
                elif key == 'qwen3.context_length':
                    info.max_position_embeddings = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.embedding_length':
                    info.hidden_size = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.block_count':
                    info.num_layers = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.feed_forward_length':
                    info.intermediate_size = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.rope.freq_base':
                    info.rope_theta = float(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.attention.layer_norm_rms_epsilon':
                    info.rms_norm_eps = float(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.attention.head_count':
                    info.num_attention_heads = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.attention.head_count_kv':
                    info.num_key_value_heads = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.attention.key_length':
                    info.attention_key_length = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen3.attention.value_length':
                    info.attention_value_length = int(value) if isinstance(value, (int, float)) else 0
                
                # Qwen2 backward compatibility
                elif key == 'qwen2.context_length':
                    info.max_position_embeddings = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.embedding_length':
                    info.hidden_size = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.block_count':
                    info.num_layers = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.feed_forward_length':
                    info.intermediate_size = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.rope.freq_base':
                    info.rope_theta = float(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.attention.head_count':
                    info.num_attention_heads = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'qwen2.attention.head_count_kv':
                    info.num_key_value_heads = int(value) if isinstance(value, (int, float)) else 0
                
                # Токенизатор
                elif key == 'tokenizer.ggml.model':
                    info.tokenizer_type = self._bytes_to_str(value)
                elif key == 'tokenizer.ggml.bos_token_id':
                    info.bos_token_id = int(value) if isinstance(value, (int, float)) else 0
                elif key == 'tokenizer.ggml.eos_token_id':
                    info.eos_token_id = int(value) if isinstance(value, (int, float)) else 0
                
                # Квантизация
                elif key == 'general.quantization_version':
                    info.quantization_version = str(value)
                elif key == 'general.file_type':
                    info.quantization_type = self._get_quantization_type(value)
            
            logger.info(f"Парсинг завершён:")
            logger.info(f"  Архитектура: {info.architecture}")
            logger.info(f"  Слоёв: {info.num_layers}")
            logger.info(f"  Hidden size: {info.hidden_size}")
            logger.info(f"  Attention heads: {info.num_attention_heads}")
            logger.info(f"  KV heads: {info.num_key_value_heads}")
            logger.info(f"  Context: {info.max_position_embeddings}")
            
        except ImportError:
            logger.warning("gguf библиотека недоступна, используем fallback")
            self._parse_fallback(info)
        except Exception as e:
            logger.warning(f"Ошибка парсинга: {e}")
            self._parse_fallback(info)
        
        return info
    
    def _get_quantization_type(self, value) -> str:
        """Получить название типа квантизации."""
        if isinstance(value, (int, float)):
            qtype = int(value)
            # Стандартные типы GGUF
            types = {
                1: "F16", 2: "Q4_0", 3: "Q5_0", 4: "Q8_0",
                7: "Q2_K", 10: "Q3_K", 11: "Q4_K", 12: "Q5_K", 13: "Q6_K",
                15: "Q4_K_M", 17: "Q5_K_M"
            }
            return types.get(qtype, f"Q{qtype}")
        return str(value)
    
    def _parse_fallback(self, info: GGUFModelInfo):
        """Fallback парсинг для qwen3 4B."""
        info.architecture = "qwen3"
        info.model_type = "RuadaptQwen3 4B"
        info.hidden_size = 2560
        info.num_layers = 36
        info.num_attention_heads = 32
        info.num_key_value_heads = 8
        info.max_position_embeddings = 262144
        info.intermediate_size = 9728
        info.rope_theta = 5000000.0
        info.rms_norm_eps = 1e-6
        info.attention_key_length = 128
        info.attention_value_length = 128
        info.quantization_type = "Q4_K_M"
    
    def extract_knowledge_for_graph(self) -> List[Dict[str, Any]]:
        """
        Извлечь знания из модели для сохранения в графе.
        
        Возвращает список узлов для добавления в граф.
        """
        info = self.parse()
        
        nodes = []
        
        # 1. Главный узел модели (статичный, защищённый)
        nodes.append({
            "type": "MODEL_ROOT",
            "level": -1,
            "is_static": True,
            "content": f"RuadaptQwen3 4B ({info.architecture})",
            "metadata": {
                "model_path": self.model_path,
                "architecture": info.architecture,
                "quantization": info.quantization_type,
                "file_size_mb": round(info.file_size / 1024 / 1024, 1)
            }
        })
        
        # 2. Архитектурные характеристики
        nodes.append({
            "type": "MODEL_A",
            "level": 0,
            "is_static": True,
            "content": f"Model A: {info.hidden_size} hidden, {info.num_layers} layers, {info.num_attention_heads} heads",
            "metadata": {
                "role": "logic",
                "hidden_size": info.hidden_size,
                "num_layers": info.num_layers,
                "attention_heads": info.num_attention_heads,
                "kv_heads": info.num_key_value_heads
            }
        })
        
        nodes.append({
            "type": "MODEL_B",
            "level": 0,
            "is_static": True,
            "content": f"Model B: {info.hidden_size} hidden, {info.num_layers} layers (context extension)",
            "metadata": {
                "role": "context",
                "hidden_size": info.hidden_size,
                "num_layers": info.num_layers
            }
        })
        
        # 3. Контекстное окно
        nodes.append({
            "type": "FACT",
            "level": 1,
            "is_static": True,
            "content": f"Контекстное окно: {info.max_position_embeddings:,} токенов (RoPE)",
            "metadata": {
                "context_length": info.max_position_embeddings,
                "rope_freq_base": info.rope_theta
            }
        })
        
        # 4. Feed-Forward Network
        if info.hidden_size > 0 and info.intermediate_size > 0:
            expansion_ratio = round(info.intermediate_size / info.hidden_size, 1)
        else:
            expansion_ratio = 3.8
        nodes.append({
            "type": "FACT",
            "level": 2,
            "is_static": True,
            "content": f"FFN: {info.intermediate_size:,} промежуточных нейронов",
            "metadata": {
                "intermediate_size": info.intermediate_size,
                "hidden_size": info.hidden_size,
                "expansion_ratio": expansion_ratio
            }
        })
        
        # 5. Attention паттерны (GQA)
        if info.num_attention_heads > 0:
            kv_ratio = round(info.num_key_value_heads / info.num_attention_heads, 2) if info.num_attention_heads > 0 else 0
        else:
            kv_ratio = 0.25
        nodes.append({
            "type": "FACT",
            "level": 2,
            "is_static": True,
            "content": f"Grouped Query Attention: {info.num_attention_heads} Q-heads, {info.num_key_value_heads} KV-heads",
            "metadata": {
                "query_heads": info.num_attention_heads,
                "kv_heads": info.num_key_value_heads,
                "kv_ratio": kv_ratio
            }
        })
        
        # 6. RMSNorm
        nodes.append({
            "type": "FACT",
            "level": 2,
            "is_static": True,
            "content": f"RMSNorm epsilon: {info.rms_norm_eps}",
            "metadata": {
                "rms_norm_eps": info.rms_norm_eps
            }
        })
        
        # 7. Размерности attention
        if info.attention_key_length > 0:
            nodes.append({
                "type": "FACT",
                "level": 2,
                "is_static": True,
                "content": f"Attention: K={info.attention_key_length}, V={info.attention_value_length} per head",
                "metadata": {
                    "key_length": info.attention_key_length,
                    "value_length": info.attention_value_length
                }
            })
        
        # 8. RoPE
        nodes.append({
            "type": "FACT",
            "level": 2,
            "is_static": True,
            "content": f"RoPE base frequency: {info.rope_theta:,.0f}",
            "metadata": {
                "rope_theta": info.rope_theta
            }
        })
        
        return nodes


def parse_gguf_model(model_path: str) -> GGUFModelInfo:
    """Фабричная функция для парсинга GGUF."""
    parser = GGUFModelParser(model_path)
    return parser.parse()


def extract_to_graph(model_path: str, graph) -> Dict[str, Any]:
    """
    Извлечь знания из GGUF модели и добавить в граф.
    
    Returns:
        Результат добавления узлов
    """
    parser = GGUFModelParser(model_path)
    knowledge_nodes = parser.extract_knowledge_for_graph()
    
    added_count = 0
    for node_data in knowledge_nodes:
        try:
            node = graph.add_node(
                content=node_data["content"],
                node_type=node_data["type"],
                level=node_data["level"],
                metadata=node_data.get("metadata", {}),
                is_static=node_data.get("is_static", False)
            )
            added_count += 1
        except Exception as e:
            logger.warning(f"Не удалось добавить узел: {e}")
    
    return {
        "model_path": model_path,
        "nodes_extracted": len(knowledge_nodes),
        "nodes_added": added_count
    }


def clear_and_reload_model_graph(graph, model_paths: List[str]) -> Dict[str, Any]:
    """
    Очистить старые данные модели и загрузить актуальные.
    
    Удаляет все узлы с node_type начинающимся на MODEL_ или FACT с metadata.model_path.
    """
    removed_count = 0
    added_count = 0
    
    # Удаляем старые MODEL_ узлы
    if hasattr(graph.storage, 'nodes'):
        nodes_to_remove = []
        for node_id, node in list(graph.storage.nodes.items()):
            if node.node_type.startswith('MODEL_'):
                nodes_to_remove.append(node_id)
            elif node.metadata.get('model_path'):
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            try:
                graph.delete_node(node_id, force=True)
                removed_count += 1
            except:
                pass
    
    # Загружаем новые данные
    for model_path in model_paths:
        if os.path.exists(model_path):
            result = extract_to_graph(model_path, graph)
            added_count += result['nodes_added']
    
    return {
        "nodes_removed": removed_count,
        "nodes_added": added_count
    }