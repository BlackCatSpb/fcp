"""
GGUF Shadow Profiler - Гибридная интеграция GGUF модели во FractalGraphV2

Создает "когнитивную тень" модели в графе:
- model_root (L0) - метаданные GGUF
- domain_profile (L1) - профили доменов
- activation_fingerprint (L1) - сжатые активации (PCA)
- routing_rule (L2) - правила маршрутизации
- quantization_profile (L2) - профили квантования
- layer_stats (L3) - статистика слоёв

Это не переносит веса в граф, а создаёт мета-представление для:
- Семантической маршрутизации запросов
- Адаптивной настройки параметров
- Прогнозирования деградации
"""

import os
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from .types import FractalNode, FractalEdge, NodeType, RelationType

logger = logging.getLogger("eva_ai.fractal_graph_v2.gguf_shadow")


class GGUFShadowProfiler:
    """
    Гибридный профилировщик GGUF модели.
    Создает когнитивную тень модели в FractalGraphV2.
    """
    
    # Домены для создания профилей
    DEFAULT_DOMAINS = [
        "python_coding",
        "formal_logic", 
        "creative_narrative",
        "scientific_explanation",
        "mathematical_reasoning",
        "general_conversation",
        "technical_documentation",
        "data_analysis"
    ]
    
    def __init__(self, fractal_graph, model_path: str = None):
        """
        Args:
            fractal_graph: FractalGraphV2 instance
            model_path: Путь к GGUF файлу (опционально)
        """
        self.graph = fractal_graph
        self.model_path = model_path
        self.model_meta: Dict[str, Any] = {}
        self.root_node_id: Optional[str] = None
        
        if model_path and os.path.exists(model_path):
            self._load_model_metadata()
    
    def _load_model_metadata(self):
        """Загрузка метаданных GGUF модели."""
        try:
            from .gguf_parser import parse_gguf_model
            info = parse_gguf_model(self.model_path)
            self.model_meta = {
                'architecture': info.architecture,
                'model_type': info.model_type,
                'vocab_size': info.vocab_size,
                'hidden_size': info.hidden_size,
                'num_layers': info.num_layers,
                'num_attention_heads': info.num_attention_heads,
                'max_position_embeddings': info.max_position_embeddings,
                'rope_theta': getattr(info, 'rope_theta', 0),
                'quantization_version': getattr(info, 'quantization_version', 'unknown'),
                'file_size': os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 0,
                'model_path': self.model_path
            }
            logger.info(f"Loaded GGUF metadata: {info.architecture}, layers={info.num_layers}, hidden={info.hidden_size}")
        except Exception as e:
            logger.warning(f"Failed to parse GGUF metadata: {e}")
            self.model_meta = {'model_path': self.model_path, 'error': str(e)}
    
    def register_model_root(self, model_name: str = "default") -> str:
        """
        Создать корневой узел модели (L0).
        
        Args:
            model_name: Имя модели для отображения
            
        Returns:
            node_id созданного узла
        """
        if self.root_node_id:
            if isinstance(self.root_node_id, FractalNode):
                return self.root_node_id.id
            return self.root_node_id
        
        # Формируем content из метаданных
        arch = self.model_meta.get('architecture', 'unknown')
        layers = self.model_meta.get('num_layers', 0)
        hidden = self.model_meta.get('hidden_size', 0)
        
        content = f"GGUF Model: {model_name} | {arch} | {layers} layers | hidden={hidden}"
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.MODEL_ROOT.value,
                level=0,
                metadata=self.model_meta,
                is_static=True,
                confidence=1.0
            )
            
            if isinstance(result, FractalNode):
                self.root_node_id = result.id
            else:
                self.root_node_id = result
            
            logger.info(f"Created model_root: {self.root_node_id}")
            
            # Создаем семантическую группу для модели
            self._create_model_shadow_group()
            
            return self.root_node_id
        except Exception as e:
            logger.error(f"Failed to create model_root: {e}")
            return None
    
    def _create_model_shadow_group(self):
        """Создать семантическую группу gguf_model_shadow."""
        if not self.root_node_id:
            return
        
        if isinstance(self.root_node_id, FractalNode):
            root_id = self.root_node_id.id
        else:
            root_id = self.root_node_id
        
        try:
            group_id = self.graph.create_semantic_group(
                name="gguf_model_shadow",
                member_ids=[root_id],
                level=2
            )
            logger.info(f"Created semantic group: gguf_model_shadow")
        except Exception as e:
            logger.warning(f"Failed to create semantic group: {e}")
    
    def _get_node_id(self, node_or_id):
        """Извлечь node_id из FractalNode или вернуть как есть."""
        if isinstance(node_or_id, FractalNode):
            return node_or_id.id
        return node_or_id
    
    def _get_root_id(self):
        """Получить root_id для использования в edges."""
        if not self.root_node_id:
            return None
        return self._get_node_id(self.root_node_id)
    
    def create_domain_profile(self, domain: str, description: str = "") -> str:
        """
        Создать профиль домена (L1).
        
        Args:
            domain: Имя домена (python_coding, creative_narrative и т.д.)
            description: Описание домена
            
        Returns:
            node_id созданного профиля
        """
        root_id = self._get_root_id()
        if not root_id:
            logger.warning("No model_root registered, cannot create domain profile")
            return None
        
        content = f"domain_profile: {domain}"
        if description:
            content += f" - {description}"
        
        metadata = {
            'domain': domain,
            'description': description,
            'created_at': time.time()
        }
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.DOMAIN_PROFILE.value,
                level=1,
                metadata=metadata,
                is_static=True,
                confidence=0.75
            )
            
            domain_id = self._get_node_id(result)
            
            # Связываем с model_root
            self.graph.add_edge(
                source_id=root_id,
                target_id=domain_id,
                relation_type=RelationType.BELONGS_TO_DOMAIN.value,
                weight=0.9
            )
            
            logger.info(f"Created domain_profile: {domain}")
            return domain_id
        except Exception as e:
            logger.error(f"Failed to create domain_profile: {e}")
            return None
    
    def create_activation_fingerprint(
        self, 
        domain: str, 
        embedding: List[float],
        sample_count: int = 0
    ) -> str:
        """
        Создать фингерпринт активаций (L1).
        
        Args:
            domain: Имя домена
            embedding: PCA-сжатый эмбеддинг (128-256D) - сохраняется в metadata
            sample_count: Количество запросов в выборке
            
        Returns:
            node_id созданного фингерпринта
        """
        # Находим domain_profile для этого домена
        domain_node_id = self._find_domain_profile(domain)
        
        content = f"activation_fingerprint: {domain}"
        
        # PCA эмбеддинг сохраняем в metadata, для semantic search используем placeholder
        # FG хранит 768D эмбеддинги, поэтому создаем placeholder для поиска
        placeholder_emb = embedding[:768] if len(embedding) >= 768 else embedding + [0.0] * (768 - len(embedding))
        
        metadata = {
            'domain': domain,
            'sample_count': sample_count,
            'pca_embedding': embedding,  # PCA compressed - для анализа
            'embedding_dim': len(embedding),
            'created_at': time.time()
        }
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.ACTIVATION_FINGERPRINT.value,
                level=1,
                embedding=placeholder_emb,  # 768D для semantic search
                metadata=metadata,
                confidence=0.75
            )
            
            fp_id = self._get_node_id(result)
            
            # Связываем с domain_profile
            if domain_node_id:
                self.graph.add_edge(
                    source_id=domain_node_id,
                    target_id=fp_id,
                    relation_type=RelationType.BELONGS_TO_DOMAIN.value,
                    weight=0.85
                )
            
            logger.info(f"Created activation_fingerprint for {domain}, pca_dim={len(embedding)}")
            return fp_id
        except Exception as e:
            logger.error(f"Failed to create activation_fingerprint: {e}")
            return None
    
    def _find_domain_profile(self, domain: str) -> Optional[str]:
        """Найти domain_profile для домена."""
        if not hasattr(self.graph, 'nodes'):
            return None
        
        for node_id, node in self.graph.nodes.items():
            if getattr(node, 'node_type', '') == NodeType.DOMAIN_PROFILE.value:
                meta = getattr(node, 'metadata', {})
                if meta.get('domain') == domain:
                    return node_id
        return None
    
    def bind_routing_rule(
        self, 
        domain: str, 
        rule_config: Dict[str, Any]
    ) -> str:
        """
        Создать правило маршрутизации (L2).
        
        Args:
            domain: Имя домена
            rule_config: Конфигурация правила:
                {
                    "trigger": {"cosine_threshold": 0.82, "min_confidence": 0.65},
                    "action": {
                        "pipeline_override": "model_a_only",
                        "parameters": {"temperature": 0.2, "repeat_penalty": 2.1}
                    }
                }
                
        Returns:
            node_id созданного правила
        """
        # Находим fingerprint для домена
        fingerprint_id = self._find_activation_fingerprint(domain)
        
        content = f"routing_rule: {domain}"
        
        # Сериализуем metadata в JSON-совместимый формат
        metadata = {
            'domain': domain,
            'trigger': self._serialize_value(rule_config.get('trigger', {})),
            'action': self._serialize_value(rule_config.get('action', {})),
            'created_at': time.time(),
            'access_count': 0
        }
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.ROUTING_RULE.value,
                level=2,
                metadata=metadata,
                confidence=0.85
            )
            
            rule_id = self._get_node_id(result)
            
            # Связываем с fingerprint
            if fingerprint_id:
                self.graph.add_edge(
                    source_id=fingerprint_id,
                    target_id=rule_id,
                    relation_type=RelationType.ROUTES_TO.value,
                    weight=0.9
                )
            
            logger.info(f"Created routing_rule for {domain}")
            return rule_id
        except Exception as e:
            logger.error(f"Failed to create routing_rule: {e}")
            return None
    
    def _serialize_value(self, val):
        """Рекурсивно сериализовать значения в JSON-совместимый формат."""
        if isinstance(val, (str, int, float, bool, type(None))):
            return val
        elif isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [self._serialize_value(v) for v in val]
        elif hasattr(val, '__dict__'):
            return str(val)
        else:
            return str(val)
    
    def _find_activation_fingerprint(self, domain: str) -> Optional[str]:
        """Найти activation_fingerprint для домена."""
        if not hasattr(self.graph, 'nodes'):
            return None
        
        for node_id, node in self.graph.nodes.items():
            if getattr(node, 'node_type', '') == NodeType.ACTIVATION_FINGERPRINT.value:
                meta = getattr(node, 'metadata', {})
                if meta.get('domain') == domain:
                    return node_id
        return None
    
    def create_quantization_profile(self, profile_data: Dict[str, Any]) -> str:
        """
        Создать профиль квантования (L2).
        
        Args:
            profile_data: {
                "scheme": "Q4_K_M",
                "domain_loss": 0.03,
                "speedup": 2.1,
                "ram_mb": 2850
            }
            
        Returns:
            node_id созданного профиля
        """
        root_id = self._get_root_id()
        if not root_id:
            return None
        
        scheme = profile_data.get('scheme', 'unknown')
        content = f"quantization_profile: {scheme}"
        
        metadata = {
            'scheme': scheme,
            'domain_loss': profile_data.get('domain_loss', 0),
            'speedup': profile_data.get('speedup', 1.0),
            'ram_mb': profile_data.get('ram_mb', 0),
            'created_at': time.time()
        }
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.QUANTIZATION_PROFILE.value,
                level=2,
                metadata=metadata,
                confidence=0.9
            )
            
            q_id = self._get_node_id(result)
            
            # Связываем с model_root
            self.graph.add_edge(
                source_id=root_id,
                target_id=q_id,
                relation_type=RelationType.HAS_PROFILE.value,
                weight=0.95
            )
            
            logger.info(f"Created quantization_profile: {scheme}")
            return q_id
        except Exception as e:
            logger.error(f"Failed to create quantization_profile: {e}")
            return None
    
    def create_layer_stats(self, stats: Dict[str, Any]) -> str:
        """
        Создать статистику слоёв (L3).
        
        Args:
            stats: {
                "layer_early": {"mean": 0.012, "std": 0.34, "sparsity": 0.18},
                "layer_middle": {...},
                "layer_late": {...}
            }
            
        Returns:
            node_id созданного узла статистики
        """
        root_id = self._get_root_id()
        if not root_id:
            return None
        
        content = "layer_stats: aggregated statistics"
        
        # Сериализуем stats для JSON
        serialized_stats = self._serialize_value(stats)
        
        metadata = {
            'layer_stats': serialized_stats,
            'created_at': time.time()
        }
        
        try:
            result = self.graph.add_node(
                content=content,
                node_type=NodeType.LAYER_STATS.value,
                level=3,
                metadata=metadata,
                confidence=0.8
            )
            
            ls_id = self._get_node_id(result)
            
            # Связываем с model_root
            self.graph.add_edge(
                source_id=root_id,
                target_id=ls_id,
                relation_type=RelationType.HAS_STATS.value,
                weight=0.85
            )
            
            logger.info(f"Created layer_stats")
            return ls_id
        except Exception as e:
            logger.error(f"Failed to create layer_stats: {e}")
            return None
    
    def log_parameter_tuning(
        self, 
        domain: str, 
        params: Dict[str, Any],
        quality_score: float
    ) -> str:
        """
        Записать настройку параметров (L3).
        
        Args:
            domain: Имя домена
            params: Использованные параметры
            quality_score: Оценка качества (0-1)
            
        Returns:
            node_id созданной записи
        """
        content = f"parameter_tuning: {domain}"
        
        metadata = {
            'domain': domain,
            'parameters': params,
            'quality_score': quality_score,
            'created_at': time.time()
        }
        
        try:
            node_id = self.graph.add_node(
                content=content,
                node_type=NodeType.PARAMETER_TUNING_RECORD.value,
                level=3,
                metadata=metadata,
                confidence=quality_score
            )
            
            logger.info(f"Logged parameter_tuning for {domain}, quality={quality_score}")
            return node_id
        except Exception as e:
            logger.error(f"Failed to log parameter_tuning: {e}")
            return None
    
    def get_routing_for_query(self, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Получить оптимальную маршрутизацию для запроса.
        
        Args:
            query_embedding: Эмбеддинг запроса (768D)
            
        Returns:
            dict с параметрами маршрутизации или None
        """
        # Семантический поиск по fingerprint
        try:
            results = self.graph.semantic_search(
                query="query_embedding",  # используем эмбеддинг напрямую
                top_k=1,
                min_level=1,
                min_similarity=0.5
            )
            
            if not results:
                return None
            
            # Находим fingerprint
            for r in results:
                if r.get('type') == 'node':
                    node_id = r.get('id')
                    node = self.graph.nodes.get(node_id)
                    if node and node.node_type == NodeType.ACTIVATION_FINGERPRINT.value:
                        # Ищем связанное routing_rule
                        return self._get_routing_from_fingerprint(node_id)
            
            return None
        except Exception as e:
            logger.warning(f"Failed to get routing: {e}")
            return None
    
    def _get_routing_from_fingerprint(self, fingerprint_id: str) -> Optional[Dict]:
        """Получить routing_rule из fingerprint."""
        if not hasattr(self.graph, 'edges'):
            return None
        
        for edge_id, edge in self.graph.edges.items():
            if edge.source_id == fingerprint_id and edge.relation_type == RelationType.ROUTES_TO.value:
                target_node = self.graph.nodes.get(edge.target_id)
                if target_node and target_node.node_type == NodeType.ROUTING_RULE.value:
                    return target_node.metadata
        
        return None
    
    def initialize_default_domains(self):
        """Инициализировать профили для всех дефолтных доменов."""
        for domain in self.DEFAULT_DOMAINS:
            domain_id = self.create_domain_profile(domain)
            if domain_id:
                # Создаем пустой fingerprint с дефолтным embedding
                default_emb = np.random.randn(128).tolist()  # PCA 128D placeholder
                self.create_activation_fingerprint(domain, default_emb, sample_count=0)
        
        logger.info(f"Initialized {len(self.DEFAULT_DOMAINS)} domain profiles")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Получить статус модели в графе."""
        if not self.root_node_id:
            return {'registered': False}
        
        node_counts = {
            'domain_profiles': 0,
            'activation_fingerprints': 0,
            'routing_rules': 0,
            'quantization_profiles': 0,
            'layer_stats': 0,
            'tuning_records': 0
        }
        
        if hasattr(self.graph, 'nodes'):
            for node in self.graph.nodes.values():
                nt = getattr(node, 'node_type', '')
                if nt == NodeType.DOMAIN_PROFILE.value:
                    node_counts['domain_profiles'] += 1
                elif nt == NodeType.ACTIVATION_FINGERPRINT.value:
                    node_counts['activation_fingerprints'] += 1
                elif nt == NodeType.ROUTING_RULE.value:
                    node_counts['routing_rules'] += 1
                elif nt == NodeType.QUANTIZATION_PROFILE.value:
                    node_counts['quantization_profiles'] += 1
                elif nt == NodeType.LAYER_STATS.value:
                    node_counts['layer_stats'] += 1
                elif nt == NodeType.PARAMETER_TUNING_RECORD.value:
                    node_counts['tuning_records'] += 1
        
        return {
            'registered': True,
            'root_node_id': self.root_node_id,
            'model_meta': self.model_meta,
            'node_counts': node_counts
        }


def create_gguf_shadow_profiler(fractal_graph, model_path: str = None) -> GGUFShadowProfiler:
    """Фабричная функция для создания профилировщика."""
    return GGUFShadowProfiler(fractal_graph, model_path)