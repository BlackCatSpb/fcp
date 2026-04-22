"""
FCP Integration - интеграция с EVA-Ai компонентами
"""
import sys
import os
import logging

# Add EVA-Ai path
EVA_PATH = "C:/Users/black/OneDrive/Desktop/EVA-Ai"
if EVA_PATH not in sys.path:
    sys.path.insert(0, EVA_PATH)

from typing import Optional, List, Dict, Any

logger = logging.getLogger("fcp.integration")


class FCPIntegration:
    """
    Интеграция FCP с лучшими компонентами EVA-Ai.
    
    Заимствуем:
    - ConceptMiner (глубокий анализ кластеров)
    - ContradictionMiner (обнаружение противоречий)  
    - FractalGraphV2 (долговременная память)
    """
    
    def __init__(self, graph_db_path: str = ""):
        self.graph_db_path = graph_db_path
        self.graph = None
        self.concept_miner = None
        self.contradiction_miner = None
    
    def load_graph(self) -> bool:
        """Загрузить FractalGraphV2."""
        try:
            from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
            
            if self.graph_db_path:
                self.graph = FractalGraphV2(self.graph_db_path)
            else:
                # Default path
                default_path = "C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
                self.graph = FractalGraphV2(default_path)
            
            logger.info("[FCP] Graph loaded")
            return True
            
        except Exception as e:
            logger.warning(f"[FCP] Graph load failed: {e}")
            self.graph = None
            return False
    
    def load_concept_miner(self) -> bool:
        """Загрузить ConceptMiner."""
        try:
            from eva_ai.knowledge.concept_miner import ConceptMiner
            
            if self.graph:
                self.concept_miner = ConceptMiner(self.graph)
            
            logger.info("[FCP] ConceptMiner loaded")
            return True
            
        except Exception as e:
            logger.warning(f"[FCP] ConceptMiner load failed: {e}")
            self.concept_miner = None
            return False
    
    def load_contradiction_miner(self) -> bool:
        """Загрузить ContradictionMiner."""
        try:
            from eva_ai.contradiction.contradiction_miner import ContradictionMiner
            
            if self.graph:
                self.contradiction_miner = ContradictionMiner(self.graph)
            
            logger.info("[FCP] ContradictionMiner loaded")
            return True
            
        except Exception as e:
            logger.warning(f"[FCP] ContradictionMiner load failed: {e}")
            self.contradiction_miner = None
            return False
    
    def load_all(self) -> bool:
        """Загрузить все компоненты."""
        results = []
        
        results.append(self.load_graph())
        results.append(self.load_concept_miner())
        results.append(self.load_contradiction_miner())
        
        return any(results)
    
    def extract_concepts(self, text: str, context: Optional[Dict] = None) -> List[Dict]:
        """
        Извлечь концепты из текста (использует EVA-Ai ConceptMiner).
        
        Args:
            text: исходный текст
            context: дополнительный контекст
            
        Returns:
            List концептов
        """
        if not self.concept_miner:
            return []
        
        try:
            # Use ConceptMiner for deep extraction
            return self.concept_miner.extract_from_text(text, context)
        except Exception as e:
            logger.warning(f"[FCP] Concept extraction: {e}")
            return []
    
    def detect_contradictions(self) -> List[Dict]:
        """
        Обнаружить противоречия в графе.
        
        Returns:
            List обнаруженных противоречий
        """
        if not self.contradiction_miner:
            return []
        
        try:
            return self.contradiction_miner.detect_and_resolve()
        except Exception as e:
            logger.warning(f"[FCP] Contradiction detection: {e}")
            return []
    
    def retrieve_subgraph(self, query_embedding: Any, top_k: int = 32) -> Any:
        """
        Извлечь подграф по запросу.
        
        Args:
            query_embedding: эмбеддинг запроса
            top_k: количество узлов
            
        Returns:
            Subgraph
        """
        if not self.graph:
            return None
        
        try:
            return self.graph.get_relevant_subgraph(query_embedding, top_k)
        except Exception as e:
            logger.warning(f"[FCP] Subgraph retrieval: {e}")
            return None
    
    def add_concept_to_graph(
        self,
        name: str,
        embedding: Any = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Добавить концепт в граф."""
        if not self.graph:
            return False
        
        try:
            self.graph.add_unique_concept(name, embedding, metadata)
            return True
        except Exception as e:
            logger.warning(f"[FCP] Add concept: {e}")
            return False
    
    @property
    def is_loaded(self) -> bool:
        return self.graph is not None
    
    @property
    def components_status(self) -> Dict[str, bool]:
        return {
            "graph": self.graph is not None,
            "concept_miner": self.concept_miner is not None,
            "contradiction_miner": self.contradiction_miner is not None
        }


# Singleton instance
_fcp_integration: Optional[FCPIntegration] = None


def get_fcp_integration(graph_db_path: str = "") -> FCPIntegration:
    """Get FCP integration singleton."""
    global _fcp_integration
    
    if _fcp_integration is None:
        _fcp_integration = FCPIntegration(graph_db_path)
    
    return _fcp_integration


class FCPFullPipeline:
    """
    Полный FCP Pipeline с интеграцией EVA-Ai.
    
    Когнитивный цикл:
    1. Восприятие (Input Layer)
    2. Гибридный стек (32 слоя)
    3. Выходной слой
    4. Запись в TCM (асинхронно)
    5. Консолидация в Graph (асинхронно)
    6. КУРАЦИЯ (асинхронно)
    """
    
    def __init__(self, config: "FCPConfig"):
        from fcp_core.config import FCPConfig
        self.config = config
        
        # EVA-Ai integration
        self.integration = get_fcp_integration(config.graph_db_path)
        
        # OpenVINO model
        self._model = None
        
        # Output layer
        from fcp_core.output_layer import OutputLayer
        self.output = OutputLayer(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim
        )
        
        # Stack
        from fcp_core.hybrid_stack import HybridStack, StackConfig
        stack_config = StackConfig(
            num_layers=config.num_layers,
            hidden_dim=config.embedding_dim,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            graph_retrieval_k=config.graph_retrieval_k,
            master_tokens=config.master_tokens,
            stop_threshold=config.stop_threshold,
            early_exit_threshold=config.early_exit_threshold
        )
        self.stack = HybridStack(stack_config)
        
        # Statistics
        self._stats = {
            "queries_processed": 0,
            "early_exits": 0,
            "total_layers_used": 0,
            "concepts_extracted": 0,
            "contradictions_resolved": 0
        }
    
    def load_model(self, model_path: str) -> bool:
        """Load OpenVINO model."""
        try:
            import openvino_genai as ov_genai
            
            self._tokenizer = ov_genai.Tokenizer(model_path)
            self._pipe = ov_genai.LLMPipeline(
                model_path,
                self._tokenizer,
                self.config.device,
                {"INFERENCE_NUM_THREADS": self.config.num_threads}
            )
            
            print(f"[FCP] Model loaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"[FCP] Model load error: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        """
        Generate response через полный FCP pipeline.
        """
        if not self._pipe:
            return "[FCP] Model not loaded"
        
        # Generate using OpenVINO (main generator)
        response = self._pipe.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        # Update stats
        self._stats["queries_processed"] += 1
        
        # Extract concepts (using EVA-Ai integration)
        if self.integration.concept_miner:
            try:
                concepts = self.integration.extract_concepts(prompt, {"response": response})
                self._stats["concepts_extracted"] += len(concepts)
            except:
                pass
        
        return response
    
    def add_layers(self, num: int) -> int:
        """Добавить слои динамически."""
        return self.stack.add_layers(num)
    
    def remove_layers(self, num: int) -> int:
        """Убрать слои динамически."""
        return self.stack.remove_layers(num)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику."""
        return {
            **self._stats,
            **self.stack.get_statistics(),
            "components": self.integration.components_status
        }


# Import config for type hints
from fcp_core.config import FCPConfig