"""
FCP Pipeline V15 - Полный пайплайн

Финальная сборка FCP v15 из "Последовательные решения.txt"
"""
import os
from typing import Optional, Dict, Any

# Import components
try:
    import openvino_genai as ov_genai
    HAS_OV_GENAI = True
except ImportError:
    HAS_OV_GENAI = False

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class FCPPipelineV15:
    """
    Полный FCP Pipeline v15.
    
    Объединяет все компоненты:
    - OpenVINO GenAI pipeline
    - GNN Encoder (OpenVINO)
    - Adaptive Fusion Injector
    - Shadow LoRA Manager
    - Tool Orchestrator
    - Thinking Controller
    - Scenario TCM
    - Expert System
    - Clarification Generator
    - Attribution Report
    - Semantic Cache Evictor
    """
    
    def __init__(
        self,
        model_path: str,
        graph_path: str,
        gnn_ov_path: Optional[str] = None,
        lora_dir: Optional[str] = None,
        draft_model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.graph_path = graph_path
        self.gnn_ov_path = gnn_ov_path
        self.lora_dir = lora_dir
        self.draft_model_path = draft_model_path
        
        # Initialize components
        self._init_tokenizer()
        self._init_pipeline()
        self._init_gnn()
        self._init_injector()
        self._init_graph()
        self._init_encoder()
        self._init_lora_manager()
        self._init_tools()
        self._init_tcm()
        self._init_experts()
        self._init_attribution()
        self._init_evictor()
    
    def _init_tokenizer(self):
        """Инициализировать токенизатор."""
        if HAS_TRANSFORMERS and os.path.exists(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            self.tokenizer = None
    
    def _init_pipeline(self):
        """Инициализировать OpenVINO pipeline."""
        if not HAS_OV_GENAI:
            self.pipeline = None
            return
        
        scheduler = self._make_scheduler()
        
        try:
            config = {"scheduler_config": scheduler}
            
            if self.draft_model_path:
                config["draft_model"] = self.draft_model_path
            
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path,
                "CPU",
                kv_cache_precision="u8",
                **config
            )
        except Exception as e:
            print(f"Pipeline init error: {e}")
            self.pipeline = None
    
    def _make_scheduler(self):
        """Создать конфиг шедулера."""
        sc = ov_genai.SchedulerConfig()
        sc.cache_size = 4
        sc.max_num_seqs = 1
        sc.max_num_batched_tokens = 2048
        sc.enable_prefix_caching = True
        sc.use_cache_eviction = True
        return sc
    
    def _init_gnn(self):
        """Инициализировать GNN."""
        try:
            from fcp_gnn.gnn_runtime_ov import GNNEncoderOV
            if self.gnn_ov_path and os.path.exists(self.gnn_ov_path):
                self.gnn = GNNEncoderOV(self.gnn_ov_path)
            else:
                self.gnn = None
        except Exception as e:
            print(f"GNN init error: {e}")
            self.gnn = None
    
    def _init_injector(self):
        """Инициализировать инъектор."""
        try:
            from fcp_gnn.injector import AdaptiveFusionInjector
            self.injector = AdaptiveFusionInjector()
        except Exception:
            self.injector = None
    
    def _init_graph(self):
        """Инициализировать граф."""
        # Используем FractalGraphV2 из EVA
        try:
            from eva_ai.memory.fractal_graph_v2 import FractalGraphV2
            self.graph = FractalGraphV2(self.graph_path)
        except ImportError:
            self.graph = None
    
    def _init_encoder(self):
        """Инициализировать энкодер."""
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.encoder = SentenceTransformer('intfloat/multilingual-e5-small')
            except Exception:
                self.encoder = None
        else:
            self.encoder = None
    
    def _init_lora_manager(self):
        """Инициализировать LoRA менеджер."""
        try:
            from fcp_lora.shadow_lora_ov import ShadowLoRAManagerOV
            self.lora_mgr = ShadowLoRAManagerOV(
                self.model_path,
                scheduler_config={}
            )
        except Exception:
            self.lora_mgr = None
    
    def _init_tools(self):
        """Инициализировать инструменты."""
        try:
            from fcp_tools.orchestrator import ToolOrchestrator
            from fcp_tools.thinking_controller import ThinkingController, SimpleRoutingEngine
            
            self.tool_orch = ToolOrchestrator(self.graph)
            self.think_ctrl = ThinkingController(
                None,
                SimpleRoutingEngine(),
                self.tokenizer
            ) if self.tokenizer else None
        except Exception as e:
            print(f"Tools init error: {e}")
            self.tool_orch = None
            self.think_ctrl = None
    
    def _init_tcm(self):
        """Инициализировать TCM."""
        try:
            from fcp_tools.scenario_tcm import ScenarioTCM
            self.tcm = ScenarioTCM(self.graph) if self.graph else None
        except Exception:
            self.tcm = None
    
    def _init_experts(self):
        """Инициализировать экспертов."""
        try:
            from fcp_tools.expert_system import ExpertSystem
            self.experts = ExpertSystem([], self.graph.contradictions if self.graph else None)
        except Exception:
            self.experts = None
    
    def _init_attribution(self):
        """Инициализировать атрибуцию."""
        try:
            from fcp_tools.attribution import AttributionReport
            self.attribution = AttributionReport()
        except Exception:
            self.attribution = None
    
    def _init_evictor(self):
        """Инициализировать evictor."""
        try:
            from fcp_tools.semantic_cache_evictor import SemanticCacheEvictor
            self.evictor = SemanticCacheEvictor(self.gnn, self.graph)
        except Exception:
            self.evictor = None
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Основной метод генерации.
        
        Args:
            prompt: пользовательский запрос
            max_tokens: макс. новых токенов
        
        Returns:
            сгенерированный ответ
        """
        # 1. Encode запрос
        q_emb = self._encode_query(prompt)
        
        # 2. Подграф и графовый вектор
        sub = self._retrieve_subgraph(q_emb)
        gv, gw = self._encode_subgraph(sub)
        
        # 3. Управление мышлением
        think = self._should_enable_thinking(prompt)
        chat = self._build_chat_prompt(prompt, think)
        
        # 4. Инъекция графового контекста
        graph_text = self._format_graph_context(sub)
        full_prompt = f"{graph_text}\n{chat}" if graph_text else chat
        
        # 5. Генерация
        if self.pipeline:
            try:
                resp = self.pipeline.generate(full_prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                resp = f"Generation error: {e}"
        else:
            resp = "Pipeline not initialized"
        
        # 6. Обработка инструментов
        if self.tool_orch:
            resp = self.tool_orch.process_response(resp)
        
        # 7. Сохранение в эпизодическую память
        self._save_to_tcm(prompt, resp, q_emb)
        
        # 8. Атрибуция
        if self.attribution:
            self.attribution.track(
                0,
                None,
                sub.get("node_ids", []),
                None
            )
        
        return resp
    
    def _encode_query(self, query: str):
        """Encode запрос."""
        if self.encoder:
            return self.encoder.encode(query, normalize_embeddings=True)
        return None
    
    def _retrieve_subgraph(self, q_emb):
        """Получить подграф."""
        if self.gnn and q_emb is not None:
            return self.gnn.retrieve_subgraph(q_emb, k=10)
        return {"x": None, "edge_index": None, "node_ids": [], "contents": []}
    
    def _encode_subgraph(self, sub):
        """Encode подграф."""
        if self.gnn and sub.get("x") is not None:
            return self.gnn.encode(sub["x"], sub["edge_index"])
        return None, None
    
    def _should_enable_thinking(self, prompt: str) -> bool:
        """Определить нужно ли мышление."""
        if self.think_ctrl:
            return self.think_ctrl.should_enable_thinking(prompt)
        return False
    
    def _build_chat_prompt(self, prompt: str, enable_thinking: bool) -> str:
        """Построить chat промпт."""
        if self.think_ctrl:
            return self.think_ctrl.build_chat_prompt(prompt, enable_thinking)
        return prompt
    
    def _format_graph_context(self, sub) -> str:
        """Форматировать контекст из графа."""
        contents = sub.get("contents", [])
        if not contents:
            return ""
        limited = contents[:5]
        lines = [f"- {t}" for t in limited]
        return "Контекст из графа:\n" + "\n".join(lines)
    
    def _save_to_tcm(self, prompt: str, response: str, q_emb):
        """Сохранить в TCM."""
        if self.tcm and q_emb is not None:
            import numpy as np
            self.tcm.add_turn("user", prompt, q_emb)
            resp_emb = self.encoder.encode(response) if self.encoder else np.zeros(384)
            self.tcm.add_turn("assistant", response, resp_emb)
    
    def set_adapter(self, adapter_name: str, alpha: float = 0.8):
        """Установить LoRA адаптер."""
        if self.lora_mgr:
            self.lora_mgr.atomic_swap(adapter_name, alpha)
    
    def get_explanation(self) -> str:
        """Получить объяснение атрибуции."""
        if self.attribution:
            return self.attribution.explain()
        return "No attribution data"


# Factory function
def create_fcp_pipeline(
    model_path: str,
    graph_path: str,
    **kwargs
) -> FCPPipelineV15:
    """Создать FCP pipeline."""
    return FCPPipelineV15(model_path, graph_path, **kwargs)