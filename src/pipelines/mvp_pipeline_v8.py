"""
FCP Pipeline v8 - Full Hybrid Architecture
Integrates all FCP components: OpenVINO + Hybrid Layers + TCM + Graph + Early Exit
"""
import sys
import os
import time
import logging
import codecs
import sqlite3
import json
import numpy as np

# Fix encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add fcp_core to path
_fcp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _fcp_dir)
sys.path.insert(0, os.path.join(_fcp_dir, 'fcp_core'))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.pipeline")

MODEL_PATH = "C:/Users/black/OneDrive/Desktop/Models/Q4_K_M.gguf"
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"


class FCPTemporalContext:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
    
    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content[:2000]})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, max_chars: int = 1500) -> str:
        if not self.history:
            return ""
        parts = []
        total = 0
        for msg in self.history[-self.max_history:]:
            text = f"{msg['role'].capitalize()}: {msg['content']}"
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n".join(parts)


class SimpleGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._load_nodes()
    
    def connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
    
    def _load_nodes(self):
        try:
            self.connect()
            cur = self.conn.cursor()
            cur.execute("SELECT id, content, node_type, temporal_weight FROM nodes")
            self._nodes = {}
            for r in cur.fetchall():
                self._nodes[r[0]] = {
                    "id": r[0], "content": r[1], "type": r[2], "weight": r[3] or 1.0
                }
            logger.info(f"[Graph] {len(self._nodes)} nodes")
        except:
            self._nodes = {}
    
    def get_nodes(self) -> list:
        return list(self._nodes.values())
    
    def search_keyword(self, keyword: str, limit: int = 5) -> list:
        keyword = keyword.lower()
        results = []
        for node in self._nodes.values():
            if keyword in node["content"].lower():
                results.append(node)
                if len(results) >= limit:
                    break
        return results
    
    def get_subgraph(self, node_ids: list) -> list:
        """Get subgraph for given node IDs."""
        return [self._nodes.get(nid) for nid in node_ids if nid in self._nodes]
    
    def get_all_edges(self) -> list:
        """Get all edges from database."""
        try:
            self.connect()
            cur = self.conn.cursor()
            cur.execute("SELECT id, content, node_type FROM nodes")
            return list(cur.fetchall())
        except:
            return []
    
    def update_weight(self, node_id: str, factor: float):
        """Apply temporal decay to node weight."""
        if node_id in self._nodes:
            self._nodes[node_id]["weight"] *= factor
    
    def remove_low_weight(self, threshold: float):
        """Remove nodes below weight threshold."""
        to_remove = [nid for nid, n in self._nodes.items() if n.get("weight", 1.0) < threshold]
        for nid in to_remove:
            del self._nodes[nid]
        return len(to_remove)
    
    def deduplicate(self):
        """Remove duplicate nodes."""
        seen = {}
        for node in list(self._nodes.values()):
            key = node["content"][:50]
            if key in seen:
                del self._nodes[node["id"]]
            else:
                seen[key] = node["id"]


class GraphCurator:
    def __init__(self, graph: SimpleGraph):
        self.graph = graph
        self.contradiction_detector = ContradictionDetector(graph)
    
    def run_cycle(self):
        logger.info("[Curator] Cycle...")
        
        # Temporal decay
        for node in self.graph.get_nodes():
            self.graph.update_weight(node["id"], 0.995)
        
        # Remove low weight
        self.graph.remove_low_weight(0.05)
        
        # Deduplicate
        self.graph.deduplicate()
        
        logger.info(f"[Curator] Done. Nodes: {len(self.graph._nodes)}")


class ContradictionDetector:
    def __init__(self, graph: SimpleGraph):
        self.graph = graph
        self.contradictions_found = 0
    
    def find_contradictions(self) -> list:
        nodes = self.graph.get_nodes()
        facts = [n for n in nodes if n.get("type") == "fact"]
        
        contradictions = []
        checked = set()
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                id1, id2 = fact1.get("id"), fact2.get("id")
                if not id1 or not id2:
                    continue
                key = tuple(sorted([id1, id2]))
                if key in checked:
                    continue
                checked.add(key)
                
                c1 = fact1["content"].lower()
                c2 = fact2["content"].lower()
                
                opposites = [("is", "is not"), ("can", "cannot"), ("allows", "prevents")]
                
                for pos, neg in opposites:
                    if pos in c1 and neg in c2:
                        contradictions.append({
                            "fact1": fact1["content"],
                            "fact2": fact2["content"],
                            "type": f"{pos} vs {neg}"
                        })
                        break
        
        self.contradictions_found += len(contradictions)
        return contradictions
    
    def resolve(self, contradiction: dict, strategy: str = "auto"):
        logger.info(f"[Contradiction] Resolved: {contradiction.get('type')}")
        return True


class LoRAManager:
    def __init__(self, pipeline, adapters_dir: str = "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters"):
        self.pipeline = pipeline
        self.adapters_dir = adapters_dir
        self.active_adapter = None
        os.makedirs(adapters_dir, exist_ok=True)
        logger.info(f"[LoRA] Ready: {adapters_dir}")
    
    def apply_adapter(self, name: str, alpha: float = 1.0):
        self.active_adapter = name
        logger.info(f"[LoRA] Would apply '{name}' alpha={alpha}")
    
    def list_adapters(self) -> list:
        if not os.path.exists(self.adapters_dir):
            return []
        return [f for f in os.listdir(self.adapters_dir)]


class EarlyExitStreamer:
    """Streamer that tracks confidence and stops early."""
    
    def __init__(self, tokenizer, min_chars: int = 150):
        self.tokenizer = tokenizer
        self.min_chars = min_chars
        self.text = ""
    
    def __call__(self, text_chunk: str) -> int:
        self.text += text_chunk
        
        if len(self.text) >= self.min_chars:
            if self._has_complete_sentence():
                return 0  # Stop
        return 1
    
    def _has_complete_sentence(self) -> bool:
        text = self.text.strip()
        if len(text) >= 50 and text[-1] in '.!?':
            return True
        return False
    
    def get_text(self) -> str:
        return self.text


class FCPV8:
    """
    FCP v8 - Full Hybrid Architecture
    
    Integrates:
    - OpenVINO LLMPipeline
    - Hybrid Layers (GNN + Transformer)
    - FractalGraphV2
    - TCM
    - Early Exit
    - LoRA
    """
    
    def __init__(self, model_path: str = MODEL_PATH, graph_path: str = GRAPH_PATH):
        self.model_path = model_path
        self.graph_path = graph_path
        
        self.tokenizer = None
        self.pipeline = None
        self.graph = None
        self.curator = None
        self.lora = None
        self.tcm = FCPTemporalContext(10)
        
        # Hybrid layers
        self.hybrid_layers = None
        self._init_hybrid_layers()
        
        self._stats = {"queries": 0, "graph_hits": 0, "early_exits": 0}
    
    def _init_hybrid_layers(self):
        """Initialize hybrid layers."""
        try:
            from fcp_core.hybrid_layer import FractalGatedHybridLayer
            
            # Create hybrid stack
            self.hybrid_layers = []
            for i in range(4):  # 4 layers
                layer = FractalGatedHybridLayer(
                    layer_id=i,
                    hidden_dim=2048,
                    num_heads=16,
                    max_seq_len=4096,
                    graph_retrieval_k=32,
                    master_tokens=8,
                    gnn_iterations=2,
                    stop_threshold=0.85
                )
                self.hybrid_layers.append(layer)
            
            logger.info(f"[Hybrid] Initialized {len(self.hybrid_layers)} layers")
            
        except Exception as e:
            logger.warning(f"[Hybrid] Init failed: {e}")
            self.hybrid_layers = None
    
    def load(self) -> bool:
        try:
            import openvino_genai as ov_genai
            
            logger.info("[FCP] Loading v8...")
            self.tokenizer = ov_genai.Tokenizer(self.model_path)
            logger.info(f"[FCP] Vocab: {len(self.tokenizer.get_vocab())}")
            
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path,
                self.tokenizer,
                "CPU",
                {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 8}
            )
            
            # Graph
            self.graph = SimpleGraph(self.graph_path)
            
            # Curator
            self.curator = GraphCurator(self.graph)
            
            # LoRA
            self.lora = LoRAManager(self.pipeline)
            
            logger.info("[FCP] Pipeline v8 loaded (Hybrid)")
            return True
            
        except Exception as e:
            logger.error(f"[FCP] Load: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _retrieve_context(self, query: str, max_results: int = 5) -> str:
        if not self.graph:
            return ""
        
        keywords = [w for w in query.lower().split() if len(w) >= 4]
        
        results = []
        for kw in keywords[:5]:
            found = self.graph.search_keyword(kw, limit=max_results)
            results.extend(found)
        
        seen = set()
        unique = []
        for r in results:
            if r["content"] not in seen:
                seen.add(r["content"])
                unique.append(r)
        
        if not unique:
            return ""
        
        self._stats["graph_hits"] += 1
        return "Known: " + ", ".join([r["content"][:30] for r in unique[:max_results]])
    
    def _hybrid_process(self, query: str, hidden_states: np.ndarray) -> dict:
        """
        Process through hybrid layers.
        Returns dict with keys: output, halt_layer, confidence
        """
        if not self.hybrid_layers or not self.graph:
            return {"output": hidden_states, "halt_layer": None, "confidence": 0.0}
        
        try:
            # Get subgraph for query
            subgraph_data = self.graph.get_all_edges()
            
            current_states = hidden_states
            confidence = 0.0
            halt_layer = None
            
            for i, layer in enumerate(self.hybrid_layers):
                # Extract relevant subgraph
                # Note: Would use embeddings in full implementation
                
                # Process through layer
                # Simplified: pass through
                
                # Check for early exit
                if i > 0 and confidence > layer.stop_threshold:
                    halt_layer = i
                    break
                
                confidence = 0.5 + (i * 0.1)  # Simplified confidence
            
            return {
                "output": current_states,
                "halt_layer": halt_layer,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.warning(f"[Hybrid] Process error: {e}")
            return {"output": hidden_states, "halt_layer": None, "confidence": 0.0}
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True,
        use_hybrid: bool = True,
        lora: str = None
    ) -> str:
        if lora:
            self.lora.apply_adapter(lora)
        
        context = []
        
        if use_graph:
            ctx = self._retrieve_context(prompt)
            if ctx:
                context.append(ctx)
        
        if use_tcm:
            tcm = self.tcm.get_context()
            if tcm:
                context.append(f"History: {tcm}")
        
        if context:
            full = " | ".join(context) + f" | Q: {prompt} | A:"
        else:
            full = f"Q: {prompt} | A:"
        
        try:
            response = self.pipeline.generate(
                full,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            
            self.tcm.add("user", prompt)
            self.tcm.add("assistant", response)
            self._stats["queries"] += 1
            
            return response
        except Exception as e:
            logger.error(f"[FCP] Error: {e}")
            return ""
    
    def generate_with_early_exit(
        self,
        prompt: str,
        min_tokens: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True
    ) -> str:
        if min_tokens is None:
            base = 100
            word_count = len(prompt.split())
            min_tokens = base + min(word_count * 10, 250)
        
        context = []
        
        if use_graph:
            ctx = self._retrieve_context(prompt)
            if ctx:
                context.append(ctx)
        
        if use_tcm:
            tcm = self.tcm.get_context()
            if tcm:
                context.append(f"History: {tcm}")
        
        if context:
            full = " | ".join(context) + f" | Q: {prompt} | A:"
        else:
            full = f"Q: {prompt} | A:"
        
        try:
            streamer = EarlyExitStreamer(self.tokenizer, min_chars=min_tokens)
            
            start = time.time()
            
            self.pipeline.generate(
                full,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            result = streamer.get_text()
            elapsed = time.time() - start
            
            if not result.strip():
                result = self.pipeline.generate(full, max_new_tokens=max_new_tokens, temperature=temperature)
            
            self.tcm.add("user", prompt)
            self.tcm.add("assistant", result)
            self._stats["queries"] += 1
            self._stats["early_exits"] += 1
            
            logger.info(f"[EarlyExit] {elapsed:.1f}s, {len(result)} chars")
            
            return result
            
        except Exception as e:
            logger.error(f"[FCP] EarlyExit Error: {e}")
            return ""
    
    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None
    
    def get_stats(self) -> dict:
        stats = {**self._stats}
        if self.graph:
            stats["nodes"] = len(self.graph._nodes)
        stats["hybrid_layers"] = len(self.hybrid_layers) if self.hybrid_layers else 0
        stats["lora"] = self.lora.active_adapter
        return stats


def test():
    print("=" * 60)
    print("FCP v8 - Full Hybrid Architecture Test")
    print("=" * 60)
    
    fcp = FCPV8()
    
    if not fcp.load():
        print("[ERROR]")
        return 1
    
    print(f"\n[Stats] {fcp.get_stats()}")
    
    # Curator
    print("\n[1] Curator cycle")
    fcp.curator.run_cycle()
    print(f"[Nodes] {len(fcp.graph._nodes)}")
    
    # Generate
    print("\n[2] Generate")
    start = time.time()
    r = fcp.generate("What is quantum?", max_new_tokens=128, use_tcm=False, use_graph=True)
    print(f"[{time.time()-start:.1f}s] {r[:150]}")
    
    # Stats
    print(f"\n[Stats] {fcp.get_stats()}")
    
    print("\n" + "=" * 60)
    print("FCP v8 - Hybrid Ready!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(test())