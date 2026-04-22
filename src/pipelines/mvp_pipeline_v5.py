"""
FCP Pipeline v5 - Graph Curator + LoRA Infrastructure
"""
import sys
import os
import time
import logging
import codecs
import sqlite3
import json
import threading

# Fix encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

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
            self._nodes = {r[0]: {"id": r[0], "content": r[1], "type": r[2], "weight": r[3] or 1.0} for r in cur.fetchall()}
            logger.info(f"[Graph] {len(self._nodes)} nodes")
        except:
            self._nodes = {}
    
    def get_nodes(self) -> list:
        return list(self._nodes.values())
    
    def search(self, keyword: str, limit: int = 5) -> list:
        keyword = keyword.lower()
        results = []
        for node in self._nodes.values():
            if keyword in node["content"].lower():
                results.append(node)
                if len(results) >= limit:
                    break
        return results
    
    def update_weight(self, node_id: str, factor: float = 0.99):
        """Apply temporal decay to node weight."""
        if node_id in self._nodes:
            self._nodes[node_id]["weight"] *= factor
            self.connect()
            cur = self.conn.cursor()
            cur.execute("UPDATE nodes SET temporal_weight = ? WHERE id = ?", 
                      (self._nodes[node_id]["weight"], node_id))
            self.conn.commit()
    
    def remove_low_weight(self, threshold: float = 0.1):
        """Remove nodes with low temporal weight."""
        to_remove = [nid for nid, n in self._nodes.items() if n["weight"] < threshold]
        if to_remove:
            self.connect()
            cur = self.conn.cursor()
            cur.execute("DELETE FROM nodes WHERE id IN ({})".format(",".join(["?"]*len(to_remove))), to_remove)
            self.conn.commit()
            for nid in to_remove:
                del self._nodes[nid]
            logger.info(f"[Graph] Removed {len(to_remove)} low-weight nodes")
    
    def deduplicate(self):
        """Remove duplicate content."""
        content_map = {}
        to_remove = []
        for node in self._nodes.values():
            content = node["content"].lower()
            if content in content_map:
                to_remove.append(node["id"])
            else:
                content_map[content] = node["id"]
        
        if to_remove:
            self.connect()
            cur = self.conn.cursor()
            cur.execute("DELETE FROM nodes WHERE id IN ({})".format(",".join(["?"]*len(to_remove))), to_remove)
            self.conn.commit()
            for nid in to_remove:
                del self._nodes[nid]
            logger.info(f"[Graph] Removed {len(to_remove)} duplicates")


class GraphCurator:
    """Graph Curator - фоновый процесс курации."""
    
    def __init__(self, graph: SimpleGraph):
        self.graph = graph
        self.running = True
    
    def run_cycle(self):
        """Run one curation cycle."""
        logger.info("[Curator] Starting cycle...")
        
        # 1. Temporal decay
        for node in self.graph.get_nodes():
            self.graph.update_weight(node["id"], 0.995)
        
        # 2. Remove low weight
        self.graph.remove_low_weight(0.05)
        
        # 3. Deduplicate
        self.graph.deduplicate()
        
        logger.info(f"[Curator] Cycle complete. Nodes: {len(self.graph._nodes)}")
    
    def start_background(self, interval_hours: float = 1.0):
        """Start background curator."""
        def worker():
            while self.running:
                try:
                    self.run_cycle()
                except Exception as e:
                    logger.warning(f"[Curator] Error: {e}")
                time.sleep(interval_hours * 3600)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        logger.info("[Curator] Background started")


class LoRAManager:
    """LoRA Manager - infrastructure for adapters."""
    
    def __init__(self, pipeline, adapters_dir: str = "C:/Users/black/OneDrive/Desktop/FCP/lora_adapters"):
        self.pipeline = pipeline
        self.adapters_dir = adapters_dir
        self.active_adapter = None
        
        os.makedirs(adapters_dir, exist_ok=True)
        
        logger.info(f"[LoRA] Manager ready. Adapters dir: {adapters_dir}")
    
    def apply_adapter(self, name: str, alpha: float = 1.0):
        """Apply LoRA adapter (prepared for future use)."""
        # Infrastructure only - actual adapter loading would need ov_genai.Adapter
        self.active_adapter = name
        logger.info(f"[LoRA] Would apply adapter '{name}' with alpha={alpha}")
    
    def list_adapters(self) -> list:
        """List available adapters."""
        if not os.path.exists(self.adapters_dir):
            return []
        return [f for f in os.listdir(self.adapters_dir) if f.endswith('. adapter')]


class FCPV5:
    """FCP v5 - с Graph Curator и LoRA infrastructure."""
    
    def __init__(self, model_path: str = MODEL_PATH, graph_path: str = GRAPH_PATH):
        self.model_path = model_path
        self.graph_path = graph_path
        
        self.tokenizer = None
        self.pipeline = None
        self.graph = None
        self.curator = None
        self.lora = None
        self.tcm = FCPTemporalContext(10)
        
        self._stats = {"queries": 0, "graph_hits": 0, "curator_runs": 0}
    
    def load(self) -> bool:
        try:
            import openvino_genai as ov_genai
            
            logger.info("[FCP] Loading pipeline...")
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
            
            self._loaded = True
            logger.info("[FCP] Pipeline v5 loaded")
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
            found = self.graph.search(kw, limit=max_results)
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True
    ) -> str:
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
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_stats(self) -> dict:
        stats = {**self._stats}
        if self.graph:
            stats["nodes"] = len(self.graph._nodes)
        stats["lora_active"] = self.lora.active_adapter if self.lora else None
        return stats


def test():
    print("=" * 50)
    print("FCP v5 - Curator + LoRA Test")
    print("=" * 50)
    
    fcp = FCPV5()
    
    if not fcp.load():
        print("[ERROR]")
        return 1
    
    # Test curator cycle
    print("\n[1] Curator cycle")
    fcp.curator.run_cycle()
    print(f"[Nodes] {len(fcp.graph._nodes)}")
    
    # Test generation
    print("\n[2] Generate")
    start = time.time()
    r = fcp.generate("What is quantum mechanics?", max_new_tokens=256, use_graph=True)
    print(f"[{time.time()-start:.1f}s]\n{r[:200]}...")
    
    # Test LoRA
    print("\n[3] LoRA adapters")
    adapters = fcp.lora.list_adapters()
    print(f"[Available] {adapters}")
    fcp.lora.apply_adapter("test_adapter", alpha=0.8)
    
    # Stats
    print(f"\n[Stats] {fcp.get_stats()}")
    
    print("\n" + "=" * 50)
    print("FCP v5 - Ready!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(test())