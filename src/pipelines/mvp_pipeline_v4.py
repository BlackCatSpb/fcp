"""
FCP Pipeline v4 - Simplified (без encoder для скорости)
"""
import sys
import os
import time
import logging
import codecs
import sqlite3

# Fix encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.pipeline")

# Paths
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
        self._load_cache()
    
    def connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
    
    def _load_cache(self):
        try:
            self.connect()
            cur = self.conn.cursor()
            cur.execute("SELECT id, content, node_type FROM nodes")
            self._nodes = [{"id": r[0], "content": r[1], "type": r[2]} for r in cur.fetchall()]
            logger.info(f"[Graph] {len(self._nodes)} nodes")
        except Exception as e:
            self._nodes = []
    
    def search(self, keyword: str, limit: int = 5) -> list:
        keyword = keyword.lower()
        results = []
        for node in self._nodes:
            if keyword in node["content"].lower():
                results.append(node)
                if len(results) >= limit:
                    break
        return results


class FCPV4:
    """FCP v4 - Simplified pipeline."""
    
    def __init__(self, model_path: str = MODEL_PATH, graph_path: str = GRAPH_PATH):
        self.model_path = model_path
        self.graph_path = graph_path
        
        self.tokenizer = None
        self.pipeline = None
        self.graph = None
        self.tcm = FCPTemporalContext(10)
        
        self._stats = {"queries": 0, "graph_hits": 0}
    
    def load(self) -> bool:
        try:
            import openvino_genai as ov_genai
            
            logger.info("[FCP] Loading...")
            self.tokenizer = ov_genai.Tokenizer(self.model_path)
            logger.info(f"[FCP] Vocab: {len(self.tokenizer.get_vocab())}")
            
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path,
                self.tokenizer,
                "CPU",
                {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 8}
            )
            
            self.graph = SimpleGraph(self.graph_path)
            logger.info("[FCP] Pipeline v4 loaded")
            return True
            
        except Exception as e:
            logger.error(f"[FCP] Load: {e}")
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
        return self.pipeline is not None
    
    def get_stats(self) -> dict:
        return self._stats


def test():
    print("=" * 50)
    print("FCP v4 Test")
    print("=" * 50)
    
    fcp = FCPV4()
    
    if not fcp.load():
        print("[ERROR]")
        return 1
    
    # Test
    print("\n[1] Search")
    results = fcp.graph.search("quantum")
    print(f"[Found] {len(results)}")
    for r in results[:3]:
        print(f"  - {r['content']}")
    
    print("\n[2] Generate")
    start = time.time()
    r = fcp.generate("What is quantum mechanics?", max_new_tokens=256, use_graph=True)
    print(f"[{time.time()-start:.1f}s] {r[:200]}...")
    
    print("\n[3] Continue")
    r2 = fcp.generate("Tell me more", max_new_tokens=256, use_tcm=True)
    print(f"[{r2[:150]}...")
    
    print(f"\n[Stats] {fcp.get_stats()}")
    
    print("\n" + "=" * 50)
    print("FCP v4 - Ready!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(test())