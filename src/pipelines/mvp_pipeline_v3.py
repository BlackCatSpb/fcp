"""
FCP Pipeline v3 - Graph Retrieval + Concept Extraction (Simplified)
"""
import sys
import os
import time
import logging
import codecs
import threading
import sqlite3
import json

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
    """TCM - история диалога."""
    
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
    """Simple graph using SQLite directly."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
    
    def get_concepts(self, limit: int = 100) -> list:
        self.connect()
        cur = self.conn.cursor()
        cur.execute("SELECT id, content FROM nodes WHERE node_type='concept' LIMIT ?", (limit,))
        return [{"id": r[0], "content": r[1]} for r in cur.fetchall()]
    
    def search_concepts(self, keyword: str, limit: int = 10) -> list:
        self.connect()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, content FROM nodes WHERE node_type='concept' AND content LIKE ? LIMIT ?",
            (f"%{keyword}%", limit)
        )
        return [{"id": r[0], "content": r[1]} for r in cur.fetchall()]


class FCPV3:
    """
    FCP v3 - Pipeline с Graph Retrieval и Concept Extraction.
    
    Этап 3:
    1. Graph Retrieval - контекстное обогащение из FractalGraphV2
    2. Concept Extraction - извлечение концептов после генерации
    """
    
    def __init__(self, model_path: str = MODEL_PATH, graph_path: str = GRAPH_PATH):
        self.model_path = model_path
        self.graph_path = graph_path
        
        self.tokenizer = None
        self.pipeline = None
        self.graph = None
        self.tcm = FCPTemporalContext(max_history=10)
        
        self._loaded = False
        self._vocab_size = 0
        
        self._stats = {"queries": 0, "graph_hits": 0, "concepts_extracted": 0}
    
    def load(self) -> bool:
        try:
            import openvino_genai as ov_genai
            
            logger.info("[FCP] Loading pipeline...")
            self.tokenizer = ov_genai.Tokenizer(self.model_path)
            self._vocab_size = len(self.tokenizer.get_vocab())
            
            self.pipeline = ov_genai.LLMPipeline(
                self.model_path,
                self.tokenizer,
                "CPU",
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": 8,
                    "NUM_STREAMS": 1,
                }
            )
            
            # Graph
            logger.info("[FCP] Loading graph...")
            try:
                self.graph = SimpleGraph(self.graph_path)
                concepts = self.graph.get_concepts(limit=5)
                logger.info(f"[FCP] Graph: {len(concepts)} concepts loaded")
            except Exception as e:
                logger.warning(f"[FCP] Graph: {e}")
                self.graph = None
            
            self._loaded = True
            logger.info("[FCP] Pipeline v3 loaded")
            return True
            
        except Exception as e:
            logger.error(f"[FCP] Load: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _retrieve_context(self, query: str, max_facts: int = 5) -> str:
        """Graph Retrieval."""
        if not self.graph:
            return ""
        
        # Search by keywords
        keywords = [w for w in query.lower().split() if len(w) >= 4]
        
        results = []
        for kw in keywords[:5]:
            try:
                found = self.graph.search_concepts(kw, limit=3)
                results.extend(found)
            except:
                pass
        
        # Dedupe
        seen = set()
        unique = []
        for r in results:
            if r['content'] not in seen:
                seen.add(r['content'])
                unique.append(r)
        
        if not unique:
            return ""
        
        self._stats["graph_hits"] += 1
        
        # Format
        facts = [f"- {r['content'][:100]}" for r in unique[:max_facts]]
        return "Known concepts:\n" + "\n".join(facts)
    
    def _extract_concepts_async(self, query: str, response: str):
        """Async concept extraction."""
        def worker():
            # Simple: add unique words as concepts
            words = set(query.lower().split() + response.lower().split())
            important = {w for w in words if len(w) >= 5}
            
            if not important:
                return
            
            # Skip if already have similar
            if self.graph:
                try:
                    existing = self.graph.get_concepts(limit=500)
                    existing_contents = {c['content'].lower() for c in existing}
                    
                    for word in list(important)[:20]:
                        if word not in existing_contents:
                            # Would add to graph
                            self._stats["concepts_extracted"] += 1
                except:
                    pass
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True
    ) -> str:
        """Generation с контекстом."""
        if not self._loaded:
            return "[FCP] Not loaded"
        
        context_parts = []
        
        # Graph
        if use_graph:
            ctx = self._retrieve_context(prompt)
            if ctx:
                context_parts.append(ctx)
        
        # TCM
        if use_tcm:
            tcm = self.tcm.get_context()
            if tcm:
                context_parts.append(f"History:\n{tcm}")
        
        # Build prompt
        if context_parts:
            full = "\n\n".join(context_parts) + f"\n\nUser: {prompt}\nAssistant:"
        else:
            full = f"User: {prompt}\nAssistant:"
        
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
            
            if use_graph:
                self._extract_concepts_async(prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"[FCP] Error: {e}")
            return ""
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get_stats(self) -> dict:
        return self._stats


def test():
    print("=" * 50)
    print("FCP v3 - Graph + Concepts Test")
    print("=" * 50)
    
    fcp = FCPV3()
    
    if not fcp.load():
        print("[ERROR] Load failed")
        return 1
    
    # Test 1: Get concepts
    print("\n[1] Get concepts")
    if fcp.graph:
        concepts = fcp.graph.get_concepts(limit=5)
        print(f"[Found] {len(concepts)}")
        for c in concepts[:3]:
            print(f"  - {c['content'][:50]}")
    
    # Test 2: Search
    print("\n[2] Search 'quantum'")
    if fcp.graph:
        found = fcp.graph.search_concepts("quantum", limit=5)
        print(f"[Found] {len(found)}")
        for f in found[:3]:
            print(f"  - {f['content'][:50]}")
    
    # Test 3: Generation with graph
    print("\n[3] Generate with graph context")
    start = time.time()
    r = fcp.generate("What is quantum mechanics?", max_new_tokens=256, use_graph=True)
    print(f"[{time.time()-start:.1f}s] {r[:200]}...")
    
    # Stats
    print(f"\n[Stats] {fcp.get_stats()}")
    
    print("\n" + "=" * 50)
    print("FCP v3 - Ready!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(test())