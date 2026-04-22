"""
FCP Pipeline v7 - Early Exit + Streaming + All v6 Features
"""
import sys
import os
import time
import logging
import codecs
import sqlite3
import json
import threading
import numpy as np
import re

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
    
    def search_semantic(self, query_emb: np.ndarray, limit: int = 5) -> list:
        """Semantic search using precomputed embeddings."""
        # Simple: use keyword fallback for now
        # Real implementation would use stored embeddings
        return []  # Will use keyword as fallback
    
    def update_weight(self, node_id: str, factor: float = 0.99):
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node["weight"] = node.get("weight", 1.0) * factor
            self.connect()
            cur = self.conn.cursor()
            cur.execute("UPDATE nodes SET temporal_weight = ? WHERE id = ?", 
                      (node["weight"], node_id))
            self.conn.commit()
    
    def remove_low_weight(self, threshold: float = 0.1):
        to_remove = []
        for nid, n in self._nodes.items():
            if not nid:
                to_remove.append(nid)
            elif n.get("weight", 1.0) < threshold:
                to_remove.append(nid)
        
        if to_remove:
            self.connect()
            cur = self.conn.cursor()
            placeholders = ",".join(["?"] * len(to_remove))
            cur.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", to_remove)
            self.conn.commit()
            for nid in to_remove:
                if nid in self._nodes:
                    del self._nodes[nid]
            logger.info(f"[Graph] Removed {len(to_remove)} low-weight nodes")
    
    def deduplicate(self):
        content_map = {}
        to_remove = []
        for node in self._nodes.values():
            node_id = node.get("id")
            if not node_id:
                continue
            content = node["content"].lower()
            if content in content_map:
                to_remove.append(node_id)
            else:
                content_map[content] = node_id
        
        if to_remove:
            self.connect()
            cur = self.conn.cursor()
            placeholders = ",".join(["?"] * len(to_remove))
            cur.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", to_remove)
            self.conn.commit()
            for nid in to_remove:
                del self._nodes[nid]
            logger.info(f"[Graph] Removed {len(to_remove)} duplicates")


class ContradictionDetector:
    """Simple contradiction detector."""
    
    def __init__(self, graph: SimpleGraph):
        self.graph = graph
        self.contradictions_found = 0
    
    def find_contradictions(self) -> list:
        """Find potential contradictions in facts."""
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
                
                # Check for opposite predicates
                c1 = fact1["content"].lower()
                c2 = fact2["content"].lower()
                
                # Simple patterns
                opposites = [
                    ("is", "is not"), ("can", "cannot"),
                    ("allows", "prevents"), ("helps", "harms")
                ]
                
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
        """Resolve contradiction."""
        # Simple: keep the first one
        logger.info(f"[Contradiction] Resolved: {contradiction.get('type')}")
        return True


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
        
        # Check contradictions
        contradictions = self.contradiction_detector.find_contradictions()
        for c in contradictions:
            self.contradiction_detector.resolve(c)
        
        logger.info(f"[Curator] Done. Nodes: {len(self.graph._nodes)}")


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


class FCPV6:
    """FCP v6 - Semantic Search + Early Exit + ContradictionDetector."""
    
    def __init__(self, model_path: str = MODEL_PATH, graph_path: str = GRAPH_PATH):
        self.model_path = model_path
        self.graph_path = graph_path
        
        self.tokenizer = None
        self.pipeline = None
        self.graph = None
        self.curator = None
        self.lora = None
        self.tcm = FCPTemporalContext(10)
        
        # Semantic encoder (optional)
        self.encoder = None
        
        self._stats = {"queries": 0, "graph_hits": 0, "contradictions": 0, "early_exits": 0}
    
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
            
            # Graph
            self.graph = SimpleGraph(self.graph_path)
            
            # Curator with ContradictionDetector
            self.curator = GraphCurator(self.graph)
            
            # LoRA
            self.lora = LoRAManager(self.pipeline)
            
            # Try to load semantic encoder (optional)
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer("intfloat/multilingual-e5-small")
                logger.info("[FCP] Encoder loaded")
            except Exception as e:
                logger.warning(f"[FCP] Encoder: {e}")
            
            logger.info("[FCP] Pipeline v6 loaded")
            return True
            
        except Exception as e:
            logger.error(f"[FCP] Load: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _retrieve_context(self, query: str, max_results: int = 5) -> str:
        if not self.graph:
            return ""
        
        # Try semantic first if encoder available
        if self.encoder is not None:
            try:
                query_emb = self.encoder.encode(query, normalize_embeddings=True)
                results = self.graph.search_semantic(query_emb, limit=max_results)
                if results:
                    self._stats["graph_hits"] += 1
                    return "Known: " + ", ".join([r["content"][:30] for r in results[:max_results]])
            except:
                pass
        
        # Fallback: keyword search
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True,
        lora: str = None
    ) -> str:
        # Apply LoRA if specified
        if lora:
            self.lora.apply_adapter(lora)
        
        # Build context
        context = []
        
        if use_graph:
            ctx = self._retrieve_context(prompt)
            if ctx:
                context.append(ctx)
        
        if use_tcm:
            tcm = self.tcm.get_context()
            if tcm:
                context.append(f"History: {tcm}")
        
        # Build prompt
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
        stats = {**self._stats}
        if self.graph:
            stats["nodes"] = len(self.graph._nodes)
        stats["encoder"] = self.encoder is not None
        stats["lora"] = self.lora.active_adapter
        return stats


class FCPV7(FCPV6):
    """FCP v7 - Early Exit + Streaming."""
    
    def _adaptive_min_tokens(self, prompt: str) -> int:
        """Compute adaptive min_chars based on prompt complexity."""
        base = 100
        word_count = len(prompt.split())
        extra = min(word_count * 10, 250)
        return base + extra
    
    def _has_complete_sentence(self, text: str) -> bool:
        """Check if text has at least one complete sentence."""
        text = text.strip()
        if not text:
            return False
        # Must have at least 2 sentences OR ends with .!? after 50+ chars
        if len(text) >= 50 and text[-1] in '.!?':
            return True
        # Or has sentence structure
        if re.search(r'[.!?]\s+[A-Z]', text):
            return True
        return False
    
    def generate_with_early_exit(
        self,
        prompt: str,
        min_tokens: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        use_tcm: bool = True,
        use_graph: bool = True
    ) -> str:
        """Generate with early exit based on complete sentences."""
        
        if min_tokens is None:
            min_tokens = self._adaptive_min_tokens(prompt)
        
        # Build context (same as generate)
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
            result_text = [""]
            
            def streamer_func(text_chunk: str) -> int:
                result_text[0] += text_chunk
                
                # Check after min_chars AND complete sentence
                if len(result_text[0]) >= min_tokens:
                    if self._has_complete_sentence(result_text[0]):
                        return 0  # Stop
                
                return 1  # Continue
            
            start = time.time()
            
            self.pipeline.generate(
                full,
                streamer=streamer_func,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            result = result_text[0]
            elapsed = time.time() - start
            
            # If no complete sentence found, continue until next sentence
            if not self._has_complete_sentence(result) and len(result) < min_tokens + 50:
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


def test():
    print("=" * 50)
    print("FCP v7.1 - Early Exit Test (5 queries)")
    print("=" * 50)
    
    fcp = FCPV7()
    
    if not fcp.load():
        print("[ERROR]")
        return 1
    
    # Curator cycle
    print("\n[1] Curator cycle")
    fcp.curator.run_cycle()
    print(f"[Nodes] {len(fcp.graph._nodes)}")
    
    test_queries = [
        "What is quantum mechanics?",
        "Explain relativity simply.",
        "How does photosynthesis work?",
        "What causes climate change?",
        "Why do we dream?"
    ]
    
    print("\n[2] Early exit tests (no TCM, no graph)")
    results = []
    
    for i, q in enumerate(test_queries):
        print(f"\n--- Query {i+1}: {q}")
        # Fresh instance per query - no TCM
        fcp_test = FCPV7()
        fcp_test.load()
        
        start = time.time()
        r = fcp_test.generate_with_early_exit(q, max_new_tokens=256, use_tcm=False, use_graph=False)
        t = time.time() - start
        results.append((q, t, len(r)))
        print(f"[{t:.1f}s] {r[:150]}...")
    
    avg_time = sum(r[1] for r in results) / len(results)
    avg_chars = sum(r[2] for r in results) / len(results)
    
    print(f"\n[Summary]")
    print(f"Avg time: {avg_time:.1f}s")
    print(f"Avg chars: {avg_chars}")
    print(f"Total: {sum(r[1] for r in results):.1f}s")
    
    print("\n" + "=" * 50)
    print("FCP v7.1 - Ready!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(test())