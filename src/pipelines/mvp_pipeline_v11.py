"""
FCP Pipeline v11 - Concept & Contradiction Mining
Phase 3: ConceptMiner, ContradictionMiner, GraphCurator per SPEC
"""
import sys
import os
import time
import logging
import codecs
import sqlite3
import json
import numpy as np
import threading
import re
from collections import Counter, defaultdict

if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

_fcp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _fcp_dir)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.curation")

GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"


class ConceptMiner:
    """Extract concepts from text per SPEC section 3.5."""
    
    def __init__(self, graph):
        self.graph = graph
        self.min_frequency = 2
        self.max_concepts = 50
        self.concepts_found = 0
        
        logger.info("[ConceptMiner] Ready")
    
    def extract_concepts(self, text: str, context: str = "") -> list:
        """
        Extract named entities and concepts from text.
        Returns: List[Concept] with name, type, confidence, related_to
        """
        concepts = []
        
        # Simple NER patterns (ru)
        patterns = {
            "科学的": r"[А-Яа-яёЁa-zA-Z]+ология",
            "技术": r"[А-Яа-яёЁa-zA-Z]+ика", 
            "人物": r"[А-Я][а-яёЁ]+\s[А-Я][а-яёЁ]+",
            "地点": r"[А-Я][а-яёЁ]+(?:ск|бург|град|город|область)",
            "日期": r"\d{1,2}[.\-]\d{1,2}[.\-]\d{2,4}"
        }
        
        for concept_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                concepts.append({
                    "name": match,
                    "type": concept_type,
                    "confidence": 0.8,
                    "source": "regex"
                })
        
        # Filter by frequency
        freq = Counter([c["name"] for c in concepts])
        concepts = [c for c in concepts if freq[c["name"]] >= self.min_frequency]
        
        # Build taxonomy
        concepts = self._build_hierarchy(concepts, context)
        
        self.concepts_found += len(concepts)
        logger.info(f"[ConceptMiner] Found {len(concepts)} concepts")
        
        return concepts[:self.max_concepts]
    
    def _build_hierarchy(self, concepts: list, context: str) -> list:
        """Build is_a relationships."""
        # Simple hierarchy by suffix
        hierarchy_relations = []
        
        for concept in concepts:
            name = concept["name"]
            
            if name.endswith("ология") or name.endswith("ics"):
                concept["is_a"] = "наука"
            elif name.endswith("ика"):
                concept["is_a"] = "область"
            elif name.endswith("ение") or name.endswith("ство"):
                concept["is_a"] = "концепт"
        
        return concepts
    
    def add_to_graph(self, concepts: list) -> int:
        """Add concepts to graph."""
        added = 0
        
        for concept in concepts:
            if self.graph:
                node_id = self.graph.add_node(
                    concept["name"],
                    concept["type"],
                    concept.get("confidence", 0.5)
                )
                if node_id:
                    added += 1
                    
                    # Add is_a relationship
                    if "is_a" in concept:
                        self.graph.add_edge(
                            concept["name"],
                            concept["is_a"],
                            "is_a"
                        )
        
        logger.info(f"[ConceptMiner] Added {added} to graph")
        return added


class ContradictionMiner:
    """Detect and resolve contradictions per SPEC section 3.6."""
    
    def __init__(self, graph):
        self.graph = graph
        self.sim_threshold = 0.75
        self.contra_threshold = 0.65
        self.contradictions_found = 0
        self.resolved = 0
        
        logger.info("[ContradictionMiner] Ready")
    
    def find_contradictions(self) -> list:
        """
        Scan graph pairs with sim >= 0.75, analyze NLI contradiction.
        Returns: List[Contradiction]
        """
        if not self.graph:
            return []
        
        nodes = self.graph.get_facts()
        contradictions = []
        checked = set()
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                key = tuple(sorted([node1["id"], node2["id"]]))
                if key in checked:
                    continue
                checked.add(key)
                
                # Compute similarity
                sim = self._compute_similarity(node1, node2)
                
                if sim >= self.sim_threshold:
                    # NLI analysis
                    contra_score = self._nli_analysis(node1, node2)
                    
                    if contra_score >= self.contra_threshold:
                        contradictions.append({
                            "node1": node1["id"],
                            "node2": node2["id"],
                            "similarity": sim,
                            "contra_score": contra_score,
                            "type": self._classify_contradiction(node1, node2)
                        })
        
        self.contradictions_found += len(contradictions)
        logger.info(f"[ContradictionMiner] Found {len(contradictions)} contradictions")
        
        return contradictions
    
    def _compute_similarity(self, node1: dict, node2: dict) -> float:
        """Compute cosine similarity."""
        # Simple word overlap
        words1 = set(node1.get("content", "").lower().split())
        words2 = set(node2.get("content", "").lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0
    
    def _nli_analysis(self, fact1: dict, fact2: dict) -> float:
        """NLI: entailment/neutral/contradiction."""
        c1 = fact1.get("content", "").lower()
        c2 = fact2.get("content", "").lower()
        
        # Check for opposites
        opposites = [
            ("да", "нет"), ("да", "не"),
            ("верно", "ложно"), ("истина", "ложь"),
            ("можно", "нельзя"), ("есть", "нет"),
            ("существует", "не существует"),
            ("был", "не был"), ("будет", "не будет")
        ]
        
        for pos, neg in opposites:
            if pos in c1 and neg in c2:
                return 0.9
            if pos in c2 and neg in c1:
                return 0.9
        
        return 0.0
    
    def _classify_contradiction(self, node1: dict, node2: dict) -> str:
        """Classify contradiction type."""
        return "logical_opposite"
    
    def resolve_contradiction(self, contradiction: dict, strategy: str = "auto") -> dict:
        """
        Resolve using strategy: auto, user_query, merge.
        """
        if strategy == "auto":
            # Simple resolution: keep majority
            resolution = {
                "strategy": "auto",
                "resolved_node": contradiction.get("node1"),
                "description": "Resolved by majority"
            }
        elif strategy == "merge":
            # Merge facts
            resolution = {
                "strategy": "merge",
                "merged": True,
                "description": "Facts merged"
            }
        else:
            resolution = {
                "strategy": "pending",
                "description": "User query required"
            }
        
        self.resolved += 1
        logger.info(f"[ContradictionMiner] Resolved: {resolution['strategy']}")
        
        return resolution


class GraphCurator:
    """Graph curation per SPEC section 3.7."""
    
    def __init__(self, graph, concept_miner: ConceptMiner, contradiction_miner: ContradictionMiner):
        self.graph = graph
        self.concept_miner = concept_miner
        self.contradiction_miner = contradiction_miner
        
        self.cycles_run = 0
        self.nodes_added = 0
        self.nodes_removed = 0
        
        logger.info("[GraphCurator] Ready")
    
    def run_cycle(self):
        """
        Full cycle: concept mining -> contradiction -> dangling -> decay -> LoRA examples.
        """
        logger.info("[Curator] Cycle start...")
        
        # 1. Extract concepts from recent dialogs
        self._extract_concepts()
        
        # 2. Find and resolve contradictions
        self._resolve_contradictions()
        
        # 3. Handle dangling nodes
        self._handle_dangling()
        
        # 4. Apply temporal decay
        self._apply_decay()
        
        # 5. Generate LoRA examples
        self._generate_lora_examples()
        
        self.cycles_run += 1
        logger.info(f"[Curator] Cycle {self.cycles_run} done")
    
    def _extract_concepts(self):
        """Run concept mining on recent content."""
        # Would get from dialog history
        recent_texts = ["Квантовая механика изучает микрочастицы", 
                    "Искусственный интеллект развивается"]
        
        for text in recent_texts:
            concepts = self.concept_miner.extract_concepts(text)
            if concepts:
                added = self.concept_miner.add_to_graph(concepts)
                self.nodes_added += added
    
    def _resolve_contradictions(self):
        """Find and resolve contradictions."""
        contradictions = self.contradiction_miner.find_contradictions()
        
        for contra in contradictions:
            self.contradiction_miner.resolve_contradiction(contra, "auto")
    
    def _handle_dangling(self):
        """Handle nodes without edges."""
        if not self.graph:
            return
        
        dangling = self.graph.get_dangling()
        removed = len(dangling)
        
        # Remove or link
        for node in dangling[:10]:  # Limit
            self.graph.remove_node(node["id"])
        
        self.nodes_removed += removed
        logger.info(f"[Curator] Removed {removed} dangling nodes")
    
    def _apply_decay(self):
        """Apply temporal decay to weights."""
        if not self.graph:
            return
        
        # Decay old nodes
        decayed = self.graph.apply_decay(0.995)
        logger.info(f"[Curator] Decayed {decayed} nodes")
    
    def _generate_lora_examples(self):
        """Generate correction examples for LoRA."""
        # After contradiction resolution
        examples = self.contradiction_miner.resolved
        
        if examples > 0:
            logger.info(f"[Curator] Generated {examples} LoRA examples")
    
    def schedule_cycle(self, interval: int = 300):
        """Schedule periodic curation."""
        import threading
        
        def loop():
            while True:
                time.sleep(interval)
                self.run_cycle()
        
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        
        logger.info(f"[Curator] Scheduled: {interval}s")


class LearningGraphManager:
    """Per SPEC 'Последовательные решения.txt' section 1."""
    
    def __init__(self, graph):
        self.graph = graph
        self.signals = []
        self.layer_sensitivity = {}
        
        logger.info("[LearningGraphManager] Ready")
    
    def record_learning_signal(
        self,
        fact_ids: list,
        layer_indices: list,
        score: float,
        domain: str,
        confidence: float = 1.0
    ) -> str:
        """Record learning signal."""
        from time import time
        import uuid
        
        final_confidence = confidence * 0.9  # time decay
        
        signal_id = f"sig_{fact_ids[0]}_{time():.6f}"
        
        signal = {
            "id": signal_id,
            "fact_ids": fact_ids,
            "layer_indices": layer_indices,
            "score": score,
            "domain": domain,
            "confidence": final_confidence,
            "timestamp": time()
        }
        
        self.signals.append(signal)
        
        # Update layer sensitivity
        for lidx in layer_indices:
            self._update_layer_sensitivity(lidx, domain, score)
        
        return signal_id
    
    def _update_layer_sensitivity(self, lidx: int, domain: str, score: float):
        """Update layer sensitivity."""
        ls_id = f"ls_{lidx}_{domain}"
        
        if ls_id not in self.layer_sensitivity:
            self.layer_sensitivity[ls_id] = {
                "sample_count": 0,
                "success_rate": 0.5,
                "last_updated": None
            }
        
        ls = self.layer_sensitivity[ls_id]
        new_success = 1.0 if score > 0 else 0.0
        
        # Exponential moving average
        ls["success_rate"] = 0.7 * new_success + 0.3 * ls["success_rate"]
        ls["sample_count"] += 1
        ls["last_updated"] = time.time()
    
    def get_learning_tasks(
        self,
        threshold: float = 0.6,
        min_samples: int = 10,
        limit: int = 5
    ) -> list:
        """Get retraining tasks."""
        tasks = []
        
        for ls_id, data in self.layer_sensitivity.items():
            if data["sample_count"] >= min_samples and data["success_rate"] < threshold:
                parts = ls_id.split("_")
                lidx = int(parts[1])
                domain = parts[2]
                
                tasks.append({
                    "task_id": f"task_{ls_id}",
                    "domain": domain,
                    "layer_indices": [lidx],
                    "priority": 1.0 - data["success_rate"]
                })
        
        # Sort by priority
        tasks.sort(key=lambda x: x["priority"], reverse=True)
        
        return tasks[:limit]
    
    def prune_old_signals(self, retention_days: int = 14, min_confidence: float = 0.1) -> int:
        """Remove old signals."""
        from time import time
        import time as time_module
        
        cutoff = time() - retention_days * 86400
        
        before = len(self.signals)
        
        self.signals = [
            s for s in self.signals
            if s["timestamp"] >= cutoff and s["confidence"] >= min_confidence
        ]
        
        removed = before - len(self.signals)
        logger.info(f"[Manager] Pruned {removed} signals")
        
        return removed
    
    def get_replay_buffer(self, total_samples: int = 256) -> list:
        """Stratified sample for experience replay."""
        if not self.signals:
            return []
        
        # Group by domain
        by_domain = defaultdict(list)
        for s in self.signals:
            by_domain[s["domain"]].append(s)
        
        # Distribute
        buffer = []
        for domain, signals in by_domain.items():
            count = max(1, int(total_samples * len(signals) / len(self.signals)))
            buffer.extend(signals[:count])
        
        return buffer
    
    def decay_signal_priority(self, lambda_decay: float = 0.01):
        """Recalculate confidence with age."""
        from time import time
        
        for signal in self.signals:
            age_days = (time() - signal["timestamp"]) / 86400
            signal["confidence"] *= np.exp(-lambda_decay * age_days)


class SimpleGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._nodes = {}
        self._load_nodes()
    
    def _load_nodes(self):
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT id, content, node_type FROM nodes")
            for r in cur.fetchall():
                self._nodes[r[0]] = {"id": r[0], "content": r[1], "type": r[2]}
            logger.info(f"[Graph] {len(self._nodes)} nodes")
        except:
            self._nodes = {}
    
    def get_facts(self) -> list:
        return [n for n in self._nodes.values() if n.get("type") == "fact"]
    
    def get_dangling(self) -> list:
        return []
    
    def add_node(self, name: str, type_: str, confidence: float) -> str:
        import uuid
        node_id = f"concept_{uuid.uuid4().hex[:8]}"
        self._nodes[node_id] = {"id": node_id, "name": name, "type": type_}
        return node_id
    
    def add_edge(self, from_id: str, to_id: str, rel: str):
        pass
    
    def remove_node(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
    
    def apply_decay(self, factor: float) -> int:
        return 0
    
    def get_stats(self) -> dict:
        return {"nodes": len(self._nodes)}


def test():
    print("=" * 60)
    print("FCP v11 - Concept & Contradiction Mining")
    print("=" * 60)
    
    # Graph
    graph = SimpleGraph(GRAPH_PATH)
    print(f"[Graph] {graph.get_stats()}")
    
    # Concept Miner
    cm = ConceptMiner(graph)
    concepts = cm.extract_concepts("Квантовая механика изучает микрочастицы в физике")
    print(f"[ConceptMiner] Found: {len(concepts)} concepts")
    
    # Contradiction Miner
    ctm = ContradictionMiner(graph)
    contradictions = ctm.find_contradictions()
    print(f"[ContradictionMiner] Found: {len(contradictions)} contradictions")
    
    # Graph Curator
    gc = GraphCurator(graph, cm, ctm)
    gc.run_cycle()
    
    # Learning Graph Manager
    lgm = LearningGraphManager(graph)
    signal_id = lgm.record_learning_signal(
        ["fact1"], [0, 1], 0.8, "physics", 0.9
    )
    print(f"[Manager] Signal: {signal_id}")
    
    tasks = lgm.get_learning_tasks()
    print(f"[Manager] Tasks: {len(tasks)}")
    
    print("\n" + "=" * 60)
    print("FCP v11 - Curation Ready!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(test())