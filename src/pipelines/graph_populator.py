"""
FCP Graph Populator - Наполнение графа знаниями
"""
import sys
import os
import sqlite3
import json
import logging
import codecs

# Fix encoding
if sys.platform == "win32":
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.populator")

# Path
GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"


# Seed knowledge
SEED_CONCEPTS = [
    # Physics
    {"content": "quantum mechanics", "type": "concept", "level": 1, "domain": "physics"},
    {"content": "quantum entanglement", "type": "concept", "level": 1, "domain": "physics"},
    {"content": "wave function", "type": "concept", "level": 2, "domain": "physics"},
    {"content": "Schrodinger equation", "type": "concept", "level": 2, "domain": "physics"},
    {"content": "Heisenberg uncertainty principle", "type": "concept", "level": 2, "domain": "physics"},
    {"content": "Planck constant", "type": "concept", "level": 2, "domain": "physics"},
    {"content": "quantum superposition", "type": "concept", "level": 2, "domain": "physics"},
    {"content": "photon", "type": "concept", "level": 1, "domain": "physics"},
    {"content": "electron", "type": "concept", "level": 1, "domain": "physics"},
    {"content": "atomic structure", "type": "concept", "level": 1, "domain": "physics"},
    
    # Computer Science
    {"content": "neural network", "type": "concept", "level": 1, "domain": "cs"},
    {"content": "deep learning", "type": "concept", "level": 1, "domain": "cs"},
    {"content": "transformer architecture", "type": "concept", "level": 2, "domain": "cs"},
    {"content": "attention mechanism", "type": "concept", "level": 2, "domain": "cs"},
    {"content": "machine learning", "type": "concept", "level": 1, "domain": "cs"},
    {"content": "natural language processing", "type": "concept", "level": 1, "domain": "cs"},
    {"content": "LLM large language model", "type": "concept", "level": 2, "domain": "cs"},
    {"content": "token embedding", "type": "concept", "level": 2, "domain": "cs"},
    {"content": "gradient descent", "type": "concept", "level": 2, "domain": "cs"},
    {"content": "backpropagation", "type": "concept", "level": 2, "domain": "cs"},
    
    # Math
    {"content": "linear algebra", "type": "concept", "level": 1, "domain": "math"},
    {"content": "matrix multiplication", "type": "concept", "level": 2, "domain": "math"},
    {"content": "vector space", "type": "concept", "level": 2, "domain": "math"},
    {"content": "eigenvalue", "type": "concept", "level": 2, "domain": "math"},
    {"content": "probability distribution", "type": "concept", "level": 1, "domain": "math"},
    {"content": "statistics", "type": "concept", "level": 1, "domain": "math"},
    
    # Philosophy
    {"content": "consciousness", "type": "concept", "level": 1, "domain": "philosophy"},
    {"content": "existentialism", "type": "concept", "level": 1, "domain": "philosophy"},
    {"content": "dualism", "type": "concept", "level": 2, "domain": "philosophy"},
    
    # General knowledge
    {"content": "artificial intelligence", "type": "concept", "level": 1, "domain": "tech"},
    {"content": "machine intelligence", "type": "concept", "level": 2, "domain": "tech"},
    {"content": "EVA AI system", "type": "concept", "level": 1, "domain": "tech"},
    {"content": "cognitive architecture", "type": "concept", "level": 2, "domain": "tech"},
]

# Facts linking concepts
SEED_FACTS = [
    {"subject": "quantum mechanics", "predicate": "describes", "object": "behavior of matter at atomic scale", "domain": "physics"},
    {"subject": "quantum entanglement", "predicate": "is a phenomenon where", "object": "particles become connected", "domain": "physics"},
    {"subject": "quantum superposition", "predicate": "allows", "object": "particles to be in multiple states", "domain": "physics"},
    {"subject": "Heisenberg principle", "predicate": "states that", "object": "position and momentum cannot both be known precisely", "domain": "physics"},
    {"subject": "neural network", "predicate": "is inspired by", "object": "biological neurons", "domain": "cs"},
    {"subject": "transformer", "predicate": "uses", "object": "self-attention mechanism", "domain": "cs"},
    {"subject": "deep learning", "predicate": "is a subset of", "object": "machine learning", "domain": "cs"},
    {"subject": "LLM", "predicate": "trained on", "object": "massive text corpora", "domain": "cs"},
    {"subject": "attention mechanism", "predicate": "allows", "object": "model to focus on relevant parts of input", "domain": "cs"},
    {"subject": "EVA AI", "predicate": "is a", "object": "cognitive architecture with graph memory", "domain": "tech"},
]


class GraphPopulator:
    """Заполняет граф знаниями."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def count_concepts(self) -> int:
        self.connect()
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes WHERE node_type='concept'")
        return cur.fetchone()[0]
    
    def add_concept(self, content: str, node_type: str = "concept", level: int = 1, domain: str = "general") -> bool:
        self.connect()
        import uuid
        node_id = f"concept_{uuid.uuid4().hex[:8]}"
        
        metadata = json.dumps({"domain": domain, "confidence": 0.8})
        
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO nodes (id, content, node_type, level, metadata) VALUES (?, ?, ?, ?, ?)",
                (node_id, content, node_type, level, metadata)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Add concept: {e}")
            return False
    
    def add_fact(self, subject: str, predicate: str, obj: str, domain: str = "general") -> bool:
        self.connect()
        import uuid
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        
        content = f"{subject} {predicate} {obj}"
        metadata = json.dumps({"domain": domain, "predicate": predicate})
        
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO nodes (id, content, node_type, level, metadata) VALUES (?, ?, ?, ?, ?)",
                (fact_id, content, "fact", 1, metadata)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.warning(f"Add fact: {e}")
            return False
    
    def seed_knowledge(self) -> dict:
        """Seed the graph with initial knowledge."""
        self.connect()
        
        stats = {"concepts": 0, "facts": 0}
        
        # Add concepts
        for item in SEED_CONCEPTS:
            if self.add_concept(
                item["content"],
                item.get("type", "concept"),
                item.get("level", 1),
                item.get("domain", "general")
            ):
                stats["concepts"] += 1
        
        # Add facts
        for item in SEED_FACTS:
            if self.add_fact(
                item["subject"],
                item["predicate"],
                item["object"],
                item.get("domain", "general")
            ):
                stats["facts"] += 1
        
        return stats


def main():
    print("=" * 50)
    print("FCP Graph Populator")
    print("=" * 50)
    
    populator = GraphPopulator(GRAPH_PATH)
    
    # Check existing
    existing = populator.count_concepts()
    print(f"\nExisting concepts: {existing}")
    
    if existing >= 50:
        print("Graph already populated, skipping...")
        return 0
    
    # Seed
    print("\nSeeding knowledge...")
    stats = populator.seed_knowledge()
    
    print(f"\nAdded:")
    print(f"  Concepts: {stats['concepts']}")
    print(f"  Facts: {stats['facts']}")
    
    # Verify
    new_count = populator.count_concepts()
    print(f"\nTotal concepts: {new_count}")
    
    print("\n" + "=" * 50)
    print("Graph Populated!")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())