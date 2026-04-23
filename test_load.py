"""Test FractalGraphV2 loading."""
import sys
sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FMF_EVA/src")
sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/EVA-Ai")

from eva_ai.memory.fractal_graph_v2.storage import FractalGraphV2

g = FractalGraphV2(
    storage_dir="C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data",
    embedding_dim=384
)
print(f"Nodes: {len(g.nodes)}")