"""
Inject Qwen token embeddings into qwen_knowledge.db

Extracts embed_tokens.weight from qwen_layer_model.pt,
reduces 2560→768, writes to nodes.embedding column.
"""
import os
import sys
import sqlite3
import logging
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger("InjectEmbs")

MODEL_PATH = "models/qwen_layer_model.pt"
DB_PATH = "eva_ai/fcp_core/data/qwen_knowledge.db"


def extract_embeddings(model_path: str) -> np.ndarray:
    """Extract embed_tokens.weight, convert to float32 numpy (vocab, 2560)."""
    logger.info(f"Loading model from {model_path}...")
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    state = sd["model_state_dict"]
    emb = state["base_model.model.embed_tokens.weight"].float().numpy()
    logger.info(f"Embedding shape: {emb.shape}, dtype={emb.dtype}")
    return emb  # (146260, 2560)


def reduce_dim(emb: np.ndarray, target_dim: int = 768) -> np.ndarray:
    """Reduce 2560→768. First target_dim dims slice + renormalize."""
    logger.info(f"Reducing {emb.shape[1]}→{target_dim} (slice)...")
    reduced = emb[:, :target_dim].copy()
    norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
    reduced = reduced / norms
    logger.info(f"Reduced shape: {reduced.shape}")
    return reduced.astype(np.float32)


def inject(embeddings: np.ndarray, db_path: str):
    """Write embeddings into nodes table.

    embeddings[tok_id] = 768-dim float32 vector.
    Node IDs are 'qwen_tok_{tok_id}'.
    """
    logger.info(f"Injecting embeddings into {db_path}...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Extract token ID from node id 'qwen_tok_{N}' -> N
    cur.execute("SELECT id FROM nodes WHERE embedding IS NULL")
    rows = cur.fetchall()
    logger.info(f"Nodes needing embeddings: {len(rows):,}")

    batch_size = 5000
    updates = []
    for row in rows:
        node_id = row[0]
        tok_id = int(node_id.split("_")[-1])
        if tok_id < embeddings.shape[0]:
            emb_bytes = embeddings[tok_id].tobytes()
            updates.append((emb_bytes, node_id))
        if len(updates) >= batch_size:
            cur.executemany("UPDATE nodes SET embedding = ? WHERE id = ?", updates)
            conn.commit()
            updates = []
            logger.info(f"  Updated {cur.rowcount} nodes...")

    if updates:
        cur.executemany("UPDATE nodes SET embedding = ? WHERE id = ?", updates)
        conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
    filled = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM nodes")
    total = cur.fetchone()[0]
    logger.info(f"Embeddings: {filled:,}/{total:,} nodes filled")
    conn.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    logger.info("=== Inject Qwen Embeddings ===")

    emb = extract_embeddings(MODEL_PATH)
    emb_768 = reduce_dim(emb, target_dim=768)
    inject(emb_768, DB_PATH)

    logger.info("=== Complete ===")


if __name__ == "__main__":
    main()
