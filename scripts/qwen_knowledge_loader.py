"""
Qwen Knowledge Graph Loader
Загружает qwen_knowledge.npz в FractalGraphV2:
- Дедуплицирует (row, col, val, count)
- Маппит ID -> токен через Qwen tokenizer
- Создаёт узлы (nodes) и связи (edges) в SQLite БД
"""
import os
import re
import sys
import time
import json
import sqlite3
import logging
import numpy as np
from typing import Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("QwenLoader")

QWEN_NPZ = r"C:\Users\black\OneDrive\Desktop\qwen_knowledge.npz"
OUTPUT_DIR = Path(__file__).parent.parent / "eva_ai" / "fcp_core" / "data"
OUTPUT_DB = OUTPUT_DIR / "qwen_knowledge.db"


def _clean_token_text(text: str) -> str:
    text = text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
    text = re.sub(r'[^\w\s\-\.,!?;:()\[\]{}<>/@#$%^&*+=~`"\'\u0400-\u04FF]', '', text)
    text = text.strip()
    if not text:
        text = f"<unk>"
    return text


def _make_node_id(token_id: int) -> str:
    return f"qwen_tok_{token_id}"


def deduplicate_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Дедупликация: схлопнуть дубликаты (row, col)."""
    logger.info("Loading npz...")
    data = np.load(npz_path)
    rows, cols, vals, counts = data['rows'], data['cols'], data['vals'], data['counts']
    logger.info(f"Raw entries: {len(rows):,}")

    dtype = [('row', np.uint32), ('col', np.uint32)]
    structured = np.empty(len(rows), dtype=dtype)
    structured['row'] = rows
    structured['col'] = cols

    logger.info("Grouping duplicates...")
    unique_keys, inverse, key_counts = np.unique(structured, return_inverse=True, return_counts=True)

    n_unique = len(unique_keys)
    sum_vals = np.zeros(n_unique, dtype=np.float64)
    sum_counts = np.zeros(n_unique, dtype=np.uint64)
    np.add.at(sum_vals, inverse, vals)
    np.add.at(sum_counts, inverse, counts)

    mean_vals = (sum_vals / key_counts).astype(np.float32)
    total_counts = sum_counts.astype(np.uint32)

    out_rows = unique_keys['row']
    out_cols = unique_keys['col']

    logger.info(f"Unique pairs: {n_unique:,} ({100 * n_unique / len(rows):.1f}%)")
    return out_rows, out_cols, mean_vals, total_counts


def build_tokenizer():
    from transformers import AutoTokenizer
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    model_path = os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B')
    snapshots = os.path.join(model_path, 'snapshots')
    if os.path.exists(snapshots):
        snaps = os.listdir(snapshots)
        if snaps:
            return AutoTokenizer.from_pretrained(
                os.path.join(snapshots, snaps[0]),
                trust_remote_code=True
            )
    raise FileNotFoundError("Qwen2.5-3B tokenizer not found in HF cache")


def create_graph_db(db_path: str, rows, cols, vals, counts, tokenizer):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=OFF;
        PRAGMA cache_size=-64000;
    """)

    cur.execute("""CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        node_type TEXT NOT NULL DEFAULT 'concept',
        level INTEGER DEFAULT 1,
        parent_group_id TEXT,
        embedding BLOB,
        confidence REAL DEFAULT 0.5,
        created_at REAL,
        updated_at REAL,
        last_accessed REAL,
        metadata TEXT,
        access_count INTEGER DEFAULT 0,
        version INTEGER DEFAULT 1,
        is_static INTEGER DEFAULT 0,
        is_contradiction INTEGER DEFAULT 0
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation_type TEXT NOT NULL DEFAULT 'related_to',
        weight REAL DEFAULT 0.5,
        created_at REAL,
        updated_at REAL,
        contradiction_flag INTEGER DEFAULT 0,
        metadata TEXT
    )""")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")

    all_ids = set(int(x) for x in rows) | set(int(x) for x in cols)
    logger.info(f"Unique token IDs: {len(all_ids):,}")

    now = time.time()

    # Insert nodes
    logger.info("Inserting nodes...")
    node_data = []
    for tid in sorted(all_ids):
        try:
            decoded = tokenizer.decode([tid])
        except Exception:
            decoded = f"<tok_{tid}>"
        clean = _clean_token_text(decoded)
        node_data.append((
            _make_node_id(tid), clean, 'concept', 1, None, None, 0.5,
            now, now, now, None, 0, 1, 0, 0
        ))

    cur.executemany(
        "INSERT OR IGNORE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        node_data
    )
    conn.commit()
    logger.info(f"Inserted {len(node_data):,} nodes")

    # Insert edges in batches
    logger.info("Inserting edges...")
    batch_size = 50000
    total = len(rows)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        edge_batch = []
        for i in range(start, end):
            src = _make_node_id(int(rows[i]))
            tgt = _make_node_id(int(cols[i]))
            weight = float(vals[i])
            freq = int(counts[i])
            meta = json.dumps({"count": freq, "source": "qwen_knowledge"})
            edge_id = f"e_{rows[i]}_{cols[i]}"
            edge_batch.append((
                edge_id, src, tgt, 'related_to', weight,
                now, now, 0, meta
            ))

        cur.executemany(
            "INSERT OR IGNORE INTO edges VALUES (?,?,?,?,?,?,?,?,?)",
            edge_batch
        )
        conn.commit()

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Edges: {end:,}/{total:,} ({100*end/total:.1f}%)")

    conn.commit()
    conn.close()
    logger.info(f"Done! Total edges: {total:,}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s'
    )
    logger.info("=== Qwen Knowledge Graph Loader ===")

    rows, cols, vals, counts = deduplicate_npz(QWEN_NPZ)

    logger.info("Loading tokenizer...")
    tokenizer = build_tokenizer()

    logger.info(f"Creating graph DB at {OUTPUT_DB}")
    create_graph_db(str(OUTPUT_DB), rows, cols, vals, counts, tokenizer)

    logger.info("=== Complete ===")


if __name__ == "__main__":
    main()
