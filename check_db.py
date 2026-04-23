"""Inspect SQLite DB."""
import sqlite3
import glob

db_paths = [
    "C:/Users/black/OneDrive/Desktop/EVA-Ai/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
    "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db",
    "C:/Users/black/OneDrive/Desktop/FCP/fmf/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
]

for db_path in db_paths:
    print(f"\n=== {db_path} ===")
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute('SELECT COUNT(*) FROM nodes')
        count = cur.fetchone()[0]
        print(f"Total nodes: {count}")
        
        if count > 0:
            cur.execute('SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type')
            print(f"Types: {cur.fetchall()}")
            
            cur.execute('SELECT id, content, node_type FROM nodes LIMIT 3')
            for row in cur.fetchall():
                print(f"  {row[2]}: {row[1][:50]}...")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")