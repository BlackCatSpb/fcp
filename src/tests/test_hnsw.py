"""
Test HNSW Graph Search integration
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_hnsw")

def test_hnsw_search():
    """Test HNSW search integration with FractalGraphV2."""
    from eva_ai.memory.fractal_graph_v2.storage import FractalGraphV2
    from src.memory.graph_search import FractalGraphSearch, GraphVectorExtractor, create_graph_search
    
    logger.info("Загрузка FractalGraphV2...")
    graph = FractalGraphV2(
        storage_dir="C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data",
        embedding_dim=384
    )
    logger.info(f"Загружено {len(graph.nodes)} узлов")
    
    logger.info("Создание FractalGraphSearch...")
    search = create_graph_search(
        graph=graph,
        encoder_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info("Построение индекса...")
    count = search.build_index(node_types=['concept', 'fact', 'entity'])
    logger.info(f"Проиндексировано {count} узлов")
    
    logger.info("Тестовый поиск: 'искусственный интеллект'")
    results = search.search("искусственный интеллект", k=5, min_score=0.3)
    
    for r in results:
        logger.info(f"  [{r.score:.3f}] {r.node_type}: {r.content[:80]}...")
    
    logger.info("Извлечение графового вектора...")
    extractor = GraphVectorExtractor(search, output_dim=384)
    vector = extractor.extract("искусственный интеллект", k=10)
    logger.info(f"  Вектор: shape={vector.shape}, norm={np.linalg.norm(vector):.3f}")
    
    logger.info("Тест HNSW пройден!")
    return True


if __name__ == "__main__":
    try:
        success = test_hnsw_search()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        sys.exit(1)