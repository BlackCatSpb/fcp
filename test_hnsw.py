"""Test HNSW Graph Search integration."""
import sys
import logging
import numpy as np

sys.path.insert(0, "C:/Users/black/OneDrive/Desktop/FCP/src")
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_hnsw")

def test_hnsw_search():
    """Test HNSW search integration with SQLite DB."""
    from memory.graph_search import FractalGraphSearch, GraphVectorExtractor, create_graph_search
    
    db_path = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
    
    logger.info(f"Создание FractalGraphSearch с {db_path}")
    search = create_graph_search(
        db_path=db_path,
        embedding_dim=384,
        encoder_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info("Построение индекса...")
    count = search.build_index(node_types=['concept', 'fact', 'entity'])
    logger.info(f"Проиндексировано {count} узлов")
    
    if count == 0:
        logger.error("Нет узлов для индексации!")
        return False
    
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