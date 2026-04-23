"""
FCP UES Auto-Tune Simple
Оптимизация параметров OpenVINO (упрощённая версия)
"""
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.auto_tune")

OPENVINO_PATH = "C:/Users/black/OneDrive/Desktop/Models"


def benchmark_compile(num_threads: int, num_streams: int) -> float:
    """Бенчмарк компиляции с параметрами"""
    from openvino import Core
    
    core = Core()
    
    if os.path.isdir(OPENVINO_PATH):
        model_xml = os.path.join(OPENVINO_PATH, "openvino_model.xml")
    else:
        model_xml = OPENVINO_PATH
    
    start = time.time()
    
    try:
        compiled = core.compile_model(model_xml, "CPU", {
            "NUM_THREADS": num_threads,
            "NUM_STREAMS": num_streams
        })
    except Exception as e:
        logger.warning(f"Compile failed: {e}")
        return 999.0
    
    elapsed = time.time() - start
    
    return elapsed


def grid_search() -> dict:
    """Grid search for best params"""
    
    logger.info("[AutoTune] Running grid search...")
    
    results = []
    
    # Test different params
    for threads in [2, 4, 8]:
        for streams in [1, 2]:
            latency = benchmark_compile(threads, streams)
            
            results.append({
                "num_threads": threads,
                "num_streams": streams,
                "latency": latency
            })
            
            logger.info(f"  threads={threads}, streams={streams}: {latency:.3f}s")
    
    # Find best
    best = min(results, key=lambda x: x["latency"])
    
    logger.info(f"[AutoTune] Best: threads={best['num_threads']}, streams={best['num_streams']}, latency={best['latency']:.3f}s")
    
    return best


def main():
    print("=" * 60)
    print("FCP UES Auto-Tune")
    print("=" * 60)
    
    # Check optuna
    try:
        import optuna
        print("[Optuna] Available")
        has_optuna = True
    except ImportError:
        print("[Optuna] Not available (will install: pip install optuna)")
        has_optuna = False
    
    if not has_optuna:
        print("\n[Grid Search] Using simple grid search...")
        best = grid_search()
    
    print("\n" + "=" * 60)
    print("Auto-Tune Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())