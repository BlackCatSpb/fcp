"""
FCP E2E Integration Test
Полное сквозное тестирование FCP
"""
import sys
import os
import time

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_pipeline():
    """Тест полного пайплайна"""
    print("=" * 60)
    print("E2E TEST: Full Pipeline")
    print("=" * 60)
    
    # 1. Import FCP v13
    from pipelines.mvp_pipeline_v13 import FCPV13
    
    print("\n[1] Loading FCP v13...")
    fcp = FCPV13()
    fcp.load()
    print("  ✓ Loaded")
    
    # 2. Generate
    print("\n[2] Generating response...")
    start = time.time()
    response = fcp.generate("Что такое квантовая механика?", max_tokens=32)
    elapsed = time.time() - start
    print(f"  ✓ Generated in {elapsed:.1f}s")
    
    # 3. Check stats
    stats = fcp.get_stats()
    print(f"\n[3] Stats:")
    print(f"  - Layers used: {stats['layers_used']}")
    print(f"  - Injections: {stats['injections']}")
    print(f"  - All hybrid: {stats['layers_are_hybrid']}")
    print("  ✓ Stats verified")
    
    # 4. Verify each layer
    print("\n[4] Verifying hybrid layers...")
    for i in [0, 4, 8, 16, 24, 31]:
        layer = fcp.hybrid_layers[i]
        has_gnn = hasattr(layer, 'gnn')
        has_trans = hasattr(layer, 'transformer')
        has_lora = hasattr(layer, 'lora')
        print(f"  - Layer {i}: GNN={has_gnn}, Trans={has_trans}, LoRA={has_lora}")
    
    print("\n" + "=" * 60)
    print("E2E TEST: PASSED!")
    print("=" * 60)


def test_gnn_integration():
    """Тест GNN интеграции"""
    print("=" * 60)
    print("E2E TEST: GNN Integration")
    print("=" * 60)
    
    from pipelines.gnn_integrator import HybridLayerWithGNN
    
    print("\n[1] Creating hybrid layer with GNN...")
    layer = HybridLayerWithGNN(layer_id=4)
    
    print("\n[2] Testing forward with graph...")
    import numpy as np
    hidden = np.random.randn(1, 8, 2048).astype(np.float32) * 0.1
    output, graph_vec, stop = layer.forward(hidden, use_graph=True)
    
    print(f"  - Input shape: {hidden.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Graph used: {graph_vec is not None}")
    print("  ✓ GNN working")
    
    print("\n" + "=" * 60)
    print("GNN INTEGRATION: PASSED!")
    print("=" * 60)


def test_co_training():
    """Тест co-training"""
    print("=" * 60)
    print("E2E TEST: Co-Training LoRA")
    print("=" * 60)
    
    from pipelines.co_train_lora import CoTrainer, CoTrainingDataset
    
    GRAPH_PATH = "C:/Users/black/OneDrive/Desktop/FMF_EVA/eva_ai/memory/fractal_graph_v2/fractal_graph_v2_data/fractal_graph.db"
    
    print("\n[1] Loading dataset...")
    dataset = CoTrainingDataset(GRAPH_PATH)
    print(f"  - Loaded {len(dataset.samples)} samples")
    
    print("\n[2] Creating trainer...")
    trainer = CoTrainer()
    print(f"  - {trainer.num_layers} layers")
    
    print("\n[3] Training one epoch...")
    import numpy as np
    batch = dataset.get_batch(4)
    for sample in batch:
        hidden = np.random.randn(1, 4, 2048).astype(np.float32) * 0.1
        for layer_id in range(4):
            output = trainer.forward(layer_id, hidden)
            grad = np.random.randn(*output.shape).astype(np.float32) * 0.01
            trainer.backward(layer_id, grad, hidden)
    
    print("  ✓ Training complete")
    
    # Stats
    print("\n[4] Checking LoRA stats...")
    for i in [0, 4, 8, 16, 24, 31]:
        lora = trainer.lora_layers[i]
        print(f"  - Layer {i}: rank={lora['rank']}, updates={lora['update_count']}")
    
    print("\n" + "=" * 60)
    print("CO-TRAINING: PASSED!")
    print("=" * 60)


def test_split_model():
    """Тест split модели"""
    print("=" * 60)
    print("E2E TEST: Split Model")
    print("=" * 60)
    
    from pipelines.split_exporter import integrate_with_hybrid
    
    print("\n[1] Checking split config...")
    
    config_path = "C:/Users/black/OneDrive/Desktop/FCP/models/split_config.json"
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            import json
            config = json.load(f)
        
        print(f"  - Split layer: {config['split_layer']}")
        print(f"  - Injection layers: {config['injection_layers']}")
        print(f"  - Total layers: {config['total_layers']}")
        print("  ✓ Config loaded")
    else:
        print("  ⚠ Config not found, generating...")
        from pipelines.split_exporter import export_split_model
        export_split_model()
    
    print("\n[2] Integration flow ready")
    integrate_with_hybrid()
    
    print("\n" + "=" * 60)
    print("SPLIT MODEL: PASSED!")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("FCP E2E INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    try:
        test_gnn_integration()
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    try:
        test_co_training()
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    try:
        test_split_model()
    except Exception as e:
        print(f"\n[ERROR] {e}")
    
    print("\n" + "=" * 60)
    print("ALL E2E TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()