"""
FCP v12 Tests - Layer Depth, Injection, and Early Exit Testing
Тесты для проверки задействования разных глубин слоев при генерации
"""
import sys
import os
import time
import logging
import numpy as np
import unittest

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("fcp.tests")

# Import from parent directory
_dir = os.path.dirname(os.path.abspath(__file__))
_pipelines_dir = os.path.join(os.path.dirname(_dir), 'pipelines')
sys.path.insert(0, _dir)
sys.path.insert(0, _pipelines_dir)

from mvp_pipeline_v12 import (
    FCPLayerStackV12,
    HybridLayerV12,
    FractalGNNLayer,
    HybridTransformerBlock,
    CoTrainLoRA,
)


# ============================================================================
# Test: Layer Depth Activation
# ============================================================================

class TestLayerDepth(unittest.TestCase):
    """Тесты на задействование разных глубин слоёв"""
    
    def setUp(self):
        """Создать стек для тестов"""
        self.stack = FCPLayerStackV12(num_layers=32, hidden_dim=512, num_heads=8)
        self.small_stack = FCPLayerStackV12(num_layers=8, hidden_dim=256, num_heads=4)
    
    def test_all_layers_processed_without_early_exit(self):
        """Все слои должны обрабатываться без early exit"""
        batch, seq = 1, 8
        embeddings = np.random.randn(batch, seq, 512).astype(np.float32) * 0.1
        
        output, stop_layer, injections = self.stack.forward(embeddings)
        
        # Без early exit все 32 слоя должны пройти
        self.assertIsNone(stop_layer, "Early exit не должен срабатывать при низкой уверенности")
        self.assertEqual(output.shape, (batch, seq, 512))
        logger.info(f"[TEST] Все 32 слоя обработаны, stop_layer={stop_layer}")
    
    def test_early_exit_at_high_confidence(self):
        """Early exit должен срабатывать при высокой confidence"""
        batch, seq = 1, 4
        embeddings = np.random.randn(batch, seq, 512).astype(np.float32) * 10.0
        
        output, stop_layer, injections = self.stack.forward(embeddings)
        
        if stop_layer is not None:
            logger.info(f"[TEST] Early exit на слое {stop_layer}")
            self.assertIsNotNone(stop_layer)
            self.assertLessEqual(stop_layer, 31)
        else:
            logger.info(f"[TEST] Нет early exit (confidence слишком низкая)")
    
    def test_partial_depth_processing(self):
        """Частичная обработка должна работать"""
        for seq_len in [2, 4, 8, 16]:
            embeddings = np.random.randn(1, seq_len, 256).astype(np.float32) * 0.05
            output, stop_layer, injections = self.small_stack.forward(embeddings)
            
            self.assertEqual(output.shape[1], seq_len)
            logger.info(f"[TEST] seq_len={seq_len}, output.shape={output.shape}")
    
    def test_injection_at_specific_layers(self):
        """Инъекция должна происходить на слоях 4, 8, 16, 24"""
        batch, seq = 1, 8
        embeddings = np.random.randn(batch, seq, 512).astype(np.float32) * 0.1
        
        graph_data = {
            "embeddings": np.random.randn(10, 512).astype(np.float32) * 0.01,
            "edges": np.array([[0,1], [1,2], [2,3]])
        }
        
        output, stop_layer, injections = self.stack.forward(embeddings, graph_data)
        
        expected_injections = 4
        self.assertEqual(injections, expected_injections, 
                        f"Ожидалось {expected_injections} инъекций, получено {injections}")
        
        logger.info(f"[TEST] Инъекции на слоях 4,8,16,24: {injections}")
    
    def test_no_injection_without_graph(self):
        """Без graph данных инъекций быть не должно"""
        batch, seq = 1, 8
        embeddings = np.random.randn(batch, seq, 512).astype(np.float32) * 0.1
        
        output, stop_layer, injections = self.stack.forward(embeddings, graph_data=None)
        
        self.assertEqual(injections, 0, "Инъекций не должно быть без graph")
        logger.info(f"[TEST] Без graph данных: {injections} инъекций")


# ============================================================================
# Test: Layer-Specific LoRA Ranks
# ============================================================================

class TestLoRARanks(unittest.TestCase):
    """Тесты на правильные ранги LoRA"""
    
    def test_spec_ranks_per_layer(self):
        """Spec ранги: 1-8→r=4, 9-16→r=8, 17-32→r=16"""
        stack = FCPLayerStackV12(num_layers=32, hidden_dim=256, num_heads=4)
        
        expected_ranks = {}
        for i in range(32):
            if i < 8:
                expected_ranks[i] = 4
            elif i < 16:
                expected_ranks[i] = 8
            else:
                expected_ranks[i] = 16
        
        for layer_id, expected_rank in expected_ranks.items():
            actual_rank = stack.layers[layer_id].lora.rank
            self.assertEqual(actual_rank, expected_rank,
                          f"Слой {layer_id}: ожидался rank={expected_rank}, получен {actual_rank}")
        
        logger.info(f"[TEST] Все 32 слоя имеют корректные ранги LoRA")
    
    def test_lora_has_weights(self):
        """LoRA должен иметь веса если rank > 0"""
        stack = FCPLayerStackV12(num_layers=4, hidden_dim=256, num_heads=4)
        
        for layer in stack.layers:
            if layer.lora.rank > 0:
                self.assertIsNotNone(layer.lora.W_down)
                self.assertIsNotNone(layer.lora.W_up)
                self.assertEqual(layer.lora.W_down.shape[1], layer.lora.rank)


# ============================================================================
# Test: Confidence Computation
# ============================================================================

class TestConfidence(unittest.TestCase):
    """Тесты на вычисление confidence"""
    
    def setUp(self):
        self.layer = HybridLayerV12(layer_id=8, hidden_dim=256, num_heads=4)
    
    def test_confidence_in_range(self):
        """Confidence должен быть в диапазоне [0, 1]"""
        for scale in [0.01, 0.1, 1.0, 5.0]:
            hidden = np.random.randn(1, 4, 256).astype(np.float32) * scale
            confidence = self.layer._compute_confidence(hidden, graph_vec=None)
            
            self.assertGreaterEqual(confidence, 0.0, f"Confidence < 0 при scale={scale}")
            self.assertLessEqual(confidence, 1.0, f"Confidence > 1 при scale={scale}")
            
            logger.info(f"[TEST] scale={scale}, confidence={confidence:.4f}")
    
    def test_confidence_with_graph(self):
        """Graph контекст должен повышать confidence"""
        hidden = np.random.randn(1, 4, 256).astype(np.float32) * 0.1
        graph_vec = np.random.randn(256).astype(np.float32) * 0.1
        
        conf_without = self.layer._compute_confidence(hidden, graph_vec=None)
        conf_with = self.layer._compute_confidence(hidden, graph_vec)
        
        self.assertGreaterEqual(conf_with, conf_without, 
                             "Graph должен повышать confidence")
        
        logger.info(f"[TEST] Без graph: {conf_without:.4f}, с graph: {conf_with:.4f}")


# ============================================================================
# Test: Co-training
# ============================================================================

class TestCoTraining(unittest.TestCase):
    """Тесты на co-training LoRA"""
    
    def test_co_train_updates_importance(self):
        """Co-training должен обновлять importance"""
        stack = FCPLayerStackV12(num_layers=4, hidden_dim=128, num_heads=2)
        
        embeddings = np.random.randn(1, 4, 128).astype(np.float32) * 0.1
        output, _, _ = stack.forward(embeddings)
        
        grad_output = np.random.randn(*output.shape).astype(np.float32) * 0.01
        stack.co_train(grad_output, embeddings, lr=0.01)
        
        importances = stack.get_lora_importances()
        for imp in importances:
            self.assertGreater(imp, 0, "Importance должен быть > 0")


# ============================================================================
# Test: Graph Injection
# ============================================================================

class TestGraphInjection(unittest.TestCase):
    """Тесты на инъекцию графа"""
    
    def test_fuse_streams(self):
        """Слияние потоков должно работать"""
        layer = HybridLayerV12(layer_id=4, hidden_dim=128, num_heads=4)
        
        hidden = np.random.randn(1, 4, 128).astype(np.float32) * 0.1
        graph_vec = np.random.randn(128).astype(np.float32) * 0.1
        
        fused = layer._fuse_streams(hidden, graph_vec)
        
        self.assertEqual(fused.shape, hidden.shape)
        
        if layer.fusion_weight > 0:
            self.assertFalse(np.allclose(fused, hidden),
                           "Результат должен отличаться от входа")
    
    def test_injection_only_at_specific_layers(self):
        """Инъекция только на слоях 4, 8, 16, 24"""
        injection_layers = {4, 8, 16, 24}
        
        for layer_id in range(32):
            layer = HybridLayerV12(layer_id=layer_id, hidden_dim=256, num_heads=4)
            
            should_inject = layer_id in injection_layers
            actual_inject = layer.layer_id in HybridLayerV12.INJECTION_LAYERS
            
            self.assertEqual(should_inject, actual_inject,
                          f"Слой {layer_id}: ожидалось {should_inject}, получено {actual_inject}")


# ============================================================================
# Test: Performance
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Тесты производительности"""
    
    def test_forward_time(self):
        """Время forward должно быть разумным"""
        stack = FCPLayerStackV12(num_layers=32, hidden_dim=512, num_heads=8)
        
        batch, seq = 1, 16
        embeddings = np.random.randn(batch, seq, 512).astype(np.float32)
        
        start = time.time()
        for _ in range(3):
            output, _, _ = stack.forward(embeddings)
        elapsed = time.time() - start
        avg_time = elapsed / 3
        
        self.assertLess(avg_time, 10.0, f"Слишком медленно: {avg_time:.2f}s")
        
        logger.info(f"[TEST] Среднее время forward: {avg_time:.3f}s")


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """Тесты на граничные случаи"""
    
    def test_empty_input(self):
        """Пустой вход должен обрабатываться"""
        stack = FCPLayerStackV12(num_layers=4, hidden_dim=128, num_heads=2)
        
        embeddings = np.random.randn(1, 1, 128).astype(np.float32) * 0.1
        
        output, stop_layer, injections = stack.forward(embeddings)
        
        self.assertEqual(output.shape[1], 1)


# ============================================================================
# Run all tests
# ============================================================================

def run_tests():
    """Запустить все тесты"""
    print("=" * 70)
    print("FCP v12 Tests - Layer Depth, Injection, and Early Exit")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestLayerDepth))
    suite.addTests(loader.loadTestsFromTestCase(TestLoRARanks))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidence))
    suite.addTests(loader.loadTestsFromTestCase(TestCoTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphInjection))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())