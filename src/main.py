"""
FCP - Fractal Cognitive Processor
Main entry point
"""
import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcp_core.config import FCPConfig
from adapters.openvino_adapter import FCPFullPipeline, get_fcp_integration


def main():
    """Main entry point."""
    print("=" * 60)
    print("FCP - Fractal Cognitive Processor")
    print("=" * 60)
    
    # Create config for BF16 model
    config = FCPConfig.from_model(
        "C:/Users/black/OneDrive/Desktop/Models/BF16.gguf",
        graph_db_path="C:/Users/black/OneDrive/Desktop/FCP/data/graph.db",
        device="CPU",
        num_threads=8
    )
    
    print(f"\n[Config]")
    print(f"  {config.summary()}")
    print(f"  Model: {config.model_path}")
    print(f"  Threads: {config.num_threads}")
    
    # Load integration (FractalGraphV2 + ConceptMiner + ContradictionMiner)
    print("\n[Loading EVA-Ai Integration]")
    integration = get_fcp_integration(config.graph_db_path)
    integration.load_all()
    
    # Show components status
    status = integration.components_status
    print(f"  Graph: {status['graph']}")
    print(f"  ConceptMiner: {status['concept_miner']}")
    print(f"  ContradictionMiner: {status['contradiction_miner']}")
    
    # Create pipeline
    print("\n[Creating FCP Pipeline]")
    pipeline = FCPFullPipeline(config)
    
    # Load model
    print("\n[Loading Model]")
    start = time.time()
    
    if not pipeline.load_model(config.model_path):
        print("[ERROR] Failed to load model")
        return 1
    
    load_time = time.time() - start
    print(f"  Loaded in {load_time:.1f}s")
    
    # Test generation
    print("\n[Test Generation]")
    prompt = "Привет! Расскажи о себе."
    
    start = time.time()
    response = pipeline.generate(prompt, max_new_tokens=256)
    latency = time.time() - start
    
    print(f"\n[Prompt]")
    print(f"  {prompt}")
    print(f"\n[Response]")
    print(f"  {response[:500]}..." if len(response) > 500 else f"  {response}")
    print(f"\n[Stats]")
    print(f"  Latency: {latency:.1f}s")
    print(f"  Tokens: ~{len(response.split())}")
    
    # Get FCP statistics
    fcp_stats = pipeline.get_statistics()
    print(f"\n[FCP Stats]")
    print(f"  Queries: {fcp_stats['queries_processed']}")
    print(f"  Concepts extracted: {fcp_stats['concepts_extracted']}")
    print(f"  Stack layers: {fcp_stats['num_layers']}")
    print(f"  Early exits: {fcp_stats['early_exits']}")
    
    # Test dynamic layer addition
    print("\n[Dynamic Layer Test]")
    current = pipeline.stack.num_layers
    new_count = pipeline.add_layers(4)
    print(f"  Added 4 layers: {current} -> {new_count}")
    
    new_count = pipeline.remove_layers(2)
    print(f"  Removed 2 layers: {new_count} -> {new_count}")
    
    print("\n" + "=" * 60)
    print("FCP Ready!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())