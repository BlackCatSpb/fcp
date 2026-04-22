"""
FMF Config - Оптимальная конфигурация OpenVINO для FMF EVA
"""
import openvino_genai as ov_genai
import os

def create_fmf_config() -> dict:
    """Создать оптимальную конфигурацию для FMF"""
    
    # === GenerationConfig ===
    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 2048
    generation_config.temperature = 0.2
    generation_config.top_p = 0.9
    generation_config.top_k = 40
    generation_config.repetition_penalty = 1.1
    generation_config.do_sample = True
    
    # === SchedulerConfig (continuous batching) ===
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.max_num_batched_tokens = 2048
    scheduler_config.max_num_seqs = 8
    scheduler_config.enable_prefix_caching = True
    
    # === Performance Hints (via env) ===
    os.environ['PERFORMANCE_HINT'] = 'LATENCY'
    os.environ['INFERENCE_NUM_THREADS'] = '8'
    
    config = {
        "generation_config": generation_config,
        "scheduler_config": scheduler_config
    }
    
    return config


# === Embeddings Manager для FMF ===
class FMFEmbeddingsManager:
    """Менеджер эмбеддингов для FMF"""
    
    _instance = None
    
    def __init__(self):
        self._model = None
        self._embedding_dim = 768
        self._cache_dir = r"C:\Users\black\OneDrive\Desktop\FMF_EVA\eva_ai\core\hf_cache"
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Use local cache from EVA
                os.environ['HF_HOME'] = self._cache_dir
                os.environ['TRANSFORMERS_CACHE'] = self._cache_dir
                
                # Find local model
                model_base = os.path.join(self._cache_dir, "models--intfloat--multilingual-e5-base", "snapshots")
                local_path = None
                
                if os.path.exists(model_base):
                    for item in os.listdir(model_base):
                        sub_path = os.path.join(model_base, item)
                        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "model.safetensors")):
                            local_path = sub_path
                            break
                
                if local_path:
                    self._model = SentenceTransformer(local_path, device="CPU")
                    print(f"FMF Embeddings loaded from: {local_path}")
                else:
                    # Fallback
                    self._model = SentenceTransformer("intfloat/multilingual-e5-base", device="CPU")
                    print("FMF Embeddings loaded from HuggingFace")
                
                self._embedding_dim = 768
            except Exception as e:
                print(f"Embeddings load failed: {e}")
    
    def encode(self, texts):
        self._ensure_model()
        if self._model is None:
            return None
        try:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get_embedding(self, text):
        result = self.encode([text])
        return result[0] if result else None


if __name__ == "__main__":
    config = create_fmf_config()
    gc = config["generation_config"]
    sc = config["scheduler_config"]
    
    print("=== FMF Config ===")
    print(f"GenerationConfig:")
    print(f"  max_new_tokens: {gc.max_new_tokens}")
    print(f"  temperature: {gc.temperature}")
    print(f"  top_p: {gc.top_p}")
    print(f"  repetition_penalty: {gc.repetition_penalty}")
    print(f"SchedulerConfig:")
    print(f"  max_num_seqs: {sc.max_num_seqs}")
    print(f"  enable_prefix_caching: {sc.enable_prefix_caching}")