"""Download Qwen3-4B PyTorch model."""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-4B"
LOCAL_DIR = "C:/Users/black/OneDrive/Desktop/Models/Qwen3-4B-PyTorch"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
print("Size: ~8GB, may take 10-20 minutes...")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
        ignore_patterns=["*.gguf", "*.safetensors"]
    )
    print(f"Done! Model saved to {LOCAL_DIR}")
except Exception as e:
    print(f"Error: {e}")