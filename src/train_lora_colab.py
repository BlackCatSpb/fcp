"""
FCP LoRA Training - For Google Colab.
Copy this to Colab and run.
"""
import sys

def main():
    print("=" * 60)
    print("FCP LoRA Training")
    print("Run on Google Colab with T4 GPU")
    print("=" * 60)
    
    print("""
# Install dependencies
!pip install peft transformers accelerate bitsandbytes datasets trl

# Create adapter directory
import os
os.makedirs('/content/fcp_adapter', exist_ok=True)

# Model (smaller for T4)
MODEL = "Qwen/Qwen2.5-3B"
OUTPUT = "/content/fcp_adapter"

# Dataset
from datasets import Dataset
import json

data = [
    {"instruction": "Что такое квантовая запутанность?", "output": "Квантовая запутанность — явление, при котором частицы связываются..."},
    # Add more pairs...
]

# Load model with QLoRA
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL)

# Apply LoRA
lora = LoraConfig(r=8, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj"])
model = get_peft_model(model, lora)

# Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(output_dir=OUTPUT, num_train_epochs=3, per_device_train_batch_size=2),
    train_dataset=Dataset.from_list(data),
    dataset_text_field="text",
)

trainer.train()
trainer.save_model(OUTPUT)

print("Done!")
""")
    return 0

if __name__ == "__main__":
    sys.exit(main())