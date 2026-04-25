# Google Colab - FCP LoRA Training v2

## Quick Start (Recommended - Using Unsloth)

1. Upload `train_lora_v2.ipynb` to Colab
2. Mount Google Drive with model
3. Run notebook

Or use these steps:

## Manual Setup

```python
# 1. Install dependencies
!pip install unsloth xformers trl peft accelerate -q

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Upload dataset
from google.colab import files
files.upload()
```

## Training Script (Unsloth)

```python
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer

# Paths
MODEL_PATH = '/content/drive/MyDrive/ruadapt_qwen3_4b'
DATA_PATH = '/content/lora_dataset.json'
OUTPUT_DIR = '/content/fcp_adapter'

# Config
RANK = 8
LORA_ALPHA = 16
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 3e-4

# Load dataset
dataset = json.load(open(DATA_PATH, 'r', encoding='utf-8'))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH, max_seq_length=512, dtype=torch.float16, load_in_4bit=True
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model, r=RANK, lora_alpha=LORA_ALPHA,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM'
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)')

# Format dataset
def formatting_func(sample):
    return [f"### Instruction\n{sample['instruction']}\n\n### Response\n{sample['output']}"]

# Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    formatting_func=formatting_func, max_seq_length=512, packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=4,
        warmup_steps=10, num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
        fp16=True, logging_steps=20, optim='adamw_8bit', weight_decay=0.01,
        lr_scheduler_type='linear', seed=42, output_dir=OUTPUT_DIR,
        save_steps=100, save_total_limit=2, report_to='none'
    )
)

trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
config = {'model_path': MODEL_PATH, 'rank': RANK, 'lora_alpha': LORA_ALPHA,
         'trainable_params': trainable, 'total_params': total, 'epochs': NUM_EPOCHS, 'fcp_version': 'v15'}
with open(f'{OUTPUT_DIR}/adapter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Saved to {OUTPUT_DIR}')
```

## Download Result

```python
import shutil
shutil.make_archive('/content/fcp_adapter', 'zip', '/content/fcp_adapter')
from google.colab import files
files.download('/content/fcp_adapter.zip')
```

## Files for Upload

| File | Description |
|------|-------------|
| `colab_upload/train_lora_v2.ipynb` | Jupyter notebook |
| `colab_upload/lora_dataset.json` | Training data (146 examples) |
| `colab_upload/train_lora_v2.py` | Standalone script |

## Model Path

Replace `/content/drive/MyDrive/ruadapt_qwen3_4b` with your actual model location in Google Drive.

## After Training

1. Download `fcp_adapter.zip`
2. Extract to `C:\Users\black\OneDrive\Desktop\FCP\lora_adapters\fcp_finetuned`
3. Test with `python test_hit2.py`