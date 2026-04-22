"""
FCP LoRA Training Script for Google Colab.
Run with T4 GPU.

Requirements:
    !pip install unsloth xformers trl peft accelerate
"""
import sys
import os
import json
import torch

def train_lora():
    print("=" * 60)
    print("FCP LoRA Training - Colab Version")
    print("=" * 60)
    
    MODEL = "Qwen/Qwen2.5-3B"
    OUTPUT = "/content/fcp_adapter"
    
    os.makedirs(OUTPUT, exist_ok=True)
    
    # Dataset
    dataset_path = "/content/lora_dataset.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Dataset: {len(dataset)} examples")
    print(f"Output: {OUTPUT}")
    print()
    
    # GPU check
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
    print()
    
    try:
        from unsloth import FastLanguageModel
        
        print("Loading model with Unsloth...")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL,
            max_seq_length=512,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        print("Adding LoRA adapters...")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print()
        
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="instruction",
            max_seq_length=512,
            dataset_num_proc=2,
            packing=True,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=100,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=42,
                output_dir=OUTPUT,
                report_to="none",
            ),
        )
        
        print("Training (100 steps)...")
        trainer.train()
        
        # Save
        print()
        print("Saving adapter...")
        trainer.save_model(OUTPUT)
        
        print()
        print("=" * 60)
        print(f"Adapter saved to: {OUTPUT}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(train_lora())