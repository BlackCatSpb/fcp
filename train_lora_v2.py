"""
FCP LoRA Training Script for Google Colab v2
Обучение LoRA адаптера на базе ruadapt_qwen3_4b

Usage in Colab:
    1. Upload this script
    2. Mount Google Drive with ruadapt_qwen3_4b
    3. Run training

Requirements:
    !pip install unsloth xformers trl peft accelerate
"""
import sys
import os
import json
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_dataset(path: str):
    """Load lora_dataset.json"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def create_prompt_formatter():
    """Create prompt formatter for FCP style"""
    def format prompt(sample):
        return f"### Instruction\n{sample['instruction']}\n\n### Response\n{sample['output']}"
    return format_prompt


class FPELDADataset:
    """Dataset for FCP LoRA training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data = load_dataset(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_prompt = create_prompt_formatter()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        text = self.format_prompt(sample)
        
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def train_lora(
    model_path: str = "/content/ruadapt_qwen3_4b",
    data_path: str = "/content/lora_dataset.json",
    output_path: str = "/content/fcp_adapter",
    rank: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 3e-4,
    max_steps: int = -1,
):
    """Train LoRA adapter"""
    print("=" * 60)
    print("FCP LoRA Training - Colab v2")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n[CONFIG]")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {data_path}")
    print(f"  Output: {output_path}")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch: {batch_size}")
    print(f"  LR: {learning_rate}")
    
    # Load dataset first
    dataset_raw = load_dataset(data_path)
    print(f"\n[DATASET] {len(dataset_raw)} examples")
    
    # GPU check
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[VRAM] {vram:.1f} GB")
    print()
    
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments, AutoTokenizer
        
        # Load tokenizer first
        print("[TOKENIZER] Loading...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        # Fix tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with Unsloth
        print("[MODEL] Loading with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        print("[LoRA] Adding adapters...")
        
        # LoRA config - FCP target modules
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[PARAMS] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print()
        
        # Prepare dataset
        dataset = FPELDADataset(data_path, tokenizer)
        
        # Calculate steps
        steps_per_epoch = len(dataset) // batch_size
        if max_steps > 0:
            total_steps = min(max_steps, steps_per_epoch * num_epochs)
        else:
            total_steps = steps_per_epoch * num_epochs
        
        print(f"[TRAINING]")
        print(f"  Samples: {len(dataset)}")
        print(f"  Batch: {batch_size}")
        print(f"  Steps/epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=total_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=20,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_path,
            save_steps=total_steps // 3,
            save_total_limit=2,
            report_to="none",
        )
        
        # Create trainer
        print("\n[START] Training...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="instruction",
            max_seq_length=512,
            dataset_num_proc=2,
            packing=True,
            args=training_args,
        )
        
        # Train
        trainer.train()
        
        # Save
        print("\n[SAVE] Saving adapter...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save config
        config = {
            "model_path": model_path,
            "rank": rank,
            "lora_alpha": lora_alpha,
            "trainable_params": trainable,
            "total_params": total,
            "num_samples": len(dataset_raw),
            "epochs": num_epochs,
            "fcp_version": "v15"
        }
        with open(os.path.join(output_path, "adapter_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Adapter: {output_path}")
        print(f"Config: {config}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


def train_adalora(
    model_path: str = "/content/ruadapt_qwen3_4b",
    data_path: str = "/content/lora_dataset.json",
    output_path: str = "/content/fcp_adapter_adalora",
):
    """Train with AdaLoRA (adaptive rank)"""
    print("=" * 60)
    print("FCP AdaLoRA Training")
    print("=" * 60)
    
    from unsloth import FastLanguageModel
    from peft import AdaLoRAConfig, get_peft_model
    
    # Adaptive rank per layer: 1-8 → r=4, 9-16 → r=8, 17-36 → r=16
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # AdaLoRA config
    lora_config = AdaLoRAConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load model
    print("[MODEL] Loading...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    
    # Apply AdaLoRA
    print("[LoRA] Applying AdaLoRA...")
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAMS] Trainable: {trainable:,}")
    
    # Dataset
    dataset = load_dataset(data_path)
    print(f"[DATASET] {len(dataset)} samples")
    
    # Train
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="instruction",
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            output_dir=output_path,
            report_to="none",
        ),
    )
    
    print("[TRAINING]...")
    trainer.train()
    
    # Save
    print("[SAVE]...")
    model.save_pretrained(output_path)
    
    print(f"[DONE] Saved to {output_path}")
    return 0


def main():
    """Main entry"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FCP LoRA Training")
    parser.add_argument("--model", default="/content/ruadapt_qwen3_4b", help="Model path")
    parser.add_argument("--data", default="/content/lora_dataset.json", help="Dataset path")
    parser.add_argument("--output", default="/content/fcp_adapter", help="Output path")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=-1, help="Max steps (-1 for full)")
    parser.add_argument("--adalora", action="store_true", help="Use AdaLoRA")
    
    args = parser.parse_args()
    
    if args.adalora:
        return train_adalora(args.model, args.data, args.output)
    else:
        return train_lora(
            model_path=args.model,
            data_path=args.data,
            output_path=args.output,
            rank=args.rank,
            lora_alpha=args.alpha,
            num_epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            max_steps=args.steps,
        )


if __name__ == "__main__":
    sys.exit(main())