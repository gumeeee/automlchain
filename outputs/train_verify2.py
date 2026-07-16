"""Training script for job {job_id}."""

import json
import os
import sys

# Training configuration
config = {
  "model": "test",
  "training_file": "/tmp/test.jsonl"
}

# Setup output
output_dir = "outputs/checkpoints/verify2"
os.makedirs(output_dir, exist_ok=True)

# Write job metadata
with open(os.path.join(output_dir, "job_info.json"), "w") as f:
    json.dump({
        "job_id": "verify2",
        "status": "running",
        "start_time": "2026-07-16 17:58:50",
    }, f, indent=2)

def load_data(file_path):
    """Load and format dataset."""
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=file_path, split="train")

    def format_example(example):
        if "messages" in example:
            # Chat format
            text = ""
            for msg in example['messages']:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += "<|" + role + "|>" + "\n" + content + "\n"
            text += "<|assistant|>" + "\n"
        elif "input" in example and "output" in example:
            # Q&A format
            text = "### Input:\n" + example["input"] + "\n### Output:\n" + example["output"] + "\n"
        else:
            text = str(example)
        return {"text": text}

    dataset = dataset.map(format_example)
    return dataset.remove_columns([c for c in dataset.column_names if c != "text"])

def main():
    print(f'Starting training job: verify2')
    print(f'Model: test')
    print(f'Dataset: /tmp/test.jsonl')

    # Load tokenizer
    print('Loading tokenizer...')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("""test""")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading dataset...")
    dataset = load_data('/tmp/test.jsonl')

    # Split for evaluation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    print("Tokenizing dataset...")
    dataset = dataset.map(tokenize, batched=True, remove_columns=['text'])

    # Load model
    print("Loading model...")
    import torch
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        'pretrained_model_name_or_path': """test""",
        'trust_remote_code': True,
        'device_map': 'auto',

        'torch_dtype': torch.float16,
    }

    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes
            print("bitsandbytes version:", bitsandbytes.__version__)
            print("Using QLoRA (4-bit quantization)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs['quantization_config'] = bnb_config
            print("quantization_config set successfully")
        except ImportError as e:
            print(f"WARNING: bitsandbytes not available: {e}")
            print("Training will continue without quantization - may cause OOM!")

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

    # Check model memory and quantization status
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        if hasattr(model, 'is_quantized') and callable(model.is_quantized):
            print(f"Model is quantized: {model.is_quantized()}")
        elif hasattr(model, 'hf_quantizer'):
            print(f"Quantization method: {model.hf_quantizer}")

    # Setup LoRA
    print("Setting up LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=0.0001,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save final model
    print("Saving model...")
    model.save_pretrained(os.path.join(output_dir, "final"))

    # Update job info
    with open(os.path.join(output_dir, "job_info.json"), "w") as f:
        json.dump({
            "job_id": "verify2",
            "status": "completed",
            "start_time": "2026-07-16 17:58:50",
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": os.path.join(output_dir, "final"),
        }, f, indent=2)

    print(f"Training completed! Model saved to {output_dir}/final")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)