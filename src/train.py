"""
Fine-tuning script for Gemma-3 model on FinQA dataset using SFTTrainer
"""

import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig

from utils import load_model_and_tokenizer
from data_loader import load_and_format_dataset, create_data_collator


def get_lora_config():
    """
    Configure LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    
    Returns:
        LoraConfig: LoRA configuration object
    """
    peft_config = LoraConfig(
        lora_alpha=16,                           # Scaling factor for LoRA
        lora_dropout=0.05,                       # Add slight dropout for regularization
        r=64,                                    # Rank of the LoRA update matrices
        bias="none",                             # No bias reparameterization
        task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target modules for LoRA
    )
    return peft_config


def get_training_arguments():
    """
    Configure training arguments for SFTTrainer.
    
    Returns:
        TrainingArguments: Training configuration object
    """
    training_arguments = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="none"
    )
    return training_arguments


def train_model():
    """
    Main training function that orchestrates the fine-tuning process.
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load and format dataset
    dataset = load_and_format_dataset(tokenizer, num_samples=500)
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Configure LoRA
    peft_config = get_lora_config()
    
    # Configure training arguments
    training_arguments = get_training_arguments()
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )
    
    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model("output/final_model")
    tokenizer.save_pretrained("output/final_model")
    
    return trainer_stats


if __name__ == "__main__":
    train_model()
