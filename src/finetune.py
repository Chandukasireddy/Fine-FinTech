"""
Fine-tuning script for Gemma3 model using TheFinAI/Fino1_Reasoning_Path_FinQA dataset
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import wandb

from data_loader import FinQADataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration class for fine-tuning parameters"""
    
    # Model configuration
    model_name: str = "google/gemma-2-9b-it"  # Gemma3 model from HF
    ollama_model_name: str = "gemma3:4b"  # Your Ollama model
    use_ollama_weights: bool = True
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Training parameters
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Data parameters
    max_seq_length: int = 2048
    dataset_text_field: str = "text"
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "gemma3-finqa-finetune"
    wandb_run_name: Optional[str] = None

class Gemma3FineTuner:
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.data_loader = FinQADataLoader()
        
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info("Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
        else:
            bnb_config = None
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        logger.info("Model and tokenizer setup complete")
        
    def load_and_prepare_data(self) -> Dict[str, Dataset]:
        """Load and prepare the training data"""
        logger.info("Loading and preparing data...")
        
        # Load datasets
        datasets = self.data_loader.load_and_prepare_all_splits(
            max_length=self.config.max_seq_length
        )
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples[self.config.dataset_text_field],
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        tokenized_datasets = {}
        for split_name, dataset in datasets.items():
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
        
        logger.info("Data preparation complete")
        return tokenized_datasets
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Trainer:
        """Setup the Trainer"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
    
    def train(self):
        """Main training function"""
        logger.info("Starting fine-tuning process...")
        
        # Setup wandb if enabled
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Load and prepare data
        datasets = self.load_and_prepare_data()
        
        # Get training and evaluation datasets
        train_dataset = datasets.get("train")
        eval_dataset = datasets.get("validation")
        
        if train_dataset is None:
            raise ValueError("Training dataset not found!")
        
        # Setup trainer
        trainer = self.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed!")
    
    def evaluate(self, test_dataset: Optional[Dataset] = None):
        """Evaluate the trained model"""
        if test_dataset is None:
            datasets = self.load_and_prepare_data()
            test_dataset = datasets.get("test")
        
        if test_dataset is None:
            logger.warning("No test dataset available for evaluation")
            return
        
        trainer = self.setup_trainer(test_dataset)
        results = trainer.evaluate(eval_dataset=test_dataset)
        
        logger.info(f"Evaluation results: {results}")
        return results

def load_config(config_path: str) -> FineTuningConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return FineTuningConfig(**config_dict)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma3 model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train",
                       help="Mode: train or evaluate")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.info("Using default configuration")
        config = FineTuningConfig()
    
    # Create fine-tuner
    fine_tuner = Gemma3FineTuner(config)
    
    if args.mode == "train":
        fine_tuner.train()
    else:
        fine_tuner.evaluate()

if __name__ == "__main__":
    main()