"""
Training utilities for fine-tuning Gemma 3 on financial datasets.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional
from pathlib import Path
from transformers import (
    TrainingArguments,
    Gemma3ForConditionalGeneration,
    AutoTokenizer
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

logger = logging.getLogger(__name__)

class FinancialTrainer:
    """Handles fine-tuning of Gemma 3 model on financial Q&A datasets."""
    
    def __init__(
        self, 
        model: Gemma3ForConditionalGeneration,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any]
    ):
        """
        Initialize FinancialTrainer.
        
        Args:
            model: Pre-loaded Gemma 3 model
            tokenizer: Pre-loaded tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_config = config.get("training", {})
        self.lora_config_dict = config.get("lora", {})
        
        # Initialize trainer-related objects
        self.trainer = None
        self.peft_config = None
        self.training_args = None
        
    def setup_lora_config(self) -> LoraConfig:
        """
        Setup LoRA (Low-Rank Adaptation) configuration.
        
        Returns:
            LoraConfig object
        """
        logger.info("Setting up LoRA configuration")
        
        self.peft_config = LoraConfig(
            r=self.lora_config_dict.get("r", 64),
            lora_alpha=self.lora_config_dict.get("lora_alpha", 16),
            lora_dropout=self.lora_config_dict.get("lora_dropout", 0.05),
            bias=self.lora_config_dict.get("bias", "none"),
            task_type=self.lora_config_dict.get("task_type", "CAUSAL_LM"),
            target_modules=self.lora_config_dict.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            fan_in_fan_out=self.lora_config_dict.get("fan_in_fan_out", False),
            inference_mode=self.lora_config_dict.get("inference_mode", False)
        )
        
        logger.info(f"LoRA config: r={self.peft_config.r}, alpha={self.peft_config.lora_alpha}")
        return self.peft_config
    
    def setup_training_arguments(self) -> TrainingArguments:
        """
        Setup training arguments for the SFTTrainer.
        
        Returns:
            TrainingArguments object
        """
        logger.info("Setting up training arguments")
        
        # Ensure output directory exists
        output_dir = self.training_config.get("output_dir", "./output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.training_config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 2),
            optim=self.training_config.get("optim", "paged_adamw_32bit"),
            num_train_epochs=self.training_config.get("num_train_epochs", 1),
            logging_steps=self.training_config.get("logging_steps", 0.2),
            warmup_steps=self.training_config.get("warmup_steps", 10),
            logging_strategy=self.training_config.get("logging_strategy", "steps"),
            learning_rate=self.training_config.get("learning_rate", 2e-4),
            fp16=self.training_config.get("fp16", False),
            bf16=self.training_config.get("bf16", False),
            group_by_length=self.training_config.get("group_by_length", True),
            report_to=self.training_config.get("report_to", "none"),
            save_steps=self.training_config.get("save_steps", 500),
            save_total_limit=self.training_config.get("save_total_limit", 2),
            evaluation_strategy=self.training_config.get("evaluation_strategy", "no"),
            dataloader_drop_last=self.training_config.get("dataloader_drop_last", False),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            remove_unused_columns=self.training_config.get("remove_unused_columns", False),
        )
        
        logger.info(f"Training args: epochs={self.training_args.num_train_epochs}, "
                   f"lr={self.training_args.learning_rate}, "
                   f"batch_size={self.training_args.per_device_train_batch_size}")
        
        return self.training_args
    
    def setup_trainer(
        self, 
        train_dataset: Dataset,
        data_collator: Any
    ) -> SFTTrainer:
        """
        Setup the SFTTrainer for fine-tuning.
        
        Args:
            train_dataset: Training dataset
            data_collator: Data collator for batching
            
        Returns:
            SFTTrainer object
        """
        logger.info("Setting up SFTTrainer")
        
        # Setup configurations if not already done
        if self.peft_config is None:
            self.setup_lora_config()
        
        if self.training_args is None:
            self.setup_training_arguments()
        
        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            peft_config=self.peft_config,
            data_collator=data_collator,
        )
        
        logger.info("SFTTrainer setup completed")
        return self.trainer
    
    def train(
        self, 
        train_dataset: Dataset,
        data_collator: Any,
        clear_cache_before: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the training process.
        
        Args:
            train_dataset: Training dataset
            data_collator: Data collator for batching
            clear_cache_before: Whether to clear CUDA cache before training
            
        Returns:
            Training statistics
        """
        logger.info("Starting training process")
        
        # Clear CUDA cache if requested
        if clear_cache_before and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        # Setup trainer if not already done
        if self.trainer is None:
            self.setup_trainer(train_dataset, data_collator)
        
        # Log training information
        self._log_training_info(train_dataset)
        
        try:
            # Start training
            trainer_stats = self.trainer.train()
            
            logger.info("Training completed successfully")
            logger.info(f"Training loss: {trainer_stats.training_loss:.4f}")
            
            return trainer_stats
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _log_training_info(self, dataset: Dataset):
        """Log information about the training setup."""
        logger.info("=== Training Information ===")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        logger.info("=== Training Starting ===")
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
        """
        if self.trainer is None:
            logger.error("No trainer available. Cannot save model.")
            return
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to: {save_path}")
        
        try:
            # Save the model and tokenizer
            self.trainer.save_model(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get training statistics if available.
        
        Returns:
            Training statistics or None if no training has been done
        """
        if self.trainer is None:
            return None
        
        # Try to get training history
        try:
            log_history = self.trainer.state.log_history
            return {
                "log_history": log_history,
                "total_steps": self.trainer.state.global_step,
                "epoch": self.trainer.state.epoch,
                "is_local_process_zero": self.trainer.is_local_process_zero(),
                "is_world_process_zero": self.trainer.is_world_process_zero(),
            }
        except Exception as e:
            logger.warning(f"Could not retrieve training stats: {str(e)}")
            return None
    
    def resume_training(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        if self.trainer is None:
            logger.error("No trainer available. Cannot resume training.")
            return
        
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            self.trainer.train(resume_from_checkpoint=checkpoint_path)
            logger.info("Training resumed successfully")
        except Exception as e:
            logger.error(f"Failed to resume training: {str(e)}")
            raise