"""
Model loading utilities for Gemma 3 fine-tuning.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import (
    AutoTokenizer, 
    Gemma3ForConditionalGeneration,
    GenerationConfig
)

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and configuration of Gemma 3 models and tokenizers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelLoader with configuration.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.tokenizer_config = config.get("tokenizer", {})
        self.generation_config = config.get("generation", {})
        
    def load_model_and_tokenizer(
        self, 
        model_path: Optional[str] = None,
        use_auth_token: Optional[str] = None
    ) -> Tuple[Gemma3ForConditionalGeneration, AutoTokenizer]:
        """
        Load Gemma 3 model and tokenizer.
        
        Args:
            model_path: Path to the model. If None, uses config.
            use_auth_token: Hugging Face auth token
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Use provided path or config
        model_name = model_path or self.model_config.get("name", "google/gemma-3-4b-it")
        
        logger.info(f"Loading model: {model_name}")
        
        # Determine torch dtype
        torch_dtype = self._get_torch_dtype()
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side=self.tokenizer_config.get("padding_side", "right"),
                trust_remote_code=self.model_config.get("trust_remote_code", True),
                token=use_auth_token
            )
            
            # Add EOS token if specified
            if self.tokenizer_config.get("add_eos_token", True) and tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_name,
                device_map=self.model_config.get("device_map", "auto"),
                torch_dtype=torch_dtype,
                attn_implementation=self.model_config.get("attention_implementation", "eager"),
                trust_remote_code=self.model_config.get("trust_remote_code", True),
                token=use_auth_token
            )
            
            # Set model to evaluation mode
            model.eval()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Model dtype: {next(model.parameters()).dtype}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {str(e)}")
            raise
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype from config."""
        dtype_str = self.model_config.get("torch_dtype", "float16")
        
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto"
        }
        
        return dtype_mapping.get(dtype_str, torch.float16)
    
    def setup_generation_config(self, model: Gemma3ForConditionalGeneration) -> GenerationConfig:
        """
        Setup generation configuration for the model.
        
        Args:
            model: The loaded model
            
        Returns:
            GenerationConfig object
        """
        generation_config = GenerationConfig(
            max_new_tokens=self.generation_config.get("max_new_tokens", 1200),
            temperature=self.generation_config.get("temperature", 0.7),
            top_p=self.generation_config.get("top_p", 0.9),
            do_sample=self.generation_config.get("do_sample", True),
            use_cache=self.generation_config.get("use_cache", True),
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
        )
        
        model.generation_config = generation_config
        logger.info("Generation config setup completed")
        
        return generation_config
    
    def get_model_info(self, model: Gemma3ForConditionalGeneration) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Args:
            model: The loaded model
            
        Returns:
            Dictionary with model information
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get memory usage
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "config": model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
        }
        
        return info
    
    def save_model_and_tokenizer(
        self, 
        model: Gemma3ForConditionalGeneration,
        tokenizer: AutoTokenizer,
        save_path: str
    ):
        """
        Save model and tokenizer to specified path.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model and tokenizer to: {save_path}")
        
        try:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info("Model and tokenizer saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model and tokenizer: {str(e)}")
            raise