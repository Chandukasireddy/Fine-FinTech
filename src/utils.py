"""
Utility functions for loading model and tokenizer
"""

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name: str = "google/gemma-3-4b-it"):
    """
    Load model and tokenizer with HuggingFace authentication
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load environment and authenticate
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    login(hf_token)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        token=hf_token
    ).eval()
    
    return model, tokenizer
