"""
Fine-tuning script for Gemma-3 model
"""

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load environment and authenticate
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

login(hf_token)
print("✓ Logged in to HuggingFace")

# Model configuration
MODEL_NAME = "google/gemma-3-4b-it"

print(f"\nLoading model: {MODEL_NAME}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

# Load model with automatic device mapping
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    token=hf_token
).eval()

print("\n✓ Model and tokenizer loaded successfully!")
print(f"Model device map: {model.hf_device_map}")
print(f"Model dtype: {model.dtype}")
