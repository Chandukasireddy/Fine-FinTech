# Gemma-3 Fine-tuning Setup Guide

## ✅ Implementation Complete

### 1. Project Structure
- ✅ Updated all imports to use `AutoModelForCausalLM` (best practice for fine-tuning)
- ✅ Configured for `google/gemma-3-4b-it` model
- ✅ Added HuggingFace authentication with `.env` file
- ✅ Updated requirements.txt with latest packages
- ✅ Created conda environment: `fine-fintech`

### 2. Key Files Updated

#### `src/finetune.py`
- ✅ Uses `AutoModelForCausalLM` instead of `Gemma3ForConditionalGeneration`
- ✅ Integrated HuggingFace authentication
- ✅ Uses `device_map="auto"` for multi-GPU support
- ✅ Uses `torch.bfloat16` for better numerical stability
- ✅ Uses `attn_implementation='eager'` for Gemma-3
- ✅ Implements LoRA fine-tuning with PEFT
- ✅ Includes 4-bit quantization support

#### `config/training_config.yaml`
- ✅ Updated to use `google/gemma-3-4b-it`
- ✅ Configured optimal LoRA parameters
- ✅ Set memory-efficient training settings

#### `requirements.txt`
- ✅ Updated all packages to latest versions
- ✅ Added transformers from Gemma-3 branch
- ✅ Includes: datasets, accelerate, peft, trl, bitsandbytes

#### `scripts/hf_login.py`
- ✅ Created authentication script
- ✅ Uses `.env` file for secure token storage

## 🔧 Setup Steps

### Step 1: Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token (or use existing one)
3. Copy the token (starts with `hf_...`)

### Step 2: Accept Gemma-3 License

1. Visit https://huggingface.co/google/gemma-3-4b-it
2. Click "Agree and access repository"
3. This gives you access to the model

### Step 3: Update .env File

```bash
# Edit the .env file and replace the token
nano .env

# Add your real token:
HUGGINGFACE_TOKEN=hf_YOUR_ACTUAL_TOKEN_HERE
```

### Step 4: Test Authentication

```bash
conda activate fine-fintech
python scripts/hf_login.py
```

You should see: "✓ Successfully logged in to Hugging Face!"

### Step 5: Test Model Loading (Optional)

Create a test script to verify everything works:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv()
login(os.getenv("HUGGINGFACE_TOKEN"))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation='eager'
)

print("✓ Model loaded successfully!")
print(f"Device map: {model.hf_device_map}")
```

### Step 6: Start Fine-tuning

```bash
# Make sure you're in the conda environment
conda activate fine-fintech

# Start training
python src/finetune.py --mode train
```

## 📊 Training Configuration

Current settings optimized for a single GPU (8GB+ VRAM):

```yaml
Model: google/gemma-3-4b-it
Epochs: 3
Batch size: 2 per device
Gradient accumulation: 4 (effective batch size = 8)
Learning rate: 2e-4
Max sequence length: 2048
LoRA r: 16
LoRA alpha: 32
Quantization: 4-bit (NF4)
```

**GPU Memory Requirements:**
- With 4-bit quantization: ~6-8 GB VRAM
- With full precision: ~16-20 GB VRAM

## 🎯 Architecture Choices

### Why AutoModelForCausalLM?
✅ Industry standard for LLM fine-tuning
✅ Compatible with all training frameworks (TRL, SFTTrainer, etc.)
✅ Works seamlessly with PEFT/LoRA
✅ Gemma-3 is a causal language model (not seq2seq)
✅ Better documentation and community support

### Why LoRA (PEFT)?
✅ Memory efficient - only trains ~1% of parameters
✅ Faster training
✅ Easy to merge back to base model
✅ Can train multiple adapters for different tasks

### Why 4-bit Quantization?
✅ Reduces memory usage by 75%
✅ Enables training on consumer GPUs
✅ Minimal accuracy loss with NF4 quantization
✅ Uses bitsandbytes for efficient implementation

## 🚀 Next Steps

1. **Update your HuggingFace token** in `.env` file
2. **Test authentication**: `python scripts/hf_login.py`
3. **Review configuration**: Check `config/training_config.yaml`
4. **Start training**: `python src/finetune.py --mode train`
5. **Monitor progress**: Watch terminal output and GPU usage
6. **Evaluate results**: `python src/finetune.py --mode eval`

## 📝 Training Tips

1. **Adjust batch size** based on your GPU memory
2. **Use gradient checkpointing** if you run out of memory
3. **Enable W&B** for better monitoring (set `use_wandb: true`)
4. **Save checkpoints frequently** (every 500 steps by default)
5. **Monitor loss values** - should decrease steadily

## ⚠️ Troubleshooting

### "Invalid user token"
- Your HuggingFace token is incorrect or expired
- Get a new token from https://huggingface.co/settings/tokens
- Update `.env` file

### "403 Forbidden"
- You haven't accepted the Gemma-3 license
- Visit https://huggingface.co/google/gemma-3-4b-it
- Click "Agree and access repository"

### "CUDA out of memory"
- Reduce `per_device_train_batch_size` to 1
- Enable gradient checkpointing
- Use 4-bit quantization (already enabled)
- Reduce `max_seq_length` to 1024

### "Model not found"
- Check your internet connection
- Verify model name: `google/gemma-3-4b-it`
- Ensure you have access to the model

## 📚 Additional Resources

- Gemma-3 Model Card: https://huggingface.co/google/gemma-3-4b-it
- FinQA Dataset: https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA
- PEFT Documentation: https://huggingface.co/docs/peft
- Transformers Documentation: https://huggingface.co/docs/transformers
