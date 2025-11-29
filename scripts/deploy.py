#!/usr/bin/env python3
"""
Model deployment script for uploading fine-tuned models to Hugging Face Hub.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, load_environment_variables, ConfigManager
from src.model_loader import ModelLoader
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

def setup_authentication():
    """Setup Hugging Face authentication."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token or hf_token == "your_hugging_face_token_here":
        logger.error("Please set HUGGINGFACE_TOKEN environment variable")
        return False
    
    try:
        login(hf_token)
        logger.info("Hugging Face authentication successful")
        return True
    except Exception as e:
        logger.error(f"Hugging Face authentication failed: {str(e)}")
        return False

def deploy_model(
    local_model_path: str,
    hub_model_name: str,
    private: bool = False,
    commit_message: str = None
):
    """
    Deploy model to Hugging Face Hub.
    
    Args:
        local_model_path: Path to local model directory
        hub_model_name: Name for the model on Hugging Face Hub
        private: Whether to make the model private
        commit_message: Optional commit message
    """
    local_path = Path(local_model_path)
    
    if not local_path.exists():
        logger.error(f"Local model path does not exist: {local_path}")
        return False
    
    logger.info(f"Deploying model from: {local_path}")
    logger.info(f"Hub model name: {hub_model_name}")
    
    try:
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(local_path),
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
        # Push to hub
        commit_msg = commit_message or f"Upload fine-tuned Gemma 3 model for financial Q&A"
        
        logger.info("Pushing model to Hugging Face Hub...")
        model.push_to_hub(
            hub_model_name,
            private=private,
            commit_message=commit_msg
        )
        
        logger.info("Pushing tokenizer to Hugging Face Hub...")
        tokenizer.push_to_hub(
            hub_model_name,
            private=private,
            commit_message=commit_msg
        )
        
        logger.info(f"Model successfully deployed to: https://huggingface.co/{hub_model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        return False

def create_model_card(hub_model_name: str, local_model_path: str):
    """
    Create and upload a model card for the deployed model.
    
    Args:
        hub_model_name: Name of the model on Hugging Face Hub
        local_model_path: Path to local model for getting info
    """
    model_card_content = f"""
---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- financial-qa
- reasoning
- gemma-3
- fine-tuned
- finance
datasets:
- TheFinAI/Fino1_Reasoning_Path_FinQA
model-index:
- name: {hub_model_name.split('/')[-1]}
  results: []
---

# Gemma 3 Fine-tuned for Financial Q&A

This model is a fine-tuned version of Google's Gemma 3-4B-IT model, specifically trained for financial question-answering tasks with enhanced reasoning capabilities.

## Model Description

- **Base Model**: google/gemma-3-4b-it
- **Fine-tuned on**: TheFinAI/Fino1_Reasoning_Path_FinQA dataset
- **Task**: Financial Question Answering with Step-by-step Reasoning
- **Language**: English

## Training Details

The model was fine-tuned using:
- **LoRA (Low-Rank Adaptation)** for efficient training
- **SFTTrainer** from the TRL library
- **Custom prompt format** with reasoning chains

### Training Parameters
- Learning Rate: 2e-4
- Batch Size: 1 (with gradient accumulation)
- Epochs: 1
- LoRA Rank: 64
- LoRA Alpha: 16

## Usage

```python
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

# Load model and tokenizer
model = Gemma3ForConditionalGeneration.from_pretrained("{hub_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{hub_model_name}")

# Format your question
prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{{your_financial_question}}

### Response:
<think>
'''

# Generate response
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=1200,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("### Response:")[1])
```

## Model Performance

This fine-tuned model demonstrates:
- Enhanced understanding of financial contexts
- Step-by-step reasoning for complex financial calculations
- Improved accuracy in financial question answering
- Better structured responses with clear thinking process

## Limitations

- Primarily trained on English financial data
- May require domain-specific fine-tuning for other financial contexts
- Should not be used for actual financial advice without expert validation

## Citation

If you use this model, please cite:

```bibtex
@misc{{gemma3-financial-finetuning,
  title={{Fine-Tuned Gemma 3 for Financial Q&A}},
  author={{Fine-FinTech Team}},
  year={{2024}},
  url={{https://huggingface.co/{hub_model_name}}}
}}
```

## License

This model is licensed under the Apache 2.0 License.
"""

    try:
        # Save model card locally first
        model_card_path = Path(local_model_path) / "README.md"
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card_content)
        
        logger.info(f"Model card created at: {model_card_path}")
        
        # Upload to hub
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=hub_model_name,
            commit_message="Add model card"
        )
        
        logger.info("Model card uploaded to Hugging Face Hub")
        
    except Exception as e:
        logger.warning(f"Failed to create/upload model card: {str(e)}")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model to Hugging Face Hub")
    parser.add_argument("--model-path", required=True, help="Path to local fine-tuned model")
    parser.add_argument("--hub-name", required=True, help="Model name on Hugging Face Hub (user/model-name)")
    parser.add_argument("--private", action="store_true", help="Make model private")
    parser.add_argument("--commit-message", help="Custom commit message")
    parser.add_argument("--create-model-card", action="store_true", help="Create and upload model card")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO", log_file="deployment")
    
    # Load environment variables
    load_environment_variables()
    
    logger.info("Starting model deployment process")
    logger.info(f"Local model path: {args.model_path}")
    logger.info(f"Hub model name: {args.hub_name}")
    
    # Setup authentication
    if not setup_authentication():
        sys.exit(1)
    
    # Deploy model
    success = deploy_model(
        local_model_path=args.model_path,
        hub_model_name=args.hub_name,
        private=args.private,
        commit_message=args.commit_message
    )
    
    if not success:
        logger.error("Model deployment failed")
        sys.exit(1)
    
    # Create model card if requested
    if args.create_model_card:
        create_model_card(args.hub_name, args.model_path)
    
    logger.info("Deployment completed successfully!")

if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    main()