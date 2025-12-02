"""
Data loader for TheFinAI/Fino1_Reasoning_Path_FinQA dataset
"""

import os
import pandas as pd
from datasets import Dataset, load_dataset
from typing import Dict, List, Optional
from pathlib import Path
from transformers import DataCollatorForLanguageModeling

# Training prompt template with reasoning
TRAIN_PROMPT_STYLE = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
</think>
{}
"""

# Inference prompt template (for testing original model - 2 placeholders)
INFERENCE_PROMPT_STYLE = """
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
{}
"""

def formatting_prompts_func(examples, tokenizer):
    """
    Format dataset examples using the training prompt style
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Tokenizer to get EOS token
        
    Returns:
        Dictionary with formatted "text" column
    """
    inputs = examples["Open-ended Verifiable Question"]
    complex_cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    
    for question, cot, response in zip(inputs, complex_cots, outputs):
        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
        
        text = TRAIN_PROMPT_STYLE.format(question, cot, response)
        texts.append(text)
    
    return {"text": texts}


def load_and_format_dataset(tokenizer, num_samples: int = 500):
    """
    Load and format the FinQA dataset
    
    Args:
        tokenizer: Tokenizer for adding EOS token
        num_samples: Number of samples to load from training split
        
    Returns:
        Formatted dataset with "text" column
    """
    # Load dataset
    dataset = load_dataset(
        "TheFinAI/Fino1_Reasoning_Path_FinQA",
        split=f"train[0:{num_samples}]",
        trust_remote_code=True
    )
    
    # Format dataset
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True
    )
    
    return dataset


def create_data_collator(tokenizer):
    """
    Create data collator for causal language modeling
    
    Args:
        tokenizer: Tokenizer for processing sequences
        
    Returns:
        DataCollatorForLanguageModeling instance
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # we're doing causal LM, not masked LM
    )
    
    return data_collator


def generate_response(model, tokenizer, question: str, max_new_tokens: int = 512):
    """
    Generate response from model for a given question
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        question: Input question
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response text
    """
    # Format question with inference prompt style
    prompt = INFERENCE_PROMPT_STYLE.format(question, "")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode tokens back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

