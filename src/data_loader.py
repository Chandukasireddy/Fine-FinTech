"""
Data loader for TheFinAI/Fino1_Reasoning_Path_FinQA dataset
"""

import os
import pandas as pd
from datasets import Dataset, load_dataset
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinQADataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, split: str = "train", cache_dir: Optional[str] = None) -> Dataset:
        """
        Load the TheFinAI/Fino1_Reasoning_Path_FinQA dataset
        
        Args:
            split: Dataset split to load ('train', 'test', 'validation')
            cache_dir: Directory to cache the dataset
            
        Returns:
            Dataset object
        """
        try:
            logger.info(f"Loading {split} split of TheFinAI/Fino1_Reasoning_Path_FinQA dataset...")
            
            # Load dataset from HuggingFace Hub
            dataset = load_dataset(
                "TheFinAI/Fino1_Reasoning_Path_FinQA",
                split=split,
                cache_dir=cache_dir or str(self.data_dir / "cache")
            )
            
            logger.info(f"Successfully loaded {len(dataset)} samples from {split} split")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def preprocess_for_training(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """
        Preprocess the dataset for fine-tuning
        
        Args:
            dataset: Raw dataset
            max_length: Maximum sequence length
            
        Returns:
            Preprocessed dataset
        """
        def format_sample(example):
            """Format each sample for instruction following"""
            
            # Extract fields - adjust based on actual dataset structure
            question = example.get('question', '')
            context = example.get('context', example.get('passage', ''))
            reasoning = example.get('reasoning_path', example.get('rationale', ''))
            answer = example.get('answer', '')
            
            # Create instruction-following format
            instruction = f"Given the following financial context, answer the question step by step.\n\nContext: {context}\n\nQuestion: {question}"
            
            if reasoning:
                response = f"Reasoning: {reasoning}\n\nAnswer: {answer}"
            else:
                response = f"Answer: {answer}"
            
            # Format for training (adjust based on your model's expected format)
            formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
            return {
                'text': formatted_text,
                'instruction': instruction,
                'response': response,
                'length': len(formatted_text)
            }
        
        logger.info("Preprocessing dataset for training...")
        processed_dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
        
        # Filter out samples that are too long
        processed_dataset = processed_dataset.filter(lambda x: x['length'] <= max_length)
        
        logger.info(f"Preprocessed dataset: {len(processed_dataset)} samples")
        return processed_dataset
    
    def create_chat_format(self, dataset: Dataset) -> Dataset:
        """
        Convert dataset to chat format for chat-based fine-tuning
        
        Args:
            dataset: Preprocessed dataset
            
        Returns:
            Dataset in chat format
        """
        def to_chat_format(example):
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example['response']}
            ]
            return {"messages": messages}
        
        logger.info("Converting to chat format...")
        chat_dataset = dataset.map(to_chat_format)
        return chat_dataset
    
    def save_processed_data(self, dataset: Dataset, filename: str):
        """Save processed dataset to disk"""
        output_path = self.data_dir / filename
        dataset.save_to_disk(str(output_path))
        logger.info(f"Saved processed dataset to {output_path}")
    
    def load_and_prepare_all_splits(self, max_length: int = 2048) -> Dict[str, Dataset]:
        """
        Load and prepare all dataset splits
        
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            try:
                # Load raw dataset
                raw_dataset = self.load_dataset(split=split_name)
                
                # Preprocess for training
                processed_dataset = self.preprocess_for_training(raw_dataset, max_length)
                
                # Convert to chat format
                chat_dataset = self.create_chat_format(processed_dataset)
                
                splits[split_name] = chat_dataset
                
                # Save processed data
                self.save_processed_data(chat_dataset, f"{split_name}_processed")
                
            except Exception as e:
                logger.warning(f"Could not load {split_name} split: {str(e)}")
                continue
        
        return splits

def main():
    """Main function to test data loading"""
    loader = FinQADataLoader()
    
    # Load and prepare all splits
    datasets = loader.load_and_prepare_all_splits(max_length=2048)
    
    # Print dataset info
    for split_name, dataset in datasets.items():
        print(f"\n{split_name.upper()} DATASET:")
        print(f"Number of samples: {len(dataset)}")
        if len(dataset) > 0:
            print("Sample data:")
            print(dataset[0])

if __name__ == "__main__":
    main()