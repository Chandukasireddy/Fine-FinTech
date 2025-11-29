"""
Data processing utilities for financial Q&A dataset formatting.
"""

import logging
from typing import Dict, List, Any, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles dataset loading, formatting, and processing for financial Q&A fine-tuning."""
    
    def __init__(self, config: Dict[str, Any], tokenizer: AutoTokenizer):
        """
        Initialize DataProcessor.
        
        Args:
            config: Configuration dictionary
            tokenizer: Tokenizer for text processing
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_config = config.get("dataset", {})
        self.prompting_config = config.get("prompting", {})
        
        # Define the training prompt style
        self.train_prompt_style = """
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

        # Define inference prompt style (without expected response)
        self.inference_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
"""

    def load_dataset(self) -> Dataset:
        """
        Load and process the financial Q&A dataset.
        
        Returns:
            Processed dataset ready for training
        """
        dataset_name = self.dataset_config.get("name", "TheFinAI/Fino1_Reasoning_Path_FinQA")
        split = self.dataset_config.get("split", "train[0:500]")
        cache_dir = self.dataset_config.get("cache_dir", "./data")
        trust_remote_code = self.dataset_config.get("trust_remote_code", True)
        
        logger.info(f"Loading dataset: {dataset_name}")
        logger.info(f"Split: {split}")
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code
            )
            
            logger.info(f"Dataset loaded successfully. Size: {len(dataset)}")
            logger.info(f"Dataset columns: {dataset.column_names}")
            
            # Apply formatting
            formatted_dataset = dataset.map(
                self._formatting_prompts_func,
                batched=True,
                desc="Formatting dataset"
            )
            
            logger.info("Dataset formatting completed")
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _formatting_prompts_func(self, examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Format dataset examples into training prompts.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Dictionary with formatted text column
        """
        inputs = examples["Open-ended Verifiable Question"]
        complex_cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        
        texts = []
        for question, cot, response in zip(inputs, complex_cots, outputs):
            # Ensure response ends with EOS token
            if not response.endswith(self.tokenizer.eos_token):
                response += self.tokenizer.eos_token
            
            # Format using training prompt style
            text = self.train_prompt_style.format(question, cot, response)
            texts.append(text)
        
        return {"text": texts}
    
    def create_data_collator(self) -> DataCollatorForLanguageModeling:
        """
        Create data collator for causal language modeling.
        
        Returns:
            DataCollatorForLanguageModeling instance
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        logger.info("Data collator created for causal language modeling")
        return data_collator
    
    def format_inference_prompt(self, question: str, reasoning: str = "") -> str:
        """
        Format a prompt for inference.
        
        Args:
            question: The financial question to ask
            reasoning: Optional reasoning context
            
        Returns:
            Formatted prompt string
        """
        return self.inference_prompt_style.format(question, reasoning) + self.tokenizer.eos_token
    
    def preprocess_for_inference(self, prompt: str, device: str = "cuda") -> Dict[str, Any]:
        """
        Preprocess prompt for model inference.
        
        Args:
            prompt: Formatted prompt string
            device: Device to move tensors to
            
        Returns:
            Tokenized inputs ready for model
        """
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            max_length=self.prompting_config.get("max_length", 2048),
            truncation=self.prompting_config.get("truncation", True),
            padding=self.prompting_config.get("padding", False)
        )
        
        # Move to device if specified
        if device:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def extract_response_from_generation(self, generated_text: str) -> str:
        """
        Extract the response part from generated text.
        
        Args:
            generated_text: Full generated text from model
            
        Returns:
            Extracted response portion
        """
        try:
            # Split on "### Response:" and take the second part
            parts = generated_text.split("### Response:")
            if len(parts) > 1:
                return parts[1].strip()
            else:
                return generated_text.strip()
        except Exception as e:
            logger.warning(f"Failed to extract response: {str(e)}")
            return generated_text.strip()
    
    def get_dataset_info(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            dataset: The loaded dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "size": len(dataset),
            "columns": dataset.column_names,
            "features": dataset.features,
        }
        
        # Sample some examples
        if len(dataset) > 0:
            sample_indices = [0, min(1, len(dataset)-1), min(2, len(dataset)-1)]
            samples = [dataset[i] for i in sample_indices if i < len(dataset)]
            info["samples"] = samples
        
        return info
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate that the dataset has required columns and format.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ["Open-ended Verifiable Question", "Complex_CoT", "Response"]
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check if dataset is not empty
        if len(dataset) == 0:
            logger.error("Dataset is empty")
            return False
        
        # Validate a few samples
        try:
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                for col in required_columns:
                    if not sample[col] or not isinstance(sample[col], str):
                        logger.error(f"Invalid data in column '{col}' at index {i}")
                        return False
        except Exception as e:
            logger.error(f"Error validating dataset: {str(e)}")
            return False
        
        logger.info("Dataset validation passed")
        return True