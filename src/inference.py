"""
Inference utilities for financial Q&A with Gemma 3 models.
"""

import logging
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from transformers import (
    Gemma3ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig
)

logger = logging.getLogger(__name__)

class FinancialInference:
    """Handles inference with fine-tuned Gemma 3 models for financial Q&A."""
    
    def __init__(
        self,
        model: Union[Gemma3ForConditionalGeneration, str],
        tokenizer: Union[AutoTokenizer, str],
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        """
        Initialize FinancialInference.
        
        Args:
            model: Pre-loaded model or path to model
            tokenizer: Pre-loaded tokenizer or path to tokenizer
            config: Optional configuration dictionary
            device: Device to run inference on
        """
        self.config = config or {}
        self.generation_config_dict = self.config.get("generation", {})
        self.device = device
        
        # Load model and tokenizer if paths are provided
        if isinstance(model, str):
            logger.info(f"Loading model from: {model}")
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            self.model = model
        
        if isinstance(tokenizer, str):
            logger.info(f"Loading tokenizer from: {tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                trust_remote_code=True
            )
        else:
            self.tokenizer = tokenizer
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Setup generation config
        self._setup_generation_config()
        
        # Define prompt templates
        self._setup_prompt_templates()
        
        logger.info("FinancialInference initialized successfully")
    
    def _setup_generation_config(self):
        """Setup generation configuration for the model."""
        self.generation_config = GenerationConfig(
            max_new_tokens=self.generation_config_dict.get("max_new_tokens", 1200),
            temperature=self.generation_config_dict.get("temperature", 0.7),
            top_p=self.generation_config_dict.get("top_p", 0.9),
            do_sample=self.generation_config_dict.get("do_sample", True),
            use_cache=self.generation_config_dict.get("use_cache", True),
            pad_token_id=self.model.config.eos_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )
        
        self.model.generation_config = self.generation_config
    
    def _setup_prompt_templates(self):
        """Setup prompt templates for different types of inference."""
        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
"""
    
    def generate_response(
        self,
        question: str,
        reasoning_context: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Generate a response for a financial question.
        
        Args:
            question: The financial question to answer
            reasoning_context: Optional reasoning context to provide
            max_new_tokens: Override max tokens for this generation
            temperature: Override temperature for this generation  
            top_p: Override top_p for this generation
            
        Returns:
            Generated response string
        """
        logger.info(f"Generating response for question: {question[:100]}...")
        
        # Format the prompt
        prompt = self.prompt_template.format(question, reasoning_context)
        if not prompt.endswith(self.tokenizer.eos_token):
            prompt += self.tokenizer.eos_token
        
        # Tokenize input
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=False
        )
        
        # Move to device
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = next(self.model.parameters()).device
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Setup generation parameters
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Override generation config if specified
        if max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generation_kwargs["max_new_tokens"] = self.generation_config.max_new_tokens
            
        if temperature is not None:
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["temperature"] = self.generation_config.temperature
            
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["top_p"] = self.generation_config.top_p
        
        generation_kwargs["do_sample"] = self.generation_config.do_sample
        generation_kwargs["use_cache"] = self.generation_config.use_cache
        
        try:
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)
            
            # Decode the generated text
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract just the response part
            response = self._extract_response_part(response)
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise
    
    def _extract_response_part(self, full_response: str) -> str:
        """
        Extract the response part from the full generated text.
        
        Args:
            full_response: Full generated text
            
        Returns:
            Extracted response part
        """
        try:
            # Split on "### Response:" and take the part after it
            parts = full_response.split("### Response:")
            if len(parts) > 1:
                return parts[1].strip()
            else:
                return full_response.strip()
        except Exception as e:
            logger.warning(f"Failed to extract response part: {str(e)}")
            return full_response.strip()
    
    def batch_generate(
        self,
        questions: List[str],
        reasoning_contexts: Optional[List[str]] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of questions.
        
        Args:
            questions: List of financial questions
            reasoning_contexts: Optional list of reasoning contexts
            **generation_kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        if reasoning_contexts is None:
            reasoning_contexts = [""] * len(questions)
        
        if len(questions) != len(reasoning_contexts):
            raise ValueError("Questions and reasoning contexts must have same length")
        
        logger.info(f"Generating responses for {len(questions)} questions")
        
        responses = []
        for question, reasoning in zip(questions, reasoning_contexts):
            try:
                response = self.generate_response(question, reasoning, **generation_kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate response for question: {question[:50]}...")
                responses.append(f"Error: {str(e)}")
        
        return responses
    
    def compare_responses(
        self,
        question: str,
        expected_response: str,
        generated_response: Optional[str] = None,
        reasoning_context: str = ""
    ) -> Dict[str, Any]:
        """
        Compare generated response with expected response.
        
        Args:
            question: The financial question
            expected_response: Expected response
            generated_response: Generated response (will generate if None)
            reasoning_context: Reasoning context for generation
            
        Returns:
            Comparison results dictionary
        """
        if generated_response is None:
            generated_response = self.generate_response(question, reasoning_context)
        
        comparison = {
            "question": question,
            "expected_response": expected_response,
            "generated_response": generated_response,
            "expected_length": len(expected_response),
            "generated_length": len(generated_response),
            "length_ratio": len(generated_response) / len(expected_response) if expected_response else 0
        }
        
        return comparison
    
    def evaluate_on_samples(
        self,
        samples: List[Dict[str, str]],
        response_key: str = "Response",
        question_key: str = "Open-ended Verifiable Question",
        reasoning_key: str = "Complex_CoT"
    ) -> Dict[str, Any]:
        """
        Evaluate model on a set of sample questions.
        
        Args:
            samples: List of sample dictionaries
            response_key: Key for expected response
            question_key: Key for question
            reasoning_key: Key for reasoning context
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {len(samples)} samples")
        
        results = []
        total_time = 0
        
        for i, sample in enumerate(samples):
            try:
                question = sample[question_key]
                expected_response = sample[response_key]
                reasoning_context = sample.get(reasoning_key, "")
                
                import time
                start_time = time.time()
                
                generated_response = self.generate_response(question, reasoning_context)
                
                end_time = time.time()
                generation_time = end_time - start_time
                total_time += generation_time
                
                comparison = self.compare_responses(
                    question, expected_response, generated_response, reasoning_context
                )
                comparison["generation_time"] = generation_time
                comparison["sample_index"] = i
                
                results.append(comparison)
                
                logger.info(f"Sample {i+1}/{len(samples)} completed in {generation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to evaluate sample {i}: {str(e)}")
                results.append({
                    "sample_index": i,
                    "error": str(e),
                    "question": sample.get(question_key, ""),
                })
        
        evaluation_summary = {
            "total_samples": len(samples),
            "successful_samples": len([r for r in results if "error" not in r]),
            "failed_samples": len([r for r in results if "error" in r]),
            "total_time": total_time,
            "average_time": total_time / len(samples) if samples else 0,
            "results": results
        }
        
        return evaluation_summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get device info
        device = next(self.model.parameters()).device
        
        info = {
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "dtype": str(next(self.model.parameters()).dtype),
            "generation_config": self.generation_config.to_dict() if hasattr(self.generation_config, 'to_dict') else str(self.generation_config)
        }
        
        return info