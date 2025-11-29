#!/usr/bin/env python3
"""
Model evaluation script for comparing pre and post fine-tuning performance.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, load_environment_variables, ConfigManager, save_results
from src.model_loader import ModelLoader
from src.data_processor import DataProcessor
from src.inference import FinancialInference

def evaluate_model(
    model_path: str,
    config: dict,
    num_samples: int = 10,
    model_name: str = "model"
) -> dict:
    """
    Evaluate a model on sample financial Q&A data.
    
    Args:
        model_path: Path to the model
        config: Configuration dictionary
        num_samples: Number of samples to evaluate
        model_name: Name for the model (for logging)
        
    Returns:
        Evaluation results dictionary
    """
    logger.info(f"Evaluating {model_name} at: {model_path}")
    
    try:
        # Load model and tokenizer
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_and_tokenizer(
            model_path=model_path,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Setup generation config
        model_loader.setup_generation_config(model)
        
        # Load and process dataset for evaluation
        data_processor = DataProcessor(config, tokenizer)
        dataset = data_processor.load_dataset()
        
        # Validate dataset
        if not data_processor.validate_dataset(dataset):
            logger.error("Dataset validation failed")
            return {}
        
        # Take a subset for evaluation
        eval_samples = dataset.select(range(min(num_samples, len(dataset))))
        samples_list = [eval_samples[i] for i in range(len(eval_samples))]
        
        # Create inference engine
        inference = FinancialInference(model, tokenizer, config)
        
        # Get model info
        model_info = inference.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Evaluate on samples
        evaluation_results = inference.evaluate_on_samples(samples_list)
        
        # Add model information to results
        evaluation_results["model_info"] = model_info
        evaluation_results["model_name"] = model_name
        evaluation_results["model_path"] = model_path
        evaluation_results["config"] = config
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_name}: {str(e)}")
        return {"error": str(e), "model_name": model_name}

def compare_models(
    base_model_path: str,
    finetuned_model_path: str,
    config: dict,
    num_samples: int = 10
) -> dict:
    """
    Compare base model and fine-tuned model performance.
    
    Args:
        base_model_path: Path to base model
        finetuned_model_path: Path to fine-tuned model
        config: Configuration dictionary
        num_samples: Number of samples to evaluate
        
    Returns:
        Comparison results dictionary
    """
    logger.info("Starting model comparison")
    
    # Evaluate base model
    logger.info("Evaluating base model...")
    base_results = evaluate_model(
        base_model_path, config, num_samples, "base_model"
    )
    
    # Evaluate fine-tuned model
    logger.info("Evaluating fine-tuned model...")
    finetuned_results = evaluate_model(
        finetuned_model_path, config, num_samples, "finetuned_model"
    )
    
    # Compare results
    comparison = {
        "base_model": base_results,
        "finetuned_model": finetuned_results,
        "comparison_summary": {}
    }
    
    # Calculate comparison metrics
    if "error" not in base_results and "error" not in finetuned_results:
        base_avg_time = base_results.get("average_time", 0)
        finetuned_avg_time = finetuned_results.get("average_time", 0)
        
        base_success_rate = base_results.get("successful_samples", 0) / base_results.get("total_samples", 1)
        finetuned_success_rate = finetuned_results.get("successful_samples", 0) / finetuned_results.get("total_samples", 1)
        
        comparison["comparison_summary"] = {
            "base_success_rate": base_success_rate,
            "finetuned_success_rate": finetuned_success_rate,
            "success_rate_improvement": finetuned_success_rate - base_success_rate,
            "base_avg_time": base_avg_time,
            "finetuned_avg_time": finetuned_avg_time,
            "time_difference": finetuned_avg_time - base_avg_time
        }
        
        logger.info(f"Success rate improvement: {comparison['comparison_summary']['success_rate_improvement']:.2%}")
        logger.info(f"Average time difference: {comparison['comparison_summary']['time_difference']:.2f}s")
    
    return comparison

def evaluate_single_question(
    model_path: str,
    config: dict,
    question: str,
    expected_response: str = None,
    reasoning_context: str = ""
) -> dict:
    """
    Evaluate model on a single question.
    
    Args:
        model_path: Path to the model
        config: Configuration dictionary
        question: Financial question to evaluate
        expected_response: Optional expected response
        reasoning_context: Optional reasoning context
        
    Returns:
        Evaluation results for the single question
    """
    logger.info("Evaluating single question")
    logger.info(f"Question: {question}")
    
    try:
        # Load model and tokenizer
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_and_tokenizer(
            model_path=model_path,
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Setup generation config
        model_loader.setup_generation_config(model)
        
        # Create inference engine
        inference = FinancialInference(model, tokenizer, config)
        
        # Generate response
        import time
        start_time = time.time()
        generated_response = inference.generate_response(question, reasoning_context)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        result = {
            "question": question,
            "generated_response": generated_response,
            "generation_time": generation_time,
            "reasoning_context": reasoning_context
        }
        
        if expected_response:
            result["expected_response"] = expected_response
            result["response_length_ratio"] = len(generated_response) / len(expected_response)
        
        logger.info(f"Generated response in {generation_time:.2f}s")
        logger.info(f"Response preview: {generated_response[:200]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to evaluate question: {str(e)}")
        return {"error": str(e), "question": question}

def print_evaluation_summary(results: dict):
    """Print a summary of evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if "error" in results:
        print(f"Evaluation failed: {results['error']}")
        return
    
    if "comparison_summary" in results:
        # Comparison results
        summary = results["comparison_summary"]
        print(f"Base Model Success Rate: {summary['base_success_rate']:.2%}")
        print(f"Fine-tuned Model Success Rate: {summary['finetuned_success_rate']:.2%}")
        print(f"Success Rate Improvement: {summary['success_rate_improvement']:.2%}")
        print(f"Base Model Avg Time: {summary['base_avg_time']:.2f}s")
        print(f"Fine-tuned Model Avg Time: {summary['finetuned_avg_time']:.2f}s")
        print(f"Time Difference: {summary['time_difference']:.2f}s")
    else:
        # Single model results
        print(f"Model: {results.get('model_name', 'Unknown')}")
        print(f"Total Samples: {results.get('total_samples', 0)}")
        print(f"Successful Samples: {results.get('successful_samples', 0)}")
        print(f"Failed Samples: {results.get('failed_samples', 0)}")
        print(f"Success Rate: {results.get('successful_samples', 0) / results.get('total_samples', 1):.2%}")
        print(f"Average Time per Sample: {results.get('average_time', 0):.2f}s")
        print(f"Total Evaluation Time: {results.get('total_time', 0):.2f}s")
    
    print("="*80)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma 3 model")
    parser.add_argument("--mode", choices=["single", "compare", "evaluate"], required=True,
                       help="Evaluation mode: single question, model comparison, or single model evaluation")
    parser.add_argument("--base-model", help="Path to base model (for comparison mode)")
    parser.add_argument("--finetuned-model", help="Path to fine-tuned model")
    parser.add_argument("--model-path", help="Path to model (for single evaluation)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--question", help="Single question to evaluate (for single mode)")
    parser.add_argument("--expected-response", help="Expected response (for single mode)")
    parser.add_argument("--reasoning-context", default="", help="Reasoning context (for single mode)")
    parser.add_argument("--save-results", action="store_true", help="Save results to file")
    parser.add_argument("--output-file", help="Output file name for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO", log_file="evaluation")
    
    # Load environment variables
    load_environment_variables()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_all_configs()
    
    logger.info(f"Starting evaluation in {args.mode} mode")
    
    results = {}
    
    if args.mode == "single":
        if not args.question:
            logger.error("Question is required for single mode")
            sys.exit(1)
        
        model_path = args.model_path or args.finetuned_model
        if not model_path:
            logger.error("Model path is required for single mode")
            sys.exit(1)
        
        results = evaluate_single_question(
            model_path=model_path,
            config=config,
            question=args.question,
            expected_response=args.expected_response,
            reasoning_context=args.reasoning_context
        )
        
    elif args.mode == "compare":
        if not args.base_model or not args.finetuned_model:
            logger.error("Both base-model and finetuned-model are required for comparison")
            sys.exit(1)
        
        results = compare_models(
            base_model_path=args.base_model,
            finetuned_model_path=args.finetuned_model,
            config=config,
            num_samples=args.num_samples
        )
        
    elif args.mode == "evaluate":
        model_path = args.model_path or args.finetuned_model
        if not model_path:
            logger.error("Model path is required for evaluation mode")
            sys.exit(1)
        
        results = evaluate_model(
            model_path=model_path,
            config=config,
            num_samples=args.num_samples,
            model_name="evaluated_model"
        )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results if requested
    if args.save_results:
        filename = args.output_file or f"evaluation_results_{args.mode}"
        save_results(results, filename)
        logger.info(f"Results saved to file: {filename}")
    
    logger.info("Evaluation completed")

if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    main()