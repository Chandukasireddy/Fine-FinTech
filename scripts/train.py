#!/usr/bin/env python3
"""
Main training script for fine-tuning Gemma 3 on financial Q&A dataset.
This script orchestrates the entire fine-tuning process.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import setup_logging, load_environment_variables, ConfigManager, get_device_info, clear_gpu_memory
from src.model_loader import ModelLoader
from src.data_processor import DataProcessor
from src.trainer import FinancialTrainer
from src.inference import FinancialInference

def pre_training_checks(config: dict) -> bool:
    """
    Perform pre-training checks to ensure everything is set up correctly.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all checks pass, False otherwise
    """
    logger.info("Performing pre-training checks...")
    
    # Check CUDA availability
    device_info = get_device_info()
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    
    if device_info["cuda_available"]:
        logger.info(f"Available GPUs: {device_info['cuda_device_count']}")
        for device in device_info["devices"]:
            logger.info(f"GPU {device['id']}: {device['name']} ({device['memory_total_gb']:.1f}GB)")
    else:
        logger.warning("CUDA not available. Training will be very slow on CPU!")
    
    # Check Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token or hf_token == "your_hugging_face_token_here":
        logger.warning("HUGGINGFACE_TOKEN not set. May have issues accessing models.")
    else:
        logger.info("Hugging Face token found")
    
    # Validate config
    from src.utils import validate_config
    if not validate_config(config):
        logger.error("Configuration validation failed")
        return False
    
    logger.info("Pre-training checks completed successfully")
    return True

def run_pre_training_inference(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    data_processor: DataProcessor,
    config: dict,
    num_samples: int = 3
) -> dict:
    """
    Run inference before training to establish baseline performance.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        dataset: Processed dataset
        data_processor: Data processor instance
        config: Configuration dictionary
        num_samples: Number of samples to test
        
    Returns:
        Pre-training inference results
    """
    logger.info("Running pre-training inference for baseline...")
    
    try:
        # Create inference engine
        inference = FinancialInference(model, tokenizer, config)
        
        # Get sample questions
        samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
        
        results = []
        for i, sample in enumerate(samples):
            question = sample["Open-ended Verifiable Question"]
            expected_cot = sample["Complex_CoT"]
            expected_response = sample["Response"]
            
            logger.info(f"Pre-training sample {i+1}/{len(samples)}")
            
            # Generate response
            generated_response = inference.generate_response(question, "")
            
            result = {
                "sample_index": i,
                "question": question,
                "expected_cot": expected_cot,
                "expected_response": expected_response,
                "generated_response": generated_response
            }
            results.append(result)
            
            logger.info(f"Generated: {generated_response[:100]}...")
        
        return {"pre_training_results": results}
        
    except Exception as e:
        logger.error(f"Pre-training inference failed: {str(e)}")
        return {"error": str(e)}

def run_post_training_inference(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    data_processor: DataProcessor,
    config: dict,
    pre_training_results: dict,
    num_samples: int = 3
) -> dict:
    """
    Run inference after training and compare with pre-training results.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        dataset: Processed dataset
        data_processor: Data processor instance
        config: Configuration dictionary
        pre_training_results: Results from pre-training inference
        num_samples: Number of samples to test
        
    Returns:
        Post-training inference results with comparison
    """
    logger.info("Running post-training inference for comparison...")
    
    try:
        # Create inference engine
        inference = FinancialInference(model, tokenizer, config)
        
        # Use same samples as pre-training
        pre_results = pre_training_results.get("pre_training_results", [])
        
        results = []
        for i, pre_result in enumerate(pre_results[:num_samples]):
            question = pre_result["question"]
            expected_response = pre_result["expected_response"]
            pre_generated = pre_result["generated_response"]
            
            logger.info(f"Post-training sample {i+1}/{len(pre_results[:num_samples])}")
            
            # Generate response with fine-tuned model
            generated_response = inference.generate_response(question, "")
            
            result = {
                "sample_index": i,
                "question": question,
                "expected_response": expected_response,
                "pre_training_response": pre_generated,
                "post_training_response": generated_response,
                "improvement_visible": len(generated_response) > len(pre_generated)
            }
            results.append(result)
            
            logger.info(f"Pre-training: {pre_generated[:100]}...")
            logger.info(f"Post-training: {generated_response[:100]}...")
        
        return {"post_training_results": results}
        
    except Exception as e:
        logger.error(f"Post-training inference failed: {str(e)}")
        return {"error": str(e)}

def main_training_pipeline(
    config: dict,
    save_model: bool = True,
    run_inference: bool = True,
    output_dir: str = "./output"
) -> dict:
    """
    Execute the complete training pipeline.
    
    Args:
        config: Configuration dictionary
        save_model: Whether to save the trained model
        run_inference: Whether to run pre/post training inference
        output_dir: Directory to save outputs
        
    Returns:
        Complete training results
    """
    results = {"pipeline_status": "started"}
    
    try:
        # Clear GPU memory
        clear_gpu_memory()
        
        # Step 1: Load model and tokenizer
        logger.info("Step 1/7: Loading model and tokenizer...")
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_model_and_tokenizer(
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Setup generation config
        model_loader.setup_generation_config(model)
        
        # Get model info
        model_info = model_loader.get_model_info(model)
        results["model_info"] = model_info
        logger.info(f"Model loaded: {model_info['total_parameters']:,} parameters")
        
        # Step 2: Load and process dataset
        logger.info("Step 2/7: Loading and processing dataset...")
        data_processor = DataProcessor(config, tokenizer)
        dataset = data_processor.load_dataset()
        
        # Validate dataset
        if not data_processor.validate_dataset(dataset):
            raise ValueError("Dataset validation failed")
        
        dataset_info = data_processor.get_dataset_info(dataset)
        results["dataset_info"] = dataset_info
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Create data collator
        data_collator = data_processor.create_data_collator()
        
        # Step 3: Run pre-training inference (if requested)
        pre_training_results = {}
        if run_inference:
            logger.info("Step 3/7: Running pre-training inference...")
            pre_training_results = run_pre_training_inference(
                model, tokenizer, dataset, data_processor, config
            )
            results["pre_training_inference"] = pre_training_results
        else:
            logger.info("Step 3/7: Skipping pre-training inference...")
        
        # Step 4: Setup trainer
        logger.info("Step 4/7: Setting up trainer...")
        trainer = FinancialTrainer(model, tokenizer, config)
        
        # Setup LoRA and training arguments
        trainer.setup_lora_config()
        trainer.setup_training_arguments()
        trainer.setup_trainer(dataset, data_collator)
        
        # Step 5: Execute training
        logger.info("Step 5/7: Starting training...")
        training_stats = trainer.train(dataset, data_collator)
        results["training_stats"] = training_stats
        logger.info("Training completed successfully!")
        
        # Step 6: Run post-training inference (if requested)
        if run_inference:
            logger.info("Step 6/7: Running post-training inference...")
            post_training_results = run_post_training_inference(
                model, tokenizer, dataset, data_processor, config, pre_training_results
            )
            results["post_training_inference"] = post_training_results
        else:
            logger.info("Step 6/7: Skipping post-training inference...")
        
        # Step 7: Save model (if requested)
        if save_model:
            logger.info("Step 7/7: Saving trained model...")
            
            # Determine save path
            save_path = Path(output_dir) / "fine_tuned_model"
            save_path.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(save_path))
            results["model_save_path"] = str(save_path)
            logger.info(f"Model saved to: {save_path}")
        else:
            logger.info("Step 7/7: Skipping model save...")
        
        results["pipeline_status"] = "completed"
        logger.info("Training pipeline completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        results["pipeline_status"] = "failed"
        results["error"] = str(e)
        return results

def print_training_summary(results: dict):
    """Print a summary of the training results."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    status = results.get("pipeline_status", "unknown")
    print(f"Pipeline Status: {status.upper()}")
    
    if status == "failed":
        print(f"Error: {results.get('error', 'Unknown error')}")
        return
    
    # Model info
    model_info = results.get("model_info", {})
    if model_info:
        print(f"Model Parameters: {model_info.get('total_parameters', 0):,}")
        print(f"Trainable Parameters: {model_info.get('trainable_parameters', 0):,}")
    
    # Dataset info
    dataset_info = results.get("dataset_info", {})
    if dataset_info:
        print(f"Training Samples: {dataset_info.get('size', 0)}")
    
    # Training stats
    training_stats = results.get("training_stats")
    if training_stats and hasattr(training_stats, 'training_loss'):
        print(f"Final Training Loss: {training_stats.training_loss:.4f}")
    
    # Model save path
    save_path = results.get("model_save_path")
    if save_path:
        print(f"Model Saved To: {save_path}")
    
    # Inference comparison
    pre_results = results.get("pre_training_inference", {}).get("pre_training_results", [])
    post_results = results.get("post_training_inference", {}).get("post_training_results", [])
    
    if pre_results and post_results:
        print(f"\nInference Comparison ({len(post_results)} samples):")
        for i, result in enumerate(post_results):
            print(f"Sample {i+1}:")
            print(f"  Pre-training response length: {len(result.get('pre_training_response', ''))}")
            print(f"  Post-training response length: {len(result.get('post_training_response', ''))}")
    
    print("="*80)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 on financial Q&A dataset")
    parser.add_argument("--config-dir", default="./config", help="Configuration directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save the trained model")
    parser.add_argument("--no-inference", action="store_true", help="Skip pre/post training inference")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level, log_file="training")
    
    # Load environment variables
    load_environment_variables()
    
    # Load configuration
    config_manager = ConfigManager(args.config_dir)
    config = config_manager.load_all_configs()
    
    if not config:
        logger.error("No configuration found. Please ensure config files exist.")
        sys.exit(1)
    
    logger.info("Starting Gemma 3 fine-tuning pipeline")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Perform pre-training checks
    if not pre_training_checks(config):
        logger.error("Pre-training checks failed")
        sys.exit(1)
    
    # Run training pipeline
    results = main_training_pipeline(
        config=config,
        save_model=not args.no_save,
        run_inference=not args.no_inference,
        output_dir=args.output_dir
    )
    
    # Print summary
    print_training_summary(results)
    
    # Save results
    from src.utils import save_results
    save_results(results, "training_results", args.output_dir)
    
    if results.get("pipeline_status") == "completed":
        logger.info("Fine-tuning completed successfully!")
        print("\nNext steps:")
        print("1. Evaluate your model: python scripts/evaluate.py --mode evaluate --model-path ./output/fine_tuned_model")
        print("2. Deploy to Hugging Face: python scripts/deploy.py --model-path ./output/fine_tuned_model --hub-name your-username/model-name")
    else:
        logger.error("Fine-tuning failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    main()