#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned model
"""

import json
import argparse
from pathlib import Path
import sys
sys.path.append('src')

from utils import ModelEvaluator, load_test_samples

def create_sample_questions():
    """Create sample financial questions for testing"""
    return [
        {
            "instruction": "Given the following financial context, answer the question step by step.\n\nContext: A company has current assets of $50,000 and current liabilities of $25,000.\n\nQuestion: What is the current ratio and what does it indicate about the company's liquidity?",
            "response": "Reasoning: The current ratio is calculated by dividing current assets by current liabilities.\n\nAnswer: Current ratio = $50,000 ÷ $25,000 = 2.0. This indicates strong liquidity as the company has twice as many current assets as current liabilities."
        },
        {
            "instruction": "Given the following financial context, answer the question step by step.\n\nContext: A stock has a beta of 1.5, the risk-free rate is 3%, and the market return is 10%.\n\nQuestion: Calculate the expected return using CAPM.",
            "response": "Reasoning: CAPM formula is: Expected Return = Risk-free rate + Beta × (Market return - Risk-free rate)\n\nAnswer: Expected Return = 3% + 1.5 × (10% - 3%) = 3% + 1.5 × 7% = 3% + 10.5% = 13.5%"
        },
        {
            "instruction": "Given the following financial context, answer the question step by step.\n\nContext: A bond has a face value of $1,000, coupon rate of 5%, and trades at $950.\n\nQuestion: What is the current yield of this bond?",
            "response": "Reasoning: Current yield is calculated by dividing annual coupon payment by current market price.\n\nAnswer: Annual coupon = $1,000 × 5% = $50. Current yield = $50 ÷ $950 = 0.0526 = 5.26%"
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, help="Path to fine-tuned model")
    parser.add_argument("--ollama_model", type=str, help="Ollama model name")
    parser.add_argument("--test_file", type=str, help="Path to test samples JSON file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    parser.add_argument("--sample_test", action="store_true", help="Run with sample questions")
    
    args = parser.parse_args()
    
    if not args.model_path and not args.ollama_model:
        print("Please provide either --model_path or --ollama_model")
        sys.exit(1)
    
    # Initialize evaluator
    if args.model_path:
        evaluator = ModelEvaluator(model_path=args.model_path)
        print(f"Using HuggingFace model: {args.model_path}")
    else:
        evaluator = ModelEvaluator(ollama_model=args.ollama_model)
        print(f"Using Ollama model: {args.ollama_model}")
    
    # Load test samples
    if args.test_file and Path(args.test_file).exists():
        test_samples = load_test_samples(args.test_file)
        print(f"Loaded {len(test_samples)} test samples from {args.test_file}")
    elif args.sample_test:
        test_samples = create_sample_questions()
        print("Using sample financial questions")
    else:
        print("No test samples provided. Use --test_file or --sample_test")
        sys.exit(1)
    
    # Run evaluation
    results = evaluator.evaluate_on_samples(test_samples, args.output)
    
    # Print summary
    print(f"\n📊 Evaluation Summary:")
    print(f"Total samples: {len(results)}")
    print(f"Average input length: {sum(r['input_length'] for r in results) / len(results):.1f} chars")
    print(f"Average output length: {sum(r['output_length'] for r in results) / len(results):.1f} chars")
    print(f"Results saved to: {args.output}")
    
    # Show sample responses
    print(f"\n📝 Sample Responses:")
    for i, result in enumerate(results[:2]):  # Show first 2 samples
        print(f"\n--- Sample {i+1} ---")
        print(f"Question: {result['instruction'][:100]}...")
        print(f"Response: {result['generated_response'][:200]}...")

if __name__ == "__main__":
    main()