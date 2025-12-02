"""
Utility functions for loading model and tokenizer
"""

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name: str = "google/gemma-3-4b-it"):
    """
    Load model and tokenizer with HuggingFace authentication
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Load environment and authenticate
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file")
    
    login(hf_token)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        token=hf_token
    ).eval()
    
    return model, tokenizer

                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_length
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error with Ollama generation: {str(e)}")
            return ""
    
    def _generate_with_transformers(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate using transformers"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error with transformers generation: {str(e)}")
            return ""
    
    def evaluate_on_samples(self, test_samples: List[Dict[str, str]], output_file: str = "evaluation_results.json"):
        """Evaluate model on test samples"""
        logger.info("Starting evaluation on test samples...")
        
        results = []
        
        for i, sample in enumerate(test_samples):
            instruction = sample.get('instruction', sample.get('question', ''))
            expected_response = sample.get('response', sample.get('answer', ''))
            
            # Generate response
            generated_response = self.generate_response(instruction)
            
            # Store result
            result = {
                'sample_id': i,
                'instruction': instruction,
                'expected_response': expected_response,
                'generated_response': generated_response,
                'input_length': len(instruction),
                'output_length': len(generated_response)
            }
            
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_samples)} samples")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {output_file}")
        return results

class OllamaIntegration:
    """Integration utilities for Ollama"""
    
    @staticmethod
    def list_models() -> List[str]:
        """List available Ollama models"""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            logger.error(f"Error listing Ollama models: {str(e)}")
            return []
    
    @staticmethod
    def pull_model(model_name: str):
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            ollama.pull(model_name)
            logger.info(f"Successfully pulled {model_name}")
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            raise
    
    @staticmethod
    def create_model_from_modelfile(model_name: str, modelfile_path: str):
        """Create Ollama model from Modelfile"""
        try:
            logger.info(f"Creating Ollama model: {model_name}")
            with open(modelfile_path, 'r') as f:
                modelfile = f.read()
            
            ollama.create(model=model_name, modelfile=modelfile)
            logger.info(f"Successfully created model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error creating Ollama model: {str(e)}")
            raise

def load_test_samples(file_path: str) -> List[Dict[str, str]]:
    """Load test samples from file"""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
        
    except Exception as e:
        logger.error(f"Error loading test samples: {str(e)}")
        return []

def compare_models(model1_path: str, model2_ollama: str, test_samples: List[Dict], output_dir: str = "comparison_results"):
    """Compare two models on the same test set"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate first model (HuggingFace format)
    evaluator1 = ModelEvaluator(model_path=model1_path)
    results1 = evaluator1.evaluate_on_samples(test_samples, f"{output_dir}/model1_results.json")
    
    # Evaluate second model (Ollama)
    evaluator2 = ModelEvaluator(ollama_model=model2_ollama)
    results2 = evaluator2.evaluate_on_samples(test_samples, f"{output_dir}/model2_results.json")
    
    # Create comparison report
    comparison = {
        'model1_path': model1_path,
        'model2_ollama': model2_ollama,
        'total_samples': len(test_samples),
        'avg_response_length_model1': sum(r['output_length'] for r in results1) / len(results1),
        'avg_response_length_model2': sum(r['output_length'] for r in results2) / len(results2),
        'results': []
    }
    
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        comparison['results'].append({
            'sample_id': i,
            'instruction': r1['instruction'],
            'model1_response': r1['generated_response'],
            'model2_response': r2['generated_response'],
            'expected_response': r1['expected_response']
        })
    
    with open(f"{output_dir}/comparison_report.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison completed. Report saved to {output_dir}/comparison_report.json")
    return comparison

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # List available Ollama models
    print("Available Ollama models:")
    models = OllamaIntegration.list_models()
    for model in models:
        print(f"  - {model}")
    
    # Example evaluation (uncomment to use)
    # evaluator = ModelEvaluator(ollama_model="gemma3:4b")
    # response = evaluator.generate_response("What is the current ratio in financial analysis?")
    # print(f"Response: {response}")