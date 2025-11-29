# Fine-Tune Gemma 3: Financial Q&A Dataset

A comprehensive guide and implementation for fine-tuning Google's Gemma 3 model on financial reasoning datasets to improve accuracy in financial question-answering tasks.

## Overview

This project demonstrates how to fine-tune the Gemma 3-4B-IT model using the TheFinAI/Fino1_Reasoning_Path_FinQA dataset to enhance the model's ability to understand complex financial questions and provide precise, contextually relevant answers with step-by-step reasoning.

## Features

- 🎯 **Fine-tuning Gemma 3** on financial reasoning data
- 📊 **Enhanced reasoning capabilities** for financial Q&A
- 🔧 **LoRA (Low-Rank Adaptation)** for efficient training
- 💡 **Step-by-step reasoning** with chain-of-thought prompting
- 📈 **Performance comparison** before and after fine-tuning
- 🚀 **Model deployment** to Hugging Face Hub

## Project Structure

```
Fine-FinTech/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── config/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── lora_config.yaml
├── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── data_processor.py
│   ├── trainer.py
│   ├── inference.py
│   └── utils.py
├── scripts/
│   ├── setup_environment.py
│   ├── train.py
│   ├── evaluate.py
│   └── deploy.py
├── notebooks/
│   └── gemma3_finetuning_demo.ipynb
├── data/
│   └── (dataset cache)
├── models/
│   └── (saved models)
└── output/
    └── (training outputs)
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/Fine-FinTech.git
cd Fine-FinTech
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your Hugging Face token
```

5. **Run setup script**:
```bash
python scripts/setup_environment.py
```

## Quick Start

### 1. Environment Setup
```python
python scripts/setup_environment.py
```

### 2. Run Training
```python
python scripts/train.py
```

### 3. Evaluate Model
```python
python scripts/evaluate.py
```

### 4. Deploy Model
```python
python scripts/deploy.py
```

## Usage

### Training the Model

```python
from src.trainer import FinancialTrainer

# Initialize trainer
trainer = FinancialTrainer(
    model_path="google/gemma-3-4b-it",
    dataset_name="TheFinAI/Fino1_Reasoning_Path_FinQA",
    output_dir="./output"
)

# Start training
trainer.train()
```

### Running Inference

```python
from src.inference import FinancialInference

# Load fine-tuned model
inference = FinancialInference("./models/gemma-3-4b-fin-qa")

# Ask a financial question
question = "What portion of the estimated amortization expense will be recognized in 2017?"
response = inference.generate_response(question)
print(response)
```

## Configuration

### Model Configuration (config/model_config.yaml)
```yaml
model:
  name: "google/gemma-3-4b-it"
  device_map: "auto"
  attention_implementation: "eager"
  torch_dtype: "float16"
```

### Training Configuration (config/training_config.yaml)
```yaml
training:
  output_dir: "./output"
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  num_train_epochs: 1
  learning_rate: 2e-4
  warmup_steps: 10
  logging_steps: 0.2
```

### LoRA Configuration (config/lora_config.yaml)
```yaml
lora:
  r: 64
  lora_alpha: 16
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
```

## Dataset

The project uses the **TheFinAI/Fino1_Reasoning_Path_FinQA** dataset, which is a financial reasoning dataset based on FinQA, enhanced with GPT-4o generated reasoning paths for structured financial question answering.

### Dataset Features:
- Financial Q&A pairs with reasoning paths
- Complex chain-of-thought annotations
- Structured financial data comprehension
- Multi-step reasoning requirements

## Model Performance

### Before Fine-tuning
- Basic financial understanding
- Limited reasoning capability
- Short, often inaccurate responses

### After Fine-tuning
- Enhanced financial reasoning
- Detailed step-by-step explanations
- Improved accuracy in financial calculations
- Better understanding of financial contexts

## Hardware Requirements

### Minimum Requirements:
- GPU: 12GB+ VRAM (RTX 3080/4070 or better)
- RAM: 16GB+ system memory
- Storage: 50GB+ available space

### Recommended Setup:
- GPU: 24GB+ VRAM (RTX 4090, A6000, or better)
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD space

## Training Tips

1. **Memory Optimization**:
   - Use gradient checkpointing
   - Enable mixed precision training
   - Adjust batch size based on available VRAM

2. **Performance Tuning**:
   - Experiment with different LoRA ranks
   - Adjust learning rate schedules
   - Monitor training loss convergence

3. **Data Preparation**:
   - Ensure consistent prompt formatting
   - Validate dataset quality
   - Consider data augmentation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google AI for the Gemma 3 model
- TheFinAI team for the financial reasoning dataset
- Hugging Face for the transformers library
- The open-source ML community

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{gemma3-financial-finetuning,
  title={Fine-Tune Gemma 3: A Step-by-Step Guide With Financial Q&A Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/Fine-FinTech}
}
```

## Support

For questions and support, please:
1. Check the [Issues](../../issues) page
2. Review the [Documentation](docs/)
3. Contact the maintainers

---

**Note**: This project is for educational and research purposes. Ensure compliance with model licenses and terms of service when using in production.