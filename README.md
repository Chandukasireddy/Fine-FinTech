# Fine-FinTech: Gemma3 Fine-tuning for Financial Q&A

A comprehensive repository for fine-tuning Ollama's Gemma3 model using the TheFinAI/Fino1_Reasoning_Path_FinQA dataset to create a specialized financial reasoning model.

## 🎯 Project Overview

This project enables you to fine-tune your locally installed Gemma3 model (via Ollama) to better understand and respond to financial questions with step-by-step reasoning. The model is trained on the FinQA dataset which contains financial question-answering pairs with detailed reasoning paths.

## 📁 Project Structure

```
Fine-FinTech/
├── src/                     # Source code
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── finetune.py         # Main fine-tuning script
│   └── utils.py            # Model conversion and evaluation utilities
├── config/                  # Configuration files
│   └── training_config.yaml # Training parameters
├── scripts/                 # Utility scripts
│   ├── setup.py            # Environment setup
│   └── evaluate.py         # Model evaluation
├── data/                    # Dataset storage (auto-created)
├── models/                  # Model storage (auto-created)
├── outputs/                 # Training outputs (auto-created)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** with pip
- **NVIDIA GPU** with 8GB+ VRAM (recommended)
- **Ollama** installed with Gemma3 model
- **Git** for cloning repositories

### 2. Install Ollama and Gemma3 (if not already done)

```bash
# Install Ollama (if not installed)
# Visit: https://ollama.ai/

# Pull Gemma3 model
ollama pull gemma3:4b
# or for larger model:
# ollama pull gemma3:9b
```

### 3. Setup Environment

```bash
# Clone or navigate to your project directory
cd Fine-FinTech

# Run setup script
python scripts/setup.py
```

This will:
- Check Python version compatibility
- Install all required dependencies
- Verify GPU availability
- Check Ollama installation
- Create necessary directories
- Optionally setup Weights & Biases

### 4. Configure Training

Edit `config/training_config.yaml` to customize:

```yaml
# Key settings to adjust:
model_name: "google/gemma-2-9b-it"  # Base model for fine-tuning
ollama_model_name: "gemma3:4b"      # Your Ollama model
num_train_epochs: 3                 # Training epochs
per_device_train_batch_size: 2      # Batch size (adjust for your GPU)
learning_rate: 2e-4                 # Learning rate
max_seq_length: 2048                # Maximum sequence length
```

### 5. Start Fine-tuning

```bash
# Start training with default config
python src/finetune.py

# Or with custom config
python src/finetune.py --config config/training_config.yaml
```

### 6. Monitor Training

Training progress will be displayed in the terminal. If you enabled Weights & Biases, you can also monitor online.

**Expected training time:**
- **4B model**: 2-4 hours on RTX 4090
- **9B model**: 6-8 hours on RTX 4090

## 📊 Evaluation

### Quick Evaluation with Sample Questions

```bash
python scripts/evaluate.py --ollama_model gemma3:4b --sample_test
```

### Evaluate Fine-tuned Model

```bash
# Evaluate HuggingFace format model
python scripts/evaluate.py --model_path ./outputs --sample_test

# Compare with original Ollama model
python scripts/evaluate.py --ollama_model gemma3:4b --sample_test
```

### Custom Test Set

```bash
python scripts/evaluate.py --model_path ./outputs --test_file path/to/test.json
```

## 🔧 Advanced Usage

### Custom Dataset

To use your own financial dataset:

1. Modify `src/data_loader.py` to load your data
2. Ensure your data has these fields:
   - `question`: The financial question
   - `context`: Relevant financial context/data
   - `reasoning_path`: Step-by-step reasoning (optional)
   - `answer`: Final answer

### Memory Optimization

For GPUs with less memory, adjust these settings in `config/training_config.yaml`:

```yaml
# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Increase to maintain effective batch size

# Enable gradient checkpointing
gradient_checkpointing: true

# Use 4-bit quantization
use_4bit: true
```

### Converting to Ollama Format

After training, convert your model for use with Ollama:

```python
from src.utils import ModelConverter

converter = ModelConverter("./outputs")
converter.convert_to_ollama_format("gemma3-finqa")

# Then in terminal:
# ollama create gemma3-finqa -f ./outputs/ollama_export/Modelfile
```

## 📈 Training Configuration Details

### LoRA (Low-Rank Adaptation)

The project uses LoRA for efficient fine-tuning:

```yaml
lora_r: 16          # Rank (higher = more parameters but better adaptation)
lora_alpha: 32      # Scaling factor
lora_dropout: 0.1   # Dropout rate
lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Quantization

4-bit quantization is used to reduce memory usage:

```yaml
use_4bit: true
bnb_4bit_compute_dtype: "float16"
bnb_4bit_use_double_quant: true
bnb_4bit_quant_type: "nf4"
```

## 🎯 Dataset: TheFinAI/Fino1_Reasoning_Path_FinQA

This dataset contains:
- **Financial Questions**: Real-world financial analysis questions
- **Context**: Relevant financial data and information
- **Reasoning Paths**: Step-by-step solutions
- **Answers**: Final numerical or textual answers

**Sample Data Format:**
```json
{
  "question": "What is the current ratio?",
  "context": "Current assets: $50,000, Current liabilities: $25,000",
  "reasoning_path": "Current ratio = Current assets / Current liabilities = $50,000 / $25,000 = 2.0",
  "answer": "2.0"
}
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing

2. **Slow Training**
   - Ensure CUDA is properly installed
   - Check GPU utilization with `nvidia-smi`
   - Consider using mixed precision training

3. **Dataset Loading Errors**
   - Check internet connection for HuggingFace downloads
   - Verify dataset name and format
   - Check available disk space

4. **Ollama Integration Issues**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is available: `ollama list`
   - Check Ollama version compatibility

### Performance Tips

- **Use SSD storage** for faster data loading
- **Monitor GPU memory** usage during training
- **Save checkpoints frequently** for long training runs
- **Use evaluation dataset** to monitor overfitting

## 📚 Additional Resources

- [Ollama Documentation](https://ollama.ai/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Gemma Model Card](https://huggingface.co/google/gemma-2-9b-it)
- [FinQA Dataset](https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA)

## 🤝 Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source. Please check individual model and dataset licenses for specific terms.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs in `./outputs/`
3. Open an issue with detailed error information
4. Include your system specifications and configuration

## 🎉 What's Next?

After successful fine-tuning:

1. **Test your model** with various financial questions
2. **Compare performance** with the original model
3. **Share your results** with the community
4. **Experiment with different configurations**
5. **Try on your own financial datasets**

Happy fine-tuning! 🚀💰