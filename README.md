# Fine-FinTech: Gemma-3 Fine-tuning for Financial Q&A

Fine-tune Google's Gemma-3-4B model on the TheFinAI/Fino1_Reasoning_Path_FinQA dataset for specialized financial reasoning.

## 📁 Project Structure

```
Fine-FinTech/
├── src/
│   ├── data_loader.py      # Dataset loading and formatting with prompt templates
│   ├── train.py            # Main training script with SFTTrainer
│   └── utils.py            # Model and tokenizer loading utilities
├── scripts/
│   └── hf_login.py         # HuggingFace authentication helper
├── config/
│   └── training_config.yaml # Training parameters (optional)
├── .env                     # HuggingFace token
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (or CPU for testing)
- HuggingFace account

### 2. Get HuggingFace Access

1. Accept Gemma-3 license: https://huggingface.co/google/gemma-3-4b-it
2. Get your token: https://huggingface.co/settings/tokens (create "Read" token)

### 3. Setup

```bash
# Clone and navigate to project
cd Fine-FinTech

# Create conda environment
conda create -n fine-fintech python=3.10 -y
conda activate fine-fintech

# Install dependencies from requirements.txt
pip install -r requirements.txt

# IMPORTANT: Reinstall transformers with Gemma-3 support
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# Add your HuggingFace token to .env
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### 4. Start Training

```bash
python src/train.py
```

That's it! The script will:
- Load the Gemma-3-4B model
- Download 500 samples from FinQA dataset
- Apply LoRA for efficient fine-tuning
- Train for 1 epoch
- Save the model to `output/final_model`

## 📊 What Gets Trained

**Model**: google/gemma-3-4b-it (4 billion parameters)
**Dataset**: TheFinAI/Fino1_Reasoning_Path_FinQA (500 samples)
**Method**: LoRA (Low-Rank Adaptation) - only trains ~0.1% of parameters
**Training Time**: ~30-60 minutes on modern GPU

### Training Configuration

```python
# LoRA Config
- Rank: 64
- Alpha: 16
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

# Training Args
- Batch size: 1 per device
- Gradient accumulation: 2 steps
- Learning rate: 2e-4
- Epochs: 1
- Optimizer: paged_adamw_32bit
```

## 🔍 Prompt Templates

The training uses special prompt formatting:

**Training Format** (with reasoning):
```
### Question:
{question}

### Response:
<think>
{reasoning}
</think>

{answer}
```

**Inference Format** (without reasoning tags):
```
### Question:
{question}

### Response:
{response}
```

## 📁 Key Files

- **src/utils.py**: Loads model and tokenizer with HF authentication
- **src/data_loader.py**: Contains prompt templates, dataset loading, and formatting functions
- **src/train.py**: Complete training pipeline with SFTTrainer
- **requirements.txt**: Minimal dependencies needed
- **.env**: Your HuggingFace token (keep private!)

## 🔧 Customization

### Use More Training Data

Edit `src/train.py`:
```python
dataset = load_and_format_dataset(tokenizer, num_samples=1000)  # Change 500 to 1000
```

### Adjust Training Epochs

Edit `src/train.py`:
```python
training_arguments = TrainingArguments(
    num_train_epochs=3,  # Change from 1 to 3
    ...
)
```

### Change Batch Size (for memory)

Edit `src/train.py`:
```python
training_arguments = TrainingArguments(
    per_device_train_batch_size=2,  # Increase if you have more GPU memory
    gradient_accumulation_steps=1,   # Decrease accordingly
    ...
)
```

## 🐛 Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size to 1
- Close other applications using GPU
- Use gradient checkpointing

**Invalid HuggingFace Token**:
- Get new token from https://huggingface.co/settings/tokens
- Update `.env` file with correct token
- Make sure you accepted Gemma-3 license

**Module Not Found or Import Errors**:
- Activate conda environment: `conda activate fine-fintech`
- Make sure transformers Gemma-3 branch is installed:
  ```bash
  pip uninstall transformers -y
  pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
  ```

**Version Compatibility Issues**:
- The project requires specific versions that work together
- Always reinstall transformers from Gemma-3 branch after installing requirements.txt
- Compatible versions: peft>=0.13.0, trl>=0.11.0

## 📈 After Training

Your fine-tuned model will be in `output/final_model/`. Use it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("output/final_model")
tokenizer = AutoTokenizer.from_pretrained("output/final_model")
```

## 📚 Resources

- [Gemma-3 Model](https://huggingface.co/google/gemma-3-4b-it)
- [FinQA Dataset](https://huggingface.co/datasets/TheFinAI/Fino1_Reasoning_Path_FinQA)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [TRL Documentation](https://huggingface.co/docs/trl)

## 📄 License

Open source. Check individual model and dataset licenses for terms.
