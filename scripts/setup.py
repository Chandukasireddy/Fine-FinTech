#!/usr/bin/env python3
"""
Setup script for the Fine-FinTech project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available - training will use CPU (slow)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def install_requirements():
    """Install Python requirements"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def check_ollama():
    """Check if Ollama is installed and running"""
    # Check if ollama command exists
    ollama_check = run_command("ollama --version", "Checking Ollama installation")
    
    if ollama_check:
        # List available models
        models_output = run_command("ollama list", "Listing Ollama models")
        if models_output:
            print("Available Ollama models:")
            print(models_output)
    else:
        print("⚠️  Ollama not found. Please install from: https://ollama.ai/")
        return False
    
    return True

def setup_wandb():
    """Setup Weights & Biases (optional)"""
    response = input("\nDo you want to set up Weights & Biases for experiment tracking? (y/n): ")
    if response.lower() == 'y':
        try:
            import wandb
            wandb.login()
            print("✅ Weights & Biases setup completed")
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
        except Exception as e:
            print(f"⚠️  W&B setup failed: {e}")

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "outputs", "models", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print("✅ Project directories created")

def main():
    """Main setup function"""
    print("🚀 Setting up Fine-FinTech project...")
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        # Check GPU availability
        check_gpu()
        
        # Check Ollama
        check_ollama()
        
        # Setup W&B (optional)
        setup_wandb()
        
        print("\n🎉 Setup completed!")
        print("\nNext steps:")
        print("1. Run: python src/data_loader.py  # Test data loading")
        print("2. Edit config/training_config.yaml if needed")
        print("3. Run: python src/finetune.py --config config/training_config.yaml")
    else:
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)

if __name__ == "__main__":
    main()