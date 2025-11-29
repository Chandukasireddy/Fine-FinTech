#!/usr/bin/env python3
"""
Environment setup script for Gemma 3 fine-tuning project.
This script handles dependency installation and environment configuration.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required. Current version: %s", sys.version)
        sys.exit(1)
    logger.info("Python version check passed: %s", sys.version)

def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found at %s", requirements_file)
        sys.exit(1)
    
    logger.info("Installing requirements from %s", requirements_file)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install requirements: %s", e)
        sys.exit(1)

def setup_environment_file():
    """Create .env file from template if it doesn't exist."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from template")
        env_file.write_text(env_example.read_text())
        logger.warning("Please edit .env file with your actual configuration values")
    elif env_file.exists():
        logger.info(".env file already exists")
    else:
        logger.warning("No .env.example file found to create .env from")

def verify_cuda_availability():
    """Check CUDA availability for GPU training."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info("CUDA available with %d GPU(s)", gpu_count)
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                logger.info("GPU %d: %s (%d GB)", i, gpu_name, gpu_memory)
        else:
            logger.warning("CUDA not available. Training will use CPU (very slow)")
    except ImportError:
        logger.warning("PyTorch not installed yet. CUDA check will be performed after installation.")

def setup_huggingface_auth():
    """Setup Hugging Face authentication."""
    try:
        from huggingface_hub import HfFolder
        
        # Check if token is already set
        token = HfFolder.get_token()
        if token:
            logger.info("Hugging Face token is already configured")
            return
        
        # Try to get token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token and hf_token != "your_hugging_face_token_here":
            from huggingface_hub import login
            login(hf_token)
            logger.info("Hugging Face authentication configured from environment")
        else:
            logger.warning("Please set HUGGINGFACE_TOKEN in your .env file")
            
    except ImportError:
        logger.warning("huggingface_hub not installed yet. Authentication will be set up after installation.")

def create_directories():
    """Create necessary project directories."""
    project_root = Path(__file__).parent.parent
    directories = [
        "data",
        "models", 
        "output",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        logger.info("Created directory: %s", dir_path)

def main():
    """Main setup function."""
    logger.info("Starting Gemma 3 Fine-tuning Environment Setup")
    
    # Check Python version
    check_python_version()
    
    # Create necessary directories
    create_directories()
    
    # Setup environment file
    setup_environment_file()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Environment variables loaded")
    except ImportError:
        logger.warning("python-dotenv not installed yet")
    
    # Install requirements
    install_requirements()
    
    # Verify CUDA after installation
    verify_cuda_availability()
    
    # Setup Hugging Face authentication
    setup_huggingface_auth()
    
    logger.info("Setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Edit .env file with your configuration")
    logger.info("2. Run: python scripts/train.py")

if __name__ == "__main__":
    main()