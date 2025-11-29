"""
Fine-FinTech: Gemma 3 Financial Q&A Fine-tuning Package

This package provides tools for fine-tuning Google's Gemma 3 model 
on financial reasoning datasets.
"""

__version__ = "1.0.0"
__author__ = "Fine-FinTech Team"
__email__ = "contact@fine-fintech.com"

from .model_loader import ModelLoader
from .data_processor import DataProcessor
from .trainer import FinancialTrainer
from .inference import FinancialInference
from .utils import setup_logging, load_config

__all__ = [
    "ModelLoader",
    "DataProcessor", 
    "FinancialTrainer",
    "FinancialInference",
    "setup_logging",
    "load_config"
]