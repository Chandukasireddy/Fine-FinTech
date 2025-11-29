"""
Utility functions for the Fine-FinTech project.
"""

import os
import sys
import yaml
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "./logs"
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory to store log files
        
    Returns:
        Configured logger
    """
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        if not log_file.endswith('.log'):
            log_file += '.log'
        
        file_path = Path(log_dir) / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {level}")
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        logging.getLogger(__name__).info(f"Configuration loaded from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {str(e)}")

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones for conflicting keys.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config:
            merged = _deep_merge(merged, config)
    
    return merged

def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Device information dictionary
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "devices": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            device_info["devices"].append({
                "id": i,
                "name": device_props.name,
                "memory_total_gb": device_props.total_memory / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
    
    return device_info

def format_model_size(num_parameters: int) -> str:
    """
    Format model size in human readable format.
    
    Args:
        num_parameters: Number of model parameters
        
    Returns:
        Formatted string (e.g., "1.2B", "345M")
    """
    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1_000_000_000:.1f}B"
    elif num_parameters >= 1_000_000:
        return f"{num_parameters / 1_000_000:.1f}M"
    elif num_parameters >= 1_000:
        return f"{num_parameters / 1_000:.1f}K"
    else:
        return str(num_parameters)

def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.2GB", "345MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()

def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def load_environment_variables():
    """
    Load environment variables from .env file if it exists.
    """
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logging.getLogger(__name__).info("Environment variables loaded from .env")
        else:
            logging.getLogger(__name__).warning(".env file not found")
    except ImportError:
        logging.getLogger(__name__).warning("python-dotenv not installed")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary for required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    required_sections = ["model", "training", "lora", "dataset"]
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        logger.error(f"Missing required config sections: {missing_sections}")
        return False
    
    # Validate model config
    model_config = config.get("model", {})
    if not model_config.get("name"):
        logger.error("Model name not specified in config")
        return False
    
    # Validate dataset config
    dataset_config = config.get("dataset", {})
    if not dataset_config.get("name"):
        logger.error("Dataset name not specified in config")
        return False
    
    logger.info("Configuration validation passed")
    return True

def count_model_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params
    }

def clear_gpu_memory():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.getLogger(__name__).info("GPU memory cache cleared")

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming this file is in src/ directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    return project_root.resolve()

def save_results(results: Dict[str, Any], filename: str, results_dir: str = "./output"):
    """
    Save results to a JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        results_dir: Directory to save results
    """
    import json
    
    results_path = ensure_directory(results_dir)
    
    # Add timestamp to filename if not present
    if not filename.endswith('.json'):
        filename += '.json'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if timestamp not in filename:
        name_parts = filename.split('.')
        filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
    
    file_path = results_path / filename
    
    # Add metadata
    results_with_meta = {
        "timestamp": get_timestamp(),
        "results": results
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
    
    logging.getLogger(__name__).info(f"Results saved to: {file_path}")

class ConfigManager:
    """Configuration manager for handling multiple config files."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
    
    def load_all_configs(self) -> Dict[str, Any]:
        """Load all configuration files from config directory."""
        config_files = {
            "model": "model_config.yaml",
            "training": "training_config.yaml", 
            "lora": "lora_config.yaml"
        }
        
        merged_config = {}
        
        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                config_data = load_config(config_path)
                merged_config.update(config_data)
                self.configs[config_name] = config_data
            else:
                logging.getLogger(__name__).warning(f"Config file not found: {config_path}")
        
        return merged_config
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get specific configuration by name."""
        return self.configs.get(config_name, {})