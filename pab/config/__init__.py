"""
Configuration management for Process-Aware Benchmarking (PAB).
"""

from .default_config import DEFAULT_CONFIG

import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Update with file config if exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Update config with file values
        if file_config:
            _update_dict(config, file_config)
    
    return config

def _update_dict(base_dict: Dict, update_dict: Dict) -> None:
    """
    Recursively update a dictionary with values from another dictionary.
    
    Args:
        base_dict: Dictionary to update
        update_dict: Dictionary with new values
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            # Recursively update nested dictionaries
            _update_dict(base_dict[key], value)
        else:
            # Update value
            base_dict[key] = value
