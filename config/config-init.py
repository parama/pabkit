"""
Configuration management for the Process-Aware Benchmarking (PAB) toolkit.

This module provides utilities for loading, saving, and managing configurations
for PAB experiments and analyses.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

from .default_config import (
    PAB_DEFAULTS,
    VISUALIZATION_DEFAULTS,
    ADVERSARIAL_DEFAULTS,
    REPRESENTATION_DEFAULTS,
    CLASS_PROGRESSION_DEFAULTS,
    EVALUATION_THRESHOLDS,
    MODEL_LAYERS,
    TRAINING_HYPERPARAMS,
    DATA_TRANSFORMS
)

class Config:
    """Configuration manager for PAB."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        config_name: str = 'default'
    ):
        """
        Initialize configuration with defaults and optional overrides.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            config_dict: Dictionary with configuration values
            config_name: Name of the configuration
        """
        self.config_name = config_name
        
        # Start with default configurations
        self.config = {
            'pab': PAB_DEFAULTS.copy(),
            'visualization': VISUALIZATION_DEFAULTS.copy(),
            'adversarial': ADVERSARIAL_DEFAULTS.copy(),
            'representation': REPRESENTATION_DEFAULTS.copy(),
            'class_progression': CLASS_PROGRESSION_DEFAULTS.copy(),
            'evaluation': EVALUATION_THRESHOLDS.copy(),
            'model_layers': MODEL_LAYERS.copy(),
            'training': TRAINING_HYPERPARAMS.copy(),
            'transforms': DATA_TRANSFORMS.copy()
        }
        
        # Load configuration from file if provided
        if config_path is not None:
            self.load_config(config_path)
        
        # Override with provided dictionary if available
        if config_dict is not None:
            self.update_config(config_dict)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file type and load accordingly
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")
        
        # Update configuration
        self.update_config(config_dict)
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        # Update each section if present
        for section, values in config_dict.items():
            if section in self.config:
                if isinstance(values, dict) and isinstance(self.config[section], dict):
                    # Deep update for nested dictionaries
                    self.config[section].update(values)
                else:
                    # Direct replacement for non-dict values
                    self.config[section] = values
            else:
                # Add new section
                self.config[section] = values
    
    def save_config(self, config_path: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save configuration file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save based on file extension
        if config_path.endswith('.json'):
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path}")
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (if None, returns entire section)
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(name={self.config_name}, sections={list(self.config.keys())})"


# Default configuration instance
default_config = Config()

def load_config(config_path: str) -> Config:
    """
    Load configuration from a file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path=config_path)

def get_config() -> Config:
    """
    Get the default configuration.
    
    Returns:
        Default Config object
    """
    return default_config
