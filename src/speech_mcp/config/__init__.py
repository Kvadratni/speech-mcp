"""
Configuration management for speech-mcp.

This module provides functions for reading and writing configuration settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Import centralized constants
from speech_mcp.constants import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

def ensure_config_dir() -> None:
    """Ensure the configuration directory exists."""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        logger.debug(f"Configuration directory ensured: {CONFIG_DIR}")
    except Exception as e:
        logger.error(f"Error creating configuration directory: {e}")


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    ensure_config_dir()
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {CONFIG_FILE}")
            
            # Merge with default config to ensure all keys exist
            merged_config = DEFAULT_CONFIG.copy()
            for section, values in config.items():
                if section in merged_config:
                    merged_config[section].update(values)
                else:
                    merged_config[section] = values
            
            return merged_config
        else:
            logger.info(f"Configuration file not found, using defaults")
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: The configuration dictionary to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    ensure_config_dir()
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """
    Get a specific setting from the configuration.
    
    Args:
        section: The configuration section
        key: The setting key
        default: Default value if not found
        
    Returns:
        The setting value or default
    """
    config = load_config()
    try:
        return config.get(section, {}).get(key, default)
    except Exception as e:
        logger.error(f"Error getting setting {section}.{key}: {e}")
        return default


def set_setting(section: str, key: str, value: Any) -> bool:
    """
    Set a specific setting in the configuration.
    
    Args:
        section: The configuration section
        key: The setting key
        value: The setting value
        
    Returns:
        bool: True if successful, False otherwise
    """
    config = load_config()
    try:
        if section not in config:
            config[section] = {}
        config[section][key] = value
        return save_config(config)
    except Exception as e:
        logger.error(f"Error setting {section}.{key} to {value}: {e}")
        return False


# Environment variable support
def get_env_setting(name: str, default: Any = None) -> Any:
    """
    Get a setting from an environment variable.
    
    Args:
        name: The environment variable name
        default: Default value if not found
        
    Returns:
        The environment variable value or default
    """
    return os.environ.get(name, default)


def set_env_setting(name: str, value: str) -> None:
    """
    Set an environment variable.
    
    Args:
        name: The environment variable name
        value: The value to set
    """
    os.environ[name] = value