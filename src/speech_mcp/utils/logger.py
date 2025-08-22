"""
Centralized logging module for speech-mcp.

This module provides a consistent logging interface for all components
of the speech-mcp project, ensuring logs are written to a deterministic
location and follow a consistent format.

Usage:
    from speech_mcp.utils.logger import get_logger
    
    # Get a logger for a specific module
    logger = get_logger(__name__)
    
    # Log messages at different levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Log exceptions with traceback
    try:
        # Some code that might raise an exception
        raise ValueError("Example error")
    except Exception as e:
        logger.exception("An error occurred")
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional

# Define constants for log file paths
LOG_DIR = os.path.expanduser("~/.speech-mcp/logs")
MAIN_LOG_FILE = os.path.join(LOG_DIR, "speech-mcp.log")
SERVER_LOG_FILE = os.path.join(LOG_DIR, "speech-mcp-server.log")
UI_LOG_FILE = os.path.join(LOG_DIR, "speech-mcp-ui.log")
TTS_LOG_FILE = os.path.join(LOG_DIR, "speech-mcp-tts.log")
STT_LOG_FILE = os.path.join(LOG_DIR, "speech-mcp-stt.log")

# Maximum log file size before rotation (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
BACKUP_COUNT = 5

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Map component names to log files
COMPONENT_LOG_FILES = {
    "server": SERVER_LOG_FILE,
    "ui": UI_LOG_FILE,
    "tts": TTS_LOG_FILE,
    "stt": STT_LOG_FILE,
}

# Cache for loggers to avoid creating duplicates
_loggers: Dict[str, logging.Logger] = {}

def ensure_log_dir() -> None:
    """Ensure the log directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger, typically __name__ of the calling module
        component: Optional component name to determine log file (server, ui, tts, stt)
                  If None, it will be inferred from the name
    
    Returns:
        A configured logger instance
    """
    # Check if we already created this logger
    if name in _loggers:
        return _loggers[name]
    
    # Create the logger
    logger = logging.getLogger(name)
    
    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        # Set the log level
        logger.setLevel(DEFAULT_LOG_LEVEL)
        
        # Ensure log directory exists
        ensure_log_dir()
        
        # Determine which log file to use based on component
        if component is None:
            # Try to infer component from name
            if "server" in name.lower():
                log_file = SERVER_LOG_FILE
            elif "ui" in name.lower():
                log_file = UI_LOG_FILE
            elif "tts" in name.lower():
                log_file = TTS_LOG_FILE
            elif "stt" in name.lower() or "recognition" in name.lower():
                log_file = STT_LOG_FILE
            else:
                log_file = MAIN_LOG_FILE
        else:
            # Use the specified component
            log_file = COMPONENT_LOG_FILES.get(component.lower(), MAIN_LOG_FILE)
        
        # Create a rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        
        # Create a console handler that writes to stderr (stdout is reserved for MCP stdio)
        console_handler = logging.StreamHandler(sys.stderr)
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set the formatter for both handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger
        logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger

def set_log_level(level: int) -> None:
    """
    Set the log level for all loggers.
    
    Args:
        level: The log level to set (e.g., logging.DEBUG, logging.INFO)
    """
    for logger in _loggers.values():
        logger.setLevel(level)

def get_log_files() -> Dict[str, str]:
    """
    Get a dictionary of log file paths.
    
    Returns:
        A dictionary mapping component names to log file paths
    """
    return {
        "main": MAIN_LOG_FILE,
        "server": SERVER_LOG_FILE,
        "ui": UI_LOG_FILE,
        "tts": TTS_LOG_FILE,
        "stt": STT_LOG_FILE,
    }