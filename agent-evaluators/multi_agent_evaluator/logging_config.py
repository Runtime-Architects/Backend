"""
Centralized logging configuration for the multi-agent evaluator system.
This ensures consistent logging format across all modules and tools.
"""

import logging
import sys
from typing import Optional

# Define our consistent format globally
CONSISTENT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
CONSISTENT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Store the original functions
_original_basicConfig = logging.basicConfig
_original_Formatter = logging.Formatter
_original_setFormatter = logging.StreamHandler.setFormatter
_original_FileHandler_setFormatter = logging.FileHandler.setFormatter

class ConsistentFormatter(logging.Formatter):
    """A formatter that always uses our consistent format."""
    def __init__(self, *args, **kwargs):
        # Ignore any format arguments and use our consistent format
        super().__init__(CONSISTENT_FORMAT, datefmt=CONSISTENT_DATE_FORMAT)

def _force_consistent_basicConfig(*args, **kwargs):
    """
    Override basicConfig to always use our consistent format.
    This prevents other modules from changing the logging format.
    """
    # Force our format regardless of what's passed
    kwargs['format'] = CONSISTENT_FORMAT
    kwargs['datefmt'] = CONSISTENT_DATE_FORMAT
    kwargs['force'] = True
    
    # Call the original basicConfig
    _original_basicConfig(*args, **kwargs)

def _force_consistent_setFormatter(self, formatter):
    """Force all handlers to use our consistent formatter."""
    # Always use our consistent formatter
    _original_setFormatter(self, ConsistentFormatter())

def _force_consistent_FileHandler_setFormatter(self, formatter):
    """Force file handlers to use our consistent formatter."""
    # Always use our consistent formatter
    _original_FileHandler_setFormatter(self, ConsistentFormatter())

# Monkey patch to enforce consistent formatting
logging.basicConfig = _force_consistent_basicConfig
logging.Formatter = ConsistentFormatter
logging.StreamHandler.setFormatter = _force_consistent_setFormatter
logging.FileHandler.setFormatter = _force_consistent_FileHandler_setFormatter

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging with a consistent format across the entire application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
    """
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsistentFormatter())
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(ConsistentFormatter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=CONSISTENT_FORMAT,
        datefmt=CONSISTENT_DATE_FORMAT,
        handlers=handlers,
        force=True
    )
    
    # Configure specific loggers to reduce noise
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('webdriver_manager').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.INFO)
    logging.getLogger('autogen_core').setLevel(logging.INFO)
    logging.getLogger('autogen_core.events').setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the module name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)