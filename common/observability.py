"""
Common observability utilities for JustNewsAgent
"""

import logging
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO, format_string: Optional[str] = None) -> None:
    """
    Setup basic logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )