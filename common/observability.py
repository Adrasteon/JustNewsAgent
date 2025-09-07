"""
Centralized Logging System for JustNewsAgent
Production-ready logging with structured output, file rotation, and environment-specific configuration.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs for production."""

    def format(self, record: logging.LogRecord) -> str:
        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.utcnow().isoformat()

        # Create structured log entry
        log_entry = {
            'timestamp': record.timestamp,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


class JustNewsLogger:
    """Centralized logger configuration for JustNewsAgent."""

    _instance: Optional['JustNewsLogger'] = None
    _initialized = False

    def __new__(cls) -> 'JustNewsLogger':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            JustNewsLogger._initialized = True

    def _setup_logging(self) -> None:
        """Setup comprehensive logging configuration."""

        # Get configuration from environment
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_format = os.getenv('LOG_FORMAT', 'structured')  # 'structured' or 'readable'
        log_dir = Path(os.getenv('LOG_DIR', './logs'))
        max_bytes = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB default
        backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))

        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Clear existing handlers to avoid duplicates
        root_logger = logging.getLogger(__name__)
        root_logger.handlers.clear()

        # Set root logger level
        root_logger.setLevel(getattr(logging, log_level))

        # Create formatters
        if log_format == 'structured':
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler with rotation for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'justnews.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Separate error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'justnews_error.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)

        # Agent-specific log files
        agent_loggers = ['scout', 'analyst', 'fact_checker', 'synthesizer', 'critic', 'chief_editor']
        for agent in agent_loggers:
            agent_handler = logging.handlers.RotatingFileHandler(
                log_dir / f'{agent}.log',
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            agent_handler.setLevel(getattr(logging, log_level))
            agent_handler.setFormatter(formatter)
            agent_handler.addFilter(lambda record, agent_name=agent: agent_name in record.name)
            root_logger.addHandler(agent_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance."""
        return logging.getLogger(name)

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance metrics."""
        logger = self.get_logger('performance')
        extra = {'extra_fields': {'operation': operation, 'duration_ms': round(duration * 1000, 2), **kwargs}}
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", extra=extra)

    def log_error(self, error: Exception, context: str = "", **kwargs) -> None:
        """Log errors with context."""
        logger = self.get_logger('error')
        extra = {'extra_fields': {'error_type': type(error).__name__, 'context': context, **kwargs}}
        logger.error(f"Error in {context}: {error}", exc_info=True, extra=extra)


# Global instance
_logger_instance = JustNewsLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given name."""
    return _logger_instance.get_logger(name)


def setup_logging() -> None:
    """Initialize the centralized logging system."""
    # This is called automatically when the module is imported
    pass


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Convenience function for performance logging."""
    _logger_instance.log_performance(operation, duration, **kwargs)


def log_error(error: Exception, context: str = "", **kwargs) -> None:
    """Convenience function for error logging."""
    _logger_instance.log_error(error, context, **kwargs)


# Initialize logging when module is imported
setup_logging()
