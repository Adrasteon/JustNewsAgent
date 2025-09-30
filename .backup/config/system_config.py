#!/usr/bin/env python3
"""
Centralized Configuration Manager for JustNewsAgent

This module provides a unified interface for loading and accessing all system
configuration settings from the centralized config file.

Usage:
    from config.system_config import config

    # Access settings
    crawl_settings = config.get('crawling')
    rate_limits = config.get('crawling.rate_limiting')

    # Environment-specific overrides
    db_host = config.get('database.host')
    gpu_enabled = config.get('gpu.enabled')
"""

import json
import os
from pathlib import Path
from typing import Any

from common.observability import get_logger

logger = get_logger(__name__)

class ConfigManager:
    """Centralized configuration manager with environment overrides"""

    def __init__(self, config_file: str | None = None):
        self.config_file = config_file or self._find_config_file()
        self._config: dict[str, Any] = {}
        self._load_config()

    def _find_config_file(self) -> str:
        """Find the system configuration file"""
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent / "system_config.json",
            Path.cwd() / "config" / "system_config.json",
            Path.cwd() / "system_config.json",
            Path.home() / ".justnews" / "system_config.json"
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Default fallback
        return str(Path(__file__).parent / "system_config.json")

    def _load_config(self):
        """Load configuration from file with environment overrides"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file) as f:
                    self._config = json.load(f)
                logger.info(f"✅ Loaded configuration from {self.config_file}")
            else:
                logger.warning(f"Configuration file not found: {self.config_file}")
                self._config = self._get_default_config()

            # Apply environment overrides
            self._apply_environment_overrides()

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = self._get_default_config()

    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # Database settings
        if os.environ.get('POSTGRES_HOST'):
            self._set_nested_value('database.host', os.environ['POSTGRES_HOST'])
        if os.environ.get('POSTGRES_DB'):
            self._set_nested_value('database.database', os.environ['POSTGRES_DB'])
        if os.environ.get('POSTGRES_USER'):
            self._set_nested_value('database.user', os.environ['POSTGRES_USER'])
        if os.environ.get('POSTGRES_PASSWORD'):
            self._set_nested_value('database.password', os.environ['POSTGRES_PASSWORD'])

        # Crawling settings
        if os.environ.get('CRAWLER_REQUESTS_PER_MINUTE'):
            self._set_nested_value('crawling.rate_limiting.requests_per_minute',
                                 int(os.environ['CRAWLER_REQUESTS_PER_MINUTE']))
        if os.environ.get('CRAWLER_DELAY_BETWEEN_REQUESTS'):
            self._set_nested_value('crawling.rate_limiting.delay_between_requests_seconds',
                                 float(os.environ['CRAWLER_DELAY_BETWEEN_REQUESTS']))
        if os.environ.get('CRAWLER_CONCURRENT_SITES'):
            self._set_nested_value('crawling.rate_limiting.concurrent_sites',
                                 int(os.environ['CRAWLER_CONCURRENT_SITES']))

        # System settings
        if os.environ.get('LOG_LEVEL'):
            self._set_nested_value('system.log_level', os.environ['LOG_LEVEL'])
        if os.environ.get('DEBUG_MODE'):
            self._set_nested_value('system.debug_mode', os.environ['DEBUG_MODE'].lower() == 'true')

        # GPU settings
        if os.environ.get('GPU_ENABLED'):
            self._set_nested_value('gpu.enabled', os.environ['GPU_ENABLED'].lower() == 'true')

    def _set_nested_value(self, key_path: str, value: Any):
        """Set a nested configuration value using dot notation"""
        keys = key_path.split('.')
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get a nested configuration value using dot notation"""
        keys = key_path.split('.')
        current = self._config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by key path"""
        return self._get_nested_value(key_path, default)

    def set(self, key_path: str, value: Any):
        """Set configuration value by key path"""
        self._set_nested_value(key_path, value)

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})

    def reload(self):
        """Reload configuration from file"""
        self._load_config()

    def save(self, file_path: str | None = None):
        """Save current configuration to file"""
        save_path = file_path or self.config_file
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"✅ Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration values"""
        return {
            "system": {
                "name": "JustNewsAgent",
                "version": "4.0",
                "environment": "development",
                "log_level": "INFO",
                "debug_mode": False
            },
            "crawling": {
                "enabled": True,
                "obey_robots_txt": True,
                "rate_limiting": {
                    "requests_per_minute": 10,
                    "delay_between_requests_seconds": 3.0
                }
            }
        }

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._config

    def keys(self):
        """Get all top-level keys"""
        return self._config.keys()

    def items(self):
        """Get all key-value pairs"""
        return self._config.items()


# Global configuration instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# Convenience instance for direct import
config = get_config()


# Utility functions for common configuration access
def get_crawling_config() -> dict[str, Any]:
    """Get crawling configuration"""
    return config.get_section('crawling')

def get_database_config() -> dict[str, Any]:
    """Get database configuration"""
    return config.get_section('database')

def get_gpu_config() -> dict[str, Any]:
    """Get GPU configuration"""
    return config.get_section('gpu')

def get_rate_limits() -> dict[str, Any]:
    """Get rate limiting configuration"""
    return config.get('crawling.rate_limiting', {})

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return config.get('system.debug_mode', False)

def get_log_level() -> str:
    """Get logging level"""
    return config.get('system.log_level', 'INFO')


if __name__ == "__main__":
    # Test the configuration system
    cfg = get_config()

    print("=== JustNewsAgent Configuration Test ===")
    print(f"System Name: {cfg.get('system.name')}")
    print(f"Version: {cfg.get('system.version')}")
    print(f"Environment: {cfg.get('system.environment')}")
    print(f"Robots.txt Compliance: {cfg.get('crawling.obey_robots_txt')}")
    print(f"Requests per Minute: {cfg.get('crawling.rate_limiting.requests_per_minute')}")
    print(f"Concurrent Sites: {cfg.get('crawling.rate_limiting.concurrent_sites')}")
    print(f"GPU Enabled: {cfg.get('gpu.enabled')}")
    print(f"Database Host: {cfg.get('database.host')}")

    # Test environment overrides
    print("\n=== Environment Override Test ===")
    print("Set CRAWLER_REQUESTS_PER_MINUTE=5 and run again to test overrides")

    print(f"\nConfiguration file: {cfg.config_file}")
    print("✅ Configuration system loaded successfully!")
