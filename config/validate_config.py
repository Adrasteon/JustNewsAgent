#!/usr/bin/env python3
"""
Configuration Validator for JustNewsAgent

Validates the centralized configuration file and provides helpful feedback
about configuration issues, missing values, and optimization suggestions.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from common.observability import get_logger

logger = get_logger(__name__)

class ConfigValidator:
    """Configuration validator with comprehensive checks"""

    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.suggestions: list[str] = []

    def load_config(self) -> bool:
        """Load and parse configuration file"""
        try:
            with open(self.config_file) as f:
                self.config = json.load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {self.config_file}")
            return False
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in configuration file: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to load configuration: {e}")
            return False

    def validate_structure(self):
        """Validate overall configuration structure"""
        required_sections = [
            'system', 'mcp_bus', 'database', 'crawling', 'gpu', 'agents'
        ]

        for section in required_sections:
            if section not in self.config:
                self.errors.append(f"Missing required section: {section}")
            elif not isinstance(self.config[section], dict):
                self.errors.append(f"Section '{section}' must be a dictionary")

    def validate_crawling_config(self):
        """Validate crawling configuration"""
        crawling = self.config.get('crawling', {})

        # Check rate limiting
        rate_limits = crawling.get('rate_limiting', {})
        rpm = rate_limits.get('requests_per_minute', 0)
        delay = rate_limits.get('delay_between_requests_seconds', 0)

        if rpm > 60:
            self.warnings.append(f"High requests per minute ({rpm}) may violate site policies")
        elif rpm < 5:
            self.suggestions.append(f"Very low requests per minute ({rpm}) may be inefficient")

        if delay < 1.0:
            self.warnings.append(f"Very short delay between requests ({delay}s) may be too aggressive")

        # Check concurrent settings
        concurrent_sites = rate_limits.get('concurrent_sites', 0)
        concurrent_browsers = rate_limits.get('concurrent_browsers', 0)

        if concurrent_sites > 5:
            self.warnings.append(f"High concurrent sites ({concurrent_sites}) may overwhelm system resources")

        if concurrent_browsers > concurrent_sites * 2:
            self.suggestions.append("Consider reducing concurrent browsers or increasing concurrent sites for better balance")

    def validate_database_config(self):
        """Validate database configuration"""
        db = self.config.get('database', {})

        # Check connection pool settings
        pool = db.get('connection_pool', {})
        min_conn = pool.get('min_connections', 0)
        max_conn = pool.get('max_connections', 0)

        if min_conn > max_conn:
            self.errors.append("Database min_connections cannot be greater than max_connections")

        if max_conn > 20:
            self.warnings.append(f"High max connections ({max_conn}) may strain database server")

        # Check for empty passwords in production
        if (self.config.get('system', {}).get('environment') == 'production' and
            not db.get('password')):
            self.warnings.append("Database password is empty in production environment")

    def validate_gpu_config(self):
        """Validate GPU configuration"""
        gpu = self.config.get('gpu', {})

        if gpu.get('enabled', False):
            devices = gpu.get('devices', {})
            preferred = devices.get('preferred', [])
            memory_limits = devices.get('memory_limits_gb', {})

            if not preferred:
                self.warnings.append("GPU enabled but no preferred devices specified")

            # Check memory limits
            memory_mgmt = gpu.get('memory_management', {})
            max_memory = memory_mgmt.get('max_memory_per_agent_gb', 0)

            for device_id, limit in memory_limits.items():
                if max_memory > limit * 0.8:
                    self.warnings.append(f"Agent memory limit ({max_memory}GB) close to device limit ({limit}GB)")

    def validate_performance_config(self):
        """Validate performance-related settings"""
        perf = self.config.get('performance', {})

        # Check cache settings
        cache = perf.get('cache_settings', {})
        if cache.get('enabled', False):
            ttl = cache.get('ttl_seconds', 0)
            max_size = cache.get('max_size_mb', 0)

            if ttl > 86400:  # 24 hours
                self.suggestions.append(f"Very long cache TTL ({ttl}s) may use excessive memory")

            if max_size > 2048:  # 2GB
                self.warnings.append(f"Large cache size ({max_size}MB) may impact system performance")

    def validate_monitoring_config(self):
        """Validate monitoring configuration"""
        monitoring = self.config.get('monitoring', {})

        if monitoring.get('enabled', False):
            alerts = monitoring.get('alert_thresholds', {})

            # Check alert thresholds are reasonable
            cpu_threshold = alerts.get('cpu_usage_percent', 0)
            memory_threshold = alerts.get('memory_usage_percent', 0)

            if cpu_threshold > 95:
                self.warnings.append(f"Very high CPU alert threshold ({cpu_threshold}%) may miss issues")

            if memory_threshold > 95:
                self.warnings.append(f"Very high memory alert threshold ({memory_threshold}%) may miss issues")

    def validate_data_minimization(self):
        """Validate data minimization settings"""
        dm = self.config.get('data_minimization', {})

        if dm.get('enabled', False):
            retention = dm.get('retention_days', {})

            # Check retention periods
            for data_type, days in retention.items():
                if days > 365:
                    self.suggestions.append(f"Long retention period for {data_type} ({days} days)")

    def generate_report(self) -> str:
        """Generate validation report"""
        report_lines = ["=== JustNewsAgent Configuration Validation Report ===\n"]

        if self.errors:
            report_lines.append("❌ ERRORS:")
            for error in self.errors:
                report_lines.append(f"  • {error}")
            report_lines.append("")

        if self.warnings:
            report_lines.append("⚠️  WARNINGS:")
            for warning in self.warnings:
                report_lines.append(f"  • {warning}")
            report_lines.append("")

        if self.suggestions:
            report_lines.append("💡 SUGGESTIONS:")
            for suggestion in self.suggestions:
                report_lines.append(f"  • {suggestion}")
            report_lines.append("")

        if not self.errors and not self.warnings and not self.suggestions:
            report_lines.append("✅ Configuration is valid with no issues found!")

        return "\n".join(report_lines)

    def validate(self) -> tuple[bool, str]:
        """Run all validation checks"""
        if not self.load_config():
            return False, self.generate_report()

        self.validate_structure()
        self.validate_crawling_config()
        self.validate_database_config()
        self.validate_gpu_config()
        self.validate_performance_config()
        self.validate_monitoring_config()
        self.validate_data_minimization()

        is_valid = len(self.errors) == 0
        return is_valid, self.generate_report()


def main():
    """Main validation function"""
    config_file = "config/system_config.json"

    # Try to find config file
    if not os.path.exists(config_file):
        # Try relative to script location
        script_dir = Path(__file__).parent
        config_file = script_dir / "system_config.json"

        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_file}")
            sys.exit(1)

    validator = ConfigValidator(str(config_file))
    is_valid, report = validator.validate()

    print(report)

    if not is_valid:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
