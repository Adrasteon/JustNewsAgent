#!/usr/bin/env python3
"""
JustNewsAgent Deployment Validator
Validates deployment readiness before CD pipeline execution.
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess

class DeploymentValidator:
    """Validates deployment prerequisites and configuration."""

    def __init__(self, config_path: str = "config/system_config.json"):
        self.config_path = Path(config_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_config_exists(self) -> bool:
        """Check if configuration file exists."""
        if not self.config_path.exists():
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False
        return True

    def validate_config_format(self) -> bool:
        """Validate configuration file format."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            required_keys = ['environment', 'agents']
            for key in required_keys:
                if key not in config:
                    self.errors.append(f"Missing required config key: {key}")
                    return False

            return True
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading config file: {e}")
            return False

    def validate_agents_config(self) -> bool:
        """Validate agents configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            agents = config.get('agents', {})
            if not agents:
                self.errors.append("No agents configured")
                return False

            required_agents = ['scout', 'analyst', 'synthesizer', 'memory']
            for agent in required_agents:
                if agent not in agents:
                    self.errors.append(f"Missing required agent: {agent}")
                    return False

                agent_config = agents[agent]
                if not isinstance(agent_config, dict):
                    self.errors.append(f"Invalid agent config for {agent}")
                    return False

                if 'enabled' not in agent_config:
                    self.warnings.append(f"Agent {agent} missing 'enabled' flag")

            return True
        except Exception as e:
            self.errors.append(f"Error validating agents config: {e}")
            return False

    def validate_python_version(self) -> bool:
        """Validate Python version compatibility."""
        try:
            result = subprocess.run(
                [sys.executable, '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            version_output = result.stdout.strip()

            # Extract version number
            if 'Python 3.' in version_output:
                version_str = version_output.split('Python ')[1]
                major, minor = map(int, version_str.split('.')[:2])

                if major != 3 or minor < 12:
                    self.errors.append(f"Python version {version_str} is not supported. Required: 3.12+")
                    return False
            else:
                self.errors.append(f"Unable to parse Python version: {version_output}")
                return False

            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to check Python version: {e}")
            return False

    def validate_dependencies(self) -> bool:
        """Validate that required dependencies are installed."""
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'sqlalchemy',
            'redis', 'psycopg2-binary', 'pytest'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.errors.append(f"Missing required packages: {', '.join(missing_packages)}")
            return False

        return True

    def validate_file_permissions(self) -> bool:
        """Validate file permissions for deployment."""
        critical_files = [
            'config/system_config.json',
            'requirements.txt',
            'pyproject.toml'
        ]

        for file_path in critical_files:
            path = Path(file_path)
            if path.exists():
                # Check if file is readable
                if not os.access(path, os.R_OK):
                    self.errors.append(f"File not readable: {file_path}")
                    return False
            else:
                self.warnings.append(f"File not found: {file_path}")

        return True

    def run_all_validations(self) -> bool:
        """Run all validation checks."""
        validations = [
            self.validate_config_exists,
            self.validate_config_format,
            self.validate_agents_config,
            self.validate_python_version,
            self.validate_dependencies,
            self.validate_file_permissions,
        ]

        all_passed = True
        for validation in validations:
            if not validation():
                all_passed = False

        return all_passed

    def report_results(self) -> None:
        """Report validation results."""
        if self.errors:
            print("❌ Deployment validation FAILED:")
            for error in self.errors:
                print(f"  - {error}")
            print()
            return

        print("✅ Deployment validation PASSED")

        if self.warnings:
            print("\n⚠️  Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print()

def main():
    """Main validation entry point."""
    validator = DeploymentValidator()

    print("🔍 Running deployment validation checks...")
    print()

    success = validator.run_all_validations()
    validator.report_results()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()