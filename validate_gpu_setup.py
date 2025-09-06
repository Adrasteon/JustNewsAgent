#!/usr/bin/env python3
"""
GPU Environment Validation and Testing Script
Validates GPU setup and tests configuration after automated setup
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

# Add project paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "agents" / "common"))

try:
    from gpu_config_manager import get_config_manager, get_gpu_config
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("⚠️  GPU configuration modules not available - limited validation mode")

class GPUEnvironmentValidator:
    """Validates GPU environment setup and configuration"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent)
        self.config_dir = self.project_root / "config" / "gpu"
        self.logs_dir = self.project_root / "logs"
        self.results = {}

    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🔍 GPU Environment Validation Suite")
        print("=" * 50)

        self.results = {
            'timestamp': time.time(),
            'checks': {},
            'summary': {}
        }

        # Run all validation checks
        checks = [
            self.check_project_structure,
            self.check_dependencies,
            self.check_gpu_hardware,
            self.check_conda_environment,
            self.check_configuration_files,
            self.check_configuration_loading,
            self.check_environment_variables,
            self.test_gpu_monitoring,
            self.test_configuration_profiles,
            self.validate_performance_settings
        ]

        for check in checks:
            check_name = check.__name__.replace('check_', '').replace('test_', '').replace('validate_', '')
            try:
                print(f"\n🧪 Running {check_name} check...")
                result = check()
                self.results['checks'][check_name] = {
                    'status': 'passed' if result else 'failed',
                    'details': result if isinstance(result, dict) else {}
                }
                status_icon = "✅" if result else "❌"
                print(f"{status_icon} {check_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.results['checks'][check_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                print(f"❌ {check_name}: ERROR - {e}")

        # Generate summary
        self._generate_summary()

        return self.results

    def check_project_structure(self) -> Dict[str, Any]:
        """Check if project structure is correct"""
        required_paths = [
            "agents/common/gpu_config_manager.py",
            "agents/common/gpu_monitoring_enhanced.py",
            "agents/common/gpu_optimizer_enhanced.py",
            "config/gpu/gpu_config.json",
            "config/gpu/environment_config.json",
            "config/gpu/model_config.json",
            "setup_gpu_environment.sh",
            ".env"
        ]

        missing_paths = []
        for path in required_paths:
            if not (self.project_root / path).exists():
                missing_paths.append(path)

        return {
            'project_root': str(self.project_root),
            'required_paths_checked': len(required_paths),
            'missing_paths': missing_paths,
            'structure_valid': len(missing_paths) == 0
        }

    def check_dependencies(self) -> Dict[str, Any]:
        """Check if required dependencies are installed"""
        dependencies = {
            'python3': 'python3 --version',
            'conda': 'conda --version',
            'nvidia-smi': 'nvidia-smi --version' if self._has_nvidia() else None,
            'git': 'git --version'
        }

        installed_deps = {}
        missing_deps = []

        for dep, command in dependencies.items():
            if command and self._run_command(command, check=False):
                installed_deps[dep] = True
            else:
                missing_deps.append(dep)
                installed_deps[dep] = False

        return {
            'dependencies_checked': list(dependencies.keys()),
            'installed': [k for k, v in installed_deps.items() if v],
            'missing': missing_deps,
            'all_available': len(missing_deps) == 0
        }

    def check_gpu_hardware(self) -> Dict[str, Any]:
        """Check GPU hardware detection"""
        gpu_info = {
            'vendor': 'unknown',
            'count': 0,
            'devices': []
        }

        # Check NVIDIA
        if self._run_command('nvidia-smi --list-gpus', check=False):
            gpu_info['vendor'] = 'nvidia'
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'],
                                      capture_output=True, text=True, timeout=10)
                gpu_count = len([line for line in result.stdout.split('\n') if line.strip()])
                gpu_info['count'] = gpu_count

                # Get memory info
                for i in range(gpu_count):
                    mem_result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                                               '--format=csv,noheader,nounits', '-i', str(i)],
                                              capture_output=True, text=True, timeout=5)
                    if mem_result.returncode == 0:
                        memory_mb = int(mem_result.stdout.strip())
                        gpu_info['devices'].append({
                            'id': i,
                            'memory_mb': memory_mb,
                            'memory_gb': memory_mb / 1024
                        })
            except Exception as e:
                gpu_info['error'] = str(e)

        return gpu_info

    def check_conda_environment(self) -> Dict[str, Any]:
        """Check conda environment setup"""
        env_name = "justnews-v2-py312"
        env_info = {
            'environment_name': env_name,
            'exists': False,
            'active': False,
            'python_version': None
        }

        # Check if environment exists
        try:
            result = subprocess.run(['conda', 'env', 'list'],
                                  capture_output=True, text=True, timeout=10)
            if env_name in result.stdout:
                env_info['exists'] = True
        except Exception:
            pass

        # Check if environment is active
        current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if current_env == env_name:
            env_info['active'] = True

        # Check Python version in environment
        if env_info['exists']:
            try:
                result = subprocess.run(['conda', 'run', '-n', env_name, 'python', '--version'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    env_info['python_version'] = result.stdout.strip()
            except Exception:
                pass

        return env_info

    def check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files"""
        config_files = [
            "gpu_config.json",
            "environment_config.json",
            "model_config.json"
        ]

        config_status = {}
        all_valid = True

        for config_file in config_files:
            file_path = self.config_dir / config_file
            config_status[config_file] = {
                'exists': file_path.exists(),
                'valid_json': False,
                'size_bytes': 0
            }

            if file_path.exists():
                config_status[config_file]['size_bytes'] = file_path.stat().st_size
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    config_status[config_file]['valid_json'] = True
                except Exception as e:
                    config_status[config_file]['error'] = str(e)
                    all_valid = False
            else:
                all_valid = False

        return {
            'config_files': config_status,
            'all_valid': all_valid
        }

    def check_configuration_loading(self) -> Dict[str, Any]:
        """Test configuration loading"""
        if not IMPORTS_AVAILABLE:
            return {'skipped': True, 'reason': 'Imports not available'}

        try:
            config = get_gpu_config()
            manager = get_config_manager()
            env_info = manager.get_environment_info()
            profiles = manager.get_available_profiles()

            return {
                'config_loaded': True,
                'config_keys': list(config.keys()),
                'environment_info': env_info,
                'available_profiles': len(profiles),
                'profiles': [p['name'] for p in profiles]
            }
        except Exception as e:
            return {
                'config_loaded': False,
                'error': str(e)
            }

    def check_environment_variables(self) -> Dict[str, Any]:
        """Check environment variables"""
        env_vars = [
            'JUSTNEWS_ENV',
            'GPU_CONFIG_PROFILE',
            'PYTHONPATH',
            'GPU_VENDOR',
            'GPU_COUNT'
        ]

        env_status = {}
        for var in env_vars:
            env_status[var] = {
                'set': var in os.environ,
                'value': os.environ.get(var, '')
            }

        # Check .env file
        env_file = self.project_root / '.env'
        env_file_exists = env_file.exists()

        return {
            'environment_variables': env_status,
            'env_file_exists': env_file_exists,
            'env_file_path': str(env_file)
        }

    def test_gpu_monitoring(self) -> Dict[str, Any]:
        """Test GPU monitoring functionality"""
        if not IMPORTS_AVAILABLE:
            return {'skipped': True, 'reason': 'Imports not available'}

        try:
            # Import monitoring module
            sys.path.insert(0, str(self.project_root / "agents" / "common"))
            from gpu_monitoring_enhanced import GPUMonitoringSystem

            # Create monitoring instance
            monitoring = GPUMonitoringSystem()

            # Get current metrics
            metrics = monitoring.get_current_metrics()

            return {
                'monitoring_available': True,
                'metrics_keys': list(metrics.keys()) if metrics else [],
                'gpu_count': len(metrics.get('gpu_devices', [])) if metrics else 0
            }
        except Exception as e:
            return {
                'monitoring_available': False,
                'error': str(e)
            }

    def test_configuration_profiles(self) -> Dict[str, Any]:
        """Test configuration profiles"""
        if not IMPORTS_AVAILABLE:
            return {'skipped': True, 'reason': 'Imports not available'}

        try:
            manager = get_config_manager()
            profiles = manager.get_available_profiles()

            profile_tests = {}
            for profile in profiles:
                profile_name = profile['name']
                config = manager.get_config(profile=profile_name)
                profile_tests[profile_name] = {
                    'loaded': True,
                    'max_memory_gb': config.get('gpu_manager', {}).get('max_memory_per_agent_gb')
                }

            return {
                'profiles_tested': len(profiles),
                'profile_results': profile_tests
            }
        except Exception as e:
            return {
                'profiles_tested': 0,
                'error': str(e)
            }

    def validate_performance_settings(self) -> Dict[str, Any]:
        """Validate performance settings"""
        if not IMPORTS_AVAILABLE:
            return {'skipped': True, 'reason': 'Imports not available'}

        try:
            config = get_gpu_config()

            performance_config = config.get('performance', {})
            gpu_config = config.get('gpu_manager', {})

            validation_results = {
                'batch_size_optimization': performance_config.get('batch_size_optimization', False),
                'memory_preallocation': performance_config.get('memory_preallocation', False),
                'async_operations': performance_config.get('async_operations', False),
                'max_memory_per_agent_gb': gpu_config.get('max_memory_per_agent_gb', 0),
                'health_check_interval': gpu_config.get('health_check_interval_seconds', 0)
            }

            # Basic validation rules
            issues = []
            if validation_results['max_memory_per_agent_gb'] <= 0:
                issues.append('Invalid max memory per agent')
            if validation_results['health_check_interval'] <= 0:
                issues.append('Invalid health check interval')

            return {
                'settings': validation_results,
                'issues': issues,
                'valid': len(issues) == 0
            }
        except Exception as e:
            return {
                'settings': {},
                'issues': [str(e)],
                'valid': False
            }

    def _generate_summary(self):
        """Generate validation summary"""
        checks = self.results['checks']
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks.values() if c['status'] == 'passed')
        failed_checks = sum(1 for c in checks.values() if c['status'] == 'failed')
        error_checks = sum(1 for c in checks.values() if c['status'] == 'error')

        self.results['summary'] = {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'errors': error_checks,
            'success_rate': (passed_checks / total_checks) * 100 if total_checks > 0 else 0,
            'overall_status': 'passed' if failed_checks == 0 and error_checks == 0 else 'failed'
        }

    def _has_nvidia(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _run_command(self, command: str, check: bool = True) -> bool:
        """Run shell command and return success status"""
        try:
            result = subprocess.run(command.split(), capture_output=True, timeout=10)
            return result.returncode == 0 if check else True
        except Exception:
            return False

    def save_report(self, output_file: str = None):
        """Save validation report to file"""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"gpu_validation_report_{timestamp}.json"

        output_path = self.project_root / output_file

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📄 Validation report saved to: {output_path}")
        return output_path

def main():
    """Main validation function"""
    validator = GPUEnvironmentValidator()

    try:
        results = validator.run_full_validation()

        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)

        summary = results['summary']
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        status_icon = "✅" if summary['overall_status'] == 'passed' else "❌"
        print(f"Overall Status: {status_icon} {summary['overall_status'].upper()}")

        # Save report
        report_file = validator.save_report()

        if summary['overall_status'] == 'failed':
            print("\n❌ Some checks failed. Please review the detailed report:")
            print(f"   {report_file}")
            return 1

        print("\n✅ GPU environment validation completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())