#!/usr/bin/env python3
"""
JustNewsAgent Health Monitor Script
Monitors system health after deployment and reports issues.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthMonitor:
    """Monitors system health after deployment."""

    def __init__(self, base_url: str, monitoring_url: str | None = None):
        self.base_url = base_url.rstrip('/')
        self.monitoring_url = monitoring_url or os.getenv('MONITORING_URL')
        self.timeout = 30
        self.max_retries = 3

    def check_service_health(self, service_name: str, endpoint: str) -> tuple[bool, dict[str, Any]]:
        """Check health of a specific service."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=self.timeout)
                response_time = time.time() - start_time

                health_data = {
                    'service': service_name,
                    'endpoint': endpoint,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'healthy': response.status_code == 200,
                    'timestamp': datetime.now().isoformat(),
                    'attempt': attempt + 1
                }

                if response.status_code == 200:
                    try:
                        data = response.json()
                        health_data['health_data'] = data
                    except json.JSONDecodeError:
                        health_data['health_data'] = {'message': 'Non-JSON response'}

                    logger.info(f"Service {service_name} is healthy (response time: {response_time:.2f}s)")
                    return True, health_data
                else:
                    logger.warning(f"Service {service_name} returned status {response.status_code}")
                    health_data['error'] = f"HTTP {response.status_code}"

            except requests.RequestException as e:
                logger.warning(f"Failed to check {service_name} health (attempt {attempt + 1}): {e}")
                health_data = {
                    'service': service_name,
                    'endpoint': endpoint,
                    'error': str(e),
                    'healthy': False,
                    'timestamp': datetime.now().isoformat(),
                    'attempt': attempt + 1
                }

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        return False, health_data

    def check_all_services(self) -> dict[str, Any]:
        """Check health of all critical services."""
        services = {
            'mcp_bus': '/health',
            'chief_editor': '/health',
            'scout': '/health',
            'analyst': '/health',
            'fact_checker': '/health',
            'synthesizer': '/health',
            'critic': '/health',
            'memory': '/health',
            'reasoning': '/health',
            'dashboard': '/health'
        }

        results = {
            'overall_healthy': True,
            'services': {},
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_services': len(services),
                'healthy_services': 0,
                'unhealthy_services': 0
            }
        }

        for service_name, endpoint in services.items():
            healthy, health_data = self.check_service_health(service_name, endpoint)
            results['services'][service_name] = health_data

            if healthy:
                results['summary']['healthy_services'] += 1
            else:
                results['summary']['unhealthy_services'] += 1
                results['overall_healthy'] = False

        logger.info(f"Health check complete: {results['summary']['healthy_services']}/{results['summary']['total_services']} services healthy")
        return results

    def check_database_connectivity(self) -> tuple[bool, dict[str, Any]]:
        """Check database connectivity."""
        try:
            # Check MCP Bus database endpoint
            url = f"{self.base_url}/api/v1/health/database"
            response = requests.get(url, timeout=self.timeout)

            db_health = {
                'component': 'database',
                'status_code': response.status_code,
                'healthy': response.status_code == 200,
                'timestamp': datetime.now().isoformat()
            }

            if response.status_code == 200:
                try:
                    data = response.json()
                    db_health['connection_info'] = data
                except json.JSONDecodeError:
                    pass

                logger.info("Database connectivity check passed")
                return True, db_health
            else:
                logger.error(f"Database connectivity check failed: HTTP {response.status_code}")
                return False, db_health

        except requests.RequestException as e:
            logger.error(f"Database connectivity check failed: {e}")
            return False, {
                'component': 'database',
                'error': str(e),
                'healthy': False,
                'timestamp': datetime.now().isoformat()
            }

    def check_external_dependencies(self) -> dict[str, Any]:
        """Check external service dependencies."""
        dependencies = {
            'gpu_orchestrator': f"{self.base_url}/api/v1/gpu/status",
            'kafka': f"{self.base_url}/api/v1/health/kafka",
            'redis': f"{self.base_url}/api/v1/health/redis"
        }

        results = {
            'dependencies': {},
            'overall_healthy': True,
            'timestamp': datetime.now().isoformat()
        }

        for dep_name, url in dependencies.items():
            try:
                response = requests.get(url, timeout=self.timeout)
                healthy = response.status_code == 200

                results['dependencies'][dep_name] = {
                    'healthy': healthy,
                    'status_code': response.status_code,
                    'timestamp': datetime.now().isoformat()
                }

                if not healthy:
                    results['overall_healthy'] = False
                    logger.warning(f"External dependency {dep_name} is unhealthy")

            except requests.RequestException as e:
                results['dependencies'][dep_name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['overall_healthy'] = False
                logger.warning(f"Failed to check external dependency {dep_name}: {e}")

        return results

    def perform_load_test(self, duration: int = 30) -> dict[str, Any]:
        """Perform basic load testing."""
        logger.info(f"Starting load test for {duration} seconds")

        start_time = time.time()
        requests_made = 0
        errors = 0
        response_times = []

        while time.time() - start_time < duration:
            try:
                # Test MCP Bus health endpoint
                test_start = time.time()
                response = requests.get(f"{self.base_url}/health", timeout=5)
                response_time = time.time() - test_start

                requests_made += 1
                response_times.append(response_time)

                if response.status_code != 200:
                    errors += 1

            except requests.RequestException:
                errors += 1
                requests_made += 1

            time.sleep(0.1)  # 10 requests per second

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        error_rate = errors / requests_made if requests_made > 0 else 0

        results = {
            'duration_seconds': duration,
            'requests_made': requests_made,
            'errors': errors,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'healthy': error_rate < 0.05,  # Less than 5% error rate
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Load test complete: {requests_made} requests, {errors} errors, avg response: {avg_response_time:.3f}s")
        return results

    def report_health_to_monitoring(self, health_data: dict[str, Any]) -> bool:
        """Report health data to monitoring system."""
        if not self.monitoring_url:
            logger.warning("No monitoring URL configured, skipping health report")
            return False

        try:
            response = requests.post(
                f"{self.monitoring_url}/api/v1/health",
                json=health_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Health data reported to monitoring system")
                return True
            else:
                logger.error(f"Failed to report health data: HTTP {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.error(f"Failed to report health data: {e}")
            return False

    def comprehensive_health_check(self, include_load_test: bool = False) -> dict[str, Any]:
        """Perform comprehensive health check."""
        logger.info("Starting comprehensive health check")

        results = {
            'services': self.check_all_services(),
            'database': self.check_database_connectivity(),
            'external_dependencies': self.check_external_dependencies(),
            'timestamp': datetime.now().isoformat()
        }

        if include_load_test:
            results['load_test'] = self.perform_load_test()

        # Overall health determination
        results['overall_healthy'] = (
            results['services']['overall_healthy'] and
            results['database'][0] and  # database returns (healthy, data)
            results['external_dependencies']['overall_healthy']
        )

        if include_load_test:
            results['overall_healthy'] = results['overall_healthy'] and results['load_test']['healthy']

        # Report to monitoring system
        self.report_health_to_monitoring(results)

        logger.info(f"Comprehensive health check complete. Overall health: {'PASS' if results['overall_healthy'] else 'FAIL'}")
        return results

def main():
    """Main health monitoring entry point."""
    parser = argparse.ArgumentParser(description='Monitor system health after deployment')
    parser.add_argument('--base-url', required=True,
                       help='Base URL of the deployed system')
    parser.add_argument('--monitoring-url',
                       help='Monitoring system URL (or set MONITORING_URL env var)')
    parser.add_argument('--check-type', default='comprehensive',
                       choices=['services', 'database', 'dependencies', 'load', 'comprehensive'],
                       help='Type of health check to perform')
    parser.add_argument('--load-test-duration', type=int, default=30,
                       help='Duration for load testing in seconds')
    parser.add_argument('--output-file',
                       help='File to save health check results (JSON)')

    args = parser.parse_args()

    monitor = HealthMonitor(args.base_url, args.monitoring_url)

    try:
        if args.check_type == 'services':
            results = monitor.check_all_services()
        elif args.check_type == 'database':
            healthy, data = monitor.check_database_connectivity()
            results = {'database': (healthy, data)}
        elif args.check_type == 'dependencies':
            results = monitor.check_external_dependencies()
        elif args.check_type == 'load':
            results = monitor.perform_load_test(args.load_test_duration)
        elif args.check_type == 'comprehensive':
            results = monitor.comprehensive_health_check(include_load_test=True)

        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Health check results saved to {args.output_file}")

        # Exit with appropriate code
        if args.check_type == 'comprehensive':
            overall_healthy = results.get('overall_healthy', False)
        elif args.check_type == 'database':
            overall_healthy = results['database'][0]
        elif args.check_type == 'load':
            overall_healthy = results.get('healthy', False)
        else:
            overall_healthy = True  # Other checks don't have overall health

        if overall_healthy:
            logger.info("Health check PASSED")
            sys.exit(0)
        else:
            logger.error("Health check FAILED")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Health monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
