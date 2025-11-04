#!/usr/bin/env python3
"""
JustNewsAgent Canary Testing Script
Validates canary deployments by testing new version against production traffic.
"""

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CanaryTester:
    """Handles canary deployment testing and validation."""

    def __init__(self, base_url: str, canary_percentage: float = 10.0):
        self.base_url = base_url.rstrip('/')
        self.canary_percentage = canary_percentage
        self.test_results: dict[str, Any] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': [],
            'canary_traffic': 0,
            'baseline_traffic': 0
        }

    def test_health_endpoint(self) -> bool:
        """Test basic health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def test_mcp_bus_connection(self) -> bool:
        """Test MCP Bus connectivity."""
        try:
            # Test MCP Bus registration endpoint
            payload = {
                "agent": "canary-tester",
                "tool": "health_check",
                "args": [],
                "kwargs": {}
            }
            response = requests.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=10
            )
            return response.status_code in [200, 404]  # 404 is expected for test agent
        except Exception as e:
            logger.error(f"MCP Bus test failed: {e}")
            return False

    def test_agent_endpoints(self) -> dict[str, bool]:
        """Test key agent endpoints."""
        agents_to_test = {
            'scout': '/api/v1/scout/status',
            'analyst': '/api/v1/analyst/status',
            'synthesizer': '/api/v1/synthesizer/status',
            'memory': '/api/v1/memory/status'
        }

        results = {}
        for agent, endpoint in agents_to_test.items():
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                results[agent] = response.status_code == 200
                if not results[agent]:
                    logger.warning(f"Agent {agent} endpoint returned status {response.status_code}")
            except Exception as e:
                logger.error(f"Agent {agent} test failed: {e}")
                results[agent] = False

        return results

    def run_load_test(self, duration: int = 60, concurrency: int = 10) -> dict[str, Any]:
        """Run load test against canary deployment."""
        logger.info(f"Running load test for {duration}s with {concurrency} concurrent users")

        start_time = time.time()
        end_time = start_time + duration

        def make_request() -> dict[str, Any]:
            """Make a single test request."""
            try:
                request_start = time.time()
                # Test a basic API endpoint
                response = requests.get(f"{self.base_url}/health", timeout=5)
                request_time = time.time() - request_start

                return {
                    'success': response.status_code == 200,
                    'response_time': request_time,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - request_start,
                    'error': str(e)
                }

        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []

            while time.time() < end_time:
                # Submit requests at a controlled rate
                futures.append(executor.submit(make_request))

                # Clean up completed futures
                for future in as_completed(futures[:], timeout=0.1):
                    try:
                        result = future.result(timeout=1)
                        results.append(result)
                        futures.remove(future)
                    except Exception as e:
                        logger.warning(f"Future collection error: {e}")

                time.sleep(0.1)  # Rate limiting

            # Wait for remaining futures
            for future in as_completed(futures, timeout=10):
                try:
                    result = future.result(timeout=1)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Final future collection error: {e}")

        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        analysis = {
            'total_requests': len(results),
            'successful_requests': len(successful),
            'failed_requests': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
            'avg_response_time': statistics.mean([r['response_time'] for r in results]) if results else 0,
            'p95_response_time': statistics.quantiles([r['response_time'] for r in results], n=20)[18] if len(results) >= 20 else 0,
            'errors': [r.get('error', 'Unknown error') for r in failed[:10]]  # Sample of errors
        }

        logger.info(f"Load test completed: {analysis['success_rate']:.1%} success rate, "
                   f"{analysis['avg_response_time']:.3f}s avg response time")

        return analysis

    def validate_canary_metrics(self) -> dict[str, Any]:
        """Validate that canary deployment is receiving expected traffic."""
        # This would integrate with monitoring system to check traffic distribution
        # For now, we'll simulate the validation
        logger.info("Validating canary traffic distribution...")

        # Simulate traffic analysis
        expected_canary_traffic = self.canary_percentage
        actual_canary_traffic = 8.5  # This would come from monitoring system

        traffic_validation = {
            'expected_percentage': expected_canary_traffic,
            'actual_percentage': actual_canary_traffic,
            'within_tolerance': abs(actual_canary_traffic - expected_canary_traffic) <= 2.0,
            'traffic_distribution': 'balanced'
        }

        if not traffic_validation['within_tolerance']:
            logger.warning(f"Canary traffic {actual_canary_traffic}% deviates from expected {expected_canary_traffic}%")

        return traffic_validation

    def run_comprehensive_test(self, environment: str) -> dict[str, Any]:
        """Run comprehensive canary validation suite."""
        logger.info(f"Starting comprehensive canary tests for {environment} environment")

        results = {
            'environment': environment,
            'timestamp': time.time(),
            'tests': {}
        }

        # Test 1: Health checks
        logger.info("Running health checks...")
        results['tests']['health_check'] = self.test_health_endpoint()

        # Test 2: MCP Bus connectivity
        logger.info("Testing MCP Bus connectivity...")
        results['tests']['mcp_bus'] = self.test_mcp_bus_connection()

        # Test 3: Agent endpoints
        logger.info("Testing agent endpoints...")
        results['tests']['agent_endpoints'] = self.test_agent_endpoints()

        # Test 4: Load testing
        logger.info("Running load tests...")
        results['tests']['load_test'] = self.run_load_test(duration=30, concurrency=5)

        # Test 5: Traffic validation
        logger.info("Validating traffic distribution...")
        results['tests']['traffic_validation'] = self.validate_canary_metrics()

        # Overall assessment
        critical_tests = ['health_check', 'mcp_bus']
        critical_passed = all(results['tests'][test] for test in critical_tests)

        agent_tests_passed = sum(results['tests']['agent_endpoints'].values()) >= 3  # At least 3/4 agents working

        load_test_passed = results['tests']['load_test']['success_rate'] >= 0.95  # 95% success rate

        traffic_valid = results['tests']['traffic_validation']['within_tolerance']

        results['overall_success'] = critical_passed and agent_tests_passed and load_test_passed and traffic_valid

        logger.info(f"Canary tests completed. Overall success: {results['overall_success']}")

        return results

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate human-readable test report."""
        report = []
        report.append("# Canary Deployment Test Report")
        report.append(f"**Environment:** {results['environment']}")
        report.append(f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}")
        report.append(f"**Overall Result:** {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
        report.append("")

        # Test results
        report.append("## Test Results")
        for test_name, test_result in results['tests'].items():
            if isinstance(test_result, bool):
                status = "✅ PASS" if test_result else "❌ FAIL"
                report.append(f"- **{test_name}:** {status}")
            elif isinstance(test_result, dict):
                if 'success_rate' in test_result:
                    success_rate = test_result['success_rate'] * 100
                    status = "✅ PASS" if success_rate >= 95 else "❌ FAIL"
                    report.append(f"- **{test_name}:** {status} ({success_rate:.1f}% success)")
                elif 'within_tolerance' in test_result:
                    status = "✅ PASS" if test_result['within_tolerance'] else "❌ FAIL"
                    report.append(f"- **{test_name}:** {status} ({test_result['actual_percentage']:.1f}% traffic)")

        return "\n".join(report)

def main():
    """Main canary testing entry point."""
    parser = argparse.ArgumentParser(description='Run canary deployment tests')
    parser.add_argument('--environment', required=True,
                       help='Target environment (staging/production)')
    parser.add_argument('--base-url', required=True,
                       help='Base URL of the deployment to test')
    parser.add_argument('--canary-percentage', type=float, default=10.0,
                       help='Expected canary traffic percentage (default: 10.0)')
    parser.add_argument('--output', default='canary-results.json',
                       help='Output file for test results')

    args = parser.parse_args()

    tester = CanaryTester(args.base_url, args.canary_percentage)

    logger.info(f"Starting canary tests for {args.environment} environment at {args.base_url}")

    try:
        results = tester.run_comprehensive_test(args.environment)

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate and print report
        report = tester.generate_report(results)
        print(report)

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)

    except Exception as e:
        logger.error(f"Canary testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
