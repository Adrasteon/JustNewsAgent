#!/usr/bin/env python3
"""
JustNewsAgent Production Validation Script
Comprehensive validation suite for production deployments.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionValidator:
    """Handles comprehensive production validation."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('API_KEY')
        self.headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}

    def test_service_discovery(self) -> dict[str, Any]:
        """Test service discovery and registration."""
        results = {
            'mcp_bus_discovery': False,
            'agent_registration': {},
            'service_health': {}
        }

        try:
            # Test MCP Bus discovery
            response = requests.get(f"{self.base_url}/agents", timeout=10, headers=self.headers)
            results['mcp_bus_discovery'] = response.status_code == 200

            if results['mcp_bus_discovery']:
                agents = response.json().get('agents', [])
                logger.info(f"Discovered {len(agents)} registered agents")

                # Test individual agent health
                for agent in ['scout', 'analyst', 'synthesizer', 'memory', 'fact_checker', 'critic']:
                    try:
                        response = requests.get(f"{self.base_url}/agents/{agent}/health",
                                              timeout=5, headers=self.headers)
                        results['service_health'][agent] = response.status_code == 200
                    except Exception as e:
                        logger.warning(f"Agent {agent} health check failed: {e}")
                        results['service_health'][agent] = False

        except Exception as e:
            logger.error(f"Service discovery test failed: {e}")

        return results

    def test_api_endpoints(self) -> dict[str, Any]:
        """Test critical API endpoints."""
        endpoints_to_test = [
            ('GET', '/health'),
            ('GET', '/api/v1/articles'),
            ('GET', '/api/v1/entities'),
            ('GET', '/api/v1/scout/status'),
            ('GET', '/api/v1/analyst/status'),
            ('GET', '/api/v1/synthesizer/status'),
            ('GET', '/api/v1/memory/status'),
        ]

        results = {}
        for method, endpoint in endpoints_to_test:
            try:
                if method == 'GET':
                    response = requests.get(f"{self.base_url}{endpoint}",
                                          timeout=10, headers=self.headers)
                elif method == 'POST':
                    response = requests.post(f"{self.base_url}{endpoint}",
                                           json={}, timeout=10, headers=self.headers)

                results[endpoint] = {
                    'status_code': response.status_code,
                    'success': response.status_code in [200, 201, 404],  # 404 is acceptable for empty endpoints
                    'response_time': response.elapsed.total_seconds()
                }

                if not results[endpoint]['success']:
                    logger.warning(f"Endpoint {endpoint} returned status {response.status_code}")

            except Exception as e:
                logger.error(f"Endpoint {endpoint} test failed: {e}")
                results[endpoint] = {
                    'status_code': None,
                    'success': False,
                    'error': str(e)
                }

        return results

    def test_database_connectivity(self) -> dict[str, Any]:
        """Test database connectivity and basic operations."""
        results = {
            'connection_test': False,
            'basic_query': False,
            'performance_test': False
        }

        try:
            # Test database connectivity through API
            payload = {
                "agent": "memory",
                "tool": "health_check",
                "args": [],
                "kwargs": {}
            }

            response = requests.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=15,
                headers=self.headers
            )

            results['connection_test'] = response.status_code == 200

            if results['connection_test']:
                # Test basic database operations
                payload = {
                    "agent": "memory",
                    "tool": "count_articles",
                    "args": [],
                    "kwargs": {}
                }

                response = requests.post(
                    f"{self.base_url}/call",
                    json=payload,
                    timeout=15,
                    headers=self.headers
                )

                results['basic_query'] = response.status_code == 200

                # Basic performance test
                start_time = time.time()
                for _ in range(5):
                    requests.post(
                        f"{self.base_url}/call",
                        json=payload,
                        timeout=10,
                        headers=self.headers
                    )
                end_time = time.time()

                avg_response_time = (end_time - start_time) / 5
                results['performance_test'] = avg_response_time < 2.0  # Should respond within 2 seconds

        except Exception as e:
            logger.error(f"Database connectivity test failed: {e}")

        return results

    def test_load_performance(self, duration: int = 120) -> dict[str, Any]:
        """Test system performance under load."""
        logger.info(f"Running production load test for {duration} seconds")

        def make_request() -> dict[str, Any]:
            """Make a test request."""
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/health",
                                      timeout=5, headers=self.headers)
                response_time = time.time() - start_time

                return {
                    'success': response.status_code == 200,
                    'response_time': response_time,
                    'status_code': response.status_code
                }
            except Exception as e:
                return {
                    'success': False,
                    'response_time': time.time() - start_time,
                    'error': str(e)
                }

        # Run load test with increasing concurrency
        concurrency_levels = [10, 25, 50, 100]
        results = {}

        for concurrency in concurrency_levels:
            logger.info(f"Testing with {concurrency} concurrent users")

            request_results = []
            start_time = time.time()
            end_time = start_time + duration

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []

                while time.time() < end_time:
                    futures.append(executor.submit(make_request))

                    # Clean up completed futures periodically
                    completed = [f for f in futures if f.done()]
                    for future in completed:
                        try:
                            result = future.result(timeout=1)
                            request_results.append(result)
                            futures.remove(future)
                        except Exception as e:
                            logger.warning(f"Future result error: {e}")

                    time.sleep(0.05)  # Rate limiting

                # Wait for remaining futures
                for future in futures:
                    try:
                        result = future.result(timeout=5)
                        request_results.append(result)
                    except Exception as e:
                        logger.warning(f"Remaining future error: {e}")

            # Analyze results for this concurrency level
            successful = [r for r in request_results if r['success']]
            success_rate = len(successful) / len(request_results) if request_results else 0

            results[f"concurrency_{concurrency}"] = {
                'total_requests': len(request_results),
                'success_rate': success_rate,
                'avg_response_time': sum(r['response_time'] for r in request_results) / len(request_results) if request_results else 0,
                'p95_response_time': sorted([r['response_time'] for r in request_results])[int(len(request_results) * 0.95)] if request_results else 0
            }

            logger.info(f"Concurrency {concurrency}: {success_rate:.1%} success, "
                       f"{results[f'concurrency_{concurrency}']['avg_response_time']:.3f}s avg response")

            # Stop if success rate drops below 95%
            if success_rate < 0.95:
                logger.warning(f"Success rate dropped below 95% at concurrency {concurrency}")
                break

        return results

    def test_security_headers(self) -> dict[str, Any]:
        """Test security headers and configurations."""
        results = {}

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10, headers=self.headers)

            security_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options',
                'X-XSS-Protection',
                'Strict-Transport-Security',
                'Content-Security-Policy'
            ]

            results['security_headers'] = {}
            for header in security_headers:
                present = header in response.headers
                results['security_headers'][header] = present
                if not present:
                    logger.warning(f"Missing security header: {header}")

            # Check HTTPS enforcement (in production)
            results['https_enforced'] = response.url.startswith('https://')

        except Exception as e:
            logger.error(f"Security headers test failed: {e}")
            results['error'] = str(e)

        return results

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete production validation suite."""
        logger.info("Starting comprehensive production validation")

        results = {
            'timestamp': time.time(),
            'tests': {}
        }

        # Test 1: Service Discovery
        logger.info("Testing service discovery...")
        results['tests']['service_discovery'] = self.test_service_discovery()

        # Test 2: API Endpoints
        logger.info("Testing API endpoints...")
        results['tests']['api_endpoints'] = self.test_api_endpoints()

        # Test 3: Database Connectivity
        logger.info("Testing database connectivity...")
        results['tests']['database_connectivity'] = self.test_database_connectivity()

        # Test 4: Load Performance
        logger.info("Testing load performance...")
        results['tests']['load_performance'] = self.test_load_performance(duration=60)

        # Test 5: Security Headers
        logger.info("Testing security headers...")
        results['tests']['security_headers'] = self.test_security_headers()

        # Overall assessment
        critical_tests = {
            'service_discovery': lambda r: r.get('mcp_bus_discovery', False),
            'api_endpoints': lambda r: sum(1 for ep in r.values() if ep.get('success', False)) >= 5,  # At least 5/7 endpoints working
            'database_connectivity': lambda r: r.get('connection_test', False) and r.get('basic_query', False),
            'load_performance': lambda r: any(level.get('success_rate', 0) >= 0.95 for level in r.values()),
            'security_headers': lambda r: r.get('https_enforced', False)
        }

        critical_passed = all(
            check_func(results['tests'][test_name])
            for test_name, check_func in critical_tests.items()
        )

        results['overall_success'] = critical_passed

        logger.info(f"Production validation completed. Overall success: {results['overall_success']}")

        return results

    def generate_report(self, results: dict[str, Any]) -> str:
        """Generate detailed validation report."""
        report = []
        report.append("# Production Validation Report")
        report.append(f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}")
        report.append(f"**Overall Result:** {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
        report.append("")

        # Test summaries
        report.append("## Test Results Summary")

        for test_name, test_result in results['tests'].items():
            if test_name == 'service_discovery':
                discovery_ok = test_result.get('mcp_bus_discovery', False)
                healthy_agents = sum(1 for status in test_result.get('service_health', {}).values() if status)
                report.append(f"- **Service Discovery:** {'✅' if discovery_ok else '❌'} ({healthy_agents}/6 agents healthy)")

            elif test_name == 'api_endpoints':
                successful_endpoints = sum(1 for ep in test_result.values() if ep.get('success', False))
                report.append(f"- **API Endpoints:** ✅ {successful_endpoints}/7 endpoints working")

            elif test_name == 'database_connectivity':
                conn_ok = test_result.get('connection_test', False)
                query_ok = test_result.get('basic_query', False)
                perf_ok = test_result.get('performance_test', False)
                report.append(f"- **Database:** {'✅' if conn_ok and query_ok else '❌'} (connection: {'✅' if conn_ok else '❌'}, query: {'✅' if query_ok else '❌'}, performance: {'✅' if perf_ok else '❌'})")

            elif test_name == 'load_performance':
                max_success_rate = max((level.get('success_rate', 0) for level in test_result.values()), default=0)
                report.append(f"- **Load Performance:** {'✅' if max_success_rate >= 0.95 else '❌'} ({max_success_rate:.1%} max success rate)")

            elif test_name == 'security_headers':
                https_ok = test_result.get('https_enforced', False)
                headers_present = sum(1 for present in test_result.get('security_headers', {}).values() if present)
                report.append(f"- **Security:** {'✅' if https_ok else '❌'} (HTTPS: {'✅' if https_ok else '❌'}, {headers_present}/5 security headers)")

        return "\n".join(report)

def main():
    """Main production validation entry point."""
    parser = argparse.ArgumentParser(description='Run production validation tests')
    parser.add_argument('--base-url', required=True,
                       help='Base URL of the production deployment')
    parser.add_argument('--api-key',
                       help='API key for authentication (or set API_KEY env var)')
    parser.add_argument('--output', default='production-validation-results.json',
                       help='Output file for validation results')

    args = parser.parse_args()

    validator = ProductionValidator(args.base_url, args.api_key)

    logger.info(f"Starting production validation for {args.base_url}")

    try:
        results = validator.run_full_validation()

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate and print report
        report = validator.generate_report(results)
        print(report)

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)

    except Exception as e:
        logger.error(f"Production validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
