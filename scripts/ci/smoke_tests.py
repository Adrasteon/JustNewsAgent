#!/usr/bin/env python3
"""
JustNewsAgent Smoke Tests
Basic functionality tests for deployed application.
"""

import sys
import time
import requests
import json
from typing import Dict, Any, Optional
import os

class SmokeTester:
    """Runs smoke tests against deployed JustNewsAgent."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        self.results: Dict[str, Any] = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'failures': []
        }

    def test_health_endpoint(self) -> bool:
        """Test basic health endpoint."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print("✅ Health endpoint responding")
                    return True
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
            return False

    def test_mcp_bus_registration(self) -> bool:
        """Test MCP bus agent registration."""
        try:
            response = self.session.get(f"{self.base_url}/agents")
            if response.status_code == 200:
                data = response.json()
                agents = data.get('agents', [])
                if len(agents) > 0:
                    print(f"✅ MCP bus has {len(agents)} registered agents")
                    return True
            print(f"❌ MCP bus registration failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ MCP bus registration error: {e}")
            return False

    def test_agent_communication(self) -> bool:
        """Test basic agent communication via MCP bus."""
        try:
            # Test scout agent if available
            payload = {
                "agent": "scout",
                "tool": "health_check",
                "args": [],
                "kwargs": {}
            }

            response = self.session.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print("✅ Agent communication working")
                    return True

            print(f"❌ Agent communication failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Agent communication error: {e}")
            return False

    def test_memory_agent(self) -> bool:
        """Test memory agent basic functionality."""
        try:
            payload = {
                "agent": "memory",
                "tool": "health_check",
                "args": [],
                "kwargs": {}
            }

            response = self.session.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print("✅ Memory agent responding")
                    return True

            print(f"❌ Memory agent failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Memory agent error: {e}")
            return False

    def test_database_connectivity(self) -> bool:
        """Test database connectivity via memory agent."""
        try:
            payload = {
                "agent": "memory",
                "tool": "test_connection",
                "args": [],
                "kwargs": {}
            }

            response = self.session.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print("✅ Database connectivity confirmed")
                    return True

            print(f"❌ Database connectivity failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Database connectivity error: {e}")
            return False

    def test_redis_connectivity(self) -> bool:
        """Test Redis connectivity."""
        try:
            payload = {
                "agent": "memory",
                "tool": "test_cache",
                "args": [],
                "kwargs": {}
            }

            response = self.session.post(
                f"{self.base_url}/call",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print("✅ Redis connectivity confirmed")
                    return True

            print(f"❌ Redis connectivity failed: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Redis connectivity error: {e}")
            return False

    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results."""
        self.results['tests_run'] += 1
        print(f"🧪 Running {test_name}...")

        try:
            result = test_func()
            if result:
                self.results['tests_passed'] += 1
                return True
            else:
                self.results['tests_failed'] += 1
                self.results['failures'].append(test_name)
                return False
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            self.results['tests_failed'] += 1
            self.results['failures'].append(f"{test_name} (crashed)")
            return False

    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        print("🚀 Starting JustNewsAgent smoke tests...")
        print(f"Target: {self.base_url}")
        print()

        # Wait a bit for services to be ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(5)

        tests = [
            ("Health Check", self.test_health_endpoint),
            ("MCP Bus Registration", self.test_mcp_bus_registration),
            ("Agent Communication", self.test_agent_communication),
            ("Memory Agent", self.test_memory_agent),
            ("Database Connectivity", self.test_database_connectivity),
            ("Redis Connectivity", self.test_redis_connectivity),
        ]

        all_passed = True
        for test_name, test_func in tests:
            result = self.run_test(test_name, test_func)
            if not result:
                all_passed = False
            print()

        return all_passed

    def report_results(self) -> None:
        """Report test results."""
        print("📊 Smoke Test Results:")
        print(f"  Tests Run: {self.results['tests_run']}")
        print(f"  Tests Passed: {self.results['tests_passed']}")
        print(f"  Tests Failed: {self.results['tests_failed']}")

        if self.results['failures']:
            print("\n❌ Failed Tests:")
            for failure in self.results['failures']:
                print(f"  - {failure}")

        success_rate = (self.results['tests_passed'] / self.results['tests_run']) * 100
        print(f"  Success Rate: {success_rate:.1f}%")

        if self.results['tests_failed'] == 0:
            print("✅ All smoke tests passed!")
        else:
            print("❌ Some smoke tests failed")

def main():
    """Main smoke test entry point."""
    # Get environment from args or environment variable
    environment = sys.argv[1] if len(sys.argv) > 1 else os.getenv('ENV', 'development')

    # Set base URL based on environment
    if environment == 'staging':
        base_url = "http://staging.justnews.example.com"
    elif environment == 'production':
        base_url = "http://api.justnews.example.com"
    else:
        base_url = "http://localhost:8000"

    tester = SmokeTester(base_url=base_url)
    success = tester.run_all_tests()
    tester.report_results()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()