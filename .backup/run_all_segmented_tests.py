#!/usr/bin/env python3
"""
JustNewsAgent Complete Segmented Test Runner

Purpose: Execute all segmented tests in proper sequence with live data.

Usage:
    python run_all_segmented_tests.py

Requirements:
- All agents running and healthy
- PostgreSQL database available
- Training coordinator operational

Success Criteria:
- All test segments pass
- End-to-end workflow validation
- Performance metrics collected
- Training loop integration verified
"""

import subprocess
import time
import logging
import json
import sys
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteTestRunner:
    def __init__(self):
        self.test_scripts = [
            "test_crawler_segmented.py",
            "test_analyst_segmented.py",
            "test_fact_checker_segmented.py",
            "test_synthesizer_segmented.py",
            "test_critic_segmented.py",
            "test_chief_editor_segmented.py",
            "test_reasoning_segmented.py",
            "test_newsreader_segmented.py",
            "test_training_loop.py"
        ]

    def run_single_test(self, script_name: str) -> Dict[str, Any]:
        """Run a single test script and capture results"""
        logger.info(f"🚀 Running {script_name}")

        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Try to parse JSON output
                try:
                    output_data = json.loads(result.stdout.strip())
                    return {
                        "script": script_name,
                        "success": True,
                        "execution_time": execution_time,
                        "metrics": output_data,
                        "error": None
                    }
                except json.JSONDecodeError:
                    return {
                        "script": script_name,
                        "success": True,
                        "execution_time": execution_time,
                        "output": result.stdout.strip(),
                        "error": None
                    }
            else:
                return {
                    "script": script_name,
                    "success": False,
                    "execution_time": execution_time,
                    "error": result.stderr.strip(),
                    "output": result.stdout.strip()
                }

        except subprocess.TimeoutExpired:
            return {
                "script": script_name,
                "success": False,
                "execution_time": 600,
                "error": "Test timed out after 10 minutes",
                "output": None
            }
        except Exception as e:
            return {
                "script": script_name,
                "success": False,
                "execution_time": 0,
                "error": str(e),
                "output": None
            }

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test scripts in sequence"""
        logger.info("🎯 Starting Complete Segmented Test Suite")

        start_time = time.time()
        results = []

        for script in self.test_scripts:
            result = self.run_single_test(script)
            results.append(result)

            if not result["success"]:
                logger.warning(f"❌ {script} failed: {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"✅ {script} passed in {result['execution_time']:.2f}s")

            # Brief pause between tests
            time.sleep(2)

        # Calculate overall metrics
        total_time = time.time() - start_time
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)

        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "average_test_time": total_time / total_tests if total_tests > 0 else 0,
            "test_results": results,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"🏁 Test suite completed: {successful_tests}/{total_tests} tests passed")
        return summary

    def save_results(self, results: Dict[str, Any], filename: str = "segmented_test_results.json"):
        """Save test results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {filename}")

if __name__ == "__main__":
    runner = CompleteTestRunner()
    results = runner.run_all_tests()
    runner.save_results(results)

    # Print summary
    print("\n" + "="*60)
    print("SEGMENTED TEST SUITE SUMMARY")
    print("="*60)
    print(f"Tests Run: {results['total_tests']}")
    print(f"Tests Passed: {results['successful_tests']}")
    print(".1f")
    print(".2f")
    print("="*60)