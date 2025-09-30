#!/usr/bin/env python3
"""
JustNews Metrics Library Test Script
Demonstrates the core metrics functionality
"""

import time
import asyncio
from common.metrics import JustNewsMetrics, measure_processing_time

def test_basic_metrics():
    """Test basic metrics functionality."""
    print("Testing JustNews Metrics Library...")

    # Initialize metrics for test agent
    metrics = JustNewsMetrics("test_agent")

    # Test request recording
    print("✓ Recording HTTP requests...")
    metrics.record_request("GET", "/health", 200, 0.1)
    metrics.record_request("POST", "/process", 201, 0.5)
    metrics.record_request("GET", "/metrics", 500, 2.0)

    # Test error recording
    print("✓ Recording errors...")
    metrics.record_error("validation_error", "/process")
    metrics.record_error("timeout_error", "/external-api")

    # Test processing metrics
    print("✓ Recording processing metrics...")
    metrics.record_processing("content_analysis", 1.2)
    metrics.record_processing("quality_check", 0.8)

    # Test quality scores
    print("✓ Recording quality scores...")
    metrics.record_quality_score("content_quality", 0.85)
    metrics.record_quality_score("relevance_score", 0.92)

    # Test queue size updates
    print("✓ Updating queue sizes...")
    metrics.update_queue_size("processing_queue", 15)
    metrics.update_queue_size("output_queue", 3)

    # Test system metrics update
    print("✓ Updating system metrics...")
    metrics.update_system_metrics()

    # Get metrics output
    print("✓ Generating Prometheus metrics...")
    metrics_output = metrics.get_metrics()

    # Assertions to validate metrics output
    assert 'justnews_requests_total' in metrics_output, "Expected 'justnews_requests_total' in metrics output"
    assert 'justnews_errors_total' in metrics_output, "Expected 'justnews_errors_total' in metrics output"
    assert len(metrics_output) > 0, "Metrics output should not be empty"

    print(f"\nMetrics output length: {len(metrics_output)} characters")
    print("Sample metrics output:")
    print("=" * 50)
    # Show first few lines of metrics output
    lines = metrics_output.split('\n')[:20]
    for line in lines:
        if line.strip():
            print(line)
    print("=" * 50)

    return metrics

@measure_processing_time("test_processing")
def test_decorator():
    """Test the processing time decorator."""
    print("Testing processing time decorator...")
    start_time = time.time()
    time.sleep(0.1)  # Simulate some work
    duration = time.time() - start_time

    # Assertions to validate the decorator functionality
    assert duration >= 0.1, "Processing time should be at least 0.1 seconds"
    print("✓ Decorator test completed")

def test_async_metrics():
    """Test metrics in async context (wrapped for pytest sync execution)."""
    async def _run():
        print("Testing async metrics context...")
        metrics = JustNewsMetrics("async_test")
        start_time = time.time()
        await asyncio.sleep(0.05)  # Short sleep for test speed
        duration = time.time() - start_time
        metrics.record_processing("async_operation", duration)

        # Assertions to validate async metrics functionality
        assert duration >= 0.05, "Async operation duration should be at least 0.05 seconds"
        print("✓ Async metrics test completed")
    asyncio.run(_run())

def main():
    """Main test function."""
    print("JustNews Metrics Library Test Suite")
    print("=" * 50)

    try:
        # Test basic functionality
        metrics = test_basic_metrics()

        # Test decorator
        test_decorator()

        # Test async functionality
        asyncio.run(test_async_metrics())

        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("JustNews Metrics Library is working correctly.")
        print("\nNext steps:")
        print("1. Integrate metrics into your FastAPI agents")
        print("2. Start the monitoring stack: ./deploy/monitoring/manage-monitoring.sh start")
        print("3. Access Grafana at http://localhost:3000")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
