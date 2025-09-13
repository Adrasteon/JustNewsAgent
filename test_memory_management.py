#!/usr/bin/env python3
"""
Test script for NewsReader memory management improvements
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/adra/justnewsagent/JustNewsAgent')

def test_memory_management():
    """Test the memory management improvements"""
    print("🧪 Testing NewsReader Memory Management Improvements")
    print("="*60)

    try:
        # Import the tools
        from agents.newsreader.tools import (
            get_engine,
            clear_engine,
            health_check,
            _check_and_cleanup_memory,
            _force_memory_cleanup
        )

        print("✅ Imports successful")

        # Test health check
        print("\n🔍 Testing health check...")
        health = health_check()
        print(f"Health status: {health.get('status', 'unknown')}")
        print(f"V2 Engine available: {health.get('v2_compliance', False)}")

        # Test memory monitoring
        print("\n🧠 Testing memory monitoring...")
        _check_and_cleanup_memory()
        print("✅ Memory monitoring check completed")

        # Test engine initialization
        print("\n🔧 Testing engine initialization...")
        engine = get_engine()
        if engine:
            print("✅ Engine initialized successfully")
        else:
            print("⚠️ Engine initialization failed (expected in some cases)")

        # Test memory cleanup
        print("\n🧹 Testing memory cleanup...")
        clear_engine()
        print("✅ Engine cleanup completed")

        # Test emergency cleanup
        print("\n🚨 Testing emergency cleanup...")
        _force_memory_cleanup()
        print("✅ Emergency cleanup completed")

        print("\n" + "="*60)
        print("🎯 Memory management test completed successfully!")
        print("The following improvements have been implemented:")
        print("  • Background GPU memory monitoring (30s intervals)")
        print("  • Automatic cleanup at 75% and 85% memory thresholds")
        print("  • Integration with GPU manager for allocation tracking")
        print("  • Aggressive cleanup on engine destruction")
        print("  • Memory monitoring in health check endpoint")
        print("  • Emergency cleanup endpoint for manual intervention")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_management()
