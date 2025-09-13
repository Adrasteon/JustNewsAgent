#!/usr/bin/env python3
"""
Direct test of Scout Agent production crawler functionality
"""

import sys
import os
import asyncio

# Add the project root to the path
sys.path.insert(0, '/home/adra/justnewsagent/JustNewsAgent')

def test_production_crawler():
    """Test the production crawler directly"""
    print("Testing production crawler initialization...")

    try:
        # Import the production crawler orchestrator
        from agents.scout.production_crawlers.orchestrator import ProductionCrawlerOrchestrator
        print("✅ ProductionCrawlerOrchestrator imported successfully")

        # Create the orchestrator
        orchestrator = ProductionCrawlerOrchestrator()
        print("✅ ProductionCrawlerOrchestrator created successfully")

        # Test getting available sites
        available_sites = orchestrator.get_available_sites()
        print(f"✅ Available sites: {available_sites}")

        # Test getting supported modes
        supported_modes = orchestrator.get_supported_modes()
        print(f"✅ Supported modes: {supported_modes}")

        # Test the tools import
        from agents.scout.tools import get_production_crawler_info, PRODUCTION_CRAWLERS_AVAILABLE
        print(f"✅ Production crawlers available: {PRODUCTION_CRAWLERS_AVAILABLE}")

        # Test the info function
        info = get_production_crawler_info()
        print(f"✅ Production crawler info: {info}")

        print("\n🎉 All production crawler tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ultra_fast_crawl():
    """Test ultra-fast crawl functionality"""
    print("\nTesting ultra-fast crawl...")

    try:
        from agents.scout.tools import production_crawl_ultra_fast

        # Test with BBC
        async def _crawl():
            return await production_crawl_ultra_fast("bbc", 5)
        result = asyncio.run(_crawl())
        print(f"✅ Ultra-fast crawl result: {result.get('error', 'Success')}")
        return True

    except Exception as e:
        print(f"❌ Ultra-fast crawl test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Direct Production Crawler Test ===\n")

    # Test basic functionality
    success = test_production_crawler()
    if not success:
        print("❌ Basic functionality test failed")
        return

    # Test ultra-fast crawl
    test_ultra_fast_crawl()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
