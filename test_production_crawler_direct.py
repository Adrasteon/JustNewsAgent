#!/usr/bin/env python3
"""
Direct test of Scout Agent production crawler functionality
"""

import sys
import asyncio

# Add the project root to the path
sys.path.insert(0, '/home/adra/justnewsagent/JustNewsAgent')

def test_production_crawler():
    """Test the production crawler directly"""
    print("Testing production crawler initialization...")

    # Import the production crawler orchestrator
    from agents.scout.production_crawlers.orchestrator import ProductionCrawlerOrchestrator
    print("✅ ProductionCrawlerOrchestrator imported successfully")

    # Create the orchestrator
    orchestrator = ProductionCrawlerOrchestrator()
    print("✅ ProductionCrawlerOrchestrator created successfully")

    # Test getting available sites
    available_sites = orchestrator.get_available_sites()
    print(f"✅ Available sites: {available_sites}")
    assert isinstance(available_sites, (list, tuple)), "Expected available_sites to be a list or tuple"

    # Test getting supported modes
    supported_modes = orchestrator.get_supported_modes()
    print(f"✅ Supported modes: {supported_modes}")
    assert isinstance(supported_modes, (list, tuple)), "Expected supported_modes to be a list or tuple"

    # Test the tools import
    from agents.scout.tools import get_production_crawler_info, PRODUCTION_CRAWLERS_AVAILABLE
    print(f"✅ Production crawlers available: {PRODUCTION_CRAWLERS_AVAILABLE}")

    # Test the info function
    info = get_production_crawler_info()
    print(f"✅ Production crawler info: {info}")
    assert isinstance(info, dict), "Expected production crawler info to be a dict"

    print("\n🎉 All production crawler tests passed!")


def test_ultra_fast_crawl():
    """Test ultra-fast crawl functionality"""
    print("\nTesting ultra-fast crawl...")

    from agents.scout.tools import production_crawl_ultra_fast

    # Test with BBC
    async def _crawl():
        return await production_crawl_ultra_fast("bbc", 5)
    result = asyncio.run(_crawl())
    print(f"✅ Ultra-fast crawl result: {result.get('error', 'Success')}")

    # Assert success or raise if there was an error
    assert isinstance(result, dict), "Expected result to be a dict"
    assert not result.get('error'), f"Ultra-fast crawl reported an error: {result.get('error')}"

def main():
    print("=== Direct Production Crawler Test ===\n")

    # Test basic functionality
    test_production_crawler()
    test_ultra_fast_crawl()

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
