#!/usr/bin/env python3
"""
Direct Article Crawler Script

Bypasses the broken crawler agent and directly crawls articles using the working
unified crawler logic. This script can achieve the 1000 article target.

Usage:
    python direct_crawler.py --target 1000 --batch-size 50

Requirements:
- Database connection (PostgreSQL)
- Memory agent running (for article storage)
- Network access for crawling
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('direct_crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database and environment setup
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_DB'] = 'justnews'
os.environ['POSTGRES_USER'] = 'justnews_user'
os.environ['POSTGRES_PASSWORD'] = 'password123'

from agents.crawler.unified_production_crawler import UnifiedProductionCrawler
from agents.crawler.crawler_utils import get_active_sources


class DirectArticleCrawler:
    """Direct crawler that bypasses the broken agent"""

    def __init__(self, target_articles: int = 1000, batch_size: int = 50):
        self.target_articles = target_articles
        self.batch_size = batch_size
        self.crawler = None
        self.total_crawled = 0
        self.start_time = None

    async def initialize(self):
        """Initialize the crawler"""
        logger.info("Initializing direct article crawler...")
        self.crawler = UnifiedProductionCrawler()
        await self.crawler._load_ai_models()
        self.start_time = datetime.now()
        logger.info(f"Target: {self.target_articles} articles")

    async def get_sources_batch(self) -> List[str]:
        """Get a batch of source domains to crawl"""
        try:
            sources = get_active_sources()
            # Shuffle to avoid always hitting the same sites
            import random
            random.shuffle(sources)

            # Return domains, limited to batch size
            domains = [s['domain'] for s in sources[:self.batch_size]]
            logger.info(f"Selected {len(domains)} domains for this batch")
            return domains

        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            return []

    async def crawl_batch(self, domains: List[str]) -> Dict[str, Any]:
        """Crawl a batch of domains"""
        logger.info(f"Starting crawl batch with {len(domains)} domains...")

        try:
            result = await self.crawler.run_unified_crawl(
                domains,
                max_articles_per_site=10,  # Conservative per site
                concurrent_sites=min(5, len(domains))  # Limit concurrency
            )

            batch_articles = result.get('total_articles', 0)
            self.total_crawled += batch_articles

            logger.info(f"Batch complete: {batch_articles} articles (total: {self.total_crawled})")
            return result

        except Exception as e:
            logger.error(f"Batch crawl failed: {e}")
            return {"error": str(e), "total_articles": 0}

    async def run_crawl_loop(self):
        """Main crawl loop to reach target"""
        logger.info("Starting main crawl loop...")

        batch_num = 1
        while self.total_crawled < self.target_articles:
            logger.info(f"\n--- Batch {batch_num} ---")
            logger.info(f"Progress: {self.total_crawled}/{self.target_articles} articles")

            # Get sources for this batch
            domains = await self.get_sources_batch()
            if not domains:
                logger.error("No domains available, stopping")
                break

            # Crawl the batch
            result = await self.crawl_batch(domains)

            # Check if we got any articles
            if result.get('total_articles', 0) == 0:
                logger.warning("No articles found in this batch, continuing...")

            # Safety check - don't run forever
            if batch_num > 50:  # Emergency stop after 50 batches
                logger.error("Emergency stop: too many batches without reaching target")
                break

            batch_num += 1

            # Small delay between batches
            await asyncio.sleep(2)

        # Final status
        duration = datetime.now() - self.start_time
        articles_per_second = self.total_crawled / duration.total_seconds() if duration.total_seconds() > 0 else 0

        logger.info("\n" + "="*50)
        logger.info("DIRECT CRAWLER COMPLETE")
        logger.info(f"Articles crawled: {self.total_crawled}")
        logger.info(f"Target: {self.target_articles}")
        logger.info(".1f")
        logger.info(".2f")
        logger.info("="*50)

        return {
            "total_articles": self.total_crawled,
            "target_reached": self.total_crawled >= self.target_articles,
            "duration_seconds": duration.total_seconds(),
            "articles_per_second": articles_per_second,
            "batches_completed": batch_num - 1
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.crawler:
            await self.crawler._cleanup()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Direct Article Crawler")
    parser.add_argument("--target", type=int, default=1000, help="Target number of articles")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of sites per batch")

    args = parser.parse_args()

    crawler = DirectArticleCrawler(
        target_articles=args.target,
        batch_size=args.batch_size
    )

    try:
        await crawler.initialize()
        result = await crawler.run_crawl_loop()

        # Exit with success/failure code
        if result['target_reached']:
            logger.info("✅ SUCCESS: Target reached!")
            sys.exit(0)
        else:
            logger.warning("⚠️ PARTIAL SUCCESS: Did not reach target")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        sys.exit(2)
    finally:
        await crawler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())