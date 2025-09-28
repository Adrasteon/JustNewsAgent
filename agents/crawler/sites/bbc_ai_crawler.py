#!/usr/bin/env python3
"""BBC AI-enhanced crawler built on Crawl4AI with AI fallback."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any

from common.observability import get_logger

from ..crawler_utils import RateLimiter
from .generic_site_crawler import GenericSiteCrawler, SiteConfig

logger = get_logger(__name__)

class ProductionBBCCrawler:
    """Production BBC crawler using Crawl4AI-first extraction with AI fallback."""

    def __init__(
        self,
        *,
        concurrent_requests: int = 4,
        requests_per_minute: int = 20,
        delay_between_requests: float = 2.0,
    ):
        source_config = {
            "id": "bbc_ai_england",
            "url": "https://www.bbc.co.uk/news/england",
            "domain": "bbc.co.uk",
            "name": "BBC",
            "description": "BBC England News feed",
            "metadata": {
                "selectors": {
                    "article_links": ["a[href*='articles/']"],
                    "title": [
                        "h1",
                        "[data-component='headline']",
                        ".story-headline",
                        "[role='main'] h1",
                    ],
                    "content": [
                        "[data-component='text-block']",
                        ".story-body__inner",
                        "[role='main'] p",
                        "main p",
                        ".article-body p",
                        "[data-testid='paragraph']",
                    ],
                }
            },
        }
        self.site_config = SiteConfig(source_config)
        self.crawler = GenericSiteCrawler(
            self.site_config,
            concurrent_browsers=concurrent_requests,
            batch_size=max(4, concurrent_requests * 2),
        )
        self.crawler.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            delay_between_requests=delay_between_requests,
        )

    async def run_production_crawl(self, max_articles: int = 100) -> dict[str, Any]:
        """Crawl the BBC England feed and return structured metadata."""

        start_time = time.time()
        logger.info(
            "🚀 Starting Crawl4AI BBC AI crawl for up to %d articles",
            max_articles,
        )

        self.crawler.processed_urls.clear()

        candidate_urls = await self.crawler.get_article_urls(
            max_urls=max_articles * 3
        )
        candidate_count = len(candidate_urls)
        if not candidate_urls:
            logger.error("❌ No candidate URLs discovered for BBC feed")
            return {
                "production_crawl": True,
                "mode": "ai_enhanced",
                "candidate_urls": 0,
                "successful_articles": 0,
                "processing_time_seconds": time.time() - start_time,
                "articles_per_second": 0.0,
                "articles": [],
            }

        articles, disallowed, failures = await self.crawler._gather_articles(
            candidate_urls,
            max_articles,
        )

        processing_time = time.time() - start_time
        articles_per_second = (
            len(articles) / processing_time if processing_time else 0.0
        )

        method_breakdown: dict[str, int] = {}
        for article in articles:
            method = article.get("extraction_method", "unknown")
            method_breakdown[method] = method_breakdown.get(method, 0) + 1

        summary = {
            "production_crawl": True,
            "mode": "ai_enhanced",
            "candidate_urls": candidate_count,
            "successful_articles": len(articles),
            "disallowed": disallowed,
            "failures": failures,
            "processing_time_seconds": processing_time,
            "articles_per_second": articles_per_second,
            "extraction_breakdown": method_breakdown,
            "timestamp": datetime.now().isoformat(),
            "articles": articles,
        }

        output_file = (
            f"production_bbc_ai_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
            logger.info("💾 Results saved to %s", output_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save results file %s: %s", output_file, exc)

        logger.info(
            "🎉 AI-enhanced crawl complete: %d articles in %.1fs (%.2f/s)",
            len(articles),
            processing_time,
            articles_per_second,
        )
        logger.info("📊 Extraction breakdown: %s", method_breakdown)

        return summary


async def main() -> None:
    """Execute a sample production crawl for manual testing."""

    crawler = ProductionBBCCrawler(concurrent_requests=4)
    await crawler.run_production_crawl(max_articles=50)

if __name__ == "__main__":
    asyncio.run(main())
