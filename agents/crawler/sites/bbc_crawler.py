#!/usr/bin/env python3
"""Ultra-fast BBC crawler using Crawl4AI with heuristic filtering."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Any

import requests

from agents.common.evidence import enqueue_human_review, snapshot_paywalled_page
from agents.common.ingest import build_article_source_map_insert, build_source_upsert
from common.observability import get_logger

# Import shared utilities
from ..crawler_utils import RateLimiter
from .generic_site_crawler import GenericSiteCrawler, SiteConfig

logger = get_logger(__name__)

class UltraFastBBCCrawler:
    """Ultra-fast crawler optimized for high-volume BBC article harvesting."""

    def __init__(
        self,
        *,
        concurrent_requests: int = 6,
        requests_per_minute: int = 30,
        delay_between_requests: float = 1.0,
    ):
        self.concurrent_requests = concurrent_requests
        self.news_keywords = {
            "high_value": [
                "arrested",
                "charged",
                "court",
                "police",
                "sentenced",
                "convicted",
                "investigation",
                "crime",
                "murder",
                "theft",
                "assault",
                "fraud",
            ],
            "medium_value": [
                "council",
                "government",
                "minister",
                "mp",
                "mayor",
                "election",
                "announced",
                "confirmed",
                "reports",
                "statement",
                "official",
            ],
            "location_indicators": [
                "england",
                "uk",
                "britain",
                "london",
                "manchester",
                "birmingham",
                "leeds",
                "liverpool",
                "bristol",
            ],
        }

        source_config = {
            "id": "bbc_ultra_fast_england",
            "url": "https://www.bbc.co.uk/news/england",
            "domain": "bbc.co.uk",
            "name": "BBC",
            "description": "BBC England News feed (ultra-fast mode)",
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
            batch_size=max(6, concurrent_requests * 2),
        )
        self.crawler.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            delay_between_requests=delay_between_requests,
        )

    def calculate_news_score(self, title: str, content: str) -> float:
        """Fast heuristic news scoring (no AI needed)."""

        text = (title + " " + content).lower()
        score = 0.0

        for keyword in self.news_keywords["high_value"]:
            if keyword in text:
                score += 0.3

        for keyword in self.news_keywords["medium_value"]:
            if keyword in text:
                score += 0.15

        for location in self.news_keywords["location_indicators"]:
            if location in text:
                score += 0.1

        if len(content) > 200:
            score += 0.2
        if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", text):
            score += 0.1
        if "bbc" in text:
            score += 0.1

        return min(score, 1.0)

    def _apply_news_scoring(
        self,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply heuristic scoring and filter low-signal articles."""

        scored: list[dict[str, Any]] = []
        for article in articles:
            title = (article.get("title") or "").strip()
            content = (article.get("content") or "").strip()
            score = self.calculate_news_score(title, content)
            if score < 0.4 or len(content) <= 100:
                continue

            article.setdefault("extraction_metadata", {}).update(
                {
                    "news_score": score,
                    "filter": "ultra_fast_heuristic",
                }
            )
            article["confidence"] = score
            article["news_score"] = score
            scored.append(article)

        scored.sort(key=lambda item: item.get("news_score", 0.0), reverse=True)
        return scored

    async def _discover_articles(
        self,
        target_articles: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Discover candidate URLs and extract article metadata."""

        self.crawler.processed_urls.clear()
        candidate_urls = await self.crawler.get_article_urls(
            max_urls=max(20, target_articles * 3)
        )
        stats = {
            "candidate_urls": len(candidate_urls),
            "disallowed": 0,
            "failures": 0,
        }
        if not candidate_urls:
            return [], stats

        articles, disallowed, failures = await self.crawler._gather_articles(
            candidate_urls,
            max(target_articles * 2, target_articles),
        )
        stats["disallowed"] = disallowed
        stats["failures"] = failures
        return articles, stats

    async def run_ultra_fast_crawl(
        self,
        target_articles: int = 200,
        skip_ingestion: bool = False,
    ) -> dict[str, Any]:
        """Main ultra-fast crawling function."""

        start_time = time.time()
        logger.info("🚀 Ultra-Fast BBC Crawl: Target %d articles", target_articles)

        articles, stats = await self._discover_articles(target_articles)
        if not articles:
            processing_time = time.time() - start_time
            return {
                "ultra_fast_crawl": True,
                "target_articles": target_articles,
                "candidate_urls": stats["candidate_urls"],
                "extracted_articles": 0,
                "filtered_low_score": 0,
                "truncated_to_target": 0,
                "delivered_articles": 0,
                "disallowed": stats["disallowed"],
                "failures": stats["failures"],
                "processing_time_seconds": processing_time,
                "articles_per_second": 0.0,
                "timestamp": datetime.now().isoformat(),
                "articles": [],
            }

        scored_articles = self._apply_news_scoring(articles)
        selected = scored_articles[:target_articles]
        filtered_low_score = len(articles) - len(scored_articles)
        truncated_to_target = max(0, len(scored_articles) - len(selected))

        processing_time = time.time() - start_time
        articles_per_second = (
            len(selected) / processing_time if processing_time else 0.0
        )

        summary = {
            "ultra_fast_crawl": True,
            "target_articles": target_articles,
            "candidate_urls": stats["candidate_urls"],
            "extracted_articles": len(articles),
            "filtered_low_score": filtered_low_score,
            "truncated_to_target": truncated_to_target,
            "delivered_articles": len(selected),
            "disallowed": stats["disallowed"],
            "failures": stats["failures"],
            "processing_time_seconds": processing_time,
            "articles_per_second": articles_per_second,
            "timestamp": datetime.now().isoformat(),
            "articles": selected,
        }

        output_file = f"ultra_fast_bbc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
            logger.info("💾 Results saved to %s", output_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save results file %s: %s", output_file, exc)

        logger.info(
            "🎉 Ultra-fast crawl complete: %d articles delivered in %.1fs (%.2f/s)",
            len(selected),
            processing_time,
            articles_per_second,
        )

        summary["output_file"] = output_file

        MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

        if not skip_ingestion and selected:

            async def dispatch_ingest(article: dict[str, Any]) -> None:
                if article.get("paywall_flag"):
                    try:
                        html_stub = article.get("content", "")
                        metadata = {
                            "title": article.get("title"),
                            "domain": article.get("domain"),
                            "url": article.get("url"),
                            "timestamp": article.get("timestamp"),
                        }
                        evidence_path = snapshot_paywalled_page(
                            article.get("url"),
                            html_stub,
                            metadata,
                        )
                        enqueue_human_review(
                            evidence_path,
                            reviewer="chief_editor",
                        )
                        logger.info(
                            "Paywalled article snapshot saved and enqueued for review: %s",
                            article.get("url"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to snapshot/enqueue paywalled article %s: %s",
                            article.get("url"),
                            exc,
                        )
                    return

                article_payload = {
                    "url": article.get("url"),
                    "url_hash": article.get("url_hash"),
                    "domain": article.get("domain"),
                    "canonical": article.get("canonical"),
                    "publisher_meta": {"publisher": "BBC"},
                    "confidence": article.get("confidence", 0.5),
                    "paywall_flag": article.get("paywall_flag", False),
                    "extraction_metadata": article.get("extraction_metadata", {}),
                    "timestamp": article.get("timestamp"),
                }

                source_sql, source_params = build_source_upsert(article_payload)
                asm_sql, asm_params = build_article_source_map_insert(
                    article_payload.get("article_id", 1),
                    article_payload,
                )
                statements = [
                    [source_sql, list(source_params)],
                    [asm_sql, list(asm_params)],
                ]

                payload = {
                    "agent": "memory",
                    "tool": "ingest_article",
                    "args": [],
                    "kwargs": {
                        "article_payload": article_payload,
                        "statements": statements,
                    },
                }

                loop = asyncio.get_running_loop()

                def do_call() -> Any:
                    try:
                        response = requests.post(
                            f"{MCP_BUS_URL}/call",
                            json=payload,
                            timeout=(2, 10),
                        )
                        response.raise_for_status()
                        return response.json()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "DB worker call failed for %s: %s",
                            article.get("url"),
                            exc,
                        )
                        return None

                await loop.run_in_executor(None, do_call)

            try:
                tasks = [dispatch_ingest(article) for article in selected]
                await asyncio.gather(*tasks, return_exceptions=True)
                summary["ingest_dispatched"] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error dispatching ingest tasks: %s", exc)
                summary["ingest_dispatched"] = False
        else:
            summary["ingest_dispatched"] = False

        return summary

async def main() -> None:
    """Run ultra-fast crawler demo."""
    crawler = UltraFastBBCCrawler(concurrent_requests=6)
    await crawler.run_ultra_fast_crawl(target_articles=100, skip_ingestion=False)

if __name__ == "__main__":
    asyncio.run(main())
