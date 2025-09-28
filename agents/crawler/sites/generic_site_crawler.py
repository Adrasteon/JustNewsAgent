#!/usr/bin/env python3
"""Generic multi-site crawler with Crawl4AI-first strategy.

This module provides:
- SiteConfig: per-site selector configuration
- GenericSiteCrawler: article discovery + extraction for a single site
- MultiSiteCrawler: orchestration across multiple sites concurrently

Strategy:
1. Prefer Crawl4AI for link discovery & article extraction (fast, cleaner text)
2. Gracefully fall back to Crawl4AI screenshot + LLaVA when direct extraction fails
3. Emit canonical metadata objects expected by downstream agents
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

try:  # pragma: no cover
    from crawl4ai import (  # type: ignore
        AsyncWebCrawler,
        CacheMode,
        CrawlerRunConfig,
        LLMConfig,
        VirtualScrollConfig,
    )
    from crawl4ai.extraction_strategy import LLMExtractionStrategy  # type: ignore
    _CRAWL4AI_AVAILABLE = True
except Exception:  # noqa: BLE001
    AsyncWebCrawler = None  # type: ignore
    CacheMode = None  # type: ignore
    CrawlerRunConfig = None  # type: ignore
    LLMConfig = None  # type: ignore
    VirtualScrollConfig = None  # type: ignore
    LLMExtractionStrategy = None  # type: ignore
    _CRAWL4AI_AVAILABLE = False

# from ...tools import _record_scout_performance  # Removed - not needed in Crawler agent
from common.observability import get_logger

from ...newsreader.tools import extract_news_from_url
from ..crawler_utils import (
    CanonicalMetadata,
    RateLimiter,
    RobotsChecker,
    get_active_sources,
    get_sources_by_domain,
)

logger = get_logger(__name__)


class SiteConfig:
    """Configuration for a specific news site."""

    def __init__(self, source_data: dict[str, Any]):
        self.source_id = source_data.get("id")
        self.url = source_data.get("url")
        self.domain = source_data.get("domain")
        self.name = source_data.get("name", "Unknown")
        self.description = source_data.get("description", "")
        self.metadata = source_data.get("metadata", {})

        # Default selectors (override via metadata["selectors"])
        self.article_link_selectors = [
            "a[href*='article']",
            "a[href*='news']",
            "a[href*='story']",
            "a[href*='/202']",  # Date patterns
            ".headline a",
            ".story-link",
            "[data-testid*='article'] a",
        ]
        self.title_selectors = [
            "h1",
            "[data-component='headline']",
            ".story-headline",
            ".article-title",
            "[role='main'] h1",
        ]
        self.content_selectors = [
            "[data-component='text-block']",
            ".story-body__inner",
            "[role='main'] p",
            "main p",
            ".article-body p",
            "[data-testid='paragraph']",
            ".content p",
        ]

        if "selectors" in self.metadata:
            selectors = self.metadata["selectors"]
            self.article_link_selectors = selectors.get(
                "article_links", self.article_link_selectors
            )
            self.title_selectors = selectors.get("title", self.title_selectors)
            self.content_selectors = selectors.get("content", self.content_selectors)

    def get_base_url(self) -> str:
        parsed = urlparse(self.url)
        return f"{parsed.scheme}://{parsed.netloc}"


class GenericSiteCrawler:
    """Crawler for a single site using Crawl4AI-first strategy with AI fallback."""

    _DATE_PATTERN = re.compile(r"/[0-9]{4}/[0-9]{2}/[0-9]{2}/")
    _EXCLUDED_KEYWORDS = ("video", "live", "audio", "sport", "gallery", "podcast")
    _ARTICLE_KEYWORDS = ("article", "story", "news")

    def __init__(
        self,
        site_config: SiteConfig,
        concurrent_browsers: int = 2,
        batch_size: int = 8,
    ):
        self.site_config = site_config
        self.concurrent_browsers = concurrent_browsers
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter()
        self.robots_checker = RobotsChecker()
        self.processed_urls: set[str] = set()
        self.session_start_time = time.time()

    def _normalize_href(self, href: str, base: str) -> str | None:
        value = href.strip()
        if not value:
            return None
        if value.startswith("/"):
            value = urljoin(base, value)
        if not value.startswith("http"):
            return None
        return value.split("#")[0]

    def _looks_like_article(self, href: str) -> bool:
        lower = href.lower()
        if any(token in lower for token in self._EXCLUDED_KEYWORDS):
            return False
        if self._DATE_PATTERN.search(href):
            return True
        return any(token in lower for token in self._ARTICLE_KEYWORDS)

    def _extract_links_from_html(self, html: str, base: str, max_urls: int) -> list[str]:
        candidates = re.findall(r'href=["\']([^"\']+)["\']', html)
        urls: list[str] = []
        seen: set[str] = set()

        for candidate in candidates:
            normalized = self._normalize_href(candidate, base)
            if not normalized or normalized in seen:
                continue
            if not self._looks_like_article(normalized):
                continue
            urls.append(normalized)
            seen.add(normalized)
            if len(urls) >= max_urls:
                break

        return urls

    def _create_virtual_scroll_config(self) -> Any:
        if not VirtualScrollConfig:
            return None
        return VirtualScrollConfig(scroll_count=30, scroll_by=800, wait_after_scroll=0.6)

    def _create_run_config(
        self,
        virtual_config: Any,
    ) -> Any:
        if not CrawlerRunConfig:
            return None

        kwargs: dict[str, Any] = {}
        if CacheMode:
            kwargs["cache_mode"] = CacheMode.BYPASS
        if virtual_config:
            kwargs["virtual_scroll_config"] = virtual_config
        if not kwargs:
            return None
        return CrawlerRunConfig(**kwargs)

    async def _run_crawl(
        self,
    *,
    url: str,
    config: Any = None,
    **kwargs: Any,
    ) -> Any:
        if not AsyncWebCrawler:
            raise RuntimeError("Crawl4AI not available")

        async with AsyncWebCrawler(verbose=False) as crawler:
            if config is not None:
                return await crawler.arun(url=url, config=config, **kwargs)
            return await crawler.arun(url=url, **kwargs)

    async def _discover_urls_with_virtual_scroll(
        self, max_urls: int, base: str
    ) -> list[str]:
        if not (_CRAWL4AI_AVAILABLE and AsyncWebCrawler):
            return []

        try:
            virtual_config = self._create_virtual_scroll_config()
            run_config = self._create_run_config(virtual_config)
            result = await self._run_crawl(
                url=self.site_config.url,
                config=run_config,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Crawl4AI virtual scroll discovery failed for %s: %s",
                self.site_config.name,
                exc,
            )
            return []

        html = (
            getattr(result, "raw_html", None)
            or getattr(result, "cleaned_html", "")
        )
        urls = self._extract_links_from_html(html, base, max_urls)
        if urls:
            logger.info(
                "✅ [Crawl4AI+Scroll] Found %d URLs for %s",
                len(urls),
                self.site_config.name,
            )
        return urls

    async def get_article_urls(self, max_urls: int = 40) -> list[str]:
        """Discover candidate article URLs using Crawl4AI strategies."""
        if not (_CRAWL4AI_AVAILABLE and AsyncWebCrawler):
            logger.error("❌ Crawl4AI not available for URL discovery")
            return []

        base = self.site_config.get_base_url()

        try:  # pragma: no cover - network
            result = await self._run_crawl(url=self.site_config.url)
            html = (
                getattr(result, "raw_html", None)
                or getattr(result, "cleaned_html", "")
            )
            urls = self._extract_links_from_html(html, base, max_urls)
            if urls:
                logger.info(
                    "✅ [Crawl4AI] Found %d URLs for %s",
                    len(urls),
                    self.site_config.name,
                )
                return urls
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Crawl4AI link discovery failed for %s: %s",
                self.site_config.name,
                exc,
            )

        fallback_urls = await self._discover_urls_with_virtual_scroll(max_urls, base)
        if fallback_urls:
            return fallback_urls

        logger.warning("⚠️ No URLs discovered for %s", self.site_config.name)
        return []

    def _build_crawl_kwargs(self, url: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "url": url,
            "css_selector": "main, article, .content, .story-body, [role='main']",
            "user_agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/118.0 Safari/537.36"
            ),
        }

        if LLMConfig and LLMExtractionStrategy:
            llm_config = LLMConfig(
                provider="ollama/llama2:7b",
                base_url="http://localhost:11434",
            )
            kwargs["extraction_strategy"] = LLMExtractionStrategy(
                llm_config=llm_config
            )
        return kwargs

    def _parse_crawl_result(
        self,
        result: Any,
        url: str,
        start_time: float,
        llm_enabled: bool,
    ) -> dict[str, Any] | None:
        candidate_blocks = [
            getattr(result, "cleaned_html", ""),
            getattr(result, "markdown", ""),
            getattr(result, "extracted_content", ""),
            getattr(result, "raw_markdown", ""),
        ]
        raw_content = next((block for block in candidate_blocks if block), "")

        clean_text = re.sub(r"<[^>]+>", " ", raw_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if len(clean_text) <= 80:
            return None

        title = getattr(result, "title", "") or f"{self.site_config.name} Article"
        canonical = getattr(result, "canonical_url", None)

        metadata = CanonicalMetadata.generate_metadata(
            url=url,
            title=title[:300],
            content=clean_text[:12000],
            extraction_method="crawl4ai_generic",
            status="success",
            paywall_flag=False,
            confidence=0.72,
            publisher=self.site_config.name,
            crawl_mode="generic_site",
            news_score=0.72,
            canonical=canonical,
        )
        metadata.setdefault("extraction_metadata", {}).update(
            {
                "strategy": "crawl4ai",
                "processing_time_s": time.time() - start_time,
                "text_length": len(clean_text),
                "llm_strategy": llm_enabled,
            }
        )
        return metadata

    def _build_robots_metadata(self, url: str, start_time: float) -> dict[str, Any]:
        elapsed = time.time() - start_time
        metadata = CanonicalMetadata.generate_metadata(
            url=url,
            title="Robots.txt Disallowed",
            content="Crawling not allowed by robots.txt",
            extraction_method="robots_guard",
            status="disallowed",
            paywall_flag=False,
            confidence=0.0,
            publisher=self.site_config.name,
            crawl_mode="generic_site",
            news_score=0.0,
        )
        metadata.setdefault("extraction_metadata", {}).update(
            {
                "robots_disallowed": True,
                "processing_time_s": elapsed,
            }
        )
        return metadata

    def _build_failure_metadata(self, url: str, start_time: float, reason: str) -> dict[str, Any]:
        elapsed = time.time() - start_time
        metadata = CanonicalMetadata.generate_metadata(
            url=url,
            title="Extraction Failed",
            content="",
            extraction_method="none_available",
            status="error",
            paywall_flag=False,
            confidence=0.0,
            publisher=self.site_config.name,
            crawl_mode="generic_site",
            news_score=0.0,
            error=reason,
        )
        metadata.setdefault("extraction_metadata", {}).update(
            {"processing_time_s": elapsed}
        )
        return metadata

    async def _llava_fallback(
        self, url: str, start_time: float
    ) -> dict[str, Any] | None:
        try:
            result = await extract_news_from_url(url=url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLaVA fallback failed for %s: %s", url, exc)
            return None

        if not (result.get("success") and result.get("article")):
            return None

        title = (
            result.get("headline")
            or f"{self.site_config.name} Article"
        ).strip()
        content = (result.get("article") or "").strip()

        metadata = CanonicalMetadata.generate_metadata(
            url=url,
            title=title[:300],
            content=content[:12000],
            extraction_method="llava_fallback",
            status="success",
            paywall_flag=False,
            confidence=0.68,
            publisher=self.site_config.name,
            crawl_mode="generic_site",
            news_score=0.68,
            canonical=None,
        )
        metadata.setdefault("extraction_metadata", {}).update(
            {
                "llava_fallback": True,
                "screenshot_path": result.get("screenshot_path"),
                "llava_confidence": result.get("confidence_score"),
                "processing_time_s": time.time() - start_time,
            }
        )
        return metadata

    async def _try_crawl4ai(
        self,
        url: str,
        start_time: float,
    ) -> dict[str, Any] | None:
        if not (_CRAWL4AI_AVAILABLE and AsyncWebCrawler):
            return None

        crawl_kwargs = self._build_crawl_kwargs(url)
        llm_enabled = "extraction_strategy" in crawl_kwargs

        try:  # pragma: no cover - network heavy
            result = await self._run_crawl(**crawl_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Crawl4AI extraction failed for %s: %s", url, exc)
            return None

        return self._parse_crawl_result(result, url, start_time, llm_enabled)

    async def process_single_url(self, url: str) -> dict[str, Any] | None:
        if url in self.processed_urls:
            logger.debug("Skipping already processed URL: %s", url)
            return None

        self.processed_urls.add(url)
        start_time = time.time()

        if not self.robots_checker.check_robots_txt(url):
            logger.info("⚠️ Robots.txt disallows crawling: %s", url)
            return self._build_robots_metadata(url, start_time)

        await self.rate_limiter.wait_if_needed(self.site_config.domain)

        crawl_metadata = await self._try_crawl4ai(url, start_time)
        if crawl_metadata:
            return crawl_metadata

        fallback_metadata = await self._llava_fallback(url, start_time)
        if fallback_metadata:
            return fallback_metadata

        logger.warning(
            "Extraction failed for %s after %.2fs",
            url,
            time.time() - start_time,
        )
        return self._build_failure_metadata(
            url,
            start_time,
            "Failed to extract via Crawl4AI and LLaVA",
        )

    async def _gather_articles(
        self,
        urls: list[str],
        max_articles: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        semaphore = asyncio.Semaphore(max(1, self.concurrent_browsers))
        results: list[dict[str, Any]] = []
        disallowed = 0
        failures = 0

        async def worker(target_url: str) -> dict[str, Any] | None:
            async with semaphore:
                return await self.process_single_url(target_url)

        tasks = [asyncio.create_task(worker(url)) for url in urls]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for item in raw_results:
            if isinstance(item, Exception):
                failures += 1
                logger.debug("Task raised exception: %s", item)
                continue
            if not isinstance(item, dict):
                continue

            status = item.get("status")
            if status == "success" and len(results) < max_articles:
                results.append(item)
            elif status == "disallowed":
                disallowed += 1
            else:
                failures += 1

        return results, disallowed, failures

    def _log_crawl_summary(
        self,
        successes: int,
        disallowed: int,
        failures: int,
        elapsed: float,
    ) -> None:
        rate = successes / elapsed if elapsed else 0.0
        logger.info(
            "🎉 Generic site crawl complete: %d success, %d disallowed, %d failed "
            "(rate %.2f/s)",
            successes,
            disallowed,
            failures,
            rate,
        )

    async def crawl_site(self, max_articles: int = 25) -> list[dict[str, Any]]:
        logger.info(
            "🚀 Starting generic crawl of %s for up to %d articles",
            self.site_config.name,
            max_articles,
        )

        urls = await self.get_article_urls(max_urls=max_articles * 3)
        if not urls:
            logger.warning("❌ No URLs found for %s", self.site_config.name)
            return []

        random.shuffle(urls)
        start_time = time.time()

        results, disallowed, failures = await self._gather_articles(urls, max_articles)
        self._log_crawl_summary(len(results), disallowed, failures, time.time() - start_time)
        return results


class MultiSiteCrawler:
    """Coordinates crawling across multiple sites"""

    def __init__(self, concurrent_sites: int = 3, articles_per_site: int = 25):
        self.concurrent_sites = concurrent_sites
        self.articles_per_site = articles_per_site
        self.site_crawlers = {}

    async def load_sources_from_db(self, domains: list[str] = None) -> list[SiteConfig]:
        """Load site configurations from database"""
        if domains:
            sources = get_sources_by_domain(domains)
        else:
            sources = get_active_sources()

        site_configs = []
        for source in sources:
            try:
                config = SiteConfig(source)
                site_configs.append(config)
                logger.info(f"📋 Loaded config for {config.name} ({config.domain})")
            except Exception as e:
                logger.warning(f"Failed to create config for {source.get('name', 'Unknown')}: {e}")

        logger.info(f"✅ Loaded {len(site_configs)} site configurations")
        return site_configs

    async def crawl_multiple_sites(self, site_configs: list[SiteConfig],
                                 max_total_articles: int = 100) -> dict[str, list[dict]]:
        """Crawl multiple sites concurrently"""
        logger.info(f"🚀 Starting multi-site crawl of {len(site_configs)} sites")

        results = {}
        semaphore = asyncio.Semaphore(self.concurrent_sites)

        async def crawl_site_with_limit(site_config: SiteConfig):
            async with semaphore:
                crawler = GenericSiteCrawler(site_config)
                articles = await crawler.crawl_site(self.articles_per_site)
                results[site_config.domain] = articles
                logger.info(f"🏁 Completed {site_config.name}: {len(articles)} articles")

        # Create tasks for all sites
        tasks = [crawl_site_with_limit(config) for config in site_configs]

        # Execute concurrently with site limit
        await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize results
        total_articles = sum(len(articles) for articles in results.values())
        logger.info("🎉 Multi-site crawl complete!")
        logger.info(f"📊 Total articles: {total_articles}")
        logger.info(f"📈 Sites crawled: {len(results)}")

        return results

    async def run_multi_site_crawl(self, domains: list[str] = None,
                                 max_total_articles: int = 100) -> dict[str, Any]:
        """Main entry point for multi-site crawling"""
        start_time = time.time()

        # Load site configurations
        site_configs = await self.load_sources_from_db(domains)
        if not site_configs:
            logger.error("❌ No site configurations loaded")
            return {"error": "No sites available"}

        # Crawl all sites
        site_results = await self.crawl_multiple_sites(site_configs, max_total_articles)

        # Flatten results for easier processing
        all_articles = []
        for _domain, articles in site_results.items():
            all_articles.extend(articles)

        processing_time = time.time() - start_time
        articles_per_second = len(all_articles) / processing_time if processing_time > 0 else 0

        summary = {
            "multi_site_crawl": True,
            "sites_crawled": len(site_results),
            "total_articles": len(all_articles),
            "processing_time_seconds": processing_time,
            "articles_per_second": articles_per_second,
            "site_breakdown": {domain: len(articles) for domain, articles in site_results.items()},
            "timestamp": datetime.now().isoformat(),
            "articles": all_articles
        }

        logger.info("🎉 Multi-site crawl summary:")
        logger.info(f"   Sites: {len(site_results)}")
        logger.info(f"   Articles: {len(all_articles)}")
        logger.info(f"   Time: {processing_time:.1f}s")
        logger.info(f"   Rate: {articles_per_second:.2f} articles/second")

        return summary


async def main():
    """Test the multi-site crawler"""
    crawler = MultiSiteCrawler(concurrent_sites=2, articles_per_site=10)
    results = await crawler.run_multi_site_crawl(max_total_articles=50)

    # Save results
    output_file = f"multi_site_crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"💾 Results saved to {output_file}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
