#!/usr/bin/env python3
"""
JustNews V4 Large-Scale Multi-Site Crawl with AI Quality Assessment

This script orchestrates a comprehensive large-scale crawl across multiple news sites
using the production crawler system with AI-powered quality assessment and analysis.

Features:
- Multi-site concurrent crawling (ultra-fast + AI-enhanced modes)
- AI quality assessment using Scout intelligence (7 AI models)
- Batch processing with configurable concurrency
- Comprehensive performance monitoring and reporting
- Database-driven source management
- Archive integration for long-term storage
- Knowledge graph processing for entity extraction
- Full pipeline validation across all agents

Usage:
    python large_scale_multi_site_crawl.py [options]

Examples:
    # Crawl all active sites from database (default)
    python large_scale_multi_site_crawl.py

    # Crawl specific sites with custom article counts
    python large_scale_multi_site_crawl.py --sites bbc cnn reuters --articles-per-site 100

    # Ultra-fast mode only (no AI enhancement)
    python large_scale_multi_site_crawl.py --mode ultra_fast --concurrent-sites 10

    # AI-enhanced mode with quality thresholds
    python large_scale_multi_site_crawl.py --mode ai_enhanced --quality-threshold 0.7

    # Test mode with small numbers
    python large_scale_multi_site_crawl.py --test-mode --articles-per-site 5
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from tqdm.asyncio import tqdm

from common.observability import get_logger
from agents.scout.production_crawlers.crawler_utils import get_active_sources

# Configure centralized logging
logger = get_logger(__name__)

# Configuration
MCP_BUS_URL = "http://localhost:8000"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONCURRENT_SITES = 5
DEFAULT_ARTICLES_PER_SITE = 50
DEFAULT_QUALITY_THRESHOLD = 0.6

@dataclass
class CrawlConfig:
    """Configuration for large-scale crawling"""
    sites: List[str] = field(default_factory=lambda: CrawlConfig._load_sites_from_database())
    mode: str = "mixed"  # 'ultra_fast', 'ai_enhanced', or 'mixed'
    articles_per_site: int = DEFAULT_ARTICLES_PER_SITE
    concurrent_sites: int = DEFAULT_CONCURRENT_SITES
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_archive: bool = True
    enable_knowledge_graph: bool = True
    test_mode: bool = False
    output_dir: str = "./large_scale_crawl_results"
    archive_port: int = 8014

    @staticmethod
    def _load_sites_from_database() -> List[str]:
        """Load active sites from database with fallback to bbc.com"""
        try:
            logger.info("🔍 Loading active sources from database...")
            sources = get_active_sources()

            if sources:
                # Extract domains from sources
                domains = [source.get('domain', '').replace('www.', '') for source in sources if source.get('domain')]
                domains = [d for d in domains if d]  # Filter out empty domains

                if domains:
                    logger.info(f"✅ Loaded {len(domains)} sites from database: {domains[:5]}{'...' if len(domains) > 5 else ''}")
                    return domains

            logger.warning("⚠️ No active sources found in database, using fallback")
            return ["bbc.com"]

        except Exception as e:
            logger.error(f"❌ Failed to load sites from database: {e}")
            logger.warning("⚠️ Using fallback site: bbc.com")
            return ["bbc.com"]

@dataclass
class SiteResult:
    """Results from crawling a single site"""
    site: str
    mode: str
    articles_found: int = 0
    articles_processed: int = 0
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    processing_time: float = 0.0
    ai_analysis_count: int = 0
    errors: List[str] = field(default_factory=list)
    articles: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LargeScaleResult:
    """Overall results from large-scale crawl"""
    config: CrawlConfig
    total_sites: int = 0
    total_articles: int = 0
    total_processing_time: float = 0.0
    overall_success_rate: float = 0.0
    overall_quality_score: float = 0.0
    site_results: List[SiteResult] = field(default_factory=list)
    archive_summary: Optional[Dict[str, Any]] = None
    knowledge_graph_stats: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    ai_models_used: List[str] = field(default_factory=list)
    timing_breakdown: Dict[str, float] = field(default_factory=dict)


def _format_duration(seconds: Optional[float]) -> str:
    """Formats a duration in seconds into a human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, rem_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(rem_seconds)}s"

    hours, rem_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem_minutes)}m {int(rem_seconds)}s"

class LargeScaleCrawler:
    """Orchestrator for large-scale multi-site crawling with AI quality assessment"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.results = LargeScaleResult(config=config)
        self.start_time = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def call_agent(self, agent: str, tool: str, args: list = None, kwargs: dict = None) -> dict:
        """Call agent through MCP bus with proper error handling"""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        payload = {
            "agent": agent,
            "tool": tool,
            "args": args,
            "kwargs": kwargs
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{MCP_BUS_URL}/call", json=payload, timeout=60)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                logger.error(f"❌ Agent call failed - {agent}.{tool}: {e}")
                return {"error": str(e), "success": False}

    async def check_system_health(self) -> Dict[str, bool]:
        """Check health of all required agents"""
        agents = {
            "mcp_bus": "http://localhost:8000/agents",
            "scout": "http://localhost:8002/health",
            "newsreader": "http://localhost:8009/health",
            "memory": "http://localhost:8007/health",
            "archive": f"http://localhost:{self.config.archive_port}/health",
        }

        async def get_health(client, agent, url):
            try:
                response = await client.get(url, timeout=5)
                return agent, response.status_code in [200, 201, 202]
            except Exception:
                return agent, False

        health_status = {}
        async with httpx.AsyncClient() as client:
            tasks = [get_health(client, agent, url) for agent, url in agents.items()]
            results = await asyncio.gather(*tasks)
            health_status = dict(results)

        healthy_count = sum(health_status.values())
        total_count = len(health_status)

        logger.info(f"🩺 System Health: {healthy_count}/{total_count} agents healthy")
        for agent, healthy in health_status.items():
            status = "✅" if healthy else "❌"
            logger.info(f"   {status} {agent}")

        return health_status

    async def get_production_crawler_info(self) -> Dict[str, Any]:
        """Get information about available production crawlers"""
        logger.info("🔍 Getting production crawler information...")
        result = await self.call_agent("scout", "get_production_crawler_info")

        if "error" in result:
            logger.warning(f"⚠️ Could not get crawler info: {result['error']}")
            return {"available": False}

        logger.info("✅ Production crawler info retrieved")
        logger.info(f"   📊 Supported sites: {len(result.get('supported_sites', []))}")
        logger.info(f"   🚀 Capabilities: {result.get('capabilities', [])}")

        return result

    async def crawl_single_site(self, site: str, mode: str) -> SiteResult:
        """Crawl a single site using specified mode"""
        start_time = time.time()
        result = SiteResult(site=site, mode=mode)

        logger.info(f"🕷️ Starting {mode} crawl of {site} for {self.config.articles_per_site} articles")

        try:
            if mode == "ultra_fast":
                crawl_result = await self.call_agent(
                    "scout",
                    "production_crawl_ultra_fast",
                    kwargs={"site": site, "target_articles": self.config.articles_per_site}
                )
            elif mode == "ai_enhanced":
                crawl_result = await self.call_agent(
                    "scout",
                    "production_crawl_ai_enhanced",
                    kwargs={"site": site, "target_articles": self.config.articles_per_site}
                )
            else:
                raise ValueError(f"Invalid mode: {mode}")

            if "error" in crawl_result:
                result.errors.append(f"Crawl failed: {crawl_result['error']}")
                logger.error(f"❌ {site} crawl failed: {crawl_result['error']}")
                return result

            # Process crawl results
            articles = crawl_result.get("articles", [])
            result.articles_found = len(articles)
            result.articles_processed = len(articles)
            result.processing_time = time.time() - start_time

            # Calculate success rate
            if articles:
                successful_articles = [a for a in articles if "error" not in a]
                result.success_rate = len(successful_articles) / len(articles)
                result.articles = successful_articles

                # Calculate average quality score if available
                quality_scores = []
                ai_analysis_count = 0
                for article in successful_articles:
                    if "scout_analysis" in article:
                        ai_analysis_count += 1
                        scout_score = article["scout_analysis"].get("scout_score", 0.0)
                        quality_scores.append(scout_score)

                result.ai_analysis_count = ai_analysis_count
                if quality_scores:
                    result.avg_quality_score = sum(quality_scores) / len(quality_scores)

            # Store performance metrics
            result.performance_metrics = {
                "articles_per_second": result.articles_found / result.processing_time if result.processing_time > 0 else 0,
                "crawl_mode": mode,
                "target_articles": self.config.articles_per_site,
                "actual_articles": result.articles_found,
                "ai_enhanced": mode == "ai_enhanced"
            }

            logger.info("✅ {} crawl complete!".format(site))
            logger.info(f"   📊 Articles: {result.articles_found}")
            logger.info(f"   ⏱️ Time: {result.processing_time:.1f}s")
            logger.info(f"   🚀 Rate: {result.performance_metrics['articles_per_second']:.2f} art/sec")
            if result.ai_analysis_count > 0:
                logger.info(f"   🧠 AI Analysis: {result.ai_analysis_count} articles")
                logger.info(f"   ⭐ Avg Quality: {result.avg_quality_score:.2f}")

        except Exception as e:
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            logger.error(f"❌ Exception during {site} crawl: {e}")

        return result

    async def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of articles through NewsReader and Memory agents"""
        logger.info(f"🔄 Processing batch of {len(articles)} articles...")

        processed_articles = []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_sites)

        async def process_single_article(article: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    url = article.get("url", "")
                    if not url:
                        return article

                    # Recommendation 1: Avoid redundant analysis if already done by scout.
                    # The 'ai_analysis' key is added by the scout's 'ai_enhanced' mode.
                    if "ai_analysis" in article and article["ai_analysis"]:
                        logger.info(f"✅ Skipping NewsReader call, analysis already present for {url}")
                        newsreader_result = article["ai_analysis"]
                    else:
                        # Step 1: Enhanced NewsReader analysis
                        logger.info(f"📰 Calling NewsReader for analysis: {url}")
                        newsreader_result = await self.call_agent(
                            "newsreader",  # FIX: Correct agent name
                            "extract_news_from_url",
                            args=[url]
                        )

                        if "error" not in newsreader_result:
                            article["newsreader_analysis"] = newsreader_result
                            article["content"] = newsreader_result.get("content", "")
                            article["headline"] = newsreader_result.get("headline", "")

                            # Step 2: Store in Memory agent
                            if article.get("content"):
                                memory_result = await self.call_agent(
                                    "memory",
                                    "save_article",
                                    kwargs={
                                        "content": article["content"],
                                        "metadata": {
                                            "url": url,
                                            "source": article.get("site", "unknown"),
                                            "timestamp": datetime.now().isoformat(),
                                            "pipeline": "large_scale_crawl",
                                            "quality_score": article.get("scout_score", 0.0)
                                        }
                                    }
                                )

                                if "error" not in memory_result:
                                    article["processing_status"] = "complete"
                                else:
                                    article["processing_status"] = "memory_failed"
                            else:
                                article["processing_status"] = "no_content"
                        else:
                            article["processing_status"] = "newsreader_failed"
                            article["error"] = newsreader_result.get("error")

                except Exception as e:
                    article["processing_status"] = "exception"
                    article["error"] = str(e)
                
                return article

        # Process articles concurrently
        tasks = [process_single_article(article) for article in articles]
        processed_articles = await asyncio.gather(*tasks)

        successful = sum(1 for a in processed_articles if a.get("processing_status") == "complete")
        logger.info(f"✅ Batch processing complete: {successful}/{len(articles)} successful")

        return processed_articles

    async def archive_and_analyze(self, all_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Archive articles and perform knowledge graph analysis"""
        archive_summary = {}
        kg_stats = {}

        if not self.config.enable_archive:
            logger.info("📋 Archive integration disabled")
            return {"archive_disabled": True}

        try:
            # Prepare articles for archiving
            crawler_results = {
                "multi_site_crawl": True,
                "sites_crawled": len(self.config.sites),
                "total_articles": len(all_articles),
                "processing_time_seconds": self.results.total_processing_time,
                "articles": all_articles
            }

            # Archive through Archive agent
            logger.info("💾 Archiving articles...")
            archive_result = await self.call_agent(
                "archive",
                "archive_from_crawler",
                kwargs={"crawler_results": crawler_results}
            )

            if "error" not in archive_result:
                archive_summary = archive_result
                logger.info("✅ Articles archived successfully")
                logger.info(f"   📊 Articles archived: {archive_result.get('articles_archived', 0)}")
                logger.info(f"   🏷️ Storage keys: {len(archive_result.get('storage_keys', []))}")

                # Knowledge Graph processing
                if self.config.enable_knowledge_graph and "knowledge_graph" in archive_result:
                    kg_result = archive_result["knowledge_graph"]
                    kg_stats = kg_result.get("graph_statistics", {})
                    logger.info("🧠 Knowledge Graph processing complete")
                    logger.info(f"   📊 Entities extracted: {kg_result.get('total_entities_extracted', 0)}")
                    logger.info(f"   🕸️ Graph nodes: {kg_stats.get('total_nodes', 0)}")
                    logger.info(f"   🔗 Graph edges: {kg_stats.get('total_edges', 0)}")
            else:
                logger.warning(f"⚠️ Archive failed: {archive_result.get('error')}")

        except Exception as e:
            logger.error(f"❌ Archive/KG processing failed: {e}")

        return {
            "archive_summary": archive_summary,
            "knowledge_graph_stats": kg_stats
        }

    async def run_large_scale_crawl(self) -> LargeScaleResult:
        """Execute the complete large-scale crawl"""
        self.start_time = time.time()
        start_dt = datetime.now()
        self.results.timestamp = start_dt.isoformat()

        logger.info("=" * 80)
        logger.info("🚀 Starting Large-Scale Multi-Site Crawl with AI Quality Assessment")
        logger.info(f"   -> Crawl started at: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: {len(self.config.sites)} sites, {self.config.articles_per_site} articles each")
        logger.info(f"⚡ Mode: {self.config.mode}")
        logger.info(f"🔄 Concurrent sites: {self.config.concurrent_sites}")
        logger.info(f"🎚️ Quality threshold: {self.config.quality_threshold}")
        logger.info("=" * 80)

        try:
            pipeline_stage_timings = {}
            # Step 1: System health check
            logger.info("🩺 Checking system health...")
            health_status = await self.check_system_health()
            unhealthy_agents = [k for k, v in health_status.items() if not v]

            if unhealthy_agents:
                logger.warning(f"⚠️ Unhealthy agents: {unhealthy_agents}")
                logger.warning("Continuing with available agents...")

            # Step 2: Get production crawler information
            crawler_info = await self.get_production_crawler_info()

            # Step 3: Determine crawl modes for each site
            crawling_start_time = time.time()
            site_modes = self._determine_site_modes()

            # Step 4: Execute concurrent site crawling
            logger.info("🕷️ Starting concurrent site crawling...")
            semaphore = asyncio.Semaphore(self.config.concurrent_sites)

            async def crawl_with_limit(site: str, mode: str) -> SiteResult:
                async with semaphore:
                    return await self.crawl_single_site(site, mode)

            # Create progress bar
            with tqdm(total=len(self.config.sites), desc="Crawling sites") as pbar:
                tasks = [crawl_with_limit(site, mode) for site, mode in site_modes.items()]
                site_results = []

                for task in asyncio.as_completed(tasks):
                    result = await task
                    site_results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({
                        'site': result.site,
                        'articles': result.articles_found,
                        'rate': f"{result.performance_metrics.get('articles_per_second', 0):.1f} art/sec"
                    })
            pipeline_stage_timings["1_site_crawling_and_discovery"] = time.time() - crawling_start_time

            # Step 5: Process articles through pipeline
            processing_start_time = time.time()
            all_articles = []
            for result in site_results:
                # Filter by quality threshold
                quality_articles = []
                for article in result.articles:
                    scout_score = article.get("scout_analysis", {}).get("scout_score", 0.0)
                    if scout_score >= self.config.quality_threshold:
                        quality_articles.append(article)

                result.articles = quality_articles
                result.articles_processed = len(quality_articles)
                all_articles.extend(quality_articles)

            logger.info(f"🎚️ Quality filtering: {len(all_articles)} articles meet threshold {self.config.quality_threshold}")

            # Step 6: Batch processing through NewsReader and Memory
            if all_articles:
                logger.info("🔄 Processing articles through NewsReader and Memory agents...")
                batch_size = self.config.batch_size

                for i in range(0, len(all_articles), batch_size):
                    batch = all_articles[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_articles) + batch_size - 1)//batch_size}")

                    processed_batch = await self.process_articles_batch(batch)
                    # Update original articles with processed data
                    for j, processed in enumerate(processed_batch):
                        original_idx = i + j
                        if original_idx < len(all_articles):
                            all_articles[original_idx].update(processed)

            pipeline_stage_timings["2_article_processing_and_storage"] = time.time() - processing_start_time

            # Step 7: Archive and Knowledge Graph analysis
            archiving_start_time = time.time()
            archive_results = await self.archive_and_analyze(all_articles)
            pipeline_stage_timings["3_archiving_and_kg"] = time.time() - archiving_start_time

            # Step 8: Calculate final statistics
            self.results.total_sites = len(site_results)
            self.results.total_articles = len(all_articles)
            self.results.total_processing_time = time.time() - self.start_time
            self.results.site_results = site_results
            self.results.archive_summary = archive_results.get("archive_summary")
            self.results.knowledge_graph_stats = archive_results.get("knowledge_graph_stats")
            self.results.timing_breakdown = pipeline_stage_timings

            # Calculate overall metrics
            if site_results:
                total_found = sum(r.articles_found for r in site_results)
                total_processed = sum(r.articles_processed for r in site_results)
                self.results.overall_success_rate = total_processed / total_found if total_found > 0 else 0

                quality_scores = []
                for result in site_results:
                    if result.avg_quality_score > 0:
                        quality_scores.append(result.avg_quality_score)

                if quality_scores:
                    self.results.overall_quality_score = sum(quality_scores) / len(quality_scores)

            # Identify AI models used
            self.results.ai_models_used = [
                "LLaMA-3-8B (Scout Classification)",
                "BERT (News Classification)",
                "BERT (Quality Assessment)",
                "RoBERTa (Sentiment Analysis)",
                "Toxic Comment Model (Bias Detection)",
                "LLaVA-OneVision (Visual Analysis)",
                "LLaVA-1.5-7B or BLIP-2 (NewsReader)"
            ]

            # Generate comprehensive report
            await self._generate_report()

            end_dt = datetime.now()
            logger.info(f"🎉 Large-scale crawl complete at: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 80)
            logger.info(f"📊 Total sites: {self.results.total_sites}")
            logger.info(f"📰 Total articles: {self.results.total_articles}")
            logger.info(f"⏱️ Total time: {_format_duration(self.results.total_processing_time)}")
            logger.info(f"🚀 Overall rate: {self.results.total_articles / self.results.total_processing_time:.2f} art/sec" if self.results.total_processing_time > 0 else "🚀 Overall rate: N/A")
            logger.info(f"✅ Success rate: {self.results.overall_success_rate * 100:.1f}%")
            logger.info(f"⭐ Quality score: {self.results.overall_quality_score:.2f}" if self.results.overall_quality_score > 0 else "⭐ Quality score: N/A")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ Large-scale crawl failed: {e}")
            self.results.total_processing_time = time.time() - self.start_time

        return self.results

    def _determine_site_modes(self) -> Dict[str, str]:
        """Determine crawl mode for each site"""
        site_modes = {}

        for site in self.config.sites:
            if self.config.mode == "mixed":
                # Alternate between ultra_fast and ai_enhanced
                site_modes[site] = "ultra_fast" if len(site_modes) % 2 == 0 else "ai_enhanced"
            else:
                site_modes[site] = self.config.mode

        return site_modes

    async def _generate_report(self):
        """Generate comprehensive report of the crawl results"""
        report_file = Path(self.config.output_dir) / f"large_scale_crawl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "crawl_configuration": {
                "sites": self.config.sites,
                "mode": self.config.mode,
                "articles_per_site": self.config.articles_per_site,
                "concurrent_sites": self.config.concurrent_sites,
                "quality_threshold": self.config.quality_threshold,
                "batch_size": self.config.batch_size,
                "test_mode": self.config.test_mode
            },
            "overall_results": {
                "total_sites": self.results.total_sites,
                "total_articles": self.results.total_articles,
                "total_processing_time": self.results.total_processing_time,
                "overall_success_rate": self.results.overall_success_rate,
                "overall_quality_score": self.results.overall_quality_score,
                "articles_per_second": self.results.total_articles / self.results.total_processing_time if self.results.total_processing_time > 0 else 0,
                "timestamp": self.results.timestamp
            },
            "ai_models_used": self.results.ai_models_used,
            "site_results": [
                {
                    "site": r.site,
                    "mode": r.mode,
                    "articles_found": r.articles_found,
                    "articles_processed": r.articles_processed,
                    "success_rate": r.success_rate,
                    "avg_quality_score": r.avg_quality_score,
                    "processing_time": r.processing_time,
                    "ai_analysis_count": r.ai_analysis_count,
                    "performance_metrics": r.performance_metrics,
                    "errors": r.errors
                }
                for r in self.results.site_results
            ],
            "archive_summary": self.results.archive_summary,
            "knowledge_graph_stats": self.results.knowledge_graph_stats
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"📄 Comprehensive report saved to: {report_file}")

        # Generate summary text file
        summary_file = Path(self.config.output_dir) / f"crawl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("JustNews V4 Large-Scale Multi-Site Crawl Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results.timestamp}\n")
            f.write(f"Sites crawled: {self.results.total_sites}\n")
            f.write(f"Articles collected: {self.results.total_articles}\n")
            f.write(f"Total processing time: {_format_duration(self.results.total_processing_time)}\n")
            f.write(f"Articles per second: {self.results.total_articles / self.results.total_processing_time:.2f}" if self.results.total_processing_time > 0 else "Articles per second: N/A\n")
            f.write(f"Overall success rate: {self.results.overall_success_rate:.1f}%\n")
            f.write(f"Average quality score: {self.results.overall_quality_score:.2f}\n\n" if self.results.overall_quality_score > 0 else "Average quality score: N/A\n\n")

            f.write("AI Models Used:\n")
            for i, model in enumerate(self.results.ai_models_used, 1):
                f.write(f"{i}. {model}\n")
            f.write("\n")

            f.write("Site-by-Site Results:\n")
            f.write("-" * 30 + "\n")
            for result in self.results.site_results:
                f.write(f"Site: {result.site} ({result.mode})\n")
                f.write(f"  Articles: {result.articles_found} found, {result.articles_processed} processed\n")
                f.write(f"  Success rate: {result.success_rate * 100:.1f}%\n")
                f.write(f"  Quality score: {result.avg_quality_score:.2f}\n" if result.avg_quality_score > 0 else "  Quality score: N/A\n")
                f.write(f"  Processing time: {_format_duration(result.processing_time)}\n")
                f.write(f"  Articles/second: {result.performance_metrics.get('articles_per_second', 0):.2f}\n")
                if result.ai_analysis_count > 0:
                    f.write(f"  AI analysis: {result.ai_analysis_count} articles\n")
                f.write("\n")

            f.write("Pipeline Timing Breakdown:\n")
            f.write("-" * 30 + "\n")
            if self.results.timing_breakdown:
                for stage, duration in sorted(self.results.timing_breakdown.items()):
                    f.write(f"  {stage.replace('_', ' ').title()}: {_format_duration(duration)}\n")
            else:
                f.write("  No timing breakdown available.\n")
            f.write("\n")

        logger.info(f"📝 Summary report saved to: {summary_file}")

async def main():
    """Main execution function"""
    # Load default sites from database
    try:
        default_sites = CrawlConfig._load_sites_from_database()
    except Exception as e:
        logger.warning(f"⚠️ Could not load default sites from database: {e}")
        default_sites = ["bbc.com"]

    parser = argparse.ArgumentParser(description="JustNews V4 Large-Scale Multi-Site Crawl")
    parser.add_argument("--sites", nargs="+", default=default_sites,
                       help="News sites to crawl (default: loaded from database, fallback: bbc.com)")
    parser.add_argument("--mode", choices=["ultra_fast", "ai_enhanced", "mixed"], default="mixed",
                       help="Crawling mode")
    parser.add_argument("--articles-per-site", type=int, default=DEFAULT_ARTICLES_PER_SITE,
                       help="Articles to crawl per site")
    parser.add_argument("--concurrent-sites", type=int, default=DEFAULT_CONCURRENT_SITES,
                       help="Number of sites to crawl concurrently")
    parser.add_argument("--quality-threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD,
                       help="Minimum quality score for articles")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Batch size for processing")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with reduced numbers")
    parser.add_argument("--output-dir", default="./large_scale_crawl_results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Adjust for test mode
    if args.test_mode:
        args.articles_per_site = 5
        args.concurrent_sites = 2
        args.sites = args.sites[:2]  # Only first 2 sites

    # Create configuration
    config = CrawlConfig(
        sites=args.sites,
        mode=args.mode,
        articles_per_site=args.articles_per_site,
        concurrent_sites=args.concurrent_sites,
        quality_threshold=args.quality_threshold,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
        output_dir=args.output_dir
    )

    # Run large-scale crawl
    crawler = LargeScaleCrawler(config)
    results = await crawler.run_large_scale_crawl()

    # Exit with appropriate code
    if results.total_articles > 0:
        print(f"\n🎉 Success! Crawled {results.total_articles} articles from {results.total_sites} sites")
        sys.exit(0)
    else:
        print("\n❌ Crawl completed but no articles were collected")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())