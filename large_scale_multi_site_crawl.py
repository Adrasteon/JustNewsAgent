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
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from agents.common.gpu_orchestrator_client import GPUOrchestratorClient
from agents.scout.production_crawlers.crawler_utils import get_active_sources
from common.dev_db_fallback import apply_test_db_env_fallback
from common.observability import get_logger

# Configure centralized logging
logger = get_logger(__name__)

# Apply development DB fallback (non-destructive) – temporary unblocker
apply_test_db_env_fallback(logger)

# Configuration
MCP_BUS_URL = "http://localhost:8000"
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONCURRENT_SITES = 5
DEFAULT_ARTICLES_PER_SITE = 50
DEFAULT_QUALITY_THRESHOLD = 0.6
# Agent call timeout configuration (seconds)
AGENT_CONNECT_TIMEOUT = float(os.getenv("MCP_CALL_CONNECT_TIMEOUT", "3"))
AGENT_READ_TIMEOUT = float(os.getenv("MCP_CLIENT_READ_TIMEOUT", "180"))

@dataclass
class CrawlConfig:
    """Configuration for large-scale crawling"""
    sites: list[str] = field(default_factory=lambda: CrawlConfig._load_sites_from_database())
    mode: str = "mixed"  # 'ultra_fast', 'ai_enhanced', or 'mixed'
    articles_per_site: int = DEFAULT_ARTICLES_PER_SITE
    concurrent_sites: int = DEFAULT_CONCURRENT_SITES
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_archive: bool = True
    enable_knowledge_graph: bool = True
    test_mode: bool = False
    output_dir: str = "./large_scale_crawl_results"
    archive_port: int = 8021  # Archive REST API port

    @staticmethod
    def _load_sites_from_database() -> list[str]:
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
    errors: list[str] = field(default_factory=list)
    articles: list[dict[str, Any]] = field(default_factory=list)
    performance_metrics: dict[str, Any] = field(default_factory=dict)

@dataclass
class LargeScaleResult:
    """Overall results from large-scale crawl"""
    config: CrawlConfig
    total_sites: int = 0
    total_articles: int = 0
    total_processing_time: float = 0.0
    overall_success_rate: float = 0.0
    overall_quality_score: float = 0.0
    site_results: list[SiteResult] = field(default_factory=list)
    archive_summary: dict[str, Any] | None = None
    knowledge_graph_stats: dict[str, Any] | None = None
    timestamp: str = ""
    ai_models_used: list[str] = field(default_factory=list)
    timing_breakdown: dict[str, float] = field(default_factory=dict)


def _format_duration(seconds: float | None) -> str:
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
        # GPU orchestrator client (fail-safe wrapper; never raises)
        try:
            self.gpu_client = GPUOrchestratorClient()
        except Exception:
            self.gpu_client = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_quality_score(article: dict[str, Any]) -> float:
        """Derive a quality score from available fields.

        Preference order:
        - scout_analysis.scout_score (if present)
        - news_score (from ultra_fast crawlers)
        - confidence (generic fallback)
        - 0.0 as last resort
        """
        try:
            scout = article.get("scout_analysis") or {}
            if isinstance(scout, dict) and "scout_score" in scout:
                return float(scout.get("scout_score") or 0.0)
        except Exception:
            pass

        for key in ("news_score", "confidence"):
            try:
                if key in article and article[key] is not None:
                    return float(article[key])
            except Exception:
                continue
        return 0.0

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
                response = await client.post(
                    f"{MCP_BUS_URL}/call",
                    json=payload,
                    timeout=httpx.Timeout(AGENT_READ_TIMEOUT, connect=AGENT_CONNECT_TIMEOUT, read=AGENT_READ_TIMEOUT, write=AGENT_CONNECT_TIMEOUT)
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                # Catches both HTTPStatusError and RequestError (connect/read timeouts)
                detail = str(e)
                if hasattr(e, "response") and e.response is not None:
                    try:
                        body_preview = e.response.text[:200]
                        detail = f"{e} | body={body_preview}"
                    except Exception:
                        pass
                logger.error(f"❌ Agent call failed - {agent}.{tool}: {detail}")
                return {"error": detail, "success": False}

    async def ensure_agents_registered(self, timeout_seconds: int = 20) -> None:
        """Ensure core agents are registered with the MCP Bus, attempt auto-register if missing.

        Best-effort: checks /agents until required names appear or timeout; if missing,
        attempts to POST /register using known localhost ports.
        """
        required = {
            "scout": "http://localhost:8002",
            "newsreader": "http://localhost:8009",
            "memory": "http://localhost:8007",
            "archive_api": f"http://localhost:{self.config.archive_port}",
            "gpu_orchestrator": "http://localhost:8014",
        }

        start = time.time()
        seen: set[str] = set()

        async with httpx.AsyncClient() as client:
            while time.time() - start < timeout_seconds:
                try:
                    resp = await client.get(f"{MCP_BUS_URL}/agents", timeout=5)
                    if resp.status_code in (200, 201):
                        data = resp.json()
                        # Bus may return a list of names OR a dict mapping name->address
                        if isinstance(data, list):
                            seen = set(data)
                        elif isinstance(data, dict):
                            seen = set(data.keys())
                        else:
                            seen = set()

                        missing = [name for name in required.keys() if name not in seen]
                        if not missing:
                            logger.info("🚌 MCP Bus has all required agents registered")
                            return

                        # Try to register missing ones
                        for name in missing:
                            address = required[name]
                            try:
                                reg = await client.post(
                                    f"{MCP_BUS_URL}/register",
                                    json={"name": name, "address": address},
                                    timeout=5,
                                )
                                if reg.status_code in (200, 201):
                                    logger.info(f"✅ Registered agent '{name}' at {address}")
                                else:
                                    logger.warning(f"⚠️ Registration failed for '{name}' ({reg.status_code}): {reg.text[:120]}")
                            except Exception as re:
                                logger.warning(f"⚠️ Could not register '{name}' at {address}: {re}")
                    else:
                        logger.warning(f"⚠️ MCP /agents returned {resp.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ MCP /agents check failed: {e}")

                await asyncio.sleep(1.0)
        logger.warning(f"⚠️ Proceeding without full MCP registration: seen={sorted(seen)}")

    async def check_system_health(self) -> dict[str, bool]:
        """Check health of all required agents"""
        agents = {
            "mcp_bus": "http://localhost:8000/agents",
            "scout": "http://localhost:8002/health",
            "newsreader": "http://localhost:8009/health",
            "memory": "http://localhost:8007/health",
            "archive": f"http://localhost:{self.config.archive_port}/health",
            "gpu_orchestrator": "http://localhost:8014/health",
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

    async def get_production_crawler_info(self) -> dict[str, Any]:
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

    async def ensure_models_ready(self, timeout_seconds: int = 180) -> bool:
        """Ensure models are preloaded and ready via the GPU orchestrator.

        Uses /models/preload to start a background warm-up and polls /models/status
        until all_ready or timeout. Returns True if ready or orchestrator unavailable.
        """
        orchestrator_base = "http://localhost:8014"
        try:
            async with httpx.AsyncClient() as client:
                # Quick health check
                try:
                    r = await client.get(f"{orchestrator_base}/health", timeout=5)
                    if r.status_code not in (200, 201, 202):
                        logger.warning("GPU orchestrator health not OK; skipping model preload gate")
                        return True
                except Exception:
                    logger.info("GPU orchestrator not reachable; skipping model preload gate")
                    return True

                # Trigger preload if needed
                try:
                    pr = await client.post(f"{orchestrator_base}/models/preload", json={"refresh": False}, timeout=10)
                    if pr.status_code == 503:
                        # Hard failure: extract detailed reasons and abort early
                        try:
                            body = pr.json()
                            detail = body.get("detail", body)
                            errs = detail.get("errors", []) if isinstance(detail, dict) else []
                            if errs:
                                for e in errs[:20]:
                                    logger.error(f"Model preload error: agent={e.get('agent')} model={e.get('model')} error={e.get('error')}")
                        except Exception:
                            logger.error(f"Model preload returned 503 with body: {pr.text[:200]}")
                        return False
                except Exception as e:
                    logger.warning(f"Failed to trigger model preload: {e}")

                # Poll status
                deadline = time.time() + timeout_seconds
                last_progress = None
                while time.time() < deadline:
                    try:
                        s = await client.get(f"{orchestrator_base}/models/status", timeout=5)
                        if s.status_code in (200, 201):
                            data = s.json()
                            all_ready = bool(data.get("all_ready", False))
                            in_progress = bool(data.get("in_progress", False))
                            summary = data.get("summary", {})
                            now_progress = (summary.get("done", 0), summary.get("failed", 0), summary.get("total", 0))
                            if now_progress != last_progress:
                                logger.info(f"Model preload progress: done={now_progress[0]} failed={now_progress[1]} total={now_progress[2]}")
                                last_progress = now_progress
                            if all_ready:
                                logger.info("✅ All models reported ready by gpu_orchestrator")
                                return True
                            if not in_progress and not all_ready:
                                # Log detailed errors if present
                                errs = data.get("errors", []) or []
                                if errs:
                                    for e in errs[:50]:
                                        logger.error(f"Model preload error: agent={e.get('agent')} model={e.get('model')} error={e.get('error')}")
                                logger.error("Model preload completed with failures; aborting to avoid unstable behavior")
                                return False
                    except Exception:
                        pass
                    await asyncio.sleep(3)
                logger.warning("Model preload did not complete before timeout; proceeding anyway")
                return False
        except Exception as e:
            logger.warning(f"Model preload gate encountered an error: {e}")
            return True

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
                # Console hint for troubleshooting when logging is muted
                print(f"[crawl_single_site] Agent error for {site} mode={mode}: {crawl_result.get('error')}")
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
                    q = self._get_quality_score(article)
                    article["quality_score"] = q
                    quality_scores.append(q)

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

            logger.info(f"✅ {site} crawl complete!")
            # Console summary for visibility in CI/test-mode
            if self.config.test_mode:
                print(f"[crawl_single_site] {site} {mode}: found={result.articles_found} time={result.processing_time:.1f}s")
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

    async def process_articles_batch(self, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a batch of articles through NewsReader and Memory agents"""
        logger.info(f"🔄 Processing batch of {len(articles)} articles...")

        processed_articles = []

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrent_sites)

        async def process_single_article(article: dict[str, Any]) -> dict[str, Any]:
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
                                            "quality_score": article.get("quality_score", self._get_quality_score(article))
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

    async def archive_and_analyze(self, all_articles: list[dict[str, Any]]) -> dict[str, Any]:
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
            # Prefer MCP bus call to archive_api tool
            archive_result = await self.call_agent(
                "archive_api",
                "archive_from_crawler",
                kwargs={"crawler_results": crawler_results}
            )

            # Fallback: direct REST call if bus/tool fails
            if (not archive_result) or (isinstance(archive_result, dict) and archive_result.get("error")):
                logger.warning("⚠️ Bus archive call failed or returned error; trying direct REST fallback")
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"http://localhost:{self.config.archive_port}/archive_from_crawler",
                            json={"args": [], "kwargs": {"crawler_results": crawler_results}},
                            timeout=300
                        )
                        resp.raise_for_status()
                        archive_result = resp.json().get("data", resp.json())
                except Exception as rest_e:
                    logger.error(f"❌ REST archive fallback failed: {rest_e}")
                    archive_result = {"error": str(rest_e), "success": False}

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
            # Step 0: Ensure models are ready to avoid on-demand spikes/OOM
            logger.info("📦 Ensuring models are preloaded and ready via gpu_orchestrator...")
            models_ready = await self.ensure_models_ready(timeout_seconds=180)
            if not models_ready:
                logger.error("❌ Models not fully ready. Aborting crawl to avoid unstable behavior.")
                # Return early with minimal results; main() will exit non-zero
                self.results.total_sites = 0
                self.results.total_articles = 0
                self.results.total_processing_time = time.time() - self.start_time
                return self.results
            # Step 1: System health check
            logger.info("🩺 Checking system health...")
            health_status = await self.check_system_health()
            unhealthy_agents = [k for k, v in health_status.items() if not v]

            if unhealthy_agents:
                logger.warning(f"⚠️ Unhealthy agents: {unhealthy_agents}")
                logger.warning("Continuing with available agents...")

            # Step 2: Ensure required agents are registered on the MCP bus
            await self.ensure_agents_registered(timeout_seconds=30)

            # Step 2b: Consult GPU Orchestrator policy to guide processing
            processing_batch_size = self.config.batch_size
            orchestrator_note = ""
            try:
                if self.gpu_client is not None:
                    decision = self.gpu_client.cpu_fallback_decision()
                    use_gpu = bool(decision.get("use_gpu", False))
                    safe_mode = bool(decision.get("safe_mode", True))
                    gpu_available = bool(decision.get("gpu_available", False))
                    orchestrator_note = (
                        f"GPU orchestrator decision: use_gpu={use_gpu} safe_mode={safe_mode} gpu_available={gpu_available}"
                    )
                    logger.info("🔐 %s", orchestrator_note)
                    # If GPU not allowed/available, reduce batch size to 1 for safety
                    if not use_gpu:
                        processing_batch_size = 1
                        logger.info("🧯 GPU not permitted/available → forcing processing batch_size=1")
                else:
                    logger.info("🔐 GPU orchestrator client unavailable; proceeding with default batch size")
            except Exception as e:
                logger.warning(f"⚠️ GPU orchestrator consult failed: {e}")

            # Step 3: Get production crawler information
            crawler_info = await self.get_production_crawler_info()

            # Step 4: Baseline BBC (legacy) + Dynamic domains separation
            crawling_start_time = time.time()
            baseline_keys = {"bbc", "bbc.com", "bbc.co.uk"}
            baseline_site = None
            remaining_domains: list[str] = []
            for s in self.config.sites:
                if s in baseline_keys and baseline_site is None:
                    baseline_site = "bbc"  # normalize to legacy key
                else:
                    remaining_domains.append(s)

            site_results: list[SiteResult] = []

            # Baseline legacy crawl (ultra_fast or ai_enhanced depending on mixed alternation)
            if baseline_site:
                mode = "ultra_fast" if self.config.mode in ("mixed", "ultra_fast") else "ai_enhanced"
                logger.info(f"🎯 Baseline legacy crawl for {baseline_site} mode={mode}")
                baseline_result = await self.crawl_single_site(baseline_site, mode)
                site_results.append(baseline_result)
            else:
                logger.info("ℹ️ No baseline BBC site found in site list")

            # Dynamic multi-domain crawl for remaining domains via new production_crawl_dynamic tool
            dynamic_articles: list[dict] = []
            if remaining_domains:
                logger.info(f"🌐 Dynamic domain crawl for {len(remaining_domains)} domains")
                try:
                    dyn = await self.call_agent(
                        "scout",
                        "production_crawl_dynamic",
                        kwargs={
                            "domains": remaining_domains,
                            "articles_per_site": self.config.articles_per_site,
                            "concurrent_sites": self.config.concurrent_sites,
                            "max_total_articles": len(remaining_domains) * self.config.articles_per_site,
                        },
                    )
                    if isinstance(dyn, dict) and "error" in dyn:
                        print(f"[dynamic_crawl] Agent error: {dyn['error']}")
                        logger.error(f"Dynamic crawl call failed: {dyn['error']}")
                    else:
                        domain_articles = dyn.get("domain_articles", {})
                        for domain, arts in domain_articles.items():
                            # Wrap each domain as a SiteResult-like object
                            sr = SiteResult(site=domain, mode="dynamic", articles_found=len(arts), articles_processed=len(arts))
                            sr.articles = arts
                            sr.success_rate = 1.0 if arts else 0.0
                            sr.performance_metrics = {
                                "articles_per_second": dyn.get("articles_per_second", 0),
                                "crawl_mode": "dynamic",
                                "target_articles": self.config.articles_per_site,
                                "actual_articles": len(arts),
                                "ai_enhanced": False,
                            }
                            site_results.append(sr)
                            dynamic_articles.extend(arts)
                except Exception as e:
                    logger.error(f"Dynamic crawl request exception: {e}")
            else:
                logger.info("ℹ️ No remaining domains for dynamic crawl")

            pipeline_stage_timings["1_site_crawling_and_discovery"] = time.time() - crawling_start_time

            # Step 5: Process articles through pipeline
            processing_start_time = time.time()
            all_articles = []
            for result in site_results:
                # Filter by quality threshold with robust scoring fallback
                quality_articles = []
                for article in result.articles:
                    q = self._get_quality_score(article)
                    article["quality_score"] = q
                    if q >= self.config.quality_threshold:
                        quality_articles.append(article)

                result.articles = quality_articles
                result.articles_processed = len(quality_articles)
                all_articles.extend(quality_articles)

            logger.info(f"🎚️ Quality filtering: {len(all_articles)} articles meet threshold {self.config.quality_threshold}")

            # Step 6: Batch processing through NewsReader and Memory
            if all_articles:
                logger.info("🔄 Processing articles through NewsReader and Memory agents...")
                batch_size = processing_batch_size

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

    def _determine_site_modes(self) -> dict[str, str]:
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
