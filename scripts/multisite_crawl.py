#!/usr/bin/env python3
"""
Script to run a unified production crawl against all sources in the JustNews database.
"""

import os
import sys

# Ensure project root is in sys.path for module imports
project_root = os.environ.get("JUSTNEWS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Auto-load global.env if environment variables are missing
env_file = os.path.join(project_root, "deploy", "systemd", "env", "global.env")
if os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ.setdefault(key, val)

import argparse
import asyncio

import psycopg2
import requests

from agents.scout.production_crawlers.unified_production_crawler import (  # still imported for fallback
    UnifiedProductionCrawler,
)
from common.observability import log_error

# CLI argument parsing
parser = argparse.ArgumentParser(description="Run unified production crawl via MCP Bus")
parser.add_argument(
    "--max-sites", type=int, default=3, help="Max concurrent sites to crawl"
)
parser.add_argument(
    "--articles-per-site", type=int, default=25, help="Max articles per site to collect"
)
parser.add_argument(
    "--domain-limit",
    type=int,
    default=0,
    help="Limit total domains to crawl (0=no limit)",
)
args = parser.parse_args()

# MCP Bus endpoint
mcp_bus_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Agent endpoints
CRAWLER_AGENT_URL = os.environ.get(
    "CRAWLER_AGENT_URL",
    f"http://localhost:{os.environ.get('CRAWLER_AGENT_PORT', '8015')}",
)
MEMORY_AGENT_URL = os.environ.get(
    "MEMORY_AGENT_URL",
    f"http://localhost:{os.environ.get('MEMORY_AGENT_PORT', '8007')}",
)


def fetch_domains():
    # Assumes DATABASE_URL is set in global.env
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in environment")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT domain FROM public.sources")
    domains = [row[0] for row in cur.fetchall()]
    conn.close()
    return domains


async def main():
    try:
        domains = fetch_domains()
        # Apply domain limit for testing
        if args.domain_limit and args.domain_limit > 0:
            domains = domains[: args.domain_limit]
        print(f"Starting unified crawl for {len(domains)} domains...")
    except Exception as e:
        log_error(e, context="fetch_domains")
        return
    try:
        # Try invoking via MCP Bus
        try:
            resp = requests.post(
                f"{mcp_bus_url}/call",
                json={
                    "agent": "crawler",
                    "tool": "unified_production_crawl",
                    "args": [domains],
                    "kwargs": {
                        "max_articles_per_site": args.articles_per_site,
                        "concurrent_sites": args.max_sites,
                    },
                },
                timeout=(5, 300),
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e_bus:
            log_error(e_bus, context="mcp_crawl_bus")
            print("Falling back to direct Crawler Agent endpoint...")
            # Direct call to Crawler Agent
            dresp = requests.post(
                f"{CRAWLER_AGENT_URL}/unified_production_crawl",
                json={
                    "args": [domains],
                    "kwargs": {
                        "max_articles_per_site": args.articles_per_site,
                        "concurrent_sites": args.max_sites,
                    },
                },
                timeout=(5, 300),
            )
            dresp.raise_for_status()
            result = dresp.json()
        print("Crawl complete via Crawler Agent", result)
        # Forward articles to Memory Agent for embedding and storage
        articles = result.get("articles", []) or []
        stored = 0
        # Use MCP Bus to call memory agent store_article tool
        for art in articles:
            try:
                mresp = requests.post(
                    f"{mcp_bus_url}/call",
                    json={
                        "agent": "memory",
                        "tool": "store_article",
                        "args": [art],
                        "kwargs": {},
                    },
                    timeout=(2, 30),
                )
                mresp.raise_for_status()
                stored += 1
            except Exception as mE:
                log_error(mE, context="mcp_memory_store_article")
        print(f"Enqueued {stored}/{len(articles)} articles via MCP Bus to Memory Agent")
    except Exception as e:
        log_error(e, context="mcp_crawl")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log_error(e, context="main_runner")
        raise
