#!/usr/bin/env python3
"""
Script to run a unified production crawl against all sources in the JustNews database.
"""
import os
import sys
# Ensure project root is in sys.path for module imports
project_root = os.environ.get("JUSTNEWS_ROOT") or os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Auto-load global.env if environment variables are missing
env_file = os.path.join(project_root, 'deploy', 'systemd', 'env', 'global.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ.setdefault(key, val)

import asyncio
import requests
import psycopg2
from common.observability import log_error

from agents.scout.production_crawlers.unified_production_crawler import UnifiedProductionCrawler  # still imported for fallback

# Agent endpoints
CRAWLER_AGENT_URL = os.environ.get("CRAWLER_AGENT_URL", f"http://localhost:{os.environ.get('CRAWLER_AGENT_PORT', '8015')}")
MEMORY_AGENT_URL = os.environ.get("MEMORY_AGENT_URL", f"http://localhost:{os.environ.get('MEMORY_AGENT_PORT', '8007')}")

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
        print(f"Starting unified crawl for {len(domains)} domains...")
    except Exception as e:
        log_error(e, context="fetch_domains")
        return
    try:
        # Invoke unified_production_crawl via MCP Bus
        mcp_bus_url = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
        resp = requests.post(
            f"{mcp_bus_url}/call",
            json={"agent": "crawler", "tool": "unified_production_crawl", "args": [domains], "kwargs": {}},
            timeout=(5, 300)
        )
        resp.raise_for_status()
        result = resp.json()
        print("Crawl complete via Crawler Agent", result)
        # Forward articles to Memory Agent for embedding and storage
        articles = result.get("articles", []) or []
        stored = 0
        # Use MCP Bus to call memory agent store_article tool
        for art in articles:
            try:
                mresp = requests.post(
                    f"{mcp_bus_url}/call",
                    json={"agent": "memory", "tool": "store_article", "args": [art], "kwargs": {}},
                    timeout=(2, 30)
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
