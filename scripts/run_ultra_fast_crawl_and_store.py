#!/usr/bin/env python3
"""Run ultra-fast BBC crawl and store articles via memory agent"""
import asyncio
import json
import os
import sys
import time
# Ensure repository root is on sys.path so `agents` packages can be imported when running as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.scout.production_crawlers.orchestrator import ProductionCrawlerOrchestrator
import requests

MCP_MEMORY_URL = os.environ.get("MEMORY_AGENT_URL", "http://localhost:8007/store_article")

async def run_and_store():
    orchestrator = ProductionCrawlerOrchestrator()
    if 'bbc' not in orchestrator.get_available_sites():
        print("BBC crawler not available")
        return

    print("Starting ultra-fast crawl for 50 articles...")
    result = await orchestrator.crawl_site_ultra_fast('bbc', target_articles=50)
    count = result.get('count', 0)
    print(f"Crawl finished: count={count}")

    articles = result.get('articles', [])
    saved = 0

    def post_with_retries(url, json_payload, timeout=15, max_attempts=3):
        attempt = 0
        backoff = 1
        last_exc = None
        while attempt < max_attempts:
            try:
                r = requests.post(url, json=json_payload, timeout=timeout)
                r.raise_for_status()
                return r
            except requests.exceptions.RequestException as e:
                last_exc = e
                attempt += 1
                if attempt >= max_attempts:
                    raise
                time.sleep(backoff)
                backoff *= 2

    for a in articles:
        content = a.get('content') or ''
        metadata = a.get('metadata')

        # Validate content
        if not content or not content.strip():
            print("Skipping article: empty content")
            continue

        # Normalize metadata to dict
        if metadata is None:
            metadata = {"source": "bbc", "note": "metadata-missing"}
        elif not isinstance(metadata, dict):
            # Try to parse if it's a JSON string
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {"source": "bbc", "note": "metadata-unparseable"}

        payload = {"args": [], "kwargs": {"content": content, "metadata": metadata}}
        try:
            r = post_with_retries(MCP_MEMORY_URL, payload, timeout=15, max_attempts=3)
            resp = r.json()
            # If the endpoint enqueued the article, count as saved for this run
            status = resp.get('status')
            article_id = resp.get('article_id')
            print(f"Stored article id: {article_id} status: {status}")
            saved += 1
        except Exception as e:
            print(f"Failed to store article after retries: {e}")

    print(f"Saved {saved}/{len(articles)} articles")

if __name__ == '__main__':
    asyncio.run(run_and_store())
