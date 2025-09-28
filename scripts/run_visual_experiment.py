#!/usr/bin/env python3
"""Run a visual-first shadow experiment by enqueuing a unified production crawl job and polling status.

Usage:
  ./scripts/run_visual_experiment.py bbc.com,reuters.com --max-articles 25 --concurrent 3

This script expects the Crawler Agent to be running locally and will POST to /unified_production_crawl.
"""

import argparse
import requests
import time

MCP_CRAWLER_URL = 'http://localhost:8015'


def enqueue_crawl(domains, max_articles=25, concurrent=3):
    endpoint = f"{MCP_CRAWLER_URL}/unified_production_crawl"
    payload = {"args": [domains], "kwargs": {"max_articles_per_site": max_articles, "concurrent_sites": concurrent}}
    resp = requests.post(endpoint, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get('job_id')


def poll_status(job_id, interval=5):
    endpoint = f"{MCP_CRAWLER_URL}/job_status/{job_id}"
    while True:
        resp = requests.get(endpoint, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get('status')
            print(f"Job {job_id} status: {status}")
            if status in ('completed', 'failed'):
                return data
        else:
            print(f"Failed to get status: {resp.status_code}")
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domains', help='Comma-separated list of domains to crawl')
    parser.add_argument('--max-articles', type=int, default=25)
    parser.add_argument('--concurrent', type=int, default=3)
    args = parser.parse_args()

    domains = args.domains.split(',')
    print('Enqueueing crawl for:', domains)
    job_id = enqueue_crawl(domains, args.max_articles, args.concurrent)
    print('Enqueued job:', job_id)
    result = poll_status(job_id)
    print('Job finished:', result)
