"""Kafka scaffold Crawler adapter.

Consumes `justnews.crawl.job.v1` events and performs rendering/extraction
(simulated in the scaffold). Uploads snapshots to the configured object
store and emits `justnews.article.created.v1` events via the transport
adapter.
"""
from __future__ import annotations

import logging
from typing import Any

from .adapter_template import EventEnvelope, get_adapter, get_object_store_client

logger = logging.getLogger(__name__)


class CrawlerAdapter:
    def __init__(self, adapter=None):
        self.adapter = adapter or get_adapter()
        self.store = get_object_store_client()

    def start(self) -> None:
        # Register consumer handler for crawl.job events
        self.adapter.consume("justnews.crawl.job.v1", self.handle_crawl_job)

    def handle_crawl_job(self, topic: str, envelope: EventEnvelope) -> None:
        # Simulate page rendering and extraction
        url = envelope.payload.get("url")
        logger.info("CrawlerAdapter handling crawl job for %s", url)
        # Simulated snapshot bytes (replace with real rendering in production)
        snapshot = f"snapshot:{url}".encode("utf-8")
        # Upload snapshot to object store
        try:
            key = self.store.upload_bytes(snapshot, key_hint="snapshot")
        except Exception:
            logger.exception("Failed to upload snapshot for %s", url)
            # Emit a failure event if desired
            return
        # Build article.created envelope
        created = EventEnvelope(
            event_id=envelope.event_id + "-article",
            event_type="justnews.article.created.v1",
            payload={
                "article_id": envelope.event_id,
                "url": url,
                "snapshot_key": key,
            },
            metadata={"producer_id": "crawler_adapter"},
        )
        # Produce event
        try:
            self.adapter.produce("justnews.article.created.v1", created)
        except Exception:
            logger.exception("Failed to produce article.created event for %s", url)


def run_adapter():
    adapter = get_adapter()
    crawler = CrawlerAdapter(adapter=adapter)
    crawler.start()
    return crawler
