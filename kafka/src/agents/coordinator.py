"""Central coordinator skeleton for the Kafka-based JustNews scaffold.

This module provides a minimal Coordinator class responsible for
orchestrating simple workflows (e.g., starting agent adapters and
scheduling periodic tasks). The real implementation will wire up Kafka
consumers/producers and job scheduling.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from .adapter_template import EventEnvelope, get_adapter, TransportAdapter, SeaweedFSClient

logger = logging.getLogger(__name__)


class Coordinator:
    """Minimal coordinator to start adapters and run lightweight tasks."""

    def __init__(self, adapters: Optional[List[TransportAdapter]] = None):
        # If no adapters are provided, create a default adapter using the
        # transport factory. This keeps the scaffold usable with default
        # settings and avoids unused-import lint warnings.
        if adapters is None:
            self.adapters = [get_adapter()]
        else:
            self.adapters = adapters
        self.tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        logger.info("Starting Coordinator")
        # Example: register simple consumer handlers to show extension points
        for adapter in self.adapters:
            adapter.consume("justnews.crawl.job.v1", self._handle_crawl_job)

        # Start background tasks
        self.tasks.append(asyncio.create_task(self._heartbeat_loop()))

    async def stop(self) -> None:
        logger.info("Stopping Coordinator")
        for t in self.tasks:
            t.cancel()

    async def _heartbeat_loop(self) -> None:
        while True:
            logger.debug("Coordinator heartbeat")
            await asyncio.sleep(15)

    def _handle_crawl_job(self, topic: str, envelope: EventEnvelope) -> None:
        url = envelope.payload.get("url")
        logger.info("Received crawl job: %s", url)
        # Simulate rendering and snapshot capture for the scaffold.
        # In the kafka scaffold, real rendering code will replace this.
        snapshot_bytes = f"snapshot-of:{url}".encode("utf-8")
        # Upload snapshot to object store (SeaweedFS placeholder)
        store = SeaweedFSClient()
        object_key = store.upload_bytes(snapshot_bytes, key_hint="snapshot")
        logger.info("Uploaded snapshot to object store: %s", object_key)
        # Emit a simplified article.created event to demonstrate the flow
        created = EventEnvelope(
            event_id=envelope.event_id + "-created",
            event_type="justnews.article.created.v1",
            payload={
                "article_id": envelope.event_id,
                "url": url,
                "content_hash": object_key,
                "snapshot_key": object_key,
            },
            metadata={"producer_id": "coordinator"},
        )
        # In a real adapter, use the adapter.produce to send to Kafka.
        for adapter in self.adapters:
            try:
                adapter.produce("justnews.article.created.v1", created)
            except Exception:
                logger.exception("Failed to produce article.created via adapter")
