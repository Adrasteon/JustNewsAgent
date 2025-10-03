"""Pilot Crawler adapter: consumes `scout.article.created` and emits `crawler.article.persisted`.

This adapter is a minimal in-memory processor for the pilot pipeline used in unit tests.
"""
from datetime import datetime
from typing import Dict

from .models import ArticleCreated, ArticlePersisted
from .adapter_template import EventEnvelope, get_adapter


def process_scout_article_created(event: Dict) -> Dict:
    """Consume Scout ArticleCreated event and return ArticlePersisted dict.

    This is a simplified pipeline step that would, in production, perform
    extraction/enrichment and then persist the article into Memory before
    emitting a persisted event.
    """
    ac = ArticleCreated(**event)
    # Simulate persistence and enrichment
    persisted = ArticlePersisted(
        id=ac.id,
        url=ac.url,
        persisted_at=datetime.utcnow().isoformat() + "Z",
        source=ac.source,
    )
    return persisted.dict()


def start_pilot_crawler(adapter=None):
    """Register the crawler handler with the transport adapter.

    Adapter should call the provided handler when messages arrive on the
    'scout.article.created' topic.
    """
    adapter = adapter or get_adapter()

    def _handler(topic: str, envelope: EventEnvelope):
        payload = envelope.payload if isinstance(envelope.payload, dict) else envelope.payload
        persisted = process_scout_article_created(payload)
        # Emit persisted event
        persisted_env = EventEnvelope(event_id=envelope.event_id + '-persisted', event_type='crawler.article.persisted', payload=persisted, metadata={'producer': 'crawler_adapter_pilot'})
        adapter.produce('crawler.article.persisted', persisted_env)

    adapter.consume('scout.article.created', _handler)
