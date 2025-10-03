"""Scout adapter skeleton: produces ArticleCreated events.

This skeleton provides a simple function that constructs a validated event payload
using the `ArticleCreated` Pydantic model. In production this would serialize to Avro
and push to Kafka; for unit tests we return a dict representation.
"""
from typing import Dict

from .models import ArticleCreated
from .adapter_template import EventEnvelope, get_adapter


def make_article_created_event(article_id: str, title: str, url: str, source: str, published_at: str = None) -> Dict:
    """Construct and validate an ArticleCreated event.

    Returns a dictionary suitable for serializing to Avro or JSON.
    """
    event = ArticleCreated(id=article_id, title=title, url=url, source=source, published_at=published_at)
    # In real system, encode with Avro and send to Kafka topic 'scout.article.created'
    return event.dict()


def produce_article_created(article_id: str, title: str, url: str, source: str, published_at: str = None) -> None:
    """Create an ArticleCreated envelope and produce it via transport adapter."""
    envelope = EventEnvelope(event_id=article_id, event_type='scout.article.created', payload=make_article_created_event(article_id, title, url, source, published_at), metadata={'producer': 'scout_adapter'})
    adapter = get_adapter()
    adapter.produce('scout.article.created', envelope)
