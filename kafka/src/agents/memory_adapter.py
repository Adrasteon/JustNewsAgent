"""Memory adapter skeleton: simulates persisting articles and emitting persisted events."""
from datetime import datetime
from typing import Dict, Optional

from .models import ArticlePersisted
from .adapter_template import EventEnvelope, get_adapter


class InMemoryStorage:
    """Simple in-memory store for pilot tests."""

    def __init__(self):
        self.store = {}

    def persist_article(self, ac: Dict) -> ArticlePersisted:
        # Generate a persisted record and store minimal data
        ap = ArticlePersisted(
            id=ac.get('id'),
            url=ac.get('url'),
            persisted_at=datetime.utcnow().isoformat() + 'Z',
            source=ac.get('source'),
        )
        self.store[ap.id] = ap.dict()
        return ap


def persist_and_emit(ac: Dict, storage: Optional[InMemoryStorage] = None) -> Dict:
    storage = storage or InMemoryStorage()
    ap = storage.persist_article(ac)
    # In production this would produce to kafka topic 'crawler.article.persisted'
    return ap.dict()


def start_memory_consumer(adapter=None, storage: Optional[InMemoryStorage] = None):
    adapter = adapter or get_adapter()
    storage = storage or InMemoryStorage()

    def _handler(topic: str, envelope: EventEnvelope):
        payload = envelope.payload if isinstance(envelope.payload, dict) else envelope.payload
        ap = storage.persist_article(payload)
        # Optionally produce an ack event or log
        ack_env = EventEnvelope(event_id=envelope.event_id + '-ack', event_type='memory.article.persisted', payload=ap.dict(), metadata={'producer': 'memory_adapter'})
        adapter.produce('memory.article.persisted', ack_env)

    adapter.consume('crawler.article.persisted', _handler)
    return storage
