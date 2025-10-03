"""Helpers for fast in-process integration testing of the pilot pipeline.

These helpers run the pilot pipeline using the in-memory MpcAdapter to avoid
starting Kafka/SR for fast developer feedback.
"""
from kafka.src.agents.adapter_template import MpcAdapter
from kafka.src.agents.scout_adapter import produce_article_created
from kafka.src.agents.crawler_adapter_pilot import start_pilot_crawler
from kafka.src.agents.memory_adapter import start_memory_consumer, InMemoryStorage


def run_pilot_in_process(article_id: str = 'test-1') -> InMemoryStorage:
    adapter = MpcAdapter()
    # Start pipeline handlers using the in-memory adapter
    start_pilot_crawler(adapter=adapter)
    storage = start_memory_consumer(adapter=adapter, storage=InMemoryStorage())

    # Produce an article.created event synchronously; handlers will be invoked
    produce_article_created(article_id, 'Title', 'http://example.com', 'test', published_at=None)

    # Return storage so tests can assert persisted records
    return storage
