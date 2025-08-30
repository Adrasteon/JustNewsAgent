import pytest

from scripts import backfill_article_sources


def test_sha256_hex():
    h = backfill_article_sources.sha256_hex('https://example.com')
    assert isinstance(h, str)
    assert len(h) == 64


# For backfill functions we provide smoke tests that run without a DB by
# passing a dummy connection with the expected cursor API.

class DummyCursor:
    def __init__(self):
        self.executed = []
        self.rows = []
    def execute(self, sql, params=None):
        # minimal emulation: record SQL and optionally set fetchall return
        self.executed.append((sql, params))
        # emulate SELECT id, url FROM public.sources WHERE url_hash IS NULL
        if 'SELECT id, url FROM public.sources' in sql:
            self.rows = [(1, 'https://example.com')]
    def fetchall(self):
        return self.rows
    def fetchone(self):
        return None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass

class DummyConn:
    def __init__(self):
        self.cursor_obj = DummyCursor()
        self.committed = False
    def cursor(self, cursor_factory=None):
        return self.cursor_obj
    def commit(self):
        self.committed = True
    def close(self):
        pass


def test_populate_url_hash_smoke():
    conn = DummyConn()
    backfill_article_sources.populate_url_hash(conn)
    assert conn.committed


def test_backfill_articles_source_id_smoke():
    conn = DummyConn()
    # This will execute the update SQL; ensure it doesn't raise
    backfill_article_sources.backfill_articles_source_id(conn)
    assert conn.committed
