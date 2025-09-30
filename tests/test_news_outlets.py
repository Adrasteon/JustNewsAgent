import io
import pytest

from scripts import news_outlets


def sample_md() -> str:
    return """
| URL | Name | Description |
| --- | --- | --- |
| https://www.example.com | Example News | A test publisher |
| https://sub.example.co.uk | Example UK | UK site |
"""


def test_parse_markdown_table_rows_basic():
    md = sample_md()
    rows = list(news_outlets.parse_markdown_table_rows(md))
    assert len(rows) == 2
    assert rows[0][0] == 'https://www.example.com'
    assert rows[0][1] == 'Example News'
    assert 'test' in rows[0][2].lower()


def test_domain_from_url():
    assert news_outlets.domain_from_url('https://www.example.com/path') == 'www.example.com'
    assert news_outlets.domain_from_url('http://example.org') == 'example.org'
    assert news_outlets.domain_from_url('not-a-url') == 'not-a-url'


# Note: upsert_outlets and create_provenance_mappings require a live DB connection.
# We include a smoke test that accepts a monkeypatched connection object which records SQL executed.

class DummyCursor:
    def __init__(self):
        self.executed = []
    def execute(self, sql, params=None):
        self.executed.append((sql, params))
    def fetchone(self):
        return None
    def fetchall(self):
        return []
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


def test_upsert_outlets_smoke(monkeypatch):
    rows = [('https://www.example.com', 'Example News', 'desc')]
    conn = DummyConn()
    ids = news_outlets.upsert_outlets(rows, conn)
    # No real DB -> no ids returned, but function should complete and commit
    assert conn.committed is True


def test_create_provenance_mappings_smoke(monkeypatch):
    conn = DummyConn()
    # Should complete without exception even when tables empty
    news_outlets.create_provenance_mappings(conn, [])
    # ensure we executed the select queries
    found_selects = any('SELECT id, domain FROM public.sources' in s for s, _ in conn.cursor_obj.executed)
    assert found_selects
