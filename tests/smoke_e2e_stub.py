#!/usr/bin/env python3
"""Smoke E2E stub for MCP Bus + DB worker (in-memory sqlite)

This script starts a minimal HTTP server that implements POST /call (MCP Bus
style). When it receives a call for agent 'db_worker' and tool 'handle_ingest'
it will execute the provided statements against an in-memory sqlite DB and
return a success response. The client side then posts a Scout-style payload to
exercise the full dispatch path used by the crawler.

No external packages required; uses built-in http.server and urllib.
"""
import json
import os
import sys

# Ensure repository root is on sys.path so imports like `agents.*` resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import sqlite3
import time
from urllib.request import Request, urlopen

PORT = 8000

# Create a single in-memory sqlite DB shared by the handler
DB = sqlite3.connect(':memory:', check_same_thread=False)
CUR = DB.cursor()
CUR.execute('''CREATE TABLE public_sources (id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, url_hash TEXT UNIQUE, domain TEXT, canonical TEXT, metadata TEXT, created_at TEXT, updated_at TEXT)''')
CUR.execute('''CREATE TABLE public_article_source_map (id INTEGER PRIMARY KEY AUTOINCREMENT, article_id INTEGER, source_url_hash TEXT, confidence REAL, paywall_flag BOOLEAN, metadata TEXT, created_at TEXT)''')
DB.commit()


class MCPStubHandler(BaseHTTPRequestHandler):
    def _set_json(self, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

    def do_POST(self):
        if self.path != '/call':
            self._set_json(404)
            self.wfile.write(json.dumps({'error': 'not found'}).encode())
            return

        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length).decode('utf-8')
        payload = json.loads(body)

        # Expecting payload: {agent, tool, args, kwargs}
        agent = payload.get('agent')
        tool = payload.get('tool')
        kwargs = payload.get('kwargs', {})

        if agent == 'db_worker' and tool == 'handle_ingest':
            article_payload = kwargs.get('article_payload', {})
            statements = kwargs.get('statements', [])

            # Execute statements mapping Postgres-upsert SQL to sqlite
            try:
                for sql, params in statements:
                    # crude routing similar to tests
                    if 'INSERT INTO public.sources' in sql:
                        # params order: url, url_hash, domain, canonical, metadata, created_at
                        DB.execute('INSERT OR REPLACE INTO public_sources (url, url_hash, domain, canonical, metadata, created_at) VALUES (?, ?, ?, ?, ?, datetime("now"))', params)
                    elif 'INSERT INTO public.article_source_map' in sql:
                        DB.execute('INSERT INTO public_article_source_map (article_id, source_url_hash, confidence, paywall_flag, metadata, created_at) VALUES (?, ?, ?, ?, ?, datetime("now"))', params)
                    else:
                        # ignore unknown SQL in this stub
                        pass
                DB.commit()

                # For the smoke path, return a fake chosen_source_id
                resp = {'status': 'ok', 'url': article_payload.get('url'), 'chosen_source_id': 1}
                self._set_json(200)
                self.wfile.write(json.dumps(resp).encode())
            except Exception as e:
                DB.rollback()
                self._set_json(500)
                self.wfile.write(json.dumps({'status': 'error', 'error': str(e)}).encode())
            return

        # Default: unknown agent/tool
        self._set_json(400)
        self.wfile.write(json.dumps({'status': 'error', 'error': 'unknown agent/tool'}).encode())


def run_server():
    httpd = HTTPServer(('localhost', PORT), MCPStubHandler)
    print(f"MCP Bus stub listening on http://localhost:{PORT}")
    httpd.serve_forever()


def client_post_call(article_payload, statements):
    payload = {
        'agent': 'db_worker',
        'tool': 'handle_ingest',
        'args': [],
        'kwargs': {
            'article_payload': article_payload,
            'statements': statements
        }
    }

    req = Request(f'http://localhost:{PORT}/call', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=5) as resp:
        body = resp.read().decode('utf-8')
        return json.loads(body)


def main():
    # Start server thread
    thr = threading.Thread(target=run_server, daemon=True)
    thr.start()
    time.sleep(0.2)

    # Build sample article and statements using the repo helpers
    from agents.common.ingest import build_source_upsert, build_article_source_map_insert

    article = {
        'url': 'https://example.com/article/42',
        'url_hash': 'abc123',
        'domain': 'example.com',
        'canonical': 'https://example.com/article/42',
        'title': 'Test Article',
        'content': '<p>Example</p>',
        'confidence': 0.7,
        'paywall_flag': False,
        'extraction_metadata': {'method': 'smoke'},
        'timestamp': '2025-08-29T00:00:00',
        'article_id': 99
    }

    source_sql, source_params = build_source_upsert(article)
    asm_sql, asm_params = build_article_source_map_insert(article.get('article_id', 1), article)

    # Convert tuples to lists for JSON
    statements = [[source_sql, list(source_params)], [asm_sql, list(asm_params)]]

    print('Posting /call to MCP Bus stub...')
    res = client_post_call(article, statements)
    print('Response:', res)

    # Verify rows in sqlite
    CUR.execute('SELECT url, url_hash, domain FROM public_sources')
    rows = CUR.fetchall()
    print('public_sources rows:', rows)

    CUR.execute('SELECT article_id, source_url_hash, confidence FROM public_article_source_map')
    rows2 = CUR.fetchall()
    print('public_article_source_map rows:', rows2)

    assert len(rows) == 1 and rows[0][1] == 'abc123'
    assert len(rows2) == 1 and rows2[0][0] == 99

    print('Smoke E2E success')


if __name__ == '__main__':
    main()
