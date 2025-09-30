"""Minimal shim for FastAPI used in tests when FastAPI isn't installed.

This shim provides tiny stand-ins for FastAPI and HTTPException so unit
tests can import modules that reference them. It is intentionally minimal and
should NOT be used in production.
"""
class FastAPI:
    def __init__(self, *args, **kwargs):
        self._routes = []

    def get(self, path):
        def _decor(fn):
            return fn
        return _decor

    def post(self, path):
        def _decor(fn):
            return fn
        return _decor


class HTTPException(Exception):
    def __init__(self, status_code=500, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
