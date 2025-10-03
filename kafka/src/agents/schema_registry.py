"""Simple Schema Registry client wrapper (HTTP) for pilot integration.

Provides minimal operations needed by the scaffold:
- register_schema(subject, schema_str) -> schema_id
- get_schema_by_id(schema_id) -> schema_str

This is intentionally small to avoid adding heavy dependencies; it uses `requests`.
"""
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

from typing import Optional

# Apicurio-aware HTTP client implementation (fallbacks for registry-compatible endpoints such as Karapace)
import requests


class SchemaRegistryClient:
    def __init__(self, url: Optional[str] = None):
        # Normalize base URL without trailing slash
        self.url = (url or "http://localhost:8081").rstrip('/')
        # Last successful candidate prefix used by the client (for ops visibility)
        self._last_successful_prefix: Optional[str] = None
        # Candidate endpoint prefixes to try for compatibility with different registries
        # Order: Apicurio native v2 API, Apicurio ccompat layer, Karapace / root-compatible API
        self._candidates = [
            '/apis/registry/v2',   # Apicurio registry v2
            '/apis/ccompat/v6',    # Apicurio ccompat compatibility layer
            '',                    # Karapace / generic registry-compatible root API
        ]

    def _try_post(self, path: str, json_payload: dict):
        for prefix in self._candidates:
            endpoint = f"{self.url}{prefix}{path}"
            try:
                resp = requests.post(endpoint, json=json_payload, timeout=10)
                if resp.status_code in (200, 201):
                    # Record which candidate prefix worked for ops visibility
                    self._last_successful_prefix = prefix or '/'  
                    return resp
            except requests.RequestException:
                # Try next candidate
                continue
        return None

    def _try_put(self, path: str, json_payload: dict):
        for prefix in self._candidates:
            endpoint = f"{self.url}{prefix}{path}"
            try:
                resp = requests.put(endpoint, json=json_payload, timeout=10)
                if resp.status_code in (200, 201):
                    self._last_successful_prefix = prefix or '/'
                    return resp
            except requests.RequestException:
                continue
        return None

    def _try_get(self, path: str):
        for prefix in self._candidates:
            endpoint = f"{self.url}{prefix}{path}"
            try:
                resp = requests.get(endpoint, timeout=10)
                if resp.status_code == 200:
                    self._last_successful_prefix = prefix or '/'
                    return resp
            except requests.RequestException:
                continue
        return None

    def register_schema(self, subject: str, schema_str: str) -> Optional[int]:
        payload = {"schema": schema_str}
        # Try standard register endpoints
        paths = [f"/subjects/{subject}/versions", f"/subjects/{subject}/versions"]
        for p in paths:
            resp = self._try_post(p, payload)
            if resp is not None:
                try:
                    data = resp.json()
                    schema_id = data.get('id') or data.get('schemaId')
                    logger.info('Registered schema for subject %s -> id %s', subject, schema_id)
                    return schema_id
                except Exception:
                    logger.exception('Failed parsing schema registry response for %s', subject)
                    return None
        logger.warning('Failed to register schema for subject %s on any known endpoint', subject)
        return None

    def get_schema_by_id(self, schema_id: int) -> Optional[str]:
        # Try common get-by-id endpoints
        paths = [f"/schemas/ids/{schema_id}", f"/ids/{schema_id}", f"/ids/{schema_id}"]
        for p in paths:
            resp = self._try_get(p)
            if resp is not None:
                try:
                    data = resp.json()
                    # Typical registry responses: {'schema': '...'} or {'content': '...'} depending on implementation
                    schema_text = data.get('schema') or data.get('content') or data.get('schemaText')
                    if schema_text:
                        logger.info('Fetched schema id %s', schema_id)
                        return schema_text
                except Exception:
                    logger.exception('Failed parsing schema registry response for id %s', schema_id)
                    return None
        logger.warning('Schema id %s not found on any known endpoint', schema_id)
        return None

    def set_subject_compatibility(self, subject: str, compatibility: str = 'BACKWARD') -> bool:
        # Try setting subject config using known endpoints: Apicurio /config/{subject} or Karapace-compatible /config/{subject}
        payload = {"compatibility": compatibility}
        paths = [f"/config/{subject}", f"/config/{subject}", f"/compatibility/subjects/{subject}"]
        for p in paths:
            resp = self._try_put(p, payload)
            if resp is not None:
                logger.info('Set compatibility for %s to %s', subject, compatibility)
                return True
        logger.warning('Failed to set compatibility for subject %s', subject)
        return False

    def get_last_successful_prefix(self) -> Optional[str]:
        """Return the last successful endpoint prefix used by the client.

        Returns a short string such as '/apis/registry/v2', '/apis/ccompat/v6',
        or '/' to indicate the root endpoint. Useful for logging and ops.
        """
        return self._last_successful_prefix
