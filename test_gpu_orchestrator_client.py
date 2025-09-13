"""Tests for GPUOrchestratorClient (fallback + caching).

These tests DO NOT require a running orchestrator. Network calls are
simulated via monkeypatching `requests.get`.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from agents.common.gpu_orchestrator_client import GPUOrchestratorClient, _FALLBACK_POLICY  # type: ignore


class DummyResponse:
    def __init__(self, status_code: int, json_data: dict[str, Any]):
        self.status_code = status_code
        self._json = json_data

    def json(self):  # noqa: D401 - simple shim
        return self._json


def test_gpu_info_unreachable(monkeypatch):
    """Client should fail closed (available False) when unreachable."""

    def fake_get(*_a, **_kw):  # noqa: ANN001, ANN002 - test shim
        raise ConnectionError("boom")

    import requests  # Local import to ensure patching the module used in client

    monkeypatch.setattr(requests, "get", fake_get)

    client = GPUOrchestratorClient(base_url="http://127.0.0.1:65500")
    info = client.get_gpu_info()
    assert info["available"] is False
    assert info["gpus"] == []
    assert "message" in info


def test_policy_caching(monkeypatch):
    """Policy should be cached until TTL expires; second call no new fetch."""
    call_count = {"n": 0}

    def fake_get(url, *args, **kwargs):  # noqa: ANN001, ANN002
        call_count["n"] += 1
        assert url.endswith("/policy")
        return DummyResponse(200, {"max_memory_per_agent_mb": 1024, "safe_mode_read_only": True})

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    client = GPUOrchestratorClient(base_url="http://localhost:8014", policy_ttl=5.0)

    first = client.get_policy()
    second = client.get_policy()

    assert call_count["n"] == 1, "Second call should use cache"
    assert first == second
    assert first["max_memory_per_agent_mb"] == 1024


def test_policy_fallback_on_error(monkeypatch):
    """If request errors, fallback policy returned with _fallback flag."""

    def fake_get(*_a, **_kw):  # noqa: ANN001, ANN002
        raise TimeoutError("timeout")

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    client = GPUOrchestratorClient(base_url="http://localhost:8014", policy_ttl=0.1)
    pol = client.get_policy(force_refresh=True)
    assert pol.get("_fallback") is True
    assert pol["safe_mode_read_only"] is True
    assert pol == _FALLBACK_POLICY


def test_cpu_fallback_decision_aggregates(monkeypatch):
    """cpu_fallback_decision aggregates gpu_info + policy assumptions."""

    def fake_get(url, *args, **kwargs):  # noqa: ANN001, ANN002
        if url.endswith("/gpu/info"):
            return DummyResponse(200, {"available": True, "gpus": [{"index": 0}]})
        if url.endswith("/policy"):
            return DummyResponse(200, {"safe_mode_read_only": True})
        raise AssertionError("Unexpected URL")

    import requests

    monkeypatch.setattr(requests, "get", fake_get)
    client = GPUOrchestratorClient(base_url="http://localhost:8014", policy_ttl=1.0)
    decision = client.cpu_fallback_decision()
    # safe_mode True should force use_gpu False even if gpu available
    assert decision["gpu_available"] is True
    assert decision["safe_mode"] is True
    assert decision["use_gpu"] is False
