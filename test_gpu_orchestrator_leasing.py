"""Tests for GPU Orchestrator lease/release + metrics endpoints.

Focused, fast tests that mock GPU snapshot to ensure deterministic behavior.
"""

import pytest
from fastapi.testclient import TestClient

from agents.gpu_orchestrator.main import ALLOCATIONS, app


@pytest.fixture(autouse=True)
def clear_allocations():
    ALLOCATIONS.clear()
    yield
    ALLOCATIONS.clear()


client = TestClient(app)


def test_metrics_basic():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "gpu_orchestrator_uptime_seconds" in body
    # requests_total may be at least 1 due to this call
    assert "justnews_requests_total" in body


def test_lease_cpu_fallback(monkeypatch):
    # Force snapshot to show no GPU available
    monkeypatch.setattr("agents.gpu_orchestrator.main.get_gpu_snapshot", lambda: {"available": False, "gpus": []})
    r = client.post("/lease", json={"agent": "analyst", "min_memory_mb": 512})
    assert r.status_code == 200
    data = r.json()
    assert data["granted"] is True
    assert data["gpu"] == "cpu"
    assert data["token"] in ALLOCATIONS


def test_lease_safe_mode_denied(monkeypatch):
    # Force SAFE_MODE=True temporarily by monkeypatching module constant
    import agents.gpu_orchestrator.main as mod
    monkeypatch.setattr(mod, "SAFE_MODE", True)
    r = client.post("/lease", json={"agent": "analyst", "min_memory_mb": 128})
    assert r.status_code == 200
    data = r.json()
    assert data["granted"] is False
    assert data.get("note") == "SAFE_MODE"


def test_lease_with_gpu(monkeypatch):
    monkeypatch.setattr(
        "agents.gpu_orchestrator.main.get_gpu_snapshot",
        lambda: {
            "available": True,
            "gpus": [
                {"index": 0, "memory_total_mb": 1000, "memory_used_mb": 400},
                {"index": 1, "memory_total_mb": 1000, "memory_used_mb": 100},
            ],
        },
    )
    r = client.post("/lease", json={"agent": "scout", "min_memory_mb": 32})
    assert r.status_code == 200
    data = r.json()
    assert data["granted"] is True
    # Should choose GPU 1 (lower used memory)
    assert data["gpu"] == 1


def test_release_unknown():
    r = client.post("/release", json={"token": "does-not-exist"})
    assert r.status_code == 404


def test_lease_and_release(monkeypatch):
    monkeypatch.setattr(
        "agents.gpu_orchestrator.main.get_gpu_snapshot",
        lambda: {"available": True, "gpus": [{"index": 0, "memory_total_mb": 1000, "memory_used_mb": 10}]},
    )
    lease_resp = client.post("/lease", json={"agent": "synthesizer"})
    token = lease_resp.json()["token"]
    assert token in ALLOCATIONS
    rel = client.post("/release", json={"token": token})
    assert rel.status_code == 200
    assert token not in ALLOCATIONS
