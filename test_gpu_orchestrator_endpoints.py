from __future__ import annotations

"""Endpoint tests for GPU Orchestrator service.

Mocks `subprocess.check_output` so tests run without real GPU / nvidia-smi.
Single authoritative test set (previous duplicate sections removed).
"""

import subprocess
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents.gpu_orchestrator.main import app, SAFE_MODE  # type: ignore


@pytest.fixture
def client():  # noqa: D401 - standard fixture
    return TestClient(app)


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "safe_mode" in body


def test_policy_get(client):
    r = client.get("/policy")
    assert r.status_code == 200
    body = r.json()
    assert "max_memory_per_agent_mb" in body
    assert "safe_mode_read_only" in body


def test_gpu_info_no_gpu(monkeypatch, client):
    def fake_check_output(*_a, **_kw):  # noqa: ANN001, ANN002
        raise FileNotFoundError("nvidia-smi not present")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    r = client.get("/gpu/info")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["gpus"] == []


def test_gpu_info_with_fake_output(monkeypatch, client):
    csv_line = "0, NVIDIA A100, 40960, 1024, 12, 50, 210"

    def fake_check_output(*_a, **_kw):  # noqa: ANN001, ANN002
        return csv_line

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    r = client.get("/gpu/info")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert len(body["gpus"]) == 1
    gpu0 = body["gpus"][0]
    assert gpu0["memory_total_mb"] == 40960.0
    assert gpu0["memory_used_mb"] == 1024.0
    assert gpu0["utilization_gpu_pct"] == 12.0


def test_policy_post_read_only(client):
    r = client.post("/policy", json={"max_memory_per_agent_mb": 9999})
    assert r.status_code == 200
    body = r.json()
    if SAFE_MODE:
        assert body.get("max_memory_per_agent_mb") != 9999
        assert "note" in body
    else:
        assert body.get("max_memory_per_agent_mb") == 9999