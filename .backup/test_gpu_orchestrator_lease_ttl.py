#!/usr/bin/env python3
"""Test lease TTL expiry logic (opportunistic purge)."""
import time
from fastapi.testclient import TestClient

from agents.gpu_orchestrator.main import app, ALLOCATIONS  # type: ignore


def test_lease_ttl_expiry(monkeypatch):
    # Force very short TTL
    import agents.gpu_orchestrator.main as mod
    monkeypatch.setattr(mod, 'LEASE_TTL_SECONDS', 1)

    client = TestClient(app)

    # Obtain a lease (SAFE_MODE may block; temporarily force SAFE_MODE False)
    monkeypatch.setattr(mod, 'SAFE_MODE', False)
    r = client.post('/lease', json={'agent': 'ttl_tester'})
    body = r.json()
    assert body.get('granted') is True
    token = body.get('token')
    assert token in ALLOCATIONS

    # Age the lease artificially by patching timestamp
    ALLOCATIONS[token]['timestamp'] = time.time() - 4000

    # Trigger purge via allocations call
    r2 = client.get('/allocations')
    allocs = r2.json()['allocations']
    assert token not in allocs  # expired

    # Check metrics show increment
    m = client.get('/metrics').text
    assert 'gpu_orchestrator_lease_expired_total' in m
