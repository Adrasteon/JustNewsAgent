#!/usr/bin/env python3
"""NVML metrics flag test (mocked).

Ensures /metrics exposes NVML gauges when ENABLE_NVML=true, SAFE_MODE=false,
and internal _NVML_SUPPORTED is True.
"""
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, '/home/adra/justnewsagent/JustNewsAgent')

def test_nvml_metrics_gauges(monkeypatch):
    # Import module
    import agents.gpu_orchestrator.main as orchestrator

    # Force conditions
    monkeypatch.setattr(orchestrator, 'ENABLE_NVML', True)
    monkeypatch.setattr(orchestrator, 'SAFE_MODE', False)
    monkeypatch.setattr(orchestrator, '_NVML_SUPPORTED', True)
    monkeypatch.setattr(orchestrator, '_NVML_INIT_ERROR', None)

    # Fake a handle cache (not used directly without real pynvml access)
    monkeypatch.setattr(orchestrator, '_NVML_HANDLE_CACHE', {0: object()})

    client = TestClient(orchestrator.app)
    resp = client.get('/metrics')
    body = resp.text
    assert 'gpu_orchestrator_nvml_supported' in body
    assert 'gpu_orchestrator_nvml_error_info' not in body  # no error expected

    # Also verify /gpu/info includes enrichment flag when NVML enabled
    gi = client.get('/gpu/info').json()
    # nvml_enriched may be False if nvidia-smi unavailable; we still expect flag keys
    assert 'nvml_enriched' in gi
    assert 'nvml_supported' in gi
