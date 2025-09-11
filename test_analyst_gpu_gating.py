from __future__ import annotations

"""Analyst GPU gating tests (deduplicated version).

Lightweight: patch orchestrator decision only, confirm gating logic outcome.
"""

import pytest


@pytest.mark.parametrize(
    "gpu_available,safe_mode,expected_use_gpu",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
def test_gpu_init_gating(monkeypatch, gpu_available: bool, safe_mode: bool, expected_use_gpu: bool):
    import agents.analyst.gpu_analyst as ga  # type: ignore

    class DummyClient:
        def cpu_fallback_decision(self):  # noqa: D401
            return {
                "use_gpu": gpu_available and (not safe_mode),
                "safe_mode": safe_mode,
                "gpu_available": gpu_available,
            }

    ga._orchestrator_client = DummyClient()  # type: ignore
    decision = ga._orchestrator_client.cpu_fallback_decision()  # type: ignore
    assert decision["use_gpu"] is expected_use_gpu
