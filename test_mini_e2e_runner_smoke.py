"""Smoke test for mini E2E orchestrator runner.

Validates that the module loads and phase configs are constructed as expected
without launching subprocesses (pure import + function call).
"""

from scripts.mini_e2e_runner import build_phase_configs


def test_phase_config_structure():
    phases = build_phase_configs(enable_nvml=True)
    assert len(phases) == 2
    assert phases[0]["SAFE_MODE"] == "true"
    assert phases[1]["SAFE_MODE"] == "false"
    assert phases[1]["ENABLE_NVML"] == "true"
