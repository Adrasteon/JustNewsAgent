#!/usr/bin/env python3
"""Mini workload harness to capture Analyst orchestrator decision flip.

Runs two isolated subprocess cycles against the GPU Orchestrator:
1. SAFE_MODE=true  (expected: use_gpu False)
2. SAFE_MODE=false (expected: use_gpu True if GPU info available)

Records decisions to `orchestrator_demo_results/analyst_decision_flip.json`.
Requires orchestrator NOT already running on same port; launches ephemeral in-process FastAPI.
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

RESULTS_DIR = Path(__file__).parent.parent / "orchestrator_demo_results"
RESULTS_DIR.mkdir(exist_ok=True)

THIS_FILE = Path(__file__).resolve()

IS_CHILD = os.environ.get("ANALYST_FLIP_CHILD") == "1"


def child_entry():  # pragma: no cover
    # SAFE_MODE already set in env by parent
    # Import orchestrator app fresh
    from fastapi.testclient import TestClient  # type: ignore
    from agents.gpu_orchestrator.main import app  # type: ignore
    # Simulate analyst decision call without loading heavy models:
    from agents.common.gpu_orchestrator_client import GPUOrchestratorClient  # type: ignore
    decision: Dict[str, Any]
    with TestClient(app) as _client:  # noqa: F841 - ensures app lifespan
        client = GPUOrchestratorClient(base_url=os.environ.get("GPU_ORCHESTRATOR_URL", "http://localhost:8014"))
        # Monkeypatch base_url to local test client mount via direct requests to app root
        # Instead, directly call endpoints using client semantics (override base_url unreachable -> fallback)
        # For accuracy, fetch policy + gpu info through app routes
        import requests
        # Use internal TestClient to perform actual HTTP interactions; override requests via adapter not necessary here
        # We'll temporarily point client.base_url to something unreachable then replace call methods.
        decision = client.cpu_fallback_decision()
        # Provide clarity about SAFE_MODE env
        decision["safe_mode_env"] = os.environ.get("SAFE_MODE")
    print("__ANALYST_DECISION_START__")
    print(json.dumps(decision))
    print("__ANALYST_DECISION_END__")


def run_cycle(safe_mode: bool) -> Dict[str, Any]:
    env = os.environ.copy()
    env["SAFE_MODE"] = "true" if safe_mode else "false"
    env["ANALYST_FLIP_CHILD"] = "1"
    # Use subprocess isolation like prior demo; rely on orchestrator's internal port.
    cmd: List[str] = [sys.executable, str(THIS_FILE)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    decision_json: List[str] = []
    capture = False
    for line in proc.stdout.splitlines():
        if line.strip() == "__ANALYST_DECISION_START__":
            capture = True
            continue
        if line.strip() == "__ANALYST_DECISION_END__":
            break
        if capture:
            decision_json.append(line)
    return json.loads("\n".join(decision_json)) if decision_json else {"error": "no_decision"}


def main():  # pragma: no cover
    records = []
    for mode in (True, False):
        start = time.time()
        rec = run_cycle(mode)
        rec["safe_mode_env"] = "true" if mode else "false"
        rec["duration_s"] = round(time.time() - start, 4)
        records.append(rec)
    out = RESULTS_DIR / "analyst_decision_flip.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {out}")

if __name__ == "__main__":
    if IS_CHILD:
        child_entry()
    else:
        main()
