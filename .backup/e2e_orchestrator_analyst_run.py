"""Automated E2E validation script for GPU Orchestrator + Analyst + Dashboard.

Steps:
1. Probe orchestrator (/health, /policy, /gpu/info)
2. Probe analyst (/health, /ready)
3. Probe dashboard (/health, /orchestrator/gpu/info)
4. Summarize decision matrix (SAFE_MODE, gpu_available, analyst gating)

Optional: If --require-gpu is passed and orchestrator disallows GPU, exit non-zero.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import requests

from agents.common.gpu_orchestrator_client import GPUOrchestratorClient


def fetch(url: str, timeout=(2, 5)) -> tuple[int, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code, r.json()
    except Exception as e:  # noqa: BLE001
        return 599, {"error": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E Orchestrator + Analyst validation")
    parser.add_argument("--orchestrator", default="http://localhost:8014")
    parser.add_argument("--analyst", default="http://localhost:8004")
    parser.add_argument("--dashboard", default="http://localhost:8013")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if orchestrator denies GPU usage")
    args = parser.parse_args()

    orch = args.orchestrator.rstrip("/")
    analyst = args.analyst.rstrip("/")
    dash = args.dashboard.rstrip("/")

    print("[1] Orchestrator probes")
    h_code, h_body = fetch(f"{orch}/health")
    p_code, p_body = fetch(f"{orch}/policy")
    gi_code, gi_body = fetch(f"{orch}/gpu/info")
    print("    /health:", h_code, h_body)
    print("    /policy:", p_code, p_body)
    print("    /gpu/info:", gi_code, gi_body)

    client = GPUOrchestratorClient(base_url=orch)
    decision = client.cpu_fallback_decision()
    print("    Decision:", decision)

    print("[2] Analyst probes")
    a_h_code, a_h_body = fetch(f"{analyst}/health")
    a_r_code, a_r_body = fetch(f"{analyst}/ready")
    print("    /health:", a_h_code, a_h_body)
    print("    /ready:", a_r_code, a_r_body)

    print("[3] Dashboard probes")
    d_h_code, d_h_body = fetch(f"{dash}/health")
    d_o_code, d_o_body = fetch(f"{dash}/orchestrator/gpu/info")
    print("    /health:", d_h_code, d_h_body)
    print("    /orchestrator/gpu/info:", d_o_code, d_o_body)

    summary = {
        "orchestrator_safe_mode": p_body.get("safe_mode_read_only", True),
        "orchestrator_gpu_available": gi_body.get("available", False),
        "analyst_ready": a_r_body.get("ready", False),
        "dashboard_ok": d_h_code == 200,
        "use_gpu_decision": decision.get("use_gpu", False),
    }
    print("[4] Summary:", summary)

    if h_code != 200 or p_code != 200 or gi_code != 200:
        print("Orchestrator not healthy enough")
        return 10
    if a_h_code != 200 or a_r_body.get("ready") is not True:
        print("Analyst not ready")
        return 11
    if d_h_code != 200:
        print("Dashboard not healthy")
        return 12
    if args.require_gpu and not summary["use_gpu_decision"]:
        print("GPU required but orchestrator denied usage")
        return 13
    print("E2E validation OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
