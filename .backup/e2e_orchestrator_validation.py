"""E2E validation runner for GPU Orchestrator + Analyst integration.

Runs a small workload (if available) while sampling orchestrator and analyst
endpoints, then writes a JSON summary artifact for documentation.

Usage:
    python e2e_orchestrator_validation.py \
        --orchestrator http://localhost:8014 \
        --analyst http://localhost:8004 \
        --output orchestrator_e2e_result.json

Exit codes:
 0 success, 2 missing required endpoint, 3 inconsistent GPU gating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

from agents.common.gpu_orchestrator_client import GPUOrchestratorClient


def fetch_json(url: str, timeout=(2, 5)) -> Dict[str, Any]:  # noqa: D401
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    p = argparse.ArgumentParser(description="GPU Orchestrator E2E validation")
    p.add_argument("--orchestrator", default="http://localhost:8014")
    p.add_argument("--analyst", default="http://localhost:8004")
    p.add_argument("--output", default="orchestrator_e2e_result.json")
    p.add_argument("--sample-interval", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=3, help="Number of telemetry samples")
    args = p.parse_args()

    orch_base = args.orchestrator.rstrip("/")
    analyst_base = args.analyst.rstrip("/")

    client = GPUOrchestratorClient(base_url=orch_base)

    result: Dict[str, Any] = {
        "orchestrator": orch_base,
        "analyst": analyst_base,
        "samples": [],
        "policy": None,
        "decision": None,
        "timestamp": time.time(),
    }

    # Baseline health
    try:
        result["health"] = fetch_json(f"{orch_base}/health")
    except Exception as e:  # noqa: BLE001
        print(f"Orchestrator health fetch failed: {e}", file=sys.stderr)
        return 2

    # Policy & decision
    result["policy"] = client.get_policy(force_refresh=True)
    result["decision"] = client.cpu_fallback_decision()

    # Telemetry sampling loop
    for i in range(args.samples):
        snap = client.get_gpu_info()
        result["samples"].append({"i": i, **snap})
        time.sleep(args.sample_interval)

    # Analyst readiness
    try:
        result["analyst_health"] = fetch_json(f"{analyst_base}/health")
        result["analyst_ready"] = fetch_json(f"{analyst_base}/ready")
    except Exception as e:  # noqa: BLE001
        print(f"Analyst endpoint failure: {e}", file=sys.stderr)
        return 2

    # Basic consistency check
    if result["decision"]["use_gpu"] and not result["decision"]["gpu_available"]:
        print("Inconsistent: decision.use_gpu True but gpu_available False", file=sys.stderr)
        return 3

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote E2E summary -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
