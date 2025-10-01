#!/usr/bin/env python3
"""Mini workload harness to capture Analyst orchestrator decision flip.

Updated version runs the orchestrator fully in-memory using FastAPI's TestClient so
`/gpu/info` and `/policy` responses are real (not network fallbacks). Two cycles:

1. SAFE_MODE=true  -> expect use_gpu False even if GPUs present.
2. SAFE_MODE=false -> expect use_gpu True only if GPUs present and available.

Outputs JSON list to an --output path (default: orchestrator_demo_results/analyst_decision_flip.json).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on sys.path when invoked directly so 'agents' can import
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent.parent / "orchestrator_demo_results"
RESULTS_DIR.mkdir(exist_ok=True)

THIS_FILE = Path(__file__).resolve()

IS_CHILD = os.environ.get("ANALYST_FLIP_CHILD") == "1"
OUTPUT_ENV_KEY = "ANALYST_FLIP_OUTPUT_PATH"


def child_entry():  # pragma: no cover
    from fastapi.testclient import TestClient  # type: ignore

    from agents.gpu_orchestrator.main import app  # type: ignore

    safe_mode_env = os.environ.get("SAFE_MODE", "false").lower() == "true"
    with TestClient(app) as client:
        gpu_resp = client.get("/gpu/info")
        policy_resp = client.get("/policy")
        gpu_info = (
            gpu_resp.json()
            if gpu_resp.status_code == 200
            else {"available": False, "gpus": []}
        )
        policy = (
            policy_resp.json()
            if policy_resp.status_code == 200
            else {"safe_mode_read_only": True}
        )

    gpu_available = bool(gpu_info.get("available"))
    decision: dict[str, Any] = {
        "use_gpu": bool(gpu_available and not safe_mode_env),
        "safe_mode": bool(policy.get("safe_mode_read_only", True)),
        "gpu_available": gpu_available,
        "device_count": len(gpu_info.get("gpus", []) or []),
        "nvml_enriched": gpu_info.get("nvml_enriched"),
        "nvml_supported": gpu_info.get("nvml_supported"),
        "safe_mode_env": str(safe_mode_env).lower(),
        "source": "in_memory_testclient",
    }
    print("__ANALYST_DECISION_START__")
    print(json.dumps(decision))
    print("__ANALYST_DECISION_END__")


def run_cycle(safe_mode: bool) -> dict[str, Any]:
    env = os.environ.copy()
    env["SAFE_MODE"] = "true" if safe_mode else "false"
    env["ANALYST_FLIP_CHILD"] = "1"
    cmd: list[str] = [sys.executable, str(THIS_FILE)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    decision_lines: list[str] = []
    capture = False
    for line in proc.stdout.splitlines():
        if line.strip() == "__ANALYST_DECISION_START__":
            capture = True
            continue
        if line.strip() == "__ANALYST_DECISION_END__":
            break
        if capture:
            decision_lines.append(line)
    return (
        json.loads("\n".join(decision_lines))
        if decision_lines
        else {"error": "no_decision"}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU Orchestrator analyst decision flip harness"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "analyst_decision_flip.json"),
        help="Output JSON file path",
    )
    return parser.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for mode in (True, False):
        start = time.time()
        rec = run_cycle(mode)
        rec["requested_safe_mode_env"] = "true" if mode else "false"
        rec["duration_s"] = round(time.time() - start, 4)
        records.append(rec)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    if IS_CHILD:
        child_entry()
    else:
        main()
