#!/usr/bin/env python3
"""Mini E2E Fresh Start Runner for GPU Orchestrator (SAFE_MODE Flip)

Purpose:
  Automates a two-phase orchestrator validation demonstrating SAFE_MODE gating
  and subsequent GPU lease allowance. Produces structured artifacts under
  `orchestrator_demo_results/` without requiring full systemd integration.

Features:
  * Fresh port check & optional force-kill of existing orchestrator on 8014
  * Phase 1: Launch orchestrator with SAFE_MODE=true → capture /health, /metrics,
             /lease (expected denial)
  * Phase 2: Relaunch with SAFE_MODE=false (and optional ENABLE_NVML=true)
             → capture /health, /metrics, /lease (expected grant when GPU present)
  * Metrics diff & lease outcome summary JSON
  * Dry-run mode (no processes started) for CI smoke testing
  * Optional invocation of the analyst decision flip harness for deeper demo

Contracts:
  Inputs: CLI flags only (no positional args)
  Outputs: JSON + raw metrics text files:
    - mini_e2e_phase1_metrics.prom
    - mini_e2e_phase2_metrics.prom
    - mini_e2e_summary.json
    - (optional) mini_e2e_analyst_decision_flip.json

Exit Codes:
  0 success, 1 unexpected error, 2 readiness timeout, 3 SAFE_MODE breach logic

Safety:
  * Fails closed if requests cannot reach orchestrator
  * Does not delete model caches or persistent DB data
  * Only targets orchestrator process lifecycle (not other agents)

Usage Examples:
  python scripts/mini_e2e_runner.py
  python scripts/mini_e2e_runner.py --enable-nvml
  python scripts/mini_e2e_runner.py --analyst-flip --analyst-output custom_flip.json
  python scripts/mini_e2e_runner.py --dry-run

"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import requests  # type: ignore
except ImportError as e:  # pragma: no cover - dependency should exist
    print("requests dependency missing. Install project requirements.", file=sys.stderr)
    raise


ORCH_PORT = int(os.environ.get("GPU_ORCHESTRATOR_PORT", "8014"))
ORCH_BASE = f"http://localhost:{ORCH_PORT}"  # canonical base
RESULTS_DIR = Path(__file__).parent.parent / "orchestrator_demo_results"


def build_phase_configs(enable_nvml: bool) -> List[Dict[str, Any]]:
    """Return ordered phase configuration for SAFE_MODE flip.

    Exposed for smoke tests (no side effects).
    """
    return [
        {"name": "phase1_safe_mode_on", "SAFE_MODE": "true", "ENABLE_NVML": "false"},
        {"name": "phase2_safe_mode_off", "SAFE_MODE": "false", "ENABLE_NVML": str(enable_nvml).lower()},
    ]


def port_in_use(port: int) -> bool:
    """Check if a TCP port is in LISTEN state using 'ss'."""
    try:
        r = subprocess.run(["ss", "-ltn", f"sport = :{port}"], capture_output=True, text=True, timeout=3)
        return "LISTEN" in r.stdout
    except Exception:
        return False


def kill_port(port: int) -> None:
    """Attempt to kill process(es) bound to port using fuser (Linux-specific)."""
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # fuser not installed; fallback best-effort with lsof
        try:
            lsof = subprocess.run(["lsof", f"-i:{port}"], capture_output=True, text=True)
            for line in lsof.stdout.splitlines()[1:]:
                parts = line.split()
                if parts:
                    pid = parts[1]
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
        except Exception:
            pass


def wait_ready(timeout: float = 25.0) -> bool:
    """Poll /ready until success or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{ORCH_BASE}/ready", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.75)
    return False


def fetch(endpoint: str, timeout: float = 4.0) -> Optional[Any]:
    url = f"{ORCH_BASE}{endpoint}" if not endpoint.startswith("http") else endpoint
    try:
        r = requests.get(url, timeout=timeout)
        if "/metrics" in endpoint:
            return r.text
        return r.json()
    except Exception as e:  # pragma: no cover - network variability
        return {"error": str(e)}


def request_lease() -> Dict[str, Any]:
    try:
        r = requests.post(f"{ORCH_BASE}/lease", json={"agent": "mini_e2e", "purpose": "test"}, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_phase(phase: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Execute a single orchestrator phase (launch → sample → terminate)."""
    env = os.environ.copy()
    env["SAFE_MODE"] = phase["SAFE_MODE"].upper()
    env["GPU_ORCHESTRATOR_PORT"] = str(ORCH_PORT)
    if phase["ENABLE_NVML"] == "true":
        env["ENABLE_NVML"] = "true"

    cmd = [sys.executable, "agents/gpu_orchestrator/main.py"]
    start_ts = time.time()

    proc = None
    if not args.dry_run:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        ready_ok = wait_ready(timeout=args.readiness_timeout)
    else:
        ready_ok = True

    phase_result: Dict[str, Any] = {
        "phase": phase["name"],
        "safe_mode": phase["SAFE_MODE"],
        "enable_nvml": phase["ENABLE_NVML"],
        "ready": ready_ok,
        "start_time": start_ts,
    }

    if not ready_ok:
        phase_result["error"] = "readiness_timeout"
    else:
        phase_result["health"] = fetch("/health")
        phase_result["policy"] = fetch("/policy")
        phase_result["gpu_info"] = fetch("/gpu/info")
        lease_resp = request_lease()
        phase_result["lease"] = lease_resp
        metrics_text = fetch("/metrics") or ""
        metrics_file = RESULTS_DIR / f"mini_e2e_{phase['name']}_metrics.prom"
        if not args.dry_run:
            metrics_file.write_text(metrics_text if isinstance(metrics_text, str) else str(metrics_text))

    # Terminate process
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    phase_result["duration_seconds"] = round(time.time() - start_ts, 3)
    return phase_result


def derive_summary(phases: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"phases": phases}
    # Basic assertions / interpretations
    p1 = phases[0]
    p2 = phases[1] if len(phases) > 1 else {}
    summary["expectation_checks"] = {
        "phase1_safe_mode_denied": bool(p1.get("lease", {}).get("safe_mode", True)),
        "phase2_gpu_possible": not bool(p2.get("lease", {}).get("safe_mode", False)),
    }
    return summary


def maybe_run_analyst_flip(args: argparse.Namespace) -> Optional[str]:
    if not args.analyst_flip or args.dry_run:
        return None
    out_path = Path(args.analyst_output)
    cmd = [sys.executable, "scripts/mini_orchestrator_analyst_flip.py", "--output", str(out_path)]
    try:
        subprocess.run(cmd, check=True)
        return str(out_path)
    except subprocess.CalledProcessError as e:  # pragma: no cover
        print(f"Analyst flip harness failed: {e}", file=sys.stderr)
        return None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini E2E Orchestrator SAFE_MODE flip runner")
    p.add_argument("--enable-nvml", action="store_true", help="Enable NVML during phase2 (SAFE_MODE=false)")
    p.add_argument("--force-kill", action="store_true", help="Force kill existing orchestrator on port before start")
    p.add_argument("--readiness-timeout", type=float, default=25.0, help="Seconds to wait for /ready")
    p.add_argument("--dry-run", action="store_true", help="Show planned actions without starting processes")
    p.add_argument("--analyst-flip", action="store_true", help="Also run analyst decision flip harness")
    p.add_argument("--analyst-output", default=str(RESULTS_DIR / "mini_e2e_analyst_decision_flip.json"))
    p.add_argument("--summary-output", default=str(RESULTS_DIR / "mini_e2e_summary.json"))
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if port_in_use(ORCH_PORT):
        if args.force_kill:
            print(f"[mini-e2e] Port {ORCH_PORT} busy → attempting kill")
            kill_port(ORCH_PORT)
            time.sleep(1.0)
            if port_in_use(ORCH_PORT):
                print(f"[mini-e2e] Port {ORCH_PORT} still in use after kill attempt", file=sys.stderr)
                return 1
        else:
            print(f"[mini-e2e] Orchestrator port {ORCH_PORT} already in use. Use --force-kill if intentional.")
            return 1

    phases_cfg = build_phase_configs(enable_nvml=args.enable_nvml)
    phase_results: List[Dict[str, Any]] = []

    for cfg in phases_cfg:
        print(f"[mini-e2e] Running {cfg['name']} SAFE_MODE={cfg['SAFE_MODE']} ENABLE_NVML={cfg['ENABLE_NVML']}")
        res = run_phase(cfg, args)
        phase_results.append(res)
        if not res.get("ready"):
            print(f"[mini-e2e] Phase {cfg['name']} failed readiness", file=sys.stderr)
            summary = derive_summary(phase_results)
            if not args.dry_run:
                Path(args.summary_output).write_text(json.dumps(summary, indent=2))
            return 2

    analyst_path = maybe_run_analyst_flip(args)
    summary = derive_summary(phase_results)
    summary["analyst_decision_flip_artifact"] = analyst_path
    summary["dry_run"] = args.dry_run
    summary["nvml_requested"] = args.enable_nvml

    if not args.dry_run:
        Path(args.summary_output).write_text(json.dumps(summary, indent=2))
        print(f"[mini-e2e] Summary written → {args.summary_output}")

    # Basic logical check: phase1 should deny (safe_mode true), phase2 should allow potential GPU
    checks = summary["expectation_checks"]
    if not checks.get("phase1_safe_mode_denied", True):
        print("[mini-e2e] Unexpected: Phase1 did not appear to be in SAFE_MODE denial state", file=sys.stderr)
        return 3
    print("[mini-e2e] Completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
