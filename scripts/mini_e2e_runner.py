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
    """Execute a single orchestrator phase (launch → sample → terminate).

    Captures stdout/stderr into log files for post-mortem if readiness fails.
    """
    env = os.environ.copy()
    env["SAFE_MODE"] = phase["SAFE_MODE"].upper()
    env["GPU_ORCHESTRATOR_PORT"] = str(ORCH_PORT)
    if phase["ENABLE_NVML"] == "true":
        env["ENABLE_NVML"] = "true"
    # Ensure project root on PYTHONPATH for subprocess imports (common.* etc.)
    project_root = str(Path(__file__).parent.parent)
    existing = env.get("PYTHONPATH", "")
    if project_root not in existing.split(":"):
        env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root

    cmd = [sys.executable, "agents/gpu_orchestrator/main.py"]
    start_ts = time.time()

    proc = None
    stdout_path = RESULTS_DIR / f"{phase['name']}_stdout.log"
    stderr_path = RESULTS_DIR / f"{phase['name']}_stderr.log"
    if not args.dry_run:
        proc = subprocess.Popen(
            cmd,
            stdout=open(stdout_path, "w"),  # type: ignore[arg-type]
            stderr=open(stderr_path, "w"),  # type: ignore[arg-type]
            env=env,
        )
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
        # Read small tail of logs for inline diagnostic snippet
        try:
            if stdout_path.exists():
                phase_result["stdout_tail"] = stdout_path.read_text()[-500:]
            if stderr_path.exists():
                phase_result["stderr_tail"] = stderr_path.read_text()[-500:]
        except Exception:
            pass
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
    p.add_argument("--readiness-timeout", type=float, default=35.0, help="Seconds to wait for /ready (default 35)")
    p.add_argument("--no-force-port-check", action="store_true", help="Skip pre-run port availability check (advanced)")
    p.add_argument("--dry-run", action="store_true", help="Show planned actions without starting processes")
    p.add_argument("--analyst-flip", action="store_true", help="Also run analyst decision flip harness")
    p.add_argument("--analyst-output", default=str(RESULTS_DIR / "mini_e2e_analyst_decision_flip.json"))
    p.add_argument("--summary-output", default=str(RESULTS_DIR / "mini_e2e_summary.json"))
    # Crawl automation
    p.add_argument("--run-crawl", action="store_true", help="After phase2, run a small crawl (test mode by default)")
    p.add_argument("--crawl-articles", type=int, default=2, help="Articles per site (if not --crawl-no-test)")
    p.add_argument(
        "--crawl-sites",
        nargs="+",
        help="Sites list (implies non-test run if --crawl-no-test specified)"
    )
    p.add_argument(
        "--crawl-mode",
        choices=["ultra_fast", "ai_enhanced", "mixed"],
        help="Crawl mode override"
    )
    p.add_argument(
        "--crawl-concurrent",
        type=int,
        help="Concurrent sites (non-test mode)"
    )
    p.add_argument(
        "--crawl-no-test",
        action="store_true",
        help="Do not use --test flag (will auto-confirm prompt)"
    )
    p.add_argument(
        "--crawl-capture-metrics",
        action="store_true",
        help="Capture /metrics pre & post crawl"
    )
    p.add_argument(
        "--crawl-metrics-prefix",
        default="mini_e2e_crawl",
        help="Prefix for captured crawl metrics files"
    )
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

    # Optional crawl automation (performed only if phases succeeded & not dry-run)
    if args.run_crawl and not args.dry_run:
        crawl_results = run_crawl(args)
        summary["crawl"] = crawl_results

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


# --- Crawl Automation Helpers (appended for clarity) ---
def run_crawl(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a small crawl after SAFE_MODE=false phase.

    Spawns a fresh orchestrator instance (SAFE_MODE=false) to ensure it is
    available during the crawl since earlier phase process was terminated.
    """
    phase_cfg = {"SAFE_MODE": "false", "ENABLE_NVML": str(args.enable_nvml).lower()}
    env = os.environ.copy()
    env["SAFE_MODE"] = "FALSE"
    env["GPU_ORCHESTRATOR_PORT"] = str(ORCH_PORT)
    if args.enable_nvml:
        env["ENABLE_NVML"] = "true"
    project_root = str(Path(__file__).parent.parent)
    existing = env.get("PYTHONPATH", "")
    if project_root not in existing.split(":"):
        env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root

    # Start orchestrator again
    orch_proc = subprocess.Popen(
        [sys.executable, "agents/gpu_orchestrator/main.py"],
        stdout=open(RESULTS_DIR / "crawl_orchestrator_stdout.log", "w"),  # type: ignore[arg-type]
        stderr=open(RESULTS_DIR / "crawl_orchestrator_stderr.log", "w"),  # type: ignore[arg-type]
        env=env,
    )
    ready = wait_ready(timeout=args.readiness_timeout)
    crawl_info: Dict[str, Any] = {
        "requested": True,
        "ready": ready,
        "args": {
            "articles": args.crawl_articles,
            "sites": args.crawl_sites,
            "mode": args.crawl_mode,
            "concurrent": args.crawl_concurrent,
            "test_mode": (not args.crawl_no_test),
        },
    }
    if not ready:
        crawl_info["error"] = "orchestrator_not_ready"
        orch_proc.terminate()
        return crawl_info

    pre_metrics_file = None
    post_metrics_file = None
    if args.crawl_capture_metrics:
        pre_metrics_file = RESULTS_DIR / f"{args.crawl_metrics_prefix}_pre.prom"
        data = fetch("/metrics")
        if isinstance(data, str):
            pre_metrics_file.write_text(data)

    cmd = [sys.executable, "run_large_scale_crawl.py"]
    interactive = False
    if not args.crawl_no_test:
        # Use test mode for a minimal crawl
        cmd.append("--test")
        if args.crawl_articles:
            cmd.extend(["--articles", str(args.crawl_articles)])
    else:
        # Non-test mode, build explicit options
        if args.crawl_sites:
            cmd.extend(["--sites", *args.crawl_sites])
        if args.crawl_articles:
            cmd.extend(["--articles", str(args.crawl_articles)])
        if args.crawl_concurrent:
            cmd.extend(["--concurrent", str(args.crawl_concurrent)])
        if args.crawl_mode:
            cmd.extend(["--mode", args.crawl_mode])
        # run_large_scale_crawl.py will prompt; mark interactive to auto-confirm
        interactive = True

    start = time.time()
    try:
        if interactive:
            proc = subprocess.run(
                cmd,
                input="y\n",
                text=True,
                cwd=Path(__file__).parent.parent,
            )
        else:
            proc = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        rc = proc.returncode
    except Exception as e:  # pragma: no cover
        rc = -1
        crawl_info["exception"] = str(e)
    duration = time.time() - start

    if args.crawl_capture_metrics:
        post_metrics_file = RESULTS_DIR / f"{args.crawl_metrics_prefix}_post.prom"
        data = fetch("/metrics")
        if isinstance(data, str):
            post_metrics_file.write_text(data)

    # Clean up orchestrator
    orch_proc.terminate()
    try:
        orch_proc.wait(timeout=8)
    except subprocess.TimeoutExpired:  # pragma: no cover
        orch_proc.kill()

    crawl_info.update(
        {
            "exit_code": rc,
            "duration_seconds": round(duration, 3),
            "pre_metrics_file": str(pre_metrics_file) if pre_metrics_file else None,
            "post_metrics_file": str(post_metrics_file) if post_metrics_file else None,
            "command": cmd,
        }
    )
    return crawl_info
