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
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - dependency should exist
    print("requests dependency missing. Install project requirements.", file=sys.stderr)
    raise


ORCH_PORT = int(os.environ.get("GPU_ORCHESTRATOR_PORT", "8014"))
ORCH_BASE = f"http://localhost:{ORCH_PORT}"  # canonical base
RESULTS_DIR = Path(__file__).parent.parent / "orchestrator_demo_results"
SYSTEMD_ENV_DIR_DEFAULT = Path(__file__).parent.parent / "deploy" / "systemd" / "env"

from common.dev_db_fallback import apply_test_db_env_fallback

# Apply centralized development DB fallback (temporary). Non-destructive.
_applied = apply_test_db_env_fallback()  # returns list of applied vars (unused but could be logged)



def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file ignoring comments and blanks.

    Performs shell-style $VAR expansion after initial collection to allow
    referencing earlier variables inside the same file (best-effort, not full bash).
    """
    data: dict[str, str] = {}
    if not path.exists():  # silent skip
        return data
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            # Strip optional surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            data[key] = val
        # Second pass expansion
        expanded: dict[str, str] = {}
        temp_env = os.environ.copy()
        temp_env.update(data)
        for k, v in data.items():
            expanded[k] = os.path.expandvars(v.replace('$PROJECT_ROOT', temp_env.get('PROJECT_ROOT', '')))
        return expanded
    except Exception as e:  # pragma: no cover - robust to malformed file
        print(f"[mini-e2e] Warning: failed parsing env file {path}: {e}")
        return data


def load_env_sources(env_dir: Path, extra_files: list[str], include_global: bool, overwrite: bool) -> dict[str, str]:
    """Load environment variables from systemd-style env directory.

    Precedence (later overrides earlier if overwrite=True):
      1. global.env (if included)
      2. extra_files in listed order
    If overwrite is False existing os.environ keys win.
    Returns dict of variables actually injected (post precedence resolution).
    """
    collected: dict[str, str] = {}
    loaded_sequence: list[Path] = []
    if include_global:
        g = env_dir / 'global.env'
        if g.exists():
            collected.update(parse_env_file(g))
            loaded_sequence.append(g)
    for name in extra_files:
        p = Path(name)
        if not p.is_absolute():
            p = env_dir / name
        if p.exists():
            # Merge respecting overwrite flag
            file_vars = parse_env_file(p)
            for k, v in file_vars.items():
                if overwrite or k not in collected:
                    collected[k] = v
            loaded_sequence.append(p)
        else:
            print(f"[mini-e2e] Note: env file not found: {p}")
    # Final application respecting existing environment if overwrite disabled
    applied: dict[str, str] = {}
    for k, v in collected.items():
        if not overwrite and k in os.environ:
            continue
        applied[k] = v
    if applied:
        os.environ.update(applied)
    if loaded_sequence:
        print("[mini-e2e] Loaded env files:")
        for lp in loaded_sequence:
            print(f"  - {lp}")
    print(f"[mini-e2e] Injected {len(applied)} variables from env directory")
    return applied


def build_phase_configs(enable_nvml: bool) -> list[dict[str, Any]]:
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


def fetch(endpoint: str, timeout: float = 4.0) -> Any | None:
    url = f"{ORCH_BASE}{endpoint}" if not endpoint.startswith("http") else endpoint
    try:
        r = requests.get(url, timeout=timeout)
        if "/metrics" in endpoint:
            return r.text
        return r.json()
    except Exception as e:  # pragma: no cover - network variability
        return {"error": str(e)}


def request_lease() -> dict[str, Any]:
    try:
        r = requests.post(f"{ORCH_BASE}/lease", json={"agent": "mini_e2e", "purpose": "test"}, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_phase(phase: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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

    phase_result: dict[str, Any] = {
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


def derive_summary(phases: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"phases": phases}
    # Basic assertions / interpretations
    p1 = phases[0]
    p2 = phases[1] if len(phases) > 1 else {}
    summary["expectation_checks"] = {
        "phase1_safe_mode_denied": bool(p1.get("lease", {}).get("safe_mode", True)),
        "phase2_gpu_possible": not bool(p2.get("lease", {}).get("safe_mode", False)),
    }
    return summary


def maybe_run_analyst_flip(args: argparse.Namespace) -> str | None:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    # Environment loading options
    p.add_argument("--env-dir", type=Path, default=SYSTEMD_ENV_DIR_DEFAULT, help="Directory containing systemd style env files (default deploy/systemd/env)")
    p.add_argument(
        "--env-files",
        nargs="+",
        default=[],
        help="Additional env file basenames to load after global.env (e.g. memory.env scout.env)"
    )
    p.add_argument("--no-global-env", action="store_true", help="Do not auto-load global.env")
    p.add_argument("--overwrite-env", action="store_true", help="Allow env files to overwrite existing environment variables")
    p.add_argument("--report-missing", action="store_true", help="Report missing critical variables (MCP_BUS_URL / DATABASE_URL)")
    # DB preflight & seeding
    p.add_argument("--preflight-db", action="store_true", help="Run DB connectivity preflight before phases")
    p.add_argument("--auto-seed-sources", action="store_true", help="Seed sources table if missing/empty (implies --preflight-db)")
    p.add_argument(
        "--sources-md",
        type=Path,
        default=Path(__file__).parent.parent / "markdown_docs" / "agent_documentation" / "potential_news_sources.md",
        help="Path to sources markdown for seeding",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load env files early so subsequent phases inherit
    injected_env: dict[str, str] = {}
    if args.env_dir:
        injected_env = load_env_sources(
            env_dir=Path(args.env_dir),
            extra_files=args.env_files,
            include_global=not args.no_global_env,
            overwrite=args.overwrite_env,
        )
    # Fallback mapping: if JUSTNEWS_DB_* present but POSTGRES_* missing, map them for consistency
    if os.environ.get("JUSTNEWS_DB_HOST") and not os.environ.get("POSTGRES_HOST"):
        os.environ.setdefault("POSTGRES_HOST", os.environ["JUSTNEWS_DB_HOST"])
    if os.environ.get("JUSTNEWS_DB_NAME") and not os.environ.get("POSTGRES_DB"):
        os.environ.setdefault("POSTGRES_DB", os.environ["JUSTNEWS_DB_NAME"])
    if os.environ.get("JUSTNEWS_DB_USER") and not os.environ.get("POSTGRES_USER"):
        os.environ.setdefault("POSTGRES_USER", os.environ["JUSTNEWS_DB_USER"])
    if os.environ.get("JUSTNEWS_DB_PASSWORD") and not os.environ.get("POSTGRES_PASSWORD"):
        os.environ.setdefault("POSTGRES_PASSWORD", os.environ["JUSTNEWS_DB_PASSWORD"])
    missing_crit: list[str] = []
    if args.report_missing:
        crit_keys = ["MCP_BUS_URL", "DATABASE_URL", "POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER"]
        for ck in crit_keys:
            if not os.environ.get(ck):
                missing_crit.append(ck)
        if missing_crit:
            print(f"[mini-e2e] Missing critical variables: {', '.join(missing_crit)}")
        else:
            print("[mini-e2e] All critical variables present")

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
    phase_results: list[dict[str, Any]] = []

    # Optional DB preflight (runs before orchestrator phases)
    db_preflight_result: dict[str, Any] = {}
    if args.auto_seed_sources:
        args.preflight_db = True  # implicit
    if args.preflight_db:
        db_preflight_result = run_db_preflight(args)

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
    summary["env_injected_count"] = len(injected_env)
    summary["env_injected_keys"] = sorted(list(injected_env.keys()))[:40]  # trim for brevity
    if injected_env and len(injected_env) > 40:
        summary["env_injected_truncated"] = True
    if 'missing_crit' in locals() and missing_crit:
        summary["missing_critical_env"] = missing_crit

    # Optional crawl automation (performed only if phases succeeded & not dry-run)
    if args.run_crawl and not args.dry_run:
        crawl_results = run_crawl(args)
        summary["crawl"] = crawl_results

    if db_preflight_result:
        summary["db_preflight"] = db_preflight_result

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


# --- Crawl Automation Helpers (defined before __main__ to avoid NameError) ---
def run_crawl(args: argparse.Namespace) -> dict[str, Any]:
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
    crawl_info: dict[str, Any] = {
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


# --- DB Preflight & Source Seeding -------------------------------------------------
def run_db_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Verify DB connectivity and optionally seed sources.

    Strategy:
      1. Determine connection parameters from DATABASE_URL or JUSTNEWS_DB_* / POSTGRES_* envs.
      2. Attempt simple SELECT 1.
      3. If --auto-seed-sources: check sources table existence & row count; seed if missing or empty.
    Returns structured diagnostic dict (never raises).
    """
    result: dict[str, Any] = {"requested": True}
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Build from component vars
        host = os.environ.get("JUSTNEWS_DB_HOST") or os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("JUSTNEWS_DB_PORT", "5432")
        name = os.environ.get("JUSTNEWS_DB_NAME") or os.environ.get("POSTGRES_DB", "justnews")
        user = os.environ.get("JUSTNEWS_DB_USER") or os.environ.get("POSTGRES_USER", "justnews_user")
        pwd = os.environ.get("JUSTNEWS_DB_PASSWORD") or os.environ.get("POSTGRES_PASSWORD", "password123")
        db_url = f"postgresql://{user}:{pwd}@{host}:{port}/{name}"
        result["composed_url"] = True
    result["db_url_present"] = bool(db_url)

    # Use psql for lightweight check if available (avoids direct psycopg2 dependency here)
    if not shutil.which("psql"):
        result["error"] = "psql_not_found"
        return result
    # tempfile may be used later if extended (placeholder for future diagnostics dumps)
    import tempfile  # noqa: F401

    check_cmd = ["psql", db_url, "-tAc", "SELECT 1"]
    try:
        proc = subprocess.run(check_cmd, capture_output=True, text=True, timeout=8)
        result["select1_rc"] = proc.returncode
        result["select1_out"] = proc.stdout.strip()
        if proc.returncode != 0:
            result["stderr"] = proc.stderr.strip()[:400]
            return result
    except Exception as e:  # pragma: no cover
        result["exception"] = str(e)
        return result

    # Optionally seed sources
    if args.auto_seed_sources:
        # Determine if table exists & row count
        table_check_cmd = [
            "psql",
            db_url,
            "-tAc",
            "SELECT to_regclass('public.sources') IS NOT NULL AS exists, COALESCE((SELECT count(*) FROM public.sources),0) AS count"
        ]
        try:
            proc2 = subprocess.run(table_check_cmd, capture_output=True, text=True, timeout=10)
            if proc2.returncode == 0:
                # Output like: 't|42' or 'f|0'
                raw = proc2.stdout.strip()
                result["sources_raw"] = raw
                if '|' in raw:
                    exists_part, count_part = raw.split('|', 1)
                    exists = exists_part.strip() in ('t', 'true', 'True')
                    try:
                        count_val = int(count_part.strip())
                    except ValueError:
                        count_val = -1
                    result["sources_exists"] = exists
                    result["sources_count"] = count_val
                    need_seed = (not exists) or count_val == 0
                else:
                    need_seed = True
            else:
                result["sources_check_error"] = proc2.stderr.strip()[:400]
                need_seed = True
        except Exception as e:  # pragma: no cover
            result["sources_check_exception"] = str(e)
            need_seed = True

        result["seed_attempted"] = False
        if need_seed:
            seed_script = Path(__file__).parent / "news_outlets.py"
            if not seed_script.exists():
                # Fallback to scripts directory (already there actually)
                seed_script = Path(__file__).parent / "news_outlets.py"
            if seed_script.exists() and args.sources_md.exists():
                seed_cmd = [sys.executable, str(seed_script), "--file", str(args.sources_md)]
                try:
                    seed_proc = subprocess.run(seed_cmd, capture_output=True, text=True, timeout=120)
                    result["seed_attempted"] = True
                    result["seed_rc"] = seed_proc.returncode
                    if seed_proc.returncode != 0:
                        result["seed_stderr"] = seed_proc.stderr[-500:]
                    else:
                        result["seed_stdout_tail"] = seed_proc.stdout[-500:]
                except Exception as e:  # pragma: no cover
                    result["seed_exception"] = str(e)
            else:
                result["seed_skipped_reason"] = "script_or_markdown_missing"
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
