"""Small workload harness to exercise Analyst gating under GPU Orchestrator.

Purpose:
- Run a tiny synthetic workload (default 12 text items) through the Analyst sentiment/bias
  paths ONLY if orchestrator policy allows GPU usage.
- Capture decision + runtime metrics to JSON inside metrics/ directory for later
  inspection & CHANGELOG confirmation.

Behavior:
1. Query orchestrator policy + gpu info via client (fail-closed SAFE_MODE semantics)
2. If use_gpu False -> log & exit 0 (not an error) producing metrics file noting CPU fallback
3. If use_gpu True -> import lightweight Analyst inference helpers (lazy import) and
   run sentiment+bias inference on synthetic articles (short constant strings) to avoid IO.
4. Record timings, batch size, number processed, and orchestrator decision data.

Notes:
- This does NOT attempt to load full production pipelines if orchestrator denies GPU.
- Metrics filename pattern: metrics/small_workload_YYYYmmdd_HHMMSS.json
- Safe to run multiple times; each run independent.

Assumptions:
- Analyst agent code exposes `run_gpu_sentiment_batch` and `run_gpu_bias_batch` like
  helpers OR we fallback to hitting Analyst HTTP endpoints if direct import unavailable.
- We keep imports minimal so that in SAFE_MODE we avoid heavy model import cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

try:  # local import – optional
    from agents.common.gpu_orchestrator_client import (
        GPUOrchestratorClient,  # type: ignore
    )
except Exception:  # pragma: no cover - hard failure only if completely missing
    GPUOrchestratorClient = None  # type: ignore

METRICS_DIR = Path("metrics")


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def analyst_http_batch(base: str, texts: list[str]) -> dict[str, Any]:
    # Minimal HTTP fallback hitting analyst endpoints if available
    # Expect an endpoint like /analyze/batch (adjust if project provides different one)
    # If 404 we degrade gracefully.
    try:
        r = requests.post(f"{base.rstrip('/')}/analyze/batch", json={"texts": texts}, timeout=30)
        if r.status_code == 200:
            return {"status": "ok", "result": r.json()}
        return {"status": "error", "http_status": r.status_code, "detail": r.text[:200]}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": str(e)}


def run_gpu_local(texts: list[str]) -> dict[str, Any]:
    # Attempt to import fast GPU analyst helper functions if they exist.
    # We keep broad except to avoid crashing harness on missing dev code.
    try:  # pragma: no cover - import side effects
        from agents.analyst import gpu_analyst  # type: ignore

        if hasattr(gpu_analyst, "run_gpu_sentiment_batch") and hasattr(gpu_analyst, "run_gpu_bias_batch"):
            s_start = time.time()
            sentiment = gpu_analyst.run_gpu_sentiment_batch(texts)
            s_dur = time.time() - s_start
            b_start = time.time()
            bias = gpu_analyst.run_gpu_bias_batch(texts)
            b_dur = time.time() - b_start
            return {
                "status": "ok",
                "sentiment_latency_s": s_dur,
                "bias_latency_s": b_dur,
                "sentiment_result_sample": sentiment[:2],
                "bias_result_sample": bias[:2],
            }
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": f"local GPU analyst import failed: {e}"}
    return {"status": "error", "detail": "No local GPU batch functions available"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small Analyst workload under orchestrator gating")
    parser.add_argument("--orchestrator", default=os.environ.get("GPU_ORCHESTRATOR_URL", "http://localhost:8014"))
    parser.add_argument("--analyst", default="http://localhost:8004")
    parser.add_argument("--items", type=int, default=12, help="Number of synthetic texts")
    parser.add_argument("--http-fallback", action="store_true", help="Use HTTP batch endpoint if local import missing")
    parser.add_argument("--outfile", default=None, help="Explicit output path (overrides metrics dir pattern)")
    args = parser.parse_args()

    if GPUOrchestratorClient is None:
        print("Orchestrator client import failed – exiting fail-closed")
        return 2

    client = GPUOrchestratorClient(base_url=args.orchestrator.rstrip("/"))
    decision = client.cpu_fallback_decision()

    texts = [f"Synthetic news text sample {i}" for i in range(args.items)]

    metrics: dict[str, Any] = {
        "timestamp": timestamp(),
        "orchestrator_url": args.orchestrator,
        "analyst_url": args.analyst,
        "decision": decision,
        "items": args.items,
    }

    if not decision.get("use_gpu", False):
        metrics["workload_skipped"] = True
        metrics["reason"] = "GPU not permitted (SAFE_MODE or denial)"
        out = Path(args.outfile) if args.outfile else METRICS_DIR / f"small_workload_{metrics['timestamp']}.json"
        write_json(out, metrics)
        print(f"Workload skipped; metrics written to {out}")
        return 0

    # GPU permitted – attempt local GPU batch first
    gpu_local = run_gpu_local(texts)
    metrics["local_gpu_attempt"] = gpu_local

    if gpu_local.get("status") != "ok" and args.http_fallback:
        metrics["http_fallback"] = analyst_http_batch(args.analyst, texts)

    out = Path(args.outfile) if args.outfile else METRICS_DIR / f"small_workload_{metrics['timestamp']}.json"
    write_json(out, metrics)
    print(f"Workload complete; metrics written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
