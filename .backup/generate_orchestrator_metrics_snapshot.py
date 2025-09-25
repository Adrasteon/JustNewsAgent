#!/usr/bin/env python3
"""Generate consolidated GPU orchestrator metrics snapshot from demo cycles.

Reads the two SAFE_MODE demo JSON outputs and extracts Prometheus metrics values,
writing a consolidated JSON + lightweight text summary for documentation.

Outputs:
- orchestrator_demo_results/metrics_snapshot.json
- orchestrator_demo_results/metrics_snapshot.txt
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any

RESULTS_DIR = Path(__file__).parent / "orchestrator_demo_results"
CYCLE_FILES = {
    "safe_mode_on": RESULTS_DIR / "safe_mode_cycle_on.json",
    "safe_mode_off": RESULTS_DIR / "safe_mode_cycle_off.json",
}

METRIC_PREFIX = "gpu_orchestrator_"
RE_LINE = re.compile(r"^(gpu_orchestrator_[a-zA-Z0-9_]+)\s+([0-9.]+)$")

SELECT_METRICS = [
    "gpu_orchestrator_active_leases",
    "gpu_orchestrator_lease_requests_total",
    "gpu_orchestrator_policy_get_requests_total",
    "gpu_orchestrator_requests_total",
]

def extract_metrics(lines: list[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in lines:
        m = RE_LINE.match(line.strip())
        if m:
            name, val = m.group(1), m.group(2)
            try:
                metrics[name] = float(val)
            except ValueError:
                continue
    return metrics


def load_cycle(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics_map = extract_metrics(data.get("metrics_text", []))
    data["parsed_metrics"] = metrics_map
    return data


def main():  # pragma: no cover - utility script
    summary: Dict[str, Any] = {}
    for label, path in CYCLE_FILES.items():
        if not path.exists():
            raise SystemExit(f"Missing expected file: {path}")
        summary[label] = load_cycle(path)

    # Build comparison table
    diff: Dict[str, float] = {}
    on_metrics = summary["safe_mode_on"]["parsed_metrics"]
    off_metrics = summary["safe_mode_off"]["parsed_metrics"]
    for metric in SELECT_METRICS:
        if metric in on_metrics and metric in off_metrics:
            diff[metric] = off_metrics[metric] - on_metrics[metric]

    # Observations
    observations = []
    if on_metrics.get("gpu_orchestrator_active_leases", 0) == 0 and off_metrics.get("gpu_orchestrator_active_leases", 0) == 1:
        observations.append("Active leases gauge increases from 0 (SAFE_MODE) to 1 (lease granted when SAFE_MODE=false).")
    if on_metrics.get("gpu_orchestrator_lease_requests_total") == 1 and off_metrics.get("gpu_orchestrator_lease_requests_total") == 1:
        observations.append("Lease requests count per cold start remains 1 in each isolated process (expected).")
    if summary["safe_mode_on"]["lease"].get("granted") is False and summary["safe_mode_off"]["lease"].get("granted") is True:
        observations.append("Lease denied with SAFE_MODE note in true cycle; granted with GPU index in false cycle.")

    out_json = {
        "cycles": {
            "safe_mode_on": {
                "safe_mode": True,
                "lease": summary["safe_mode_on"]["lease"],
                "metrics": {k: on_metrics.get(k) for k in SELECT_METRICS},
            },
            "safe_mode_off": {
                "safe_mode": False,
                "lease": summary["safe_mode_off"]["lease"],
                "metrics": {k: off_metrics.get(k) for k in SELECT_METRICS},
            },
        },
        "diff_off_minus_on": diff,
        "observations": observations,
    }

    json_path = RESULTS_DIR / "metrics_snapshot.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    txt_path = RESULTS_DIR / "metrics_snapshot.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("GPU Orchestrator Metrics Snapshot (SAFE_MODE toggle)\n")
        f.write("====================================================\n\n")
        f.write("Selected Metrics (per isolated cold start)\n")
        for metric in SELECT_METRICS:
            f.write(f"{metric}: on={on_metrics.get(metric)} off={off_metrics.get(metric)} diff(off-on)={diff.get(metric)}\n")
        f.write("\nObservations:\n")
        for obs in observations:
            f.write(f"- {obs}\n")

    print(f"Wrote {json_path}\nWrote {txt_path}")

if __name__ == "__main__":
    main()
