# GPU Dashboard and Watcher Ingestion — Operations Guide

This document captures how to use the Dashboard agent to visualize GPU metrics and how to ingest the GPU watcher output for historical charts.

## Overview

- Dashboard agent provides a web UI and JSON APIs for GPU status, historical trends, and agent usage.
- A new endpoint accepts GPU watcher JSONL for ingestion into the dashboard’s SQLite history.

## URLs and Ports

- Web UI: http://localhost:8013 (default) or http://localhost:8014 (ad-hoc user run)
- Health: GET /health
- Ready: GET /ready

Config/notes:
- Default port is 8013, configurable via `agents/dashboard/dashboard_config.json` and `agents/dashboard/config.py`.
- Systemd service name: `justnews@dashboard`. Restart to pick up new endpoints.

## Key Endpoints

- GET /gpu/dashboard — consolidated payload (summary, gpu_info, agent_usage, manager metrics)
- GET /gpu/history/db
  - Query params: `hours` (int), `metric` in {`utilization`, `memory`, `temperature`, `performance`}
  - Returns aligned timestamp/value series for Chart.js
- GET /gpu/info — current GPU snapshot (NVML/manager or nvidia-smi fallback)
- GET /gpu/agents — per-agent GPU usage (manager-backed or fallback)
- POST /gpu/ingest_jsonl — ingest GPU watcher JSONL into history
  - Body: `{ "path": "/absolute/or/relative/path.jsonl", "max_lines": 10000 }`
  - Accepts JSONL (one record per line) and also tries to parse older wrapped formats that contain a top-level `samples` array.

### Ingestion JSON shape (tolerant)

Expected per-record keys (unknown fields ignored):
- `time` or `timestamp`: ISO8601 or epoch seconds
- `gpus`: array with items containing some of:
  - `index`, `name`
  - `memory_used_mb` or `memory_used_mib`, `memory_total_mb` or `memory_total_mib`, `memory_free_mb` or `memory_free_mib`
  - `gpu_utilization_percent` or `utilization_percent`
  - `temperature_celsius` or `temperature_c`
  - `power_draw_watts` or `power_watts`

The dashboard computes `memory_utilization_percent` if total/used are provided.

## Typical Flows

- View dashboard UI:
  - http://localhost:8013 or http://localhost:8014 (for a user-run instance)
- Query history APIs:
  - `/gpu/history/db?hours=24&metric=utilization`
  - `/gpu/history/db?hours=24&metric=memory`
  - `/gpu/history/db?hours=24&metric=temperature`
- Ingest watcher output:
  - POST `/gpu/ingest_jsonl` with `{ "path": "gpu_watch_gpu_analyst.jsonl" }`
  - For older “wrapped” files (with `samples`): the endpoint attempts to parse automatically; if needed, pre-clean by extracting `samples` into JSONL and re-post.

## File References (Repo)

- Dashboard service:
  - `agents/dashboard/main.py` — FastAPI app, endpoints (incl. `/gpu/ingest_jsonl`)
  - `agents/dashboard/storage.py` — SQLite schema and read/write helpers
  - `agents/dashboard/templates/dashboard.html` — Web UI (Chart.js)
  - `agents/dashboard/dashboard_config.json`, `agents/dashboard/config.py` — configuration
- Systemd / ops:
  - `deploy/systemd/justnews-start-agent.sh` — standardized agent startup (SAFE_MODE, GPU env)
- GPU watcher (sampler):
  - `deploy/systemd/scripts/gpu_watch.sh` — emits per-line JSON (JSONL)

## Run/Restart Notes

- Systemd restart (requires sudo):
  - `sudo systemctl restart justnews@dashboard`
- User-run (no sudo), useful for testing new endpoints quickly:
  - `DASHBOARD_PORT=8014 python -m agents.dashboard.main`

## Troubleshooting

- 404 on `/gpu/ingest_jsonl`:
  - The service hasn’t picked up the new code. Restart the dashboard agent (systemd or user-run instance).
- No data on `/gpu/history/db` after ingest:
  - Verify path is correct and readable; check logs. Ensure records have `gpus` with required fields.
- UI loads but charts empty:
  - Use the history DB endpoints above to confirm data points exist (look for non-zero `data_points`).

## Next Steps

- Automate periodic ingestion by posting watcher outputs to `/gpu/ingest_jsonl` (e.g., cron/systemd timers).
- Optionally export Prometheus metrics and wire a Grafana dashboard for unified ops alongside the built-in UI.

---

Last updated: 2025-09-10
