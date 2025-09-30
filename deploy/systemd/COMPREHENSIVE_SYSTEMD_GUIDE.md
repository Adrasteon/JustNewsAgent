---
title: Comprehensive systemd guide
description: Gating internals, drop-ins, environment tuning for JustNews
---

# Comprehensive systemd guide

This guide explains how the native systemd deployment works, with special focus on gating, environment files, and unit drop-ins.

## Gating model preload (why orchestrator-first)

- Units use `ExecStartPre` to run `preflight.sh --gate-only <instance>`.
- In gate-only mode, the script waits for the GPU Orchestrator on `127.0.0.1:8014` and ensures `/models/preload` completes (or is already “all_ready”).
- Therefore, start `justnews@gpu_orchestrator` first; once READY, other services start cleanly.

Relevant env/tuning:
- `GATE_TIMEOUT` (seconds): how long to wait for orchestrator and preload.
- `REQUIRE_BUS=0` to bypass bus wait in `wait_for_mcp.sh` (rarely needed).

## Environment files

Loaded by the unit template:

```
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/%i.env
```

Minimum keys (examples):

```
JUSTNEWS_PYTHON=/home/adra/miniconda3/envs/justnews-v2-py312/bin/python
SERVICE_DIR=/home/adra/justnewsagent/JustNewsAgent
JUSTNEWS_DB_URL=postgresql://user:pass@localhost:5432/justnews
ENABLE_MPS=true
```

Per-instance overrides (e.g., `/etc/justnews/analyst.env`):

```
CUDA_VISIBLE_DEVICES=0
# EXEC_START can override the module if necessary
# EXEC_START="$JUSTNEWS_PYTHON -m agents.analyst.main"
```

## NVIDIA MPS Configuration (Enterprise GPU Isolation)

Enable NVIDIA Multi-Process Service for GPU resource isolation across agents:

### MPS Setup Steps

1. **Start MPS Control Daemon** (run at system boot):
```bash
sudo nvidia-cuda-mps-control -d
```

2. **Verify MPS Operation**:
```bash
pgrep -x nvidia-cuda-mps-control
ls -la /tmp/nvidia-mps/
```

3. **Environment Configuration**:
   - Global: `ENABLE_MPS=true` in `/etc/justnews/global.env`
   - GPU Orchestrator: `ENABLE_MPS=true` and `ENABLE_NVML=true` in `/etc/justnews/gpu_orchestrator.env`

4. **Validate Configuration**:
```bash
curl -s http://127.0.0.1:8014/mps/allocation | jq '.mps_resource_allocation.system_summary'
curl -s http://127.0.0.1:8014/gpu/info | jq '{mps_enabled, mps}'
```

### MPS Troubleshooting

- **MPS daemon not running**: `sudo nvidia-cuda-mps-control -d`
- **Pipe directory missing**: Check `/tmp/nvidia-mps/` permissions
- **GPU isolation issues**: Verify MPS control process and client connections
- **Memory limits**: Check `config/gpu/mps_allocation_config.json` and restart services

## Unit drop-ins

Place per-instance overrides in `/etc/systemd/system/justnews@<name>.service.d/`.

Templates provided under `units/drop-ins/`:

- `05-gate-timeout.conf` – adjust `Environment=GATE_TIMEOUT=180`
- `10-preflight-gating.conf` – enforce gate-only ExecStartPre
- `20-restart-policy.conf` – tune `Restart=` and `RestartSec=`

After changes: `sudo systemctl daemon-reload`.

## Operations scripts

- `enable_all.sh` – orchestration of enable/disable/start/stop/restart/fresh.
- `health_check.sh` – consolidated status table of systemd/ports/HTTP/READY.
- `preflight.sh` – full validations and model preload gate.

PATH wrappers (optional): small shims installed to `/usr/local/bin` so operators can run commands from any CWD:

```
enable_all.sh, health_check.sh, reset_and_start.sh, cold_start.sh
```

Install examples are in the Quick Reference.

Helpers (optional):
- `helpers/orchestrator-ready.sh` – poll 8014 `/ready` with backoff.
- `helpers/tail-logs.sh` – multi-service log follow with labels.
- `helpers/diag-dump.sh` – capture status/logs/ports and optional `nvidia-smi`.
- `helpers/db-check.sh` – assert DB connectivity based on `JUSTNEWS_DB_URL`.

## Logs and troubleshooting

```
sudo systemctl status justnews@analyst
sudo journalctl -u justnews@analyst -e -n 200 -f
sudo ./deploy/systemd/preflight.sh --stop     # to free occupied ports
sudo ./deploy/systemd/health_check.sh -v
```

If many services fail on first boot, verify `justnews@gpu_orchestrator` is READY.

## Orderly shutdown

Shut down the system cleanly using the orchestration script which issues systemd stops in reverse order to avoid dependency issues:

```
sudo ./deploy/systemd/enable_all.sh stop
```

Behavior:
- Stops all configured `justnews@<instance>` services in reverse dependency order.
- Leaves PostgreSQL running (database is managed separately).
- Respectful timeouts; emits summary and exit code suitable for automation.

Per-instance stop (alternative):

```
sudo systemctl stop justnews@analyst
sudo systemctl stop justnews@scout
```

Also stop the GPU orchestrator if desired:

```
sudo systemctl stop justnews@gpu_orchestrator
```

Troubleshooting:
- If a service hangs, check logs: `journalctl -u justnews@<name> -e -n 200 -f`.
- Free ports and dangling processes: `sudo ./deploy/systemd/preflight.sh --stop`.
- After changes, confirm all ports are free with `deploy/systemd/health_check.sh` (it reports port usage).

## Status panel (auto-refresh)

Launch a non-interactive, auto-refreshing health panel for operators:

```
sudo health_check.sh --panel
```

Behavior:
- Opens a new terminal window when available; otherwise uses tmux or runs inline with `watch`.
- Refresh interval is configurable with `--refresh SEC` (default 2).
- Honors `--host`, `-t/--timeout`, and optional service filters.

Examples:

```
sudo health_check.sh --panel --refresh 3
sudo health_check.sh --panel mcp_bus analyst
```

Requirements:
- `watch` (procps) must be installed on headless servers.
- For GUI terminals, one of x-terminal-emulator/gnome-terminal/konsole/xfce4-terminal/xterm.

## Orchestrator-first and single-command restart

This project gates agent startup on the GPU Orchestrator’s model preload, which avoids cascading failures and noisy restarts. There are two supported paths:

1) One-command fresh restart (recommended)

```
sudo ./deploy/systemd/reset_and_start.sh
```

What it does:
- Stops and disables all services, frees ports in the canonical range
- Optionally reinstalls unit template and helper scripts (see flags in the script)
- Ensures `justnews@gpu_orchestrator` is started and `/ready` reports ready
- Starts MCP Bus, then the rest of the agents in dependency order
- Runs `health_check.sh` and exits non-zero on failure

2) Manual sequence (more control)

```
sudo systemctl enable --now justnews@gpu_orchestrator
curl -fsS http://127.0.0.1:8014/ready
sudo ./deploy/systemd/enable_all.sh start
sudo ./deploy/systemd/health_check.sh
```

Notes and tuning:
- `enable_all.sh` now starts `gpu_orchestrator` first and waits on `/ready` (up to 120s), then MCP Bus, then all remaining services. It accepts `fresh` and the alias `--fresh`.
- `preflight.sh --gate-only <instance>` is invoked by unit drop-ins; it will wait up to `GATE_TIMEOUT` seconds (default 180) for orchestrator and model preload.
- If you must bypass bus wait (e.g., maintenance), set `REQUIRE_BUS=0` in the environment for `wait_for_mcp.sh` (rare).
- Increase timeouts for cold-start scenarios or slow disks/GPUs by setting a drop-in with `Environment=GATE_TIMEOUT=300`.

Failure handling:
- If the orchestrator `READY` probe doesn’t succeed within the timeout, `enable_all.sh` aborts with a clear message. Check `journalctl -u justnews@gpu_orchestrator -f`.
- If MCP Bus health isn’t ready, the script logs a warning and continues; subsequent services will still start due to systemd gating.
- Always run `sudo ./deploy/systemd/health_check.sh -v` after changes to confirm all agents are healthy.

## Cold start (machine reboot)

Use the one-command cold boot to bring the system up from a clean machine restart:

```
sudo ./deploy/systemd/cold_start.sh
```

What it does:
- Enables unit template instances (idempotent)
- Starts PostgreSQL if present (best-effort)
- Ensures GPU Orchestrator is up and `/ready` before starting other agents
- Starts MCP Bus, then all remaining services in order
- Runs `health_check.sh` and returns non-zero on failures

Notes:
- If your installation manages PostgreSQL externally, the script skips it safely.
- If helper scripts or unit template are missing, the script installs them from this repository path when available.
- For slow cold GPU initialization, consider increasing `GATE_TIMEOUT` via a systemd drop-in.

### Auto-start at boot (timer)

Install the service/timer pair to trigger a cold start shortly after boot:

```
sudo cp deploy/systemd/scripts/justnews-cold-start.sh /usr/local/bin/
sudo cp deploy/systemd/scripts/justnews-boot-smoke.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/justnews-cold-start.sh
sudo chmod +x /usr/local/bin/justnews-boot-smoke.sh
sudo cp deploy/systemd/units/justnews-cold-start.service /etc/systemd/system/
sudo cp deploy/systemd/units/justnews-cold-start.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now justnews-cold-start.timer
```

This schedules a one-shot cold start ~45s after boot, after `network-online.target`.

### Optional: Boot-time smoke test (timer)

Install a lightweight smoke test that runs ~2 minutes after boot to verify orchestrator, MCP Bus, and agent /health endpoints. It logs a concise summary to the journal and always exits 0 (so it never flaps):

```
sudo cp deploy/systemd/helpers/boot_smoke_test.sh /usr/local/bin/
sudo cp deploy/systemd/scripts/justnews-boot-smoke.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/justnews-boot-smoke.sh /usr/local/bin/boot_smoke_test.sh
sudo cp deploy/systemd/units/justnews-boot-smoke.service /etc/systemd/system/
sudo cp deploy/systemd/units/justnews-boot-smoke.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now justnews-boot-smoke.timer
```

View results:

```
systemctl list-timers | grep boot-smoke
journalctl -u justnews-boot-smoke.service -e -n 200
```

Tuning (optional):
- `SMOKE_TIMEOUT_SEC`, `SMOKE_RETRIES`, `SMOKE_SLEEP_BETWEEN` can be exported in the environment or set via a systemd drop-in for `justnews-boot-smoke.service`.
- To delay further, increase `OnBootSec` in the timer unit.

