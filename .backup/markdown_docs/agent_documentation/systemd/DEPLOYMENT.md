---
title: JustNews native deployment (systemd)
description: Auto-generated description for JustNews native deployment (systemd)
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews native deployment (systemd)

This scaffold lets you run the MCP Bus and all agents natively on Ubuntu using
systemd units and simple per-service environment files.

## What you get
- Unit template: `justnews@.service` (instanced units like `justnews@mcp_bus.service`)
- Global env template: `deploy/systemd/env/global.env`
- Per-service env templates in `deploy/systemd/env/*.env`
- Minimal, stable, and observable deployment without Docker/Kubernetes

## Prepare directories (one-time)
- Create `/etc/justnews/` for environment files (root-owned)
- Optionally create `/var/log/justnews/` for centralized logging

### Model store (optional but recommended)
- If you use centralized per-agent model copies (recommended for live training), create a model store directory on a shared filesystem, for example `/opt/justnews/models`.
- Ensure trainers (writers) and agents (readers) have correct unix permissions. Example (run as root):

	mkdir -p /opt/justnews/models
	chgrp -R justnews /opt/justnews/models
	chmod -R g+rwX /opt/justnews/models

Set `MODEL_STORE_ROOT=/opt/justnews/models` in `/etc/justnews/global.env` (see example in `deploy/systemd/examples/justnews.env.example`).

## Install environment files
- Copy `deploy/systemd/env/global.env` to `/etc/justnews/global.env`
- Copy the per-service `*.env` files to `/etc/justnews/`
- Edit paths and GPU settings as needed

## Install unit template
- Copy `deploy/systemd/units/justnews@.service` to `/etc/systemd/system/`
- Reload: `sudo systemctl daemon-reload`

## Enable and start services
Example for MCP Bus and Analyst:
- `sudo systemctl enable --now justnews@mcp_bus`
- `sudo systemctl enable --now justnews@analyst`

Start all known services at once (starts MCP Bus first and waits for health):
- `sudo ./deploy/systemd/enable_all.sh`
Start a subset:
- `sudo ./deploy/systemd/enable_all.sh analyst scout`

Start from a clean slate (stop everything, free ports, then start):
- `sudo ./deploy/systemd/enable_all.sh --fresh`

Preflight checks only:
- `./deploy/systemd/preflight.sh` (summary)
- `./deploy/systemd/preflight.sh --stop` (stop and wait for ports to free)

## Logs and status
- Status: `systemctl status justnews@analyst`
- Follow logs: `journalctl -u justnews@analyst -f`

## Health checks
- Run: `./deploy/systemd/health_check.sh` (optionally pass instance names)
- Exits non-zero if any service or HTTP health fails; prints a summary table

## Rollback
- Stop/disable specific instances: `sudo ./deploy/systemd/rollback_native.sh analyst scout`
- Stop/disable all known instances: `sudo ./deploy/systemd/rollback_native.sh --all`
- Purge unit/env files (remove from /etc): `sudo ./deploy/systemd/rollback_native.sh --all --purge`

## Notes
- The template reads `/etc/justnews/global.env` and `/etc/justnews/<instance>.env`
- ExecStart uses a Bash subshell to `cd` into the `SERVICE_DIR` before launching
- Prefer absolute paths for the Python interpreter in the env files
- Pin GPUs per service via `CUDA_VISIBLE_DEVICES` in the service env file
- Implement `/health`, `/ready`, `/warmup` in each service to leverage Restart and readiness checks

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md
