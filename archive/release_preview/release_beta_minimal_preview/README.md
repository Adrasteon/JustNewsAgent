---
title: Release: Beta Minimal Preview
description: Auto-generated description for Release: Beta Minimal Preview
tags: [documentation]
status: current
last_updated: 2025-10-18
---

# Release: Beta Minimal Preview

This folder contains the new template and helper files required to assemble a minimal native (systemd) preview release of JustNews.

Files in this folder are intended to be copied into the final preview tree or used as templates by operators during a native (non-Docker) install.

- `requirements.txt` — pip dependency list (fallback)
- `stop_services.sh` — graceful shutdown helper
- `run_db_migrations.sh` — wrapper that applies SQL migrations
- `bootstrap_models_from_store.sh` — symlink/bootstrap helpers for Model Store usage
- `create_justnews_user.sh` — preflight user & directory creation helper

Note: this preview intentionally does not include model weights. The Model Store integration scripts create symlinks to externally-managed model artifacts so the preview remains lightweight.
