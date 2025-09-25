---
title: JustNews V4 - Low-risk Action Plan (2025-09-12)
description: Auto-generated description for JustNews V4 - Low-risk Action Plan (2025-09-12)
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews V4 - Low-risk Action Plan (2025-09-12)

This document captures the high-ROI, low-risk improvements agreed for incremental hardening. It mirrors the working todo list to keep intent and implementation aligned.

## Items

1. Gate services with preflight in systemd
   - Add ExecStartPre to run deploy/systemd/preflight.sh before GPU agents/crawler start
   - Use drop-in overrides under /etc/systemd/system/justnews@<instance>.service.d/
   - Set After=network-online.target, TimeoutStartSec=300; block startup on failure
   - Validate with systemd-analyze verify and a dry run

2. Enrich /gpu/info with NVML + MPS
   - Add optional NVML metrics and MPS state to orchestrator /gpu/info
   - Keep CPU fallback and avoid errors when NVML is unavailable

3. Test preload error/refresh paths
   - Extend unit tests to cover 503 on prior failures, refresh clearing, and status errors

4. Harden sudo NOPASSWD helper (Done)
   - Add --dry-run, backups/rollback, XDG cache logging, clearer outputs

5. Operator runbook for preload failures
   - Document common causes and fixes; link from README

6. Backoff and max-wait tuning
   - Unify exponential backoff and sane max wait in preflight and crawler

7. Preflight success summary logging
   - One-line JSON summary with durations; write to stdout and cache file

8. Systemd restart policy hygiene
   - Restart=on-failure; tune StartLimit*; ensure orchestrator ordered before dependents

## Notes
- All changes adhere to the documentation placement rules (markdown_docs/*) and systemd best practices (drop-in overrides, no unit file edits in-repo).
- For production hosts, prefer bare metal with systemd; MPS optional, MIG not available on RTX 3090.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

