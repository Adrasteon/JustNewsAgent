---
title: Operations Quick Reference
description: Fast, copy-pasteable commands for health checks, service management, preflight gating, GPU orchestrator, and MCP bus operations in production.
tags: [operations, systemd, preflight, orchestrator, mcp, health]
status: current
last_updated: 2025-09-12
---

# Operations Quick Reference

See also: [Systemd Operations Guide](OPERATOR_GUIDE_SYSTEMD.md), [GPU Orchestrator Operations](GPU_ORCHESTRATOR_OPERATIONS.md), [MCP Bus Operations](MCP_BUS_OPERATIONS.md), [Preflight Runbook](preflight_runbook.md)

## Fast checks
```bash
# Ports and health
for p in 8000 8014 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012; do 
  echo "--- :$p/health"; curl -s http://127.0.0.1:$p/health; echo; done

# Bus agents
curl -s http://127.0.0.1:8000/agents | jq

# Orchestrator readiness
curl -s http://127.0.0.1:8014/models/status | jq '{in_progress, all_ready, summary}'
```

## Manage services
```bash
sudo systemctl status justnews@scout.service
sudo systemctl restart justnews@scout.service
sudo journalctl -u justnews@scout.service -e
```

## Preflight
```bash
# Gate-only (systemd pre-start)
deploy/systemd/preflight.sh --gate-only mcp_bus

# Full validation (requires sudo)
sudo deploy/systemd/preflight.sh

# Install NOPASSWD helper
sudo deploy/systemd/setup_preflight_nopasswd.sh --install
```

## Orchestrator
```bash
# Start warmup
curl -s -X POST -H 'Content-Type: application/json' \
 -d '{"refresh": false}' http://127.0.0.1:8014/models/preload
# Status
curl -s http://127.0.0.1:8014/models/status | jq
```

## MCP Bus
```bash
# Agents
curl -s http://127.0.0.1:8000/agents | jq
# Health
curl -s http://127.0.0.1:8000/health
```

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

