---
name: P0 Hardening — Reliability & Safety
about: Track P0 tasks to harden MCP Bus, health contracts, and observability
labels: [P0, reliability, MCP, observability]
assignees: []
---

## Summary

Hardening tasks for production reliability and safety. Derived from System_Assessment_2025-08-09.

## Tasks

- [ ] MCP Bus resilience
  - [ ] Add request timeouts for `/call` (client + server)
  - [ ] Implement retries with exponential backoff and jitter
  - [ ] Add circuit breaker around downstream agent calls
  - [ ] Support idempotency keys for `/call`
  - [ ] Define per-agent SLAs and failure budgets

- [ ] Health/Readiness/Warmup
  - [ ] Standardize `/health` (liveness), `/ready` (dependencies OK), `/warmup` (preload engines)
  - [ ] Ensure responses are non-sensitive; include minimal status codes/flags only
  - [ ] Add startup warm-up routine to pre-load TensorRT engines and caches

- [ ] Observability stack
  - [ ] OpenTelemetry tracing across MCP Bus and agents
  - [ ] Prometheus metrics (latency p50/p95/p99, error rates, queue depth)
  - [ ] GPU metrics (utilization, allocated/reserved memory)
  - [ ] Grafana dashboard with service and GPU panels

- [ ] Schema contracts
  - [ ] Centralize/version Pydantic schemas for tool args/kwargs
  - [ ] Add contract tests to prevent MCP interface drift

- [ ] Image/dependency hygiene
  - [ ] Pin per-agent requirements
  - [ ] Multi-stage Docker builds with slim CUDA base images
  - [ ] Generate SBOM and enable CVE scanning in CI
  - [ ] Enforce `trust_remote_code=False` model loads

## Acceptance Criteria

- All P0 endpoints have standardized health/readiness/warmup behavior
- `/call` has timeouts, retries with backoff, and circuit breaker protections
- Traces/metrics visible in Grafana with correct p95/p99 and GPU telemetry
- Pydantic schemas are versioned and validated by contract tests
- CI produces SBOM and passes vulnerability scan gates

## References

- `markdown_docs/development_reports/System_Assessment_2025-08-09.md`
- `README.md` (Architecture & Metrics)
- `.github/copilot-instructions.md` (Standards)
