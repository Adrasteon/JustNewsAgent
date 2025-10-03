# Kafka Migration Actionable Todo List

This file mirrors `KAFKA_MIGRATION_TODO.yaml` in human-readable markdown checklist form. Use this in PR descriptions or issue templates.

- [ ] 1. Project kickoff & design
  - Deliverable: `kafka/docs/topic-schema-policy.md` + sample topic configs.

- [x] 2. Dev scaffold & infra
  - Deliverable: `kafka/docker/docker-compose.kafka.yml` and bootstrap script.

- [x] 3. Core developer ergonomics
  - Deliverable: `kafka/src/agents/adapter_template.py`, `kafka/scripts/port_agent.sh`.

- [ ] 4. Topic & schema contract tests
  - Deliverable: `kafka/config/schemas/*` and `kafka/tests/test_schemas.py`.

- [ ] 5. Pilot pipeline: Scout -> Crawler -> Memory
  - Deliverable: scout_adapter, crawler_adapter, memory_consumer, integration test.

- [ ] 6. Contract parity & validation
  - Deliverable: `kafka/tests/test_dual_parity_pilot.py`.

- [ ] 7. Training loop & model registry integration
  - Deliverable: `kafka/src/agents/training/*`, `justnews.training.*` topics.

- [ ] 8. Multimedia ingestion & processing pipeline
  - Deliverable: Transcoder, ASR, Vision consumers; segment indexes persisted.

- [ ] 9. Port analytic agents (incremental)
  - Deliverable: adapter per agent + tests and CI entries.

- [ ] 10. GPU orchestration & inference cluster
  - Deliverable: `gpu_orchestrator` and inference endpoint patterns.

- [ ] 11. Observability, monitoring & DLQ
  - Deliverable: Prometheus metrics, dashboards, DLQ alerts.

- [ ] 12. Security, signing & schema governance
  - Deliverable: signing library, governance docs, ACLs for schema registry.

- [ ] 13. CI, tests & release gates
  - Deliverable: `ci/kafka-poc.yml` and contract tests in CI.

- [ ] 14. Documentation & runbooks
  - Deliverable: `kafka/docs/DEVELOPER_GUIDE.md` and runbooks.

- [ ] 15. Migration & cutover plan
  - Deliverable: `kafka/docs/cutover-checklist.md` and pilot reports.

- [ ] 16. Housekeeping & governance
  - Deliverable: CHANGELOG entries, security scan integration.

---

Metadata:
- created_by: GitHub Copilot
- created_at: 2025-10-02
- source: `kafka/KAFKA_MIGRATION_TODO.yaml`