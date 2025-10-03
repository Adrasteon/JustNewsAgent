# Topic & Schema Policy (Pilot + Project-wide)

This document defines the topic naming conventions, schema placement, compatibility policy,
review and approval process, and sample topic configurations for the JustNews Kafka
migration. Follow these rules for all pilot and production topics.

## Goals
- Ensure consistent, discoverable topic names across services and agents.
- Provide a clear schema strategy (formats, evolution rules, locations).
- Enforce compatibility and governance to avoid silent breakages.
- Provide sample configuration for pilot and core topics used in integration tests.

## Topic naming conventions
- Use dot-separated namespaces: <domain>.<resource>.<action>.<version-optional>
- Lowercase only; use hyphens or underscores only for internal resource segments when necessary.
- Prefer short, explicit names with a clear resource verb:
  - `scout.article.created` — Scout produced an article creation event
  - `crawler.article.persisted` — Crawler/Memory acknowledged persistence
  - `training.example` — Training example emitted for ML workflows
- Versioning strategy:
  - Keep logical topic names version-agnostic where possible (e.g., `scout.article.created`).
  - Encode schema versions in schema file names and in schema registry subjects (e.g., `article.created_v1.avsc`).
  - When incompatible message format changes are required, bump schema version and register new schema versions under the same subject following compatibility policy.

## Topic lifetime & configuration
- Partitioning:
  - Key topics by deterministic entity id (e.g., article_id) to keep per-entity ordering.
  - Default pilot partitions: 3 for high-volume topics (scout/crawler), 1 for low-volume topics (training.example).
- Retention:
  - Default retention: 7 days for created data; extended retention for persisted canonical data (14 days) during pilot.
  - Archive long-term data into object store + canonical DB instead of relying solely on topic retention.
- Replication:
  - For dev/pilot: replication 1. For production: replication >= 3 across broker nodes for durability.

## Schema strategy
- Preferred formats:
  - Avro for strict, typed event schemas that require controlled evolution (article lifecycle, canonical events).
  - JSON Schema for flexible artifact-like payloads and training examples.
- Schema files:
  - Store canonical schema files under `kafka/config/schemas/` with a clear version suffix: `resource.action_vN.avsc` or `.json`.
  - Example: `article.created_v1.avsc`, `training.example_v1.json`.
- Registry:
  - Use Apicurio Registry (dev stack) or any registry compatible with the registry wire format (Apicurio or Karapace in production).
  - Subject naming in registry: `<topic>-value` (e.g., `scout.article.created-value`).

## Compatibility policy
- Default compatibility: BACKWARD (consumers built for older schemas can read newer records).
- Rules for Avro JSON/Schema changes:
  - Backward compatible change examples: adding an optional field with default or nullable union including `null`.
  - Incompatible examples: removing a required field, narrowing a field type.
- Register each schema version in the registry and run compatibility checks before merging schema changes to `main`.

## Governance & approvals
- All schema changes must be proposed via pull requests that include:
  - Schema file under `kafka/config/schemas/`.
  - A short rationale in the PR description describing why the change is needed and compatibility impact.
  - Unit tests validating backwards compatibility where appropriate.
- Approvals:
  - Schema changes affecting core topics require review from data governance owner(s) (owner TBD).
  - Security changes (signing, agent identities) require security team approval.

## Testing & CI
- Unit tests:
  - Must validate each schema parses and passes a conservative compatibility check (see `kafka/tests/test_schemas.py`).
- CI:
  - PRs targeting `kafka/**` must run the `ci/kafka-poc.yml` workflow which:
    - Boots the dev kafka stack (Apicurio registry included)
    - Runs bootstrap to register schemas and set compatibility
    - Executes unit tests and the (optional) integration tests

## Operational runbook (schema & topic changes)
- Local verification:
  - Run `python kafka/scripts/validate_topic_configs.py kafka/config/topics/*.yaml` to assert schema files exist and names follow the policy.
- Registration & bootstrap:
  - Use `python kafka/scripts/bootstrap_pilot.py --bootstrap <bootstrap> --registry <registry-url>` to create topics (dev) and register schemas.
- Rollout steps for schema changes:
  1. Implement schema change in `kafka/config/schemas/` with a clear versioned filename.
  2. Add unit tests to validate parsing/compatibility.
  3. Open a PR, assign data governance reviewer, and run CI.
  4. Once CI passes and owners approve, merge to `main` and run bootstrap in staging and production under controlled release windows.

## Sample topic configs
- See `kafka/config/topics/pilot_topics.yaml` and `kafka/config/topics/core_topics.yaml` for sample pilot and core topic configurations.
- Pilot topics are used by integration tests; core topics are canonical examples for production settings.

## Examples and common topics
- Core topic list (examples):
  - scout.article.created — Scout discovery & extraction events
  - crawler.article.persisted — Crawler/Memory persistence acknowledgement
  - memory.article.persisted — Memory canonical persisted event
  - analysis.request, analysis.result — Analytical requests and results
  - training.example — Emitted training examples for ML loops
  - model.registry.update — Model registry artifact updates
  - dlq.* — Dead-letter topics for unprocessable events
  - gpu.lease.request, gpu.lease.granted, gpu.lease.released — GPU orchestration lifecycle

## Contact & owners
- Data Governance Owner: TBD — assign an owner in the project board for schema approvals.
- Developer owner: dev (for scaffolding and helper scripts)
- CI owner: ci (for ensuring the CI flows are maintained)

## Frequently asked questions
- Q: How do I perform incompatible schema changes?
  - A: Incompatible changes require coordination and a plan: (1) publish new schema under new subject/version, (2) deploy consumers that can read new subject, (3) migrate producers to new subject, (4) deprecate old subject.
- Q: Can I auto-register schemas from my client at runtime?
  - A: For production avoid auto-registering unreviewed schemas; use the bootstrap approach in controlled windows.
