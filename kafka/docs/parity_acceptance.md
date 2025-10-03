# Parity Acceptance Criteria

This document defines the acceptance criteria for Task 6: Contract parity & validation.

Goals

- Assert that the legacy MCP processing path and the Kafka-based path produce
  semantically equivalent persisted article records for the pilot dataset.

Definitions

- Canonical record: The persisted article record as stored by the Memory adapter.
- Parity: Canonical records are equivalent for the set of canonical fields defined
  in this document, allowing for documented tolerances on non-deterministic fields.

Canonical fields to compare

- id (string) — must match exactly
- url (string) — compared after canonicalization (trailing slashes removed)
- source (string) — compared after normalization (lower-cased)
- Any additional metadata fields must be explicitly documented and included
  in `kafka/docs/parity_field_map.md` if they are required for parity checks.

Non-deterministic tolerance rules

- Timestamp fields (keys matching `_at`, `_time`, or `timestamp`) are compared
  using a tolerance window. Default tolerance: 5 seconds.
- If timestamps differ by more than the configured tolerance, this is considered
  a parity failure and will be surfaced in the diff artifact.

Acceptance criteria (for marking Task 6 complete)

1. Unit parity tests (in-memory adapters) pass locally and in CI.
2. A parity E2E job runs against the dev-stack and produces no diffs for the
   canonical sample dataset using the default tolerance.
3. The CI produces artifact(s) on failure with a diff.json and JUnit xml for
   further triage.
4. Stakeholders (QA/Lead) review the parity report and sign off on parity for
   the pilot topics.

If any diffs remain, the job must fail and the team must evaluate whether the
variance is acceptable (update tolerance or normalize rules) or indicates a
functional bug that must be fixed before cutover.

Configuration

- Tolerance can be adjusted in generate_parity_report via `--tolerance`.
- The parity-e2e CI job includes a Kafka admin probe and bootstrap step to ensure
  the dev-stack is fully prepared before running tests.
