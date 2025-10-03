Kafka scaffold license policy

Goal
- Maintain a fully self-hosted, open-source stack composed only of
  components whose licenses match our approved policy.

Approved licenses (automatically allowed):
- Apache-2.0
- MIT
- BSD-2-Clause
- BSD-3-Clause
- ISC
- PostgreSQL License (Postgres)

Licenses that require explicit review (flagged in CI):
- MPL-2.0
- LGPL
- EPL
- AGPL (strong copyleft) -- requires legal review and explicit approval

Disallowed for inclusion without executive approval:
- Proprietary/closed-source components
- SaaS-only connectors or managed services that embed vendor-only logic

Model & artifact governance
- Every model binary (weights) must include a MODEL_MANIFEST.yaml entry
  under `kafka/models/` with `name`, `version`, `source_url`, `license`,
  `sha256`, and `allowed_uses`.
- CI will reject model artifacts that declare disallowed licenses.

Object storage and decentralization
- The recommended object store options are SeaweedFS (Apache-2.0) and
  IPFS for public, tamper-evident snapshots. Both options avoid managed
  cloud lock-in and align with the project's autonomy goals.
- By default, the scaffold uses SeaweedFS (self-hosted S3-like store).
  IPFS may be used for public, signed snapshot publishing (pinning and
  retention must be operated by the team).
- The project expressly avoids relying on AWS S3 as a managed service
  to maintain full operational control and minimize 3rd-party service
  dependencies. The S3 API is acceptable as an interface only when
  used against self-hosted OSS-compatible backends.

Enforcement in CI
- A license-check workflow will run for PRs; any new dependency or
  artifact with a disallowed license will cause the job to fail.
- The license policy file is authoritative for the accepted/reviewed
  licenses.

Review process
- To include a flagged license (e.g., AGPL), create an issue describing
  the risk, get legal/ops approval, and update the policy with the
  specific component and exception referenced by issue number.
