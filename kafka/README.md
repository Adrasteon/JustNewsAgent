Kafka-First Greenfield Scaffold for JustNews

This folder is a greenfield scaffold that implements a Kafka-centric
re-implementation of the JustNews platform while keeping the original
codebase available for reference and porting. The goal is to provide a
clean, isolated workspace inside the same repository so developers can
build the new system without cluttering the legacy implementation.

Structure
- kafka/
  - src/agents/         # New agent implementations and adapters
  - config/             # Configuration templates for Kafka + services
  - docker/             # docker-compose for local dev (lightweight)
  - tests/              # Integration & contract tests for kafka stack
  - scripts/            # Helper scripts (worktree, porting helpers)

Guidelines
- Keep adapters and business logic separate. Copy code from the
  legacy `agents/` tree only when the kafka agent needs it; prefer
  porting small functions rather than wholesale copying.
- Use the TRANSPORT config pattern in agent adapters if you need to
  maintain both MCP and Kafka transports temporarily.
- Heavy integration tests run in CI; local tests use lightweight
  mocks or a single-node broker.

Next steps
- Implement per-agent kafka adapters in `src/agents`.
- Add developer helper scripts to copy or port minimal agent logic into
  the kafka scaffold.
- Add CI job that runs the kafka dev stack and executes contract tests.