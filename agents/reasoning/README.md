# Reasoning Agent

This package contains the reasoning agent (Nucleoid) for JustNews.

Structure
- `nucleoid_implementation.py` — low-level engine implementation (AST parsing, state, graph)
- `main.py` — FastAPI runtime, MCP Bus integration, HTTP endpoints
- `enhanced_reasoning_architecture.py` — domain rules and `EnhancedReasoningEngine` wrapper

Design Notes
- Rules and higher-level orchestration live in `enhanced_reasoning_architecture.py` to keep policy separate from the runtime server.
- `main.py` instantiates a single `NucleoidEngine` and passes it into `EnhancedReasoningEngine` so rules are loaded once and the runtime uses a shared engine instance.

Testing and development
- To run unit tests for this package, add tests under `tests/` that import `agents.reasoning.enhanced_reasoning_architecture` and `agents.reasoning.main`.
