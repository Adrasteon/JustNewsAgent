---
title: MODEL USAGE
description: Auto-generated description for MODEL USAGE
tags: [documentation]
status: current
last_updated: 2025-09-12
---

Model Usage, Caching and GPU Guidelines
======================================

This document describes model usage patterns, caching, and GPU allocation conventions used across the JustNewsAgentic repository. It complements `EMBEDDING_HELPER.md` and provides a concise reference for developers and operators.

1. Model caching and per-agent directories
----------------------------------------
- Each agent should use a per-agent model cache directory to avoid permission conflicts and to support atomic installs. Default pattern:
  - `agents/<agent>/models`
- You may override per-agent caches via environment variables (agent-specific):
  - `SYNTHESIZER_MODEL_CACHE`, `SYNTHESIZER_V2_MODEL_CACHE`, `FACT_CHECKER_MODEL_PATH`, etc.
- The recommended flow for agents is:
  1. At startup, call `ensure_agent_model_exists(model_name, agent_cache_dir)` to ensure the model files are present.
  2. Load the model via `get_shared_embedding_model(...)` so the process reuses an in-process cached instance when possible.

2. Atomic download and install semantics
----------------------------------------
- The embedding helper uses a `.tmp` staging directory and a file lock to coordinate downloads across processes.
- If a process finds a `.tmp` directory or a lock, it will wait for completion or use the existing complete model directory.
- This avoids race conditions where two processes write the same files concurrently causing permission errors or partial installs.

3. Prefer helper APIs over direct constructors
---------------------------------------------
- Do not instantiate heavy models directly with library constructors (e.g., `SentenceTransformer(...)`) in agent modules. Instead use:
  - `from agents.common.embedding import get_shared_embedding_model`
  - `from agents.common.embedding import ensure_agent_model_exists`
- This centralizes filesystem and concurrency behavior and reduces memory usage by reusing in-process instances.

4. GPU allocation and the GPU manager
------------------------------------
- Agents that request GPUs should use the GPU manager API where available:
  - `from agents.common.gpu_manager import request_agent_gpu, release_agent_gpu, get_gpu_manager`
- For local dev or lint/test runs the repository includes a lightweight shim `agents/common/gpu_manager.py` that simulates allocation (returns GPU index 0). Production deployments may replace this with a real multi-agent GPU allocation service.

5. Pre-download strategies for restricted environments
-----------------------------------------------------
- If agents run in air-gapped or restricted networks, pre-download models on a machine with network access and copy the model folder to the per-agent cache location.
- Use `ensure_agent_model_exists()` in startup to detect missing models and optionally fail or log a clear error instructing operators to install the model manually.

6. Environment variables and tuning
-----------------------------------
- Common env vars observed across the repo:
  - `SYNTHESIZER_MODEL_CACHE`, `SYNTHESIZER_V2_MODEL_CACHE`, `FACT_CHECKER_MODEL_PATH` — per-agent cache directories
  - `MCP_BUS_URL` — address of the MCP Bus (used for agent registration)
  - `FACT_CHECKER_AGENT_PORT`, `SYNTHESIZER_AGENT_PORT` — agent ports used in dev
- When running on GPU-enabled machines, set `device` values appropriately (e.g., `cuda:0`, `cuda:1`) when calling `get_shared_embedding_model(..., device='cuda:0')`.

7. Troubleshooting common errors
--------------------------------
- Permission errors while downloading models:
  - Ensure per-agent cache directories are owned by the agent user and writable.
  - Set per-agent cache environment variables to directories on writable volumes.
- Concurrent download failures across agents:
  - Check for the presence of `.tmp` staging directories. If present for long periods, investigate aborted downloads.
- Missing heavy deps in tests (torch, transformers):
  - For CI, either install the extra dependencies in the test environment or mock model loading in unit tests.

8. Examples
-----------
- Startup snippet (synthesizer):

```python
from agents.common.embedding import ensure_agent_model_exists, get_shared_embedding_model
from pathlib import Path

agent_cache = os.environ.get('SYNTHESIZER_V2_MODEL_CACHE') or str(Path('./agents/synthesizer/models').resolve())
ensure_agent_model_exists('sentence-transformers/all-MiniLM-L6-v2', agent_cache)
embedder = get_shared_embedding_model('sentence-transformers/all-MiniLM-L6-v2', cache_folder=agent_cache, device='cpu')
```

9. Further reading
------------------
- `markdown_docs/agent_documentation/EMBEDDING_HELPER.md` — details about the helper functions and usage patterns.
- `agents/common/embedding.py` — implementation and comments for the helper.
- `agents/common/gpu_manager.py` — lightweight GPU manager shim used in dev and testing.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

