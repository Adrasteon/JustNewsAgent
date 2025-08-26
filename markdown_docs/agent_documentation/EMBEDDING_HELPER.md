Embedding Helper (agents/common/embedding.py)

Purpose
-------
Centralize downloading, caching, and instantiation of sentence-transformers embedding models so multiple agents (processes) avoid race conditions and repeated heavy in-process loads. The helper uses process-local caching and atomic file operations so concurrent agents can safely ensure a model exists on disk and then load a shared in-process instance.

Key functions
-------------
- get_shared_embedding_model(model_name: str = "all-MiniLM-L6-v2", cache_folder: Optional[str] = None, device: Optional[str] = None) -> SentenceTransformer-like
  - Returns a process-local cached model instance. Prefer this as the primary entry point for agents that need embeddings.
  - Parameters:
    - model_name: huggingface model id (e.g. 'sentence-transformers/all-MiniLM-L6-v2').
    - cache_folder: directory to store per-agent model files (defaults to agents/<agent>/models or current working dir).
    - device: 'cpu' or 'cuda:0' etc. If omitted, the helper will pick a sensible default.

- ensure_agent_model_exists(model_name: str, agent_cache_dir: str) -> str
  - Ensures the model is downloaded into the provided agent-specific cache dir using atomic install semantics.
  - Returns the path to the model on disk.

Usage pattern
-------------
Prefer the `get_shared_embedding_model()` call. Example in an agent:

```python
from agents.common.embedding import get_shared_embedding_model

agent_cache = os.environ.get('SYNTHESIZER_MODEL_CACHE') or str(Path('./agents/synthesizer/models').resolve())
model = get_shared_embedding_model('sentence-transformers/all-MiniLM-L6-v2', cache_folder=agent_cache, device='cpu')
embeddings = model.encode(['text1', 'text2'], convert_to_numpy=True)
```

Fallback and atomic install
---------------------------
If your environment restricts direct downloads at runtime (air-gapped or pre-installed artifacts), use `ensure_agent_model_exists()` during startup to assert the model is present (it will attempt to download if missing). The helper performs a cross-process lock and atomic directory replacement to prevent partial installs being observed by other agents.

Best practices
--------------
- Set per-agent cache directories to avoid permission conflicts: `agents/<agent>/models`.
- Avoid calling `SentenceTransformer(...)` directly; use the helper to benefit from the process-level cache and atomic download semantics.
- If you need to control storage location via environment variables, set `SYNTHESIZER_MODEL_CACHE`, `BALANCER_MODEL_CACHE`, etc. to per-agent directories.

Troubleshooting
---------------
- If you see permission errors in huggingface cache paths, ensure the per-agent cache directory exists and is writable by the agent process.
- In environments with strict network policies, pre-download the model using `ensure_agent_model_exists()` on a machine with access and then commit/cache the model directory to a shared volume.

Contact
-------
For issues related to the helper or GPU allocation behavior, see `markdown_docs/production_status/` and open an issue in the repository describing the environment and error logs.
