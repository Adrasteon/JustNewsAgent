---
title: Hugging Face model caching and pre-download for Memory Agent
description: Auto-generated description for Hugging Face model caching and pre-download for Memory Agent
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Hugging Face model caching and pre-download for Memory Agent

This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cache SentenceTransformer models used by the Memory agent.

Why
---
The Memory agent uses `sentence-transformers` (default model `all-MiniLM-L6-v2`) and downloads model files from Hugging Face on first use. In production runs this can hit rate limits or slow startup.

Options to avoid 429 / slow downloads
------------------------------------
1. Provide an HF token (recommended)
   - Export HF_HUB_TOKEN (or HUGGINGFACE_HUB_TOKEN) in the environment used by the agent.
   - Example:

```bash
export HF_HUB_TOKEN="<your_token_here>"
```

   - The agent will attempt to login via `huggingface_hub.login()` at startup when this token is present.

2. Use a local cache directory
   - Set HF_HOME or HUGGINGFACE_HUB_CACHE to a path where model files should be cached.
   - Example:

```bash
export HF_HOME="/var/cache/hf"
---
title: Hugging Face model caching — Memory agent
description: Practical steps to avoid HF 429s by caching/pre-downloading SentenceTransformer models and authenticating with tokens.
tags: [memory, models, huggingface, caching, deployment]
status: active
last_updated: 2025-09-12
---

# Hugging Face model caching and pre-download for Memory Agent

This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cache SentenceTransformer models used by the Memory agent.

## Why
The Memory agent uses sentence-transformers (default model all-MiniLM-L6-v2) and downloads model files from Hugging Face on first use. In production runs this can hit rate limits or slow startup.

## Options to avoid 429 / slow downloads
1. Provide an HF token (recommended)
   - Export HF_HUB_TOKEN (or HUGGINGFACE_HUB_TOKEN) in the environment used by the agent.
   - Example:

```bash
export HF_HUB_TOKEN="<your_token_here>"
```

   - The agent will attempt to login via huggingface_hub.login() at startup when this token is present.

2. Use a local cache directory
   - Set HF_HOME or HUGGINGFACE_HUB_CACHE to a path where model files should be cached.
   - Example:

```bash
export HF_HOME="/var/cache/hf"
mkdir -p /var/cache/hf
chown justnews:justnews /var/cache/hf
```

3. Pre-download the model during deployment
   - From a machine with network access and HF token, run a short Python snippet to download the model into the cache directory:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/var/cache/hf')
```

4. Bundle the model into your container or VM image
   - For fully offline deployments, download the model and bake it into the image used by the service.

## Notes
- If huggingface_hub is not installed, the agent will still function but cannot authenticate or pre-download via the hub API.
- The code now honors HF_HUB_TOKEN and HF_HOME environment variables at agent startup and will try to authenticate when the token is present.

## Troubleshooting
- If you still see repeated model download logs in journalctl and 429 errors, verify:
  - HF token is present and valid
  - Cache directory is writable by the agent process
  - Network access to huggingface.co is available from the host

## See also

- MODEL_STORE_GUIDELINES.md
- GPU_ORCHESTRATOR_OPERATIONS.md
