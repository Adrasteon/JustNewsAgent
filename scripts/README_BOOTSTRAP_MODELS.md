---
title: "Readme Bootstrap Models"
description: "Documentation for model bootstrap helpers and usage (auto-created placeholder). See `scripts/bootstrap_models.py` for implementation details."
tags:
  - models
  - scripts
  - bootstrap
status: current
last_updated: 2025-10-07
---


# Readme: Bootstrap Models

This document explains the model bootstrap and provisioning helpers used by
JustNewsAgent. These utilities ensure reproducible model downloads, atomic
installs into per-agent caches, and basic verification of model artifacts.

Note: `scripts/bootstrap_models.py` is the primary convenience wrapper; the
repository also contains lower-level utilities such as
`scripts/download_agent_models.py` and `scripts/verify_models.py`.

## Goals

- Provide a simple, repeatable workflow to prepare model artifacts for local
  development and CI.
- Ensure per-agent model caches with atomic staging to avoid race conditions
  when multiple processes download the same model.
- Offer verification steps to confirm model integrity and expected layout.

## Prerequisites

- Python 3.10+ (project uses 3.12 in dev environments) and pip-installed
  dependencies from `requirements.txt`.
- Recommended: conda environment `justnews-v2-py312` for reproducible runs.
- Network access to model repositories (HuggingFace, Ollama, or local mirror
  depending on your setup).

## Model cache layout (recommended)

Top-level layout used by the scripts in this repo:

- `models/` — optional local copy of frequently used models (not required)
- `model_cache/agents/<agent_name>/` — per-agent cache directories
- `model_cache/shared/` — shared caches for large base models used across
  agents

Examples:

```
model_cache/
├─ agents/
│  ├─ synthesizer/
│  └─ scout/
└─ shared/
   └─ sentence-transformers/all-MiniLM-L6-v2/
```

Agents should never write into each other's cache directories; the
bootstrap/download utilities create and manage per-agent directories by
default.

## Environment variables

The scripts honor the following environment variables (with sensible
defaults):

- `MODEL_CACHE_ROOT` — root directory for model caches (default: `./model_cache`)
- `AGENT_MODEL_CACHE` — per-agent override (e.g. `model_cache/agents/scout`)
- `HF_HUB_TOKEN` — HuggingFace token when accessing private models
- `OLLAMA_HOST` — host URL for local Ollama server (if applicable)

Export example (bash):

```bash
export MODEL_CACHE_ROOT="$PWD/model_cache"
export HF_HUB_TOKEN="<your-token>"
```

## Typical workflows and examples

1) Prepare models for local development (per-agent cache):

```bash
# activate recommended conda env
conda run -n justnews-v2-py312 python3 scripts/bootstrap_models.py \
  --agent synthesizer \
  --models sentence-transformers/all-MiniLM-L6-v2 transformers/llama-3-small \
  --cache-root "$PWD/model_cache"
```

2) Download a single model to shared cache (suitable for CI):

```bash
python3 scripts/download_agent_models.py --model sentence-transformers/all-MiniLM-L6-v2 \
  --target "$PWD/model_cache/shared"
```

3) Verify integrity and expected files for a cache directory:

```bash
python3 scripts/verify_models.py --path "$PWD/model_cache/agents/synthesizer"
```

4) Running a dry-run to preview actions without network writes:

```bash
python3 scripts/bootstrap_models.py --agent scout --models all-MiniLM-L6-v2 --dry-run
```

## Behavior and implementation notes

- Atomic installs: scripts should download into a `.tmp` staging folder then
  rename to the final path once complete. This prevents partial installs when
  concurrent processes run.
- Locking: a lightweight file lock (flock) is used around downloads to avoid
  concurrent writes to the same staging area.
- Idempotency: re-running the same bootstrap command should not re-download
  or re-write existing files unless `--force` is supplied.

## Troubleshooting

- Permission errors when writing caches:
  - Ensure the user running the script owns the `MODEL_CACHE_ROOT` or use
    `sudo` only for system-wide caches.
- Partial model installs (leftover `.tmp` directories):
  - Inspect `.tmp` directories and remove them if no active download is
    running. The `verify_models.py` script provides a `--repair` flag to
    attempt recovery.
- Missing model files after download:
  - Confirm `HF_HUB_TOKEN` if model is private; check network access and
    proxy settings.

## Tests & CI

Add unit tests that simulate concurrent downloads to assert atomicity and
locking behavior. Suggested test names and locations:

- `tests/test_bootstrap_atomicity.py`
- `tests/test_model_verify.py`

CI should run `scripts/verify_models.py --path <ci-cache>` during the build
step to ensure required artifacts are present.

## Maintainership notes

- If you update `bootstrap_models.py` API (CLI flags, env var names), update
  this README and the catalogue entry in `docs/docs_catalogue_v2.json`.
- Keep `download_agent_models.py` and `verify_models.py` lightweight and
  focused so they can be used independently in CI and air-gapped setups.

## Recommended additions

- Add examples of how to pre-seed `model_cache/` for air-gapped CI runners.
- Document expected disk usage per model so operators can plan storage.
- Provide a small `scripts/list_cached_models.py` helper to inspect caches.

If this file still needs more detail or should reference a different
documentation location, tell me where to pull content from and I will update
the README accordingly.
