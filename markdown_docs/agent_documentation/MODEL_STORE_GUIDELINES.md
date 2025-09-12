---
title: Model store guidelines
description: Auto-generated description for Model store guidelines
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Model store guidelines

This document explains the canonical model-store layout and safe update patterns for
per-agent model copies used by the JustNewsAgent system. The implementation is
backed by `agents/common/model_store.py` which provides a minimal atomic staging
and finalize API.

Goals
- Ensure agents always load consistent, fully-written model artifacts.
- Support per-agent model copies (fine-tuned variants) with versioning.
- Provide atomic swaps so readers never see partially-written files.
- Keep deployment and permissions simple.

Directory layout

Root model store (example): `/opt/justnews/models`

Structure:

---
title: Model Store — canonical layout and atomic updates
description: Canonical model store design, atomic versioning, and safe reader/writer patterns for agent models.
tags: [models, storage, atomic, versions, deployment]
status: active
last_updated: 2025-09-12
---

# Model store guidelines

This document explains the canonical model-store layout and safe update patterns for per-agent model copies used by the JustNewsAgent system. The implementation is backed by agents/common/model_store.py which provides a minimal atomic staging and finalize API.

## Goals
- Ensure agents always load consistent, fully-written model artifacts.
- Support per-agent model copies (fine-tuned variants) with versioning.
- Provide atomic swaps so readers never see partially-written files.
- Keep deployment and permissions simple.

## Directory layout

Root model store (example): /opt/justnews/models

Structure:

```
/opt/justnews/models/
  scout/
    versions/
      v2025-08-26/...
      v2025-08-27/...
    current -> versions/v2025-08-27
  synthesizer/
    versions/
      v2025-05-10/...
    current -> versions/v2025-05-10
```

## Safe update pattern (recommended)

1. Trainer writes new model into a staged directory:

   /opt/justnews/models/{agent}/versions/{version}.tmp

2. After writing and validating files, call ModelStore.finalize(agent, version). This:
   - computes a checksum and writes manifest.json into the version dir,
   - renames {version}.tmp -> {version} (atomic on same filesystem),
   - creates a temporary symlink and atomically replaces current to point to the new version.

3. Readers load from /opt/justnews/models/{agent}/current.

## Notes
- Use the agents/common/ModelStore helper where possible. See examples in agents/common/model_store.py.
- Ensure the model store is on a single filesystem to allow atomic renames.
- Keep the trainer UID and agent UIDs in the same unix group or configure permissions so trainers can write versions and agents can read.
- Use offline mode (local_files_only) in production to avoid background downloads.

## Manifest format

- manifest.json placed inside each version directory contains:
  - version: string tag
  - checksum: sha256 checksum of directory contents
  - metadata: free-form object for training info (epoch, commit, author)

## Cleanup policy
- Keep a small number of versions (for example, 3). Provide a cleanup job that removes older versions after verifying they are not pointed to by current.

## Example code snippet (writer)

```python
from pathlib import Path
from agents.common.model_store import ModelStore

store = ModelStore(Path('/opt/justnews/models'))
with store.stage_new('scout', 'v2025-08-27') as tmp:
    # write model files into tmp
    pass
store.finalize('scout', 'v2025-08-27')
```

## Example code snippet (reader)

```python
from pathlib import Path
from agents.common.model_store import ModelStore
from transformers import AutoModel

store = ModelStore(Path('/opt/justnews/models'))
current = store.get_current('scout')
if current:
    # load model from current (path to directory)
    model = AutoModel.from_pretrained(str(current))
```

## Security and permissions
- Set group permissions to allow trainers to write and agents to read:

```bash
chgrp -R justnews /opt/justnews/models
chmod -R g+rwX /opt/justnews/models
```

This file should be kept in markdown_docs/agent_documentation and referenced from deployment docs.

## See also

- GPU_ORCHESTRATOR_OPERATIONS.md
- AGENT_MODEL_MAP.md
- preflight_runbook.md
