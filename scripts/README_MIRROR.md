Mirror per-agent models to large volume
=====================================

This folder contains a small utility to migrate `agents/<agent>/models` folders to a shared, large-volume target directory and then create symlinks back to the agents. It is intentionally conservative and idempotent.

Usage
-----

Preview (no changes):

```bash
# If you omit --target, the script will use the DATA_DRIVE_TARGET env var or fall back to the
# canonical path: /media/adra/Data/justnews/agents
python3 scripts/mirror_agent_models.py --dry-run
```

Perform migration (destructive to local `agents/<agent>/models` — removed after copy):

```bash
# Recommended: run against the canonical agents path on the shared drive
python3 scripts/mirror_agent_models.py --target /media/adra/Data/justnews/agents --yes
```

Per-agent migration example:

```bash
python3 scripts/mirror_agent_models.py --agent synthesizer --yes
```

Using an explicit target path:

```bash
# Prefer the canonical agents-aware path
python3 scripts/mirror_agent_models.py --target /media/adra/Data/justnews/agents --dry-run
python3 scripts/mirror_agent_models.py --target /media/adra/Data/justnews/agents --yes
```

Environment variable:

- You can set DATA_DRIVE_TARGET to point to another location (for example in your shell profile):

```bash
export DATA_DRIVE_TARGET=/media/adra/Data/justnews/agents
```

Notes and safety
----------------
- The script copies each `agents/<agent>/models` into `TARGET/<agent>/models` in a staging folder and then performs an atomic rename where possible.
- The original folder in `agents/<agent>/models` is removed after a successful copy+rename and replaced with a symlink to the new target.
- If the target is on a different filesystem, the script falls back to a safe move/copy but atomicity across filesystems is not guaranteed.
- Always run with `--dry-run` first. The script refuses to perform actions unless `--yes` is passed.

Permission & ownership
----------------------
After migration, ensure the permissions/ownership on the target volume allow the agent processes to read/write as required. Use chown/chmod as appropriate.

Rollback
--------
If you need to roll back after running the script:
1. Remove the symlink `agents/<agent>/models`.
2. Move `TARGET/<agent>/models` back into `agents/<agent>/models` using `mv`.
3. Adjust ownership/permissions accordingly.
