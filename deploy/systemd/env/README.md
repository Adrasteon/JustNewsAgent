# Environment files for system installation

Place service-wide or per-agent env files here. When
`reset_and_start.sh --sync-env` (or `--force-sync-env`) is run,
these files are copied into `/etc/justnews/`.

Recommended files:

- `global.env` — global settings used by all agents. Typical keys:
  - database credentials
  - PYTHON_BIN (absolute path to interpreter)
  - SAFE_MODE

- `<agent>.env` — per-agent overrides, for example
  `gpu_orchestrator.env`.

PYTHON_BIN

- Set `PYTHON_BIN` to the absolute path of the Python
  interpreter you want services to use (typically a conda
  environment). Example:
  `PYTHON_BIN=/home/adra/miniconda3/envs/justnews-v2-py312/bin/python`

- If `PYTHON_BIN` is not present when syncing env files,
  `reset_and_start.sh` will attempt to auto-detect a Python
  interpreter and inject a `PYTHON_BIN` into
  `/etc/justnews/global.env` when run as root.

Permissions and group

- The installer will create a dedicated group `justnews` and
  set ownership of `/etc/justnews` to `root:justnews` with
  restrictive file permissions (typical files: `0640`).

- Service users (configured in the systemd unit template)
  are added to the `justnews` group during installation so
  they can read env files without exposing them to other users.

Service user and admin group

- Recommended: create a dedicated system account `justnews` and a group `justnews`.
  The installer can create these automatically when run as root. Example usage:

  sudo ./deploy/systemd/reset_and_start.sh --reinstall --sync-env \
    --set-python-bin /home/adra/miniconda3/envs/justnews-v2-py312/bin/python \
    --justnews-group justnews \
    --service-user justnews --admin-user alice --admin-user bob

- `justnews` is a non-login system user used to run services.

- `justnews-admins` is a group for human administrators. Admin users passed via
  `--admin-user` will be added to both `justnews-admins` and the `justnews` group
  so they can manage service files under `/etc/justnews`.

Security notes

- The installer sets `/etc/justnews` ownership to `root:justnews` and file permissions
  to `0640` so only root and members of the `justnews` group can read env files.
- For production, avoid storing secrets in source-controlled files and use a secret
  manager instead.
