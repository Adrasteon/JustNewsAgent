# Examples for systemd native deployment

Files in this directory are examples and helpers to install the JustNews systemd units.

1. Copy env files to /etc/justnews/
   sudo mkdir -p /etc/justnews
   sudo cp deploy/systemd/env/*.env /etc/justnews/

2. Install unit template
   sudo cp deploy/systemd/units/justnews@.service /etc/systemd/system/justnews@.service
   sudo systemctl daemon-reload

3. Install the wrapper script
   sudo cp deploy/systemd/examples/justnews-start-agent.sh /usr/local/bin/justnews-start-agent.sh
   sudo chmod +x /usr/local/bin/justnews-start-agent.sh

4. Enable and start an instance
   sudo systemctl enable --now justnews@scout

5. Inspect status and logs
   sudo systemctl status justnews@scout
   journalctl -u justnews@scout -f

Notes:
- The wrapper script will attempt to use `conda run -n ${CONDA_ENV}` if CONDA_ENV is set in the env files.
- The unit template already includes a hook to wait for the MCP Bus via `wait_for_mcp.sh` helper; provide that script if needed in /usr/local/bin/
