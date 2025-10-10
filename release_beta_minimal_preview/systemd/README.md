# JustNews Preview - Systemd Templates

This directory contains minimal systemd unit files and configuration templates for deploying JustNews agents in production.

## Directory Structure

```
systemd/
├── units/              # Systemd service unit templates
│   └── justnews@.service
└── logrotate/          # Log rotation configuration
    └── justnews
```

## Installation

### 1. Service Unit Installation

Copy the service template to systemd:

```bash
sudo cp units/justnews@.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 2. Enable Agents

Enable the agents you need:

```bash
# Core agents (minimal deployment)
sudo systemctl enable justnews@mcp_bus
sudo systemctl enable justnews@chief_editor
sudo systemctl enable justnews@scout
sudo systemctl enable justnews@analyst

# Additional agents (full deployment)
sudo systemctl enable justnews@synthesizer
sudo systemctl enable justnews@fact_checker
sudo systemctl enable justnews@critic
sudo systemctl enable justnews@memory
sudo systemctl enable justnews@reasoning
```

### 3. Start Services

```bash
# Start in dependency order
sudo systemctl start justnews@mcp_bus
sudo systemctl start justnews@chief_editor
sudo systemctl start justnews@scout
# ... continue with other agents
```

Or use the helper script:

```bash
/opt/justnews/JustNewsAgent/deploy/systemd/scripts/enable_all.sh
```

### 4. Logrotate Configuration (Optional)

If using file-based logging (not systemd journal):

```bash
sudo cp logrotate/justnews /etc/logrotate.d/justnews
sudo mkdir -p /var/log/justnews
sudo chown -R justnews:justnews /var/log/justnews
```

**Note**: If using systemd journal (recommended), log rotation is handled automatically and the logrotate configuration is not needed.

## Environment Configuration

### Global Environment File

Create `/etc/justnews/global.env` with your configuration:

```bash
# JustNews Global Configuration
JUSTNEWS_ROOT=/opt/justnews/JustNewsAgent
PYTHON_BIN=/opt/justnews/venv/bin/python
MCP_BUS_URL=http://127.0.0.1:8000
GPU_ORCHESTRATOR_URL=http://127.0.0.1:8014

# Database (for Memory agent)
POSTGRES_HOST=localhost
POSTGRES_DB=justnews
POSTGRES_USER=justnews
POSTGRES_PASSWORD=your_secure_password

# Optional: GPU settings
ENABLE_NVML=true
SAFE_MODE=false
```

### Per-Agent Configuration

You can override settings per agent by creating drop-in files:

```bash
sudo mkdir -p /etc/systemd/system/justnews@mcp_bus.service.d/
sudo tee /etc/systemd/system/justnews@mcp_bus.service.d/override.conf << EOF
[Service]
Environment="MCP_BUS_PORT=8000"
MemoryLimit=1G
EOF

sudo systemctl daemon-reload
```

## Building the Virtual Environment

Before starting services, build the production virtual environment:

```bash
sudo /opt/justnews/JustNewsAgent/deploy/systemd/scripts/build_service_venv.sh \
  --venv /opt/justnews/venv \
  --requirements /opt/justnews/JustNewsAgent/release_beta_minimal_preview/requirements-runtime.txt
```

## Service Management

### Check Status

```bash
# Check all JustNews services
systemctl list-units 'justnews@*'

# Check specific agent
sudo systemctl status justnews@mcp_bus
```

### View Logs

```bash
# Follow logs for an agent
sudo journalctl -u justnews@mcp_bus -f

# View logs since boot
sudo journalctl -u justnews@scout -b

# View logs from last hour
sudo journalctl -u justnews@analyst --since "1 hour ago"
```

### Restart Services

```bash
# Restart single agent
sudo systemctl restart justnews@analyst

# Restart all agents
sudo systemctl restart 'justnews@*.service'
```

## Troubleshooting

### Service Won't Start

1. Check the service status:
   ```bash
   sudo systemctl status justnews@mcp_bus
   ```

2. View detailed logs:
   ```bash
   sudo journalctl -u justnews@mcp_bus -n 100 --no-pager
   ```

3. Verify environment:
   ```bash
   sudo systemctl show justnews@mcp_bus | grep -i env
   ```

4. Check file permissions:
   ```bash
   ls -la /opt/justnews/JustNewsAgent
   ls -la /opt/justnews/venv
   ```

### Dependency Issues

Run the dependency checker:

```bash
/opt/justnews/venv/bin/python \
  /opt/justnews/JustNewsAgent/deploy/systemd/scripts/ci_check_deps.py
```

### Health Checks

```bash
# Check MCP Bus
curl http://127.0.0.1:8000/health

# List registered agents
curl http://127.0.0.1:8000/agents
```

## Production Deployment Checklist

- [ ] Create `justnews` user and group
- [ ] Install JustNews to `/opt/justnews/JustNewsAgent`
- [ ] Build virtual environment at `/opt/justnews/venv`
- [ ] Create `/etc/justnews/global.env` with configuration
- [ ] Copy systemd unit to `/etc/systemd/system/`
- [ ] Set up PostgreSQL database (for Memory agent)
- [ ] Enable and start services
- [ ] Verify all services are running: `systemctl list-units 'justnews@*'`
- [ ] Check health endpoints
- [ ] Configure firewall rules if needed
- [ ] Set up monitoring/alerting

## See Also

- [Comprehensive Systemd Guide](../../deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md)
- [Quick Reference](../../deploy/systemd/QUICK_REFERENCE.md)
- [PostgreSQL Integration](../../deploy/systemd/postgresql_integration.md)
