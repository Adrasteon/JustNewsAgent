# JustNews Systemd Unit Examples
#
# These examples show how to configure systemd units for JustNews agents.
# They use the standardized justnews-start-agent.sh launcher which provides:
# - Runtime dependency checking with actionable error messages
# - Consistent environment setup
# - Proper startup sequencing

# Example 1: MCP Bus (foundational service)
# File: /etc/systemd/system/justnews@mcp_bus.service
[Unit]
Description=JustNews MCP Bus (Agent Communication Hub)
Documentation=https://github.com/Adrasteon/JustNewsAgent
After=network-online.target
Wants=network-online.target
# MCP Bus should wait for GPU Orchestrator to be ready
After=justnews@gpu_orchestrator.service

[Service]
Type=simple
User=justnews
Group=justnews
WorkingDirectory=/opt/justnews/JustNewsAgent

# Environment configuration
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/mcp_bus.env
Environment=PYTHON_BIN=/opt/justnews/venv/bin/python
Environment=MCP_BUS_PORT=8000

# Preflight check before starting (validates GPU models are loaded)
ExecStartPre=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-preflight-check.sh --gate-only

# Use the standardized launcher
ExecStart=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-start-agent.sh mcp_bus

# Restart configuration
Restart=on-failure
RestartSec=10
TimeoutStartSec=600
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/justnews/JustNewsAgent
ReadWritePaths=/var/log/justnews

[Install]
WantedBy=multi-user.target

# Example 2: GPU Orchestrator (must start before MCP Bus)
# File: /etc/systemd/system/justnews@gpu_orchestrator.service
[Unit]
Description=JustNews GPU Orchestrator (GPU Resource Manager)
Documentation=https://github.com/Adrasteon/JustNewsAgent
After=network-online.target
Wants=network-online.target
# No MCP Bus dependency - this starts independently

[Service]
Type=simple
User=justnews
Group=justnews
WorkingDirectory=/opt/justnews/JustNewsAgent

# Environment configuration
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/gpu_orchestrator.env
Environment=PYTHON_BIN=/opt/justnews/venv/bin/python
Environment=GPU_ORCHESTRATOR_PORT=8014
Environment=USE_GPU=true

# Use the standardized launcher (no MCP Bus dependency via REQUIRE_BUS=0)
Environment=REQUIRE_BUS=0
ExecStart=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-start-agent.sh gpu_orchestrator

# Restart configuration
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=30

# Resource limits (GPU service needs more resources)
LimitNOFILE=65536
LimitNPROC=4096

# Security (relaxed for GPU access)
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target

# Example 3: Scout Agent (depends on MCP Bus)
# File: /etc/systemd/system/justnews@scout.service
[Unit]
Description=JustNews Scout Agent (Content Discovery)
Documentation=https://github.com/Adrasteon/JustNewsAgent
After=network-online.target justnews@mcp_bus.service
Wants=network-online.target
Requires=justnews@mcp_bus.service

[Service]
Type=simple
User=justnews
Group=justnews
WorkingDirectory=/opt/justnews/JustNewsAgent

# Environment configuration
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/scout.env
Environment=PYTHON_BIN=/opt/justnews/venv/bin/python
Environment=SCOUT_AGENT_PORT=8002

# Preflight check (validates MCP Bus is ready)
ExecStartPre=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-preflight-check.sh

# Use the standardized launcher
ExecStart=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-start-agent.sh scout

# Restart configuration
Restart=on-failure
RestartSec=15
TimeoutStartSec=120
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/justnews/JustNewsAgent
ReadWritePaths=/var/log/justnews

[Install]
WantedBy=multi-user.target

# Example 4: Template unit (for all agents)
# File: /etc/systemd/system/justnews@.service
# Usage: systemctl start justnews@scout.service, justnews@analyst.service, etc.
[Unit]
Description=JustNews %I Agent
Documentation=https://github.com/Adrasteon/JustNewsAgent
After=network-online.target
Wants=network-online.target

# All agents except gpu_orchestrator depend on MCP Bus
After=justnews@mcp_bus.service
# Use ConditionPathExists to make the dependency conditional
ConditionPathExists=!/etc/justnews/skip_mcp_dep

[Service]
Type=simple
User=justnews
Group=justnews
WorkingDirectory=/opt/justnews/JustNewsAgent

# Environment configuration (loads global + agent-specific)
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/%i.env
Environment=PYTHON_BIN=/opt/justnews/venv/bin/python

# Use the standardized launcher with agent name from instance
ExecStart=/opt/justnews/JustNewsAgent/deploy/systemd/scripts/justnews-start-agent.sh %i

# Restart configuration
Restart=on-failure
RestartSec=15
TimeoutStartSec=120
TimeoutStopSec=30

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/justnews/JustNewsAgent
ReadWritePaths=/var/log/justnews

[Install]
WantedBy=multi-user.target

# Deployment notes:
# 1. Create /etc/justnews/global.env with shared configuration
# 2. Create agent-specific env files: /etc/justnews/{agent}.env
# 3. Install the template unit: sudo cp justnews@.service /etc/systemd/system/
# 4. Enable agents: sudo systemctl enable justnews@scout.service justnews@analyst.service
# 5. Start in dependency order:
#    sudo systemctl start justnews@gpu_orchestrator.service
#    sudo systemctl start justnews@mcp_bus.service
#    sudo systemctl start justnews@scout.service
# 6. Check status: sudo systemctl status 'justnews@*'
# 7. View logs: sudo journalctl -u 'justnews@*' -f
