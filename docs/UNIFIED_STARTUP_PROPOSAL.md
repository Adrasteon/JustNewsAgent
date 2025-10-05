# Unified JustNews Startup Service Proposal

## Objective
To create a single, comprehensive startup service that:
1. Validates system readiness.
2. Starts all agents and services in the correct order.
3. Monitors and verifies the system's health post-startup.
4. Provides detailed logging and error handling.

---

## Components to Integrate

### 1. Preflight Validation
- **Script**: `justnews-preflight-check.sh`
- **Purpose**: Validate the readiness of core components (e.g., GPU Orchestrator, MCP Bus).
- **Integration**: Run as a `ExecStartPre` step for critical services.

### 2. Service Management
- **Script**: `enable_all.sh`
- **Purpose**: Manage the lifecycle of all JustNews services.
- **Integration**: Use as the main entry point for enabling, starting, and stopping services.

### 3. Agent Startup
- **Script**: `justnews-start-agent.sh`
- **Purpose**: Start individual agents with proper environment setup.
- **Integration**: Use as the `ExecStart` command for all agent services.

### 4. Status Verification
- **Script**: `justnews-system-status.sh`
- **Purpose**: Verify the statuses of agents, services, MPS, and NVML.
- **Integration**: Run as a post-startup health check.

### 5. GPU Monitoring
- **Script**: `gpu_watch.sh`
- **Purpose**: Collect GPU telemetry data.
- **Integration**: Run as a background service for continuous monitoring.

### 6. Cold Start
- **Script**: `justnews-cold-start.sh`
- **Purpose**: Perform a one-shot bring-up after a reboot.
- **Integration**: Triggered by a systemd timer.

### 7. Smoke Test
- **Script**: `justnews-boot-smoke.sh`
- **Purpose**: Perform a lightweight smoke test during boot.
- **Integration**: Triggered by a systemd timer.

### 8. Dependency Management
- **Script**: `wait_for_mcp.sh`
- **Purpose**: Wait for the MCP Bus to be ready before starting dependent services.
- **Integration**: Run as a `ExecStartPre` step for dependent services.

---

## Proposed Systemd Unit File

```plaintext
[Unit]
Description=Unified JustNews Startup Service
After=network-online.target
Wants=network-online.target postgresql.service

[Service]
Type=oneshot
User=root
ExecStartPre=/bin/bash -lc '/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/preflight.sh --gate-only'
ExecStart=/bin/bash -lc '/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/enable_all.sh start'
ExecStartPost=/bin/bash -lc '/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/justnews-system-status.sh'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

---

## Proposed Timers

### 1. Cold Start Timer
- **Unit**: `justnews-cold-start.timer`
- **Purpose**: Trigger the cold start script shortly after boot.

### 2. Smoke Test Timer
- **Unit**: `justnews-boot-smoke.timer`
- **Purpose**: Trigger the smoke test script after boot.

---

## Proposed Directory Structure

```
deploy/systemd/
├── scripts/
│   ├── enable_all.sh
│   ├── gpu_watch.sh
│   ├── health_check.sh
│   ├── justnews-boot-smoke.sh
│   ├── justnews-cold-start.sh
│   ├── justnews-start-agent.sh
│   ├── reset_and_start.sh
│   ├── wait_for_mcp.sh
├── units/
│   ├── justnews@.service
│   ├── justnews-cold-start.service
│   ├── justnews-cold-start.timer
│   ├── justnews-boot-smoke.service
│   ├── justnews-boot-smoke.timer
│   ├── drop-ins/
│   │   ├── 05-gate-timeout.conf
│   │   ├── 10-preflight-gating.conf
│   │   ├── 20-restart-policy.conf
│   ├── overrides/
│       ├── 10-preflight-gating.conf
│       ├── 20-restart-policy.conf
```

---

## Unified Command

To simplify usage, the unified startup service can be invoked from anywhere using the following command:

```bash
square-one
```

This command will:
1. Validate system readiness.
2. Start all agents and services in the correct order.
3. Monitor and verify the system's health post-startup.

---

## Benefits of the Unified Service

### 1. Simplified Management
- A single entry point for starting, stopping, and verifying the system.

### 2. Improved Reliability
- Preflight checks ensure the system is ready before starting services.

### 3. Enhanced Monitoring
- Continuous GPU telemetry and post-startup health checks.

### 4. Modular Design
- Individual scripts can still be used independently if needed.

---

**End of Proposal**

---

**Last Updated**: September 25, 2025  
**Copyright**: © 2025 JustNews Inc. All rights reserved.
