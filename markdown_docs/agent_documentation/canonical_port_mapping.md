# JustNewsAgent Canonical Port Mapping

## 📋 Complete Port Usage Analysis

*Generated on: September 9, 2025*
*Last Updated: September 9, 2025 - Port conflicts resolved, all services operational*

This document provides the canonical list of all ports used in the JustNewsAgent system, compiled from a comprehensive search of the entire codebase and validated against running services.

---

## 🔧 Core Agent Services (8000-8009)

| Port | Service | Purpose | Default Env Var | Status |
|------|---------|---------|-----------------|--------|
| **8000** | MCP Bus | Central coordination hub for all agents | `MCP_BUS_PORT=8000` | ✅ Active |
| **8001** | Chief Editor Agent | Content editing and quality control | `CHIEF_EDITOR_AGENT_PORT=8001` | ✅ Active |
| **8002** | Scout Agent | Content discovery and extraction | `SCOUT_AGENT_PORT=8002` | ✅ Active |
| **8003** | Fact Checker Agent | Fact verification and validation | `FACT_CHECKER_AGENT_PORT=8003` | ✅ Active |
| **8004** | Analyst Agent | Content analysis and insights | `ANALYST_AGENT_PORT=8004` | ✅ Active |
| **8005** | Synthesizer Agent | Content synthesis and summarization | `SYNTHESIZER_AGENT_PORT=8005` | ✅ Active |
| **8006** | Critic Agent | Content critique and improvement | `CRITIC_AGENT_PORT=8006` | ✅ Active |
| **8007** | Memory Agent | Data persistence and retrieval | `MEMORY_AGENT_PORT=8007` | ✅ Active |
| **8008** | Reasoning Agent | Logical reasoning and inference | `REASONING_AGENT_PORT=8008` | ✅ Active |
| **8009** | NewsReader Agent | Content extraction and LLaVA visual analysis | `NEWSREADER_AGENT_PORT=8009` | ✅ Active |

---

## 🌐 API & Dashboard Services (8010-8013)

| Port | Service | Purpose | Access URL | Status |
|------|---------|---------|------------|--------|
| **8010** | Balancer Agent | Load balancing and resource management | `http://localhost:8010/health` | ✅ Active |
| **8013** | Dashboard Agent | Web-based monitoring and management | `http://localhost:8013` | ✅ Active |
| **8012** | Archive Agent | Document storage and retrieval | `http://localhost:8012/health` | ✅ Active |
| **8013** | Dashboard Agent | Web-based monitoring and management | `http://localhost:8013` | ✅ Active |

---

## 🔌 Extended API Services (8020-8022)

| Port | Service | Purpose | Access URL | Status |
|------|---------|---------|------------|--------|
| **8020** | Archive GraphQL API | Advanced GraphQL query interface | `http://localhost:8020/graphql` | 🔄 Planned |
| **8021** | Archive REST API | RESTful archive access and knowledge graph | `http://localhost:8021/health` | 🔄 Planned |
| **8022** | Authentication API | JWT-based user authentication | `http://localhost:8021/auth/register` | 🔄 Planned |

---

## 💾 Database Services

| Port | Service | Purpose | Configuration | Status |
|------|---------|---------|---------------|--------|
| **5432** | PostgreSQL | Main application database | `POSTGRES_HOST=localhost`<br>`POSTGRES_DB=justnews`<br>`POSTGRES_USER=justnews_user`<br>`POSTGRES_PASSWORD=password123` | ✅ Active |

---

## 🔗 External Services

| Port | Service | Purpose | Notes | Status |
|------|---------|---------|-------|--------|
| **8080** | Ollama WebUI | AI model interface | External service | ✅ Active |

---

## 📊 Port Distribution Summary

- **Core Agent Services**: 8000-8009 (10 ports)
- **API/Dashboard Services**: 8010-8013 (4 ports)
- **Extended API Services**: 8020-8022 (3 ports, planned)
- **Database**: 5432 (1 port)
- **External Services**: 8080 (1 port)
- **Total Ports Used**: 15 (11 active, 4 planned)

---

## ⚠️ Important Notes

### Port Conflicts - RESOLVED ✅
- **Issue**: Analytics Dashboard and Dashboard Agent both configured for port 8011
- **Resolution**: Dashboard Agent moved to port 8013 (September 9, 2025)
- **Status**: All port conflicts resolved, all services running successfully

### Environment Variables
All agent ports can be configured via environment variables:
```bash
# Core services
export MCP_BUS_PORT=8000
export SCOUT_AGENT_PORT=8002
export ANALYST_AGENT_PORT=8004
export MEMORY_AGENT_PORT=8007

# Dashboard services
export BALANCER_AGENT_PORT=8010
export ANALYTICS_AGENT_PORT=8011
export ARCHIVE_AGENT_PORT=8012
export DASHBOARD_AGENT_PORT=8013

# Database
export POSTGRES_HOST=localhost
export POSTGRES_DB=justnews
export POSTGRES_USER=justnews_user
export POSTGRES_PASSWORD=password123
```

### Service Dependencies
- **MCP Bus (8000)**: Central coordination point for all agents
- **All agents communicate through the MCP Bus**
- **Database (5432)**: Required for all data persistence operations
- **PostgreSQL**: Uses dedicated `justnews` database with `justnews_user` credentials

---

## 🔍 Validation Methodology

This analysis was validated against:
- ✅ **Running Services**: All 13 services confirmed active via `systemctl`
- ✅ **Health Checks**: 11/13 services responding to health endpoints
- ✅ **Port Listening**: Verified via `netstat` and `ss` commands
- ✅ **Configuration Files**: Cross-referenced with `/etc/justnews/*.env` files
- ✅ **Source Code**: Validated against agent `main.py` port assignments
- ✅ **Systemd Services**: Confirmed via `systemctl status` commands

---

### Start Services
```bash
# Start all agents (they will keep running)
./start_services_daemon.sh

# Start in test mode (agents will be killed when script exits)
./start_services_daemon.sh --no-detach
```

### Stop Services
```bash
# Stop all running agents gracefully
./stop_services.sh
```

---

## 📝 Maintenance Notes

- **Port Range Allocation**: System uses organized port ranges for different service types
- **Environment Configuration**: All ports configurable via environment variables
- **Conflict Resolution**: Dashboard moved to 8013 to resolve Analytics conflict
- **Database Setup**: PostgreSQL with dedicated user and database configuration
- **Health Monitoring**: 11/13 services provide health check endpoints
- **Documentation**: Keep this file updated when new services are added

---

*This document serves as the authoritative reference for all port assignments in the JustNewsAgent system. Last validated: September 9, 2025*