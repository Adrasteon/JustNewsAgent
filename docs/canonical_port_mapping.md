# JustNewsAgent Canonical Port Mapping

## 📋 Complete Port Usage Analysis

*Generated on: September 1, 2025*

This document provides the canonical list of all ports used in the JustNewsAgent system, compiled from a comprehensive search of the entire codebase.

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
| **8013** | Balancer Agent | Load balancing and resource management | `BALANCER_AGENT_PORT=8013` | ✅ Active |

---

## 🌐 API & Dashboard Services (8010-8022)

| Port | Service | Purpose | Access URL | Status |
|------|---------|---------|------------|--------|
| **8010** | DB Worker / Editor UI | Database operations and content editing | `http://localhost:8010` | ✅ Active |
| **8011** | GPU Dashboard | GPU monitoring and management | `http://localhost:8011/gpu/dashboard` | ✅ Active |
| **8012** | Analytics Dashboard | System analytics and reporting | `http://localhost:8012/api/health` | ✅ Active |
| **8013** | Analytics Dashboard (Alt) | Alternative analytics interface | `http://localhost:8013` | ✅ Active |
| **8020** | GraphQL API | Advanced GraphQL query interface | `http://localhost:8020/graphql` | ✅ Active |
| **8021** | REST Archive API | RESTful archive access and knowledge graph | `http://localhost:8021/health` | ✅ Active |
| **8022** | Authentication API | JWT-based user authentication | `http://localhost:8021/auth/register` | ✅ Active (integrated into Archive API) |

---

## 💾 Database Services

| Port | Service | Purpose | Configuration | Status |
|------|---------|---------|---------------|--------|
| **5432** | PostgreSQL | Main application database | `JUSTNEWS_DB_PORT=5432` | ✅ Active |

---

## 🔗 External Services

| Port | Service | Purpose | Notes | Status |
|------|---------|---------|-------|--------|
| **8080** | Ollama WebUI | AI model interface | External service | ✅ Active |

---

## 📊 Port Distribution Summary

- **Agent Services**: 8000-8009 (10 ports)
- **API/Dashboard Services**: 8010-8022 (7 ports)
- **Database**: 5432 (1 port)
- **External Services**: 8080 (1 port)
- **Total Ports Used**: 19

---

## ⚠️ Important Notes

### Port Conflicts
- **Port 8009**: Originally assigned to NewsReader/Balancer but conflicts with main system agents
- **Solution**: Authentication API integrated into Archive API on port 8021 to avoid conflicts
- **✅ RESOLVED**: NewsReader and Balancer agents port conflict fixed - Balancer moved to port 8013

### Environment Variables
Most agent ports can be configured via environment variables:
```bash
export MCP_BUS_PORT=8000
export SCOUT_AGENT_PORT=8002
export ANALYTICS_PORT=8012
export DASHBOARD_PORT=8011
# ... etc
```

### Service Dependencies
- **MCP Bus (8000)**: Central coordination point for all agents
- **All agents communicate through the MCP Bus**
- **API services (8020-8021)**: Provide external access to the system
- **Database (5432)**: Required for all data persistence operations

---

## 🔍 Search Methodology

This analysis was compiled from a comprehensive search of:
- ✅ `localhost` references across all files
- ✅ `127.0.0.1` IP address usage
- ✅ Port number patterns (`:8000`, `:8001`, etc.)
- ✅ Environment variable definitions
- ✅ Configuration files and scripts
- ✅ Documentation and README files
- ✅ Source code port assignments

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
- **Conflict Resolution**: Authentication API integrated into Archive API on port 8021
- **Documentation**: Keep this file updated when new services are added

---

*This document serves as the authoritative reference for all port assignments in the JustNewsAgent system.*