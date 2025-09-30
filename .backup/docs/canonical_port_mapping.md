---
title: canonical port mapping
description: Auto-generated description for canonical port mapping
tags: [documentation]
status: current
last_updated: 2025-09-15
---

# Canonical Port Mapping

This document defines the canonical port assignments for all JustNews agents and services.

## Agent Port Assignments

| Port | Agent/Service | Description | Status |
|------|---------------|-------------|--------|
| 8000 | MCP Bus | Model Context Protocol Bus for agent communication | ✅ Active |
| 8001 | *(Available)* | Reserved for future Chief Editor agent | Reserved |
| 8002 | Scout Agent | Web content discovery and initial analysis | ✅ Active |
| 8003 | Fact Checker Agent | Multi-model fact verification and evidence analysis | ✅ Active |
| 8004 | Analyst Agent | AI-powered news analysis with TensorRT optimization | ✅ Active |
| 8005 | Synthesizer Agent | Article synthesis from verified data | ✅ Active |
| 8006 | Critic Agent | Quality assessment and neutrality analysis | ✅ Active |
| 8007 | *(Available)* | Reserved for future Memory agent | Reserved |
| 8008 | Reasoning Agent | Nucleoid symbolic logic processing | ✅ Active |
| 8009 | *(Previously Crawler)* | Port was assigned to Crawler but conflicts detected | ❌ Deprecated |
| 8010-8014 | *(Available)* | Reserved for future agents | Reserved |
| 8015 | Crawler Agent | Unified production crawling with AI analysis pipeline | ✅ Active |
| 8016 | Crawler Control | Web interface for crawler management and monitoring | ✅ Active | ✅ Active |

## Port Assignment Rules

1. **Range**: All agent ports use the 8000-8999 range
2. **Increment**: Ports are assigned in increments of 1, skipping only when conflicts exist
3. **Documentation**: All port assignments must be documented in this canonical file
4. **Conflicts**: If a port conflict is detected, the agent must be reassigned to the next available port
5. **Reservation**: Available ports are marked as "Reserved for future agent" to prevent accidental usage
6. **Verification**: Port assignments are verified against actual agent code, not just documentation

## Current Agent Status

### ✅ Active Agents (Ports 8000-8008, 8010-8016)
- **MCP Bus** (8000): Central communication hub
- **Chief Editor** (8001): Workflow orchestration
- **Scout** (8002): Content discovery with LLaMA-3-8B GPU acceleration
- **Fact Checker** (8003): Multi-model verification system
- **Analyst** (8004): TensorRT-accelerated sentiment/bias analysis
- **Synthesizer** (8005): 4-model synthesis stack (BERTopic, BART, FLAN-T5, SentenceTransformers)
- **Critic** (8006): Quality assessment and review
- **Memory** (8007): PostgreSQL + vector search storage
- **Reasoning** (8008): Nucleoid symbolic logic engine
- **NewsReader** (8009): LLaVA visual analysis
- **Balancer** (8010): Load balancing and resource management
- **Analytics** (8011): Advanced performance & analytics API
- **Archive** (8012): Document storage and retrieval
- **Dashboard** (8013): Web-based monitoring and management
- **GPU Orchestrator** (8014): Central GPU coordination and telemetry
- **Crawler** (8015): Unified production crawling with AI pipeline
- **Crawler Control** (8016): Web interface for crawler management and monitoring

### 🔄 Reserved Ports (None currently)
- All ports in the 8000-8015 range are now actively assigned

### ❌ Deprecated Ports (8009)
- **8009**: Previously assigned to Crawler agent, moved to 8015 due to conflicts

## Recent Changes

- **2025-09-16**: Assigned port 8016 to Crawler Control web interface
- **2025-09-15**: Complete port mapping audit - discovered actual assignments differ from documentation
- **2025-09-15**: Corrected all port assignments based on actual agent code verification
- **2025-09-15**: Moved Crawler Agent from port 8009 to 8015 due to port conflict resolution
- **2025-09-15**: Added comprehensive port mapping with all active services (8000-8015)
- **2025-09-12**: Added canonical port mapping documentation

## Verification Methodology

Port assignments were verified by:
1. **Code Audit**: Examining `*_AGENT_PORT` environment variables in each agent's `main.py`
2. **Default Values**: Checking fallback port assignments in agent startup code
3. **Conflict Resolution**: Identifying and resolving port conflicts (8009 → 8015)
4. **Documentation Sync**: Ensuring canonical documentation matches actual implementation

## Future Port Assignments

When adding new agents, follow this priority order:
1. Use next available reserved port (8001, 8007, 8010-8014)
2. If all reserved ports are used, assign next sequential port (8016+)
3. Update this canonical document immediately after assignment
4. Test for conflicts before deployment

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

