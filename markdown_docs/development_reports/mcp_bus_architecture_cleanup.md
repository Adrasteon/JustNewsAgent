---
title: MCP Bus Architecture Cleanup - August 2, 2025
description: Auto-generated description for MCP Bus Architecture Cleanup - August 2, 2025
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# MCP Bus Architecture Cleanup - August 2, 2025

## 🎯 Issue Identified

Found **two `mcp_bus` folders** in the JustNews V4 project:
1. `/mcp_bus/` (root level) - **ACTIVE**
2. `/agents/mcp_bus/` (agents folder) - **LEGACY**

## 🔍 Investigation Results

### Active MCP Bus: `/mcp_bus/` ✅
- **Docker Integration**: Referenced in `docker-compose.yml` 
- **Production Usage**: Has activity logs (`mcp_bus.log`) and `__pycache__/`
- **Clean Design**: Focused 70-line implementation
- **Proper Lifecycle**: Context managers and error handling
- **Current Architecture**: Matches V4 design patterns

### Legacy MCP Bus: `/agents/mcp_bus/` ❌
- **Unused**: No activity logs or runtime artifacts
- **Complex**: 115-line implementation with redundant code
- **Hardcoded URLs**: Legacy agent addressing patterns
- **Inconsistent API**: Different registration model
- **Architectural Misplacement**: Infrastructure in agents folder

## 🧹 Resolution

### Action Taken
```bash
mv agents/mcp_bus archive_obsolete_files/development_session_20250802/legacy_mcp_bus_agents_folder
```

### Architecture Clarification
- **MCP Bus Location**: Root level (`/mcp_bus/`) as infrastructure component
- **Agent Location**: Agent-specific code in (`/agents/*/`) 
- **Docker Build**: Uses `dockerfile: mcp_bus/Dockerfile` (root level)
- **Clean Separation**: Infrastructure vs application logic

## 📊 Impact Assessment

### Benefits
- ✅ **Single Source of Truth**: One MCP bus implementation
- ✅ **Clear Architecture**: Infrastructure at root, agents in agents/
- ✅ **Reduced Confusion**: Eliminates duplicate folders
- ✅ **Simplified Maintenance**: One codebase to maintain

### Validation
- ✅ **Docker Build**: Still references correct path
- ✅ **Agent Communication**: Unaffected (agents call root MCP bus)
- ✅ **System Function**: No operational impact

## 🎯 Architectural Clarity

### Correct Structure
```
/mcp_bus/                    # Infrastructure - Message bus system
├── main.py                  # Active FastAPI MCP bus
├── Dockerfile              # Docker build configuration
└── requirements.txt        # Dependencies

/agents/                     # Application Logic - Business agents
├── scout/                   # Content discovery agent
├── analyst/                 # Content analysis agent
├── memory/                  # Storage agent
└── [other agents]/         # Additional specialized agents
```

### Design Principle
**Infrastructure** (MCP bus, databases, message queues) belongs at **root level**.
**Application logic** (agents, business logic) belongs in **agents/** folder.

## ✅ Conclusion

Successfully resolved architectural duplication by archiving legacy MCP bus implementation. The system now has a single, clean MCP bus architecture that properly separates infrastructure from application logic.

**Result**: Clean architecture with single MCP bus implementation! 🚀

---
*Cleanup completed: August 2, 2025*
*Architecture validated: Single source of truth established*

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

