# JustNewsAgent Documentation Catalogue

**Version:** 2.0
**Last Updated:** 2025-09-15
**Total Documents:** 182
**Categories:** 28

## Table of Contents

1. [Main Documentation](#main_documentation)
2. [Architecture & Design](#architecture_design)
3. [Agent Documentation](#agent_documentation)
4. [GPU Setup & Configuration](#gpu_configuration)
5. [Production & Deployment](#production_deployment)
6. [API & Integration](#api_integration)
7. [Training & Learning](#training_learning)
8. [Monitoring & Analytics](#monitoring_analytics)
9. [Compliance & Security](#compliance_security)
10. [Development Reports](#development_reports)
11. [Scripts Tools](#scripts_tools)
12. [Deployment System](#deployment_system)
13. [General Documentation](#general_documentation)
14. [Performance Optimization](#performance_optimization)
15. [Architecture & Design Reports](#development_reports_architecture)
16. [Implementation Reports](#development_reports_implementation)
17. [Performance & Optimization Reports](#development_reports_performance)
18. [Testing & Quality Assurance Reports](#development_reports_testing)
19. [Deployment & Operations Reports](#development_reports_deployment)
20. [Training & Learning Reports](#development_reports_training)
21. [Integration & Workflow Reports](#development_reports_integration)
22. [Maintenance & Housekeeping Reports](#development_reports_maintenance)
23. [Core Agent Documentation](#agent_documentation_core_agents)
24. [Specialized Agent Documentation](#agent_documentation_specialized_agents)
25. [Deprecated Agent Documentation](#agent_documentation_deprecated_agents)
26. [Agent Management & Tools](#agent_documentation_agent_management)
27. [Model Integration Documentation](#agent_documentation_model_integration)
28. [Crawling & Data Collection](#agent_documentation_crawling_systems)

---

## Main Documentation

**Category ID:** main_documentation
**Priority:** critical
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Main Project Documentation](README.md) | Complete system overview, installation, usage, and deployment guide with RTX3090 GPU support This co... | overview, installation, deployment | production_ready |
| [Version History & Changelog](CHANGELOG.md) | Detailed changelog including PyTorch 2.6.0+cu124 upgrade and GPU optimization achievements This comp... | versions, history, releases | current |

---

## Architecture & Design

**Category ID:** architecture_design
**Priority:** critical
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [JustNews V4 Architecture Proposal](docs/JustNews_Proposal_V4.md) | Hybrid architecture proposal with specialized models and continuous learning This comprehensive guid... | proposal, hybrid-architecture, continuous-learning | current |
| [JustNews V4 Implementation Plan](docs/JustNews_Plan_V4.md) | Native GPU-accelerated architecture migration plan with specialized models This document includes de... | planning, migration, specialized-models | current |
| [MCP Bus Architecture](markdown_docs/development_reports/mcp_bus_architecture_cleanup.md) | Central communication hub design and implementation for agent coordination This comprehensive guide ... | mcp, communication, agents | current |
| [Technical Architecture Overview](markdown_docs/TECHNICAL_ARCHITECTURE.md) | Complete system architecture with RTX3090 GPU allocation and PyTorch 2.6.0+cu124 integration This co... | architecture, gpu, pytorch | current |

---

## Agent Documentation

**Category ID:** agent_documentation
**Priority:** high
**Documents:** 80

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [AGENT_MODEL_MAP — Definitive Agent → Model mapping](markdown_docs/agent_documentation/AGENT_MODEL_MAP.md) | Authoritative mapping of agents to model dependencies | models, reasoning, multi-agent | current |
| [API Documentation](markdown_docs/agent_documentation/api_documentation.md) | API reference and usage examples for agents | api, gpu, synthesizer | current |
| [Crawler Agent](agents/crawler/README.md) | Overview and operational notes for the Crawler Agent | scout, gpu, memory | current |
| [Deployment - Systemd Quick Reference](markdown_docs/agent_documentation/systemd/QUICK_REFERENCE.md) | Quick operational commands and common workflows for systemd deployment | deployment, systemd | current |
| [Systemd Comprehensive Guide](markdown_docs/agent_documentation/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md) | Detailed systemd deployment and unit management guide | deployment, systemd | current |
| [GPU Orchestrator Operations (Port 8014)](markdown_docs/agent_documentation/GPU_ORCHESTRATOR_OPERATIONS.md) | Operations notes for GPU Orchestrator | gpu, operations | current |
| [NewsReader Agent - Production-Validated Configuration](agents/newsreader/README.md) | Production configuration and known issues/resolutions | newsreader, production | current |
| [Preflight Runbook: Model Gating and Preload Failures](markdown_docs/agent_documentation/preflight_runbook.md) | Runbook for preflight gating and handling preload failures | preflight, operations | current |
| [Operator Guide — Systemd Deployment and Operations](markdown_docs/agent_documentation/OPERATOR_GUIDE_SYSTEMD.md) | Operator-focused guide for systemd deployment | operator, systemd | current |
| [Model store guidelines](markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md) | Guidelines for model-store layout and safe updates | models, deployment | current |
| [Agent Communication Protocols Documentation](markdown_docs/agent_documentation/agent_communication_protocols.md) | Comprehensive documentation covering agent communication protocols documentation with detailed techn... | optimization, deployment, mcp | current |
| [Agent Documentation index](markdown_docs/agent_documentation/README.md) | This folder contains agent-specific documentation used by operators and
developers. Key documents:..... | ai-agents, multi-agent, agents | current |
| [Balancer Agent V1 - Integration & Debugging Guide](markdown_docs/agent_documentation/BALANCER_AGENT_INTEGRATION_GUIDE.md) | ## Overview
The Balancer Agent is a production-grade component of the JustNews V4 system, designed t... | compliance, mcp, api | current |
| [Balancer Agent V1 Documentation](markdown_docs/agent_documentation/BALANCER_AGENT_V1.md) | ## Overview
The Balancer Agent is a production-ready component of the JustNews V4 system, designed t... | optimization, compliance, training | current |
| [Configuration Management Documentation](markdown_docs/agent_documentation/configuration_management_guide.md) | Comprehensive documentation covering configuration management documentation with detailed technical ... | security, gpu, version-specific | current |
| [Contributing](agents/reasoning/nucleoid_repo/CONTRIBUTING.md) | Thanks to declarative programming, we have a brand-new approach to data and logic. As we are still d... | development, guidelines, contribution | current |
| [Contributor Covenant Code of Conduct](agents/reasoning/nucleoid_repo/CODE_OF_CONDUCT.md) | Documentation for Contributor Covenant Code of Conduct providing essential information and technical... | security | current |
| [Crawl4AI vs Playwright — feature-by-feature comparison](markdown_docs/agent_documentation/Crawl4AI_vs_Playwright_Comparison.md) | Documentation for Crawl4AI vs Playwright — feature-by-feature comparison providing essential informa... | compliance, deployment, api | current |
| [Crawler Consolidation Plan — JustNewsAgent](markdown_docs/agent_documentation/Crawler_Consolidation_Plan.md) | Date: 2025-08-27
Author: Consolidation plan generated from interactive session...... | api, archive, models | current |
| [Data Pipeline Documentation](markdown_docs/agent_documentation/data_pipeline_documentation.md) | Comprehensive documentation covering data pipeline documentation with detailed technical information... | gpu, version-specific, cuda | current |
| [Database Schema & Operations Documentation](markdown_docs/agent_documentation/database_schema_operations.md) | Comprehensive documentation covering database schema & operations documentation with detailed techni... | optimization, security, training | current |
| [Deployment Procedures Documentation](markdown_docs/agent_documentation/deployment_procedures_guide.md) | Comprehensive documentation covering deployment procedures documentation with detailed technical inf... | security, gpu, version-specific | current |
| [Embedding Helper](markdown_docs/agent_documentation/EMBEDDING_HELPER.md) | Documentation for Embedding Helper providing essential information and technical details for the Jus... | models, gpu, multi-agent | current |
| [Examples for systemd native deployment](markdown_docs/agent_documentation/systemd/examples/README.md) | Files in this directory are examples and helpers to install the JustNews systemd units. | mcp, ai-agents, scout | current |
| [GPU Acceleration Documentation](markdown_docs/agent_documentation/gpu_acceleration_guide.md) | Comprehensive documentation covering gpu acceleration documentation with detailed technical informat... | gpu, version-specific, cuda | current |
| [GPU Dashboard and Watcher Ingestion — Operations Guide](markdown_docs/agent_documentation/gpu_dashboard_and_watcher_ingestion.md) | Documentation for GPU Dashboard and Watcher Ingestion — Operations Guide... | performance, gpu, dashboard | current |
| [GPU Orchestrator Operations (Port 8014)](markdown_docs/agent_documentation/GPU_ORCHESTRATOR_OPERATIONS.md) | Documentation for GPU Orchestrator Operations (Port 8014)... | production, gpu, models | current |
| [Gpu Orchestrator](markdown_docs/agent_documentation/gpu_orchestrator.md) | Documentation for Gpu Orchestrator... | gpu, architecture | current |
| [Gpu Orchestrator Checklist](markdown_docs/agent_documentation/gpu_orchestrator_checklist.md) | Documentation for Gpu Orchestrator Checklist... | production, archive, gpu | current |
| [Hugging Face model caching and pre-download for Memory Agent](markdown_docs/agent_documentation/HF_MODEL_CACHING.md) | This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cach... | deployment, api, models | current |
| [Implementation Summary](agents/newsreader/IMPLEMENTATION_SUMMARY.md) | Detailed report documenting implementation summary with analysis, findings, and recommendations for ... | agent, implementation, summary | current |
| [JustNews Advanced Monitoring - Phase 5 Implementation Plan](markdown_docs/agent_documentation/systemd/phase5_advanced_monitoring_plan.md) | Documentation for JustNews Advanced Monitoring - Phase 5 Implementation Plan... | production, monitoring, version-specific | current |
| [JustNews Advanced Monitoring System - PHASE 5 COMPLETE ✅](markdown_docs/agent_documentation/monitoring/README.md) | Documentation for JustNews Advanced Monitoring System - PHASE 5 COMPLETE ✅... | production, performance, gpu | current |
| [JustNews Agent - API Documentation](markdown_docs/agent_documentation/PHASE3_API_DOCUMENTATION.md) | ## Phase 3 Sprint 3-4: Advanced Knowledge Graph APIs...... | security, gpu, synthesizer | current |
| [JustNews PostgreSQL Integration Guide](markdown_docs/agent_documentation/systemd/postgresql_integration.md) | Documentation for JustNews PostgreSQL Integration Guide... | production, performance, archive | current |
| [JustNews Systemd Quick Reference](markdown_docs/agent_documentation/systemd/QUICK_REFERENCE.md) | Documentation for JustNews Systemd Quick Reference... | monitoring, deployment, architecture | current |
| [JustNews V4 - Low-risk Action Plan (2025-09-12)](markdown_docs/agent_documentation/action_plan_2025-09-12.md) | Documentation for JustNews V4 - Low-risk Action Plan (2025-09-12)... | production, gpu, ai-agents | current |
| [JustNews V4 Operator Guide — Systemd Deployment and Operations](markdown_docs/agent_documentation/OPERATOR_GUIDE_SYSTEMD.md) | Documentation for JustNews V4 Operator Guide — Systemd Deployment and Operations... | production, performance, archive | current |
| [JustNews V4 Systemd Implementation Guide](markdown_docs/agent_documentation/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md) | Documentation for JustNews V4 Systemd Implementation Guide... | production, performance, gpu | current |
| [JustNews native deployment (systemd)](markdown_docs/agent_documentation/systemd/DEPLOYMENT.md) | Documentation for JustNews native deployment (systemd)... | mcp, models, logging | current |
| [JustNewsAgent Canonical Port Mapping](markdown_docs/agent_documentation/canonical_port_mapping.md) | Complete documentation for JustNewsAgent Canonical Port Mapping including implementation details, co... | analytics, mcp, api | current |
| [LLaVA NewsReader Agent Implementation Summary](agents/newsreader/documentation/IMPLEMENTATION_SUMMARY.md) | Detailed report documenting llava newsreader agent implementation summary with analysis, findings, a... | optimization, pytorch, mcp | current |
| [Later: resume](markdown_docs/agent_documentation/Crawl4AI_API_SUMMARY.md) | This short reference summarises the Crawl4AI programmatic APIs, dispatcher classes, REST endpoints, ... | deployment, logging, api | current |
| [Lifespan Migration](agents/newsreader/LIFESPAN_MIGRATION.md) | Documentation for Lifespan Migration providing essential information and technical details for the J... | agent, migration, architecture | current |
| [Lifespan Migration](agents/newsreader/documentation/LIFESPAN_MIGRATION.md) | Documentation for Lifespan Migration providing essential information and technical details for the J... | agent, migration, architecture | current |
| [MCP Bus Architecture Documentation](markdown_docs/agent_documentation/mcp_bus_architecture.md) | Comprehensive documentation covering mcp bus architecture documentation with detailed technical info... | deployment, mcp, security | current |
| [MCP Bus Operations Guide (Port 8000)](markdown_docs/agent_documentation/MCP_BUS_OPERATIONS.md) | Documentation for MCP Bus Operations Guide (Port 8000)... | production, scout, archive | current |
| [Model Usage](markdown_docs/agent_documentation/MODEL_USAGE.md) | Documentation for Model Usage providing essential information and technical details for the JustNews... | deployment, mcp, api | current |
| [Model store guidelines](markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md) | This document explains the canonical model-store layout and safe update patterns for
per-agent model... | deployment, training, security | current |
| [Monitoring and Observability Documentation](markdown_docs/agent_documentation/monitoring_observability_guide.md) | Comprehensive documentation covering monitoring and observability documentation with detailed techni... | security, gpu, version-specific | current |
| [Native TensorRT Analyst Agent - Production Ready](agents/analyst/NATIVE_TENSORRT_README.md) | ## 🏆 **Production Status: VALIDATED & DEPLOYED**...... | optimization, deployment, mcp | current |
| [Native TensorRT Analyst Agent - Quick Start Guide](agents/analyst/NATIVE_AGENT_README.md) | Comprehensive documentation covering native tensorrt analyst agent - quick start guide with detailed... | deployment, mcp, api | current |
| [New Blueprint Agents](markdown_docs/agent_documentation/New_Blueprint_Agents.md) | Complete documentation for New Blueprint Agents including implementation details, configuration, and... | security, knowledge-graph, gpu | current |
| [News Outlets Loader & Backfill Runbook](markdown_docs/agent_documentation/NEWS_OUTLETS_RUNBOOK.md) | This runbook explains how to safely run the canonical sources loader (`scripts/news_outlets.py`) and... | performance, multi-agent, ai-agents | current |
| [NewsReader Agent - Production-Validated Configuration](agents/newsreader/README.md) | ## 🚨 **CRITICAL UPDATE: GPU Crash Resolution - August 13, 2025**...... | optimization, deployment, api | current |
| [NewsReader V2 Vision-Language Model Fallback Logic](markdown_docs/agent_documentation/NEWSREADER_V2_MODEL_FALLBACK.md) | ## Overview
The NewsReader V2 agent now implements robust fallback logic for vision-language model i... | optimization, mcp, gpu | current |
| [Next-Generation AI-First Scout Agent V2 Documentation](markdown_docs/agent_documentation/SCOUT_AGENT_V2_DOCUMENTATION.md) | Comprehensive documentation covering next-generation ai-first scout agent v2 documentation with deta... | gpu, cuda, agents | current |
| [Nucleoid](agents/reasoning/nucleoid_repo/arc/src/instruct_dataset/nucleoid.md) | Nucleoid extends JavaScript syntax for declarative (logic) programming.
Nucleoid has two modes; decl... | agent, reasoning, logic | current |
| [Operations Quick Reference](markdown_docs/agent_documentation/OPERATIONS_QUICK_REFERENCE.md) | Documentation for Operations Quick Reference... | production, scout, gpu | current |
| [Performance Optimization Documentation](markdown_docs/agent_documentation/performance_optimization_documentation.md) | Comprehensive documentation covering performance optimization documentation with detailed technical ... | optimization, pytorch, training | current |
| [Potential Development Paths](markdown_docs/agent_documentation/potential_development_paths.md) | This document captures a compact summary of recent analysis and recommendations about the project's ... | analytics, mcp, api | current |
| [Potential News Sources](markdown_docs/agent_documentation/potential_news_sources.md) | Documentation for Potential News Sources providing essential information and technical details for t... | security | current |
| [Preflight Runbook: Model Gating and Preload Failures](markdown_docs/agent_documentation/preflight_runbook.md) | Documentation for Preflight Runbook: Model Gating and Preload Failures... | gpu, models, ai-agents | current |
| [Product Modalities Comparison](markdown_docs/agent_documentation/product_modalities_comparison.md) | This document compares three high-level product modalities the JustNews system can pursue, aligns ea... | gpu, synthesizer, fact-checker | current |
| [Readme](agents/reasoning/nucleoid_repo/README.md) | Documentation for Readme providing essential information and technical details for the JustNews syst... | reasoning, memory, architecture | current |
| [Reasoning Agent](agents/reasoning/README.md) | This package contains the reasoning agent (Nucleoid) for JustNews....... | mcp, api, reasoning | current |
| [Reasoning Agent Complete Implementation Documentation](markdown_docs/agent_documentation/REASONING_AGENT_COMPLETE_IMPLEMENTATION.md) | Comprehensive documentation covering reasoning agent complete implementation documentation with deta... | optimization, deployment, mcp | current |
| [Scout Agent - Enhanced Deep Crawl Documentation](markdown_docs/agent_documentation/SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md) | **JustNews V4 Scout Agent with Native Crawl4AI Integration**...... | optimization, analytics, deployment | current |
| [Scout Agent V2 - Next-Generation AI-First Content Analysis System](agents/scout/README.md) | Complete documentation for Scout Agent V2 - Next-Generation AI-First Content Analysis System includi... | optimization, deployment, training | current |
| [Scout → Memory Pipeline Success Summary](markdown_docs/agent_documentation/SCOUT_MEMORY_PIPELINE_SUCCESS.md) | **Date**: January 29, 2025  
**Milestone**: Core JustNews V4 pipeline operational with native deploy... | deployment, training, mcp | current |
| [Security Implementation Documentation](markdown_docs/agent_documentation/security_implementation_documentation.md) | Comprehensive documentation covering security implementation documentation with detailed technical i... | optimization, compliance, security | current |
| [Security Implementation Guide](markdown_docs/agent_documentation/security_implementation_guide.md) | Comprehensive documentation covering security implementation guide with detailed technical informati... | compliance, deployment, mcp | current |
| [Sources Schema and Workflow](markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md) | This document specifies the `sources` schema, provenance mapping (`article_source_map`), ingestion w... | analytics, deployment, security | current |
| [Systemd scaffold for JustNews](markdown_docs/agent_documentation/systemd/README.md) | Documentation for Systemd scaffold for JustNews... | ai-agents, multi-agent, monitoring | current |
| [TensorRT Quickstart (safe, no-GPU stub)](agents/analyst/TENSORRT_QUICKSTART.md) | This file explains how to run a safe, developer-friendly stub for the TensorRT engine build process.... | tensorrt, gpu, multi-agent | current |
| [Training System API Documentation](markdown_docs/agent_documentation/training_system_api.md) | Comprehensive documentation covering training system api documentation with detailed technical infor... | training, api, performance | current |
| [Unified Production Crawler](agents/scout/production_crawlers/README.md) | ## Overview...... | production, performance, scout | current |
| [Use the main RAPIDS environment](markdown_docs/agent_documentation/gpu_runner_README.md) | Documentation for Use the main RAPIDS environment providing essential information and technical deta... | pytorch, analytics, api | current |
| [Why INT8 Quantization Should Be Implemented Immediately](agents/newsreader/documentation/INT8_QUANTIZATION_RATIONALE.md) | Documentation for Why INT8 Quantization Should Be Implemented Immediately providing essential inform... | optimization, deployment, tensorrt | current |
| [all-MiniLM-L6-v2](agents/memory/models/all-MiniLM-L6-v2/README.md) | Documentation for all-MiniLM-L6-v2 providing essential information and technical details for the Jus... | version-specific, training, models | current |
| [all-MiniLM-L6-v2](agents/memory/models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md) | Documentation for all-MiniLM-L6-v2 providing essential information and technical details for the Jus... | version-specific, training, models | current |
| [all-MiniLM-L6-v2](agents/memory/agents/memory/models/all-MiniLM-L6-v2/README.md) | Documentation for all-MiniLM-L6-v2... | training, version-specific, models | current |
| [all-MiniLM-L6-v2](agents/memory/agents/memory/models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/README.md) | Documentation for all-MiniLM-L6-v2... | training, version-specific, models | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/README.md) | Documentation for all-mpnet-base-v2 providing essential information and technical details for the Ju... | training, api, models | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md) | Documentation for all-mpnet-base-v2 providing essential information and technical details for the Ju... | training, api, models | current |
| [nuc-arc](agents/reasoning/nucleoid_repo/arc/README.md) | Documentation for nuc-arc providing essential information and technical details for the JustNews sys... | reasoning, training, performance | current |

---

## GPU Setup & Configuration

**Category ID:** gpu_configuration
**Priority:** high
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [GPU Model Store Assessment](docs/GPU_ModelStore_Assessment.md) | Model performance analysis and GPU resource optimization assessment This comprehensive guide provide... | gpu, models, assessment | current |
| [GPU Usage Audit Report](docs/GPU_Audit_Report.md) | Comprehensive GPU usage audit with performance metrics and optimization recommendations, providing c... | gpu, audit, performance | completed |

---

## Production & Deployment

**Category ID:** production_deployment
**Priority:** high
**Documents:** 13

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Canonical Port Mapping](docs/canonical_port_mapping.md) | Complete service port allocation reference with status and configuration details This comprehensive ... | ports, services, configuration | current |
| [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) | Detailed implementation roadmap with phase breakdowns and success criteria, including detailed imple... | implementation, roadmap, phases | current |
| [JustNews V4 Memory Optimization - Mission Accomplished](markdown_docs/production_status/MEMORY_OPTIMIZATION_SUCCESS_SUMMARY.md) | Production deployment and operational documentation including service management, configuration, sca... | analyst, version-specific, memory | current |
| [JustNewsAgentic System Assessment Summary](markdown_docs/production_status/SYSTEM_OVERLAP_ANALYSIS.md) | **Assessment Date**: 7th August 2025 
**System Version**: V4 Hybrid Architecture  
**Lead Assessment... | analyst, version-specific, training | current |
| [Legal Compliance Framework Documentation](markdown_docs/production_status/LEGAL_COMPLIANCE_FRAMEWORK.md) | Comprehensive documentation covering legal compliance framework documentation with detailed technica... | optimization, compliance, deployment | current |
| [Package Management & Environment Optimization - PRODUCTION READY](markdown_docs/production_status/PACKAGE_MANAGEMENT_SUCCESS.md) | **Date**: September 2, 2025
**Status**: ✅ COMPLETE - All core packages installed, tested, and produc... | analyst, dashboard, version-specific | current |
| [Production Deployment Status](markdown_docs/production_status/PRODUCTION_DEPLOYMENT_STATUS.md) | Current operational status with RTX3090 GPU utilization and performance metrics Defines production d... | production, deployment, operational | current |
| [Project Status Report](docs/PROJECT_STATUS.md) | Current development status, milestones, and roadmap with version tracking, providing comprehensive a... | status, milestones, roadmap | current |
| [Prometheus Metrics Integration - Complete Implementation](markdown_docs/production_status/PROMETHEUS_METRICS_INTEGRATION_COMPLETE.md) | Documentation for Prometheus Metrics Integration - Complete Implementation... | scout, gpu, memory | current |
| [RAPIDS Integration Guide](markdown_docs/production_status/RAPIDS_USAGE_GUIDE.md) | Comprehensive documentation covering rapids integration guide with detailed technical information, i... | optimization, pytorch, analytics | current |
| [Synthesizer V3 Production Success Summary](markdown_docs/production_status/SYNTHESIZER_V3_PRODUCTION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ PRODUCTION READY  
**Version**: V4.16.0 This comprehensive ... | version-specific, training, memory | current |
| [🎉 JustNews V4 Memory Optimization - DEPLOYMENT SUCCESS](markdown_docs/production_status/DEPLOYMENT_SUCCESS_SUMMARY.md) | ## 🏆 Mission Accomplished - Memory Crisis Resolved! Defines production deployment procedures, monito... | version-specific, memory, models | current |
| [🎯 **USER INSIGHT VALIDATION: COMPLETE SUCCESS**](markdown_docs/production_status/USER_INSIGHT_VALIDATION_SUCCESS.md) | ## **✅ Key Achievement: Your INT8 Quantization Approach Works!** This comprehensive guide provides d... | memory, multi-agent, ai-agents | current |

---

## API & Integration

**Category ID:** api_integration
**Priority:** medium
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Phase 3 Archive Integration Plan](docs/phase3_archive_integration_plan.md) | Research-scale archiving infrastructure with provenance tracking and legal compliance This comprehen... | archive, research, provenance | planning |

---

## Training & Learning

**Category ID:** training_learning
**Priority:** medium
**Documents:** 0


---

## Monitoring & Analytics

**Category ID:** monitoring_analytics
**Priority:** medium
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Centralized Logging Migration](docs/LOGGING_MIGRATION.md) | Centralized logging system with structured JSON logging and performance tracking This comprehensive ... | logging, centralized, structured | current |

---

## Compliance & Security

**Category ID:** compliance_security
**Priority:** high
**Documents:** 0


---

## Development Reports

**Category ID:** development_reports
**Priority:** medium
**Documents:** 48

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Action Plan Implementation Status (Code/Tests Evidence Only)](markdown_docs/development_reports/action_plan_implementation_status.md) | This document maps the actions listed in the action plan to their current implementation status in t... | mcp, api, tensorrt | current |
| [Action Plan: JustNews V4 RTX-Accelerated Development](markdown_docs/development_reports/action_plan.md) | **Current Status**: Enhanced Scout Agent + TensorRT-LLM Integration Complete - Ready for Multi-Agent... | gpu, version-specific, cuda | current |
| [Added CPU fallback for meta tensor issues](markdown_docs/development_reports/FACT_CHECKER_FIXES_SUCCESS.md) | Documentation for Added CPU fallback for meta tensor issues providing essential information and tech... | optimization, training, performance | current |
| [Advanced Topic Modeling Enhancement Research Report](markdown_docs/development_reports/Advanced_Topic_Modeling_Enhancement_Research.md) | **Date:** September 15, 2025  
**Researcher:** GitHub Copilot Analysis  
**Foc#### **Option 4: Neura... | production, performance, synthesizer | current |
| [Agent Assessment — 2025-08-18](markdown_docs/development_reports/agent_assessment_2025-08-18.md) | This document summarizes an inspection of the `agents/` directory and how each agent maps to the Jus... | gpu, version-specific, synthesizer | current |
| [Analytics Dashboard Fixes Summary](markdown_docs/development_reports/ANALYTICS_DASHBOARD_FIXES_SUMMARY.md) | ## Overview
This document summarizes all the fixes and improvements made to the JustNewsAgent Analyt... | optimization, analytics, api | current |
| [BBC Crawler Duplicates - Complete Resolution ✅](markdown_docs/development_reports/bbc_crawler_duplicates_complete_resolution.md) | Documentation for BBC Crawler Duplicates - Complete Resolution ✅ providing essential information and... | mcp, performance, archive | current |
| [Centralized Logging Migration Guide](markdown_docs/development_reports/LOGGING_MIGRATION.md) | ## Overview
JustNewsAgent now has a centralized logging system that provides:
- Structured JSON logg... | multi-agent, production, monitoring | current |
| [Cross Reference Repair Report](markdown_docs/development_reports/CROSS_REFERENCE_REPAIR_REPORT.md) | Detailed report documenting cross reference repair report with analysis, findings, and recommendatio... | security, gpu, version-specific | current |
| [Development Reports Reorganization Plan](markdown_docs/development_reports/DEVELOPMENT_REPORTS_REORGANIZATION_PLAN.md) | Documentation for Development Reports Reorganization Plan... | optimization, deployment, training | current |
| [Entrypoints and Orchestration Flows — 2025-08-18](markdown_docs/development_reports/entrypoints_assessment_2025-08-18.md) | This document lists entry points into the JustNewsAgentic system that accept a URL or "news topic as... | mcp, api, performance | current |
| [Full GPU Implementation Action Plan](markdown_docs/development_reports/full_gpu_implementation_action_plan.md) | Goal: take JustNewsAgent from the current hybrid/partial TensorRT implementation to a robust, reprod... | pytorch, deployment, security | current |
| [GPU Crash Investigation - Final Report](markdown_docs/development_reports/GPU-Crash-Investigation-Final-Report.md) | **Investigation Period**: August 13, 2025  
**Status**: ✅ **RESOLVED - Production Validated**  
**Im... | optimization, deployment, training | current |
| [GPU Model Store Assessment](markdown_docs/development_reports/GPU_ModelStore_Assessment.md) | **Assessment Date:** September 7, 2025
**Last Updated:** September 7, 2025
**System:** JustNewsAgent... | gpu, version-specific, cuda | current |
| [GPU Orchestrator Migration Plan (V4)](markdown_docs/development_reports/gpu_orchestrator_plan.md) | Documentation for GPU Orchestrator Migration Plan (V4)... | gpu, memory, multi-agent | current |
| [GPU Usage Audit Report - JustNewsAgent](markdown_docs/development_reports/GPU_Audit_Report.md) | Detailed report documenting gpu usage audit report - justnewsagent with analysis, findings, and reco... | gpu, version-specific, cuda | current |
| [GitHub Copilot Instructions Update Summary - August 2, 2025](markdown_docs/development_reports/COPILOT_INSTRUCTIONS_UPDATE_SUMMARY.md) | ## 🎯 **Key Updates Made to `.github/copilot-instructions.md`**...... | deployment, tensorrt, archive | current |
| [JustNews Documentation Maintenance Action Plan - COMPLETED](markdown_docs/development_reports/MAINTENANCE_COMPLETION_REPORT.md) | **Date**: September 7, 2025  
**Status**: ✅ **100% COMPLETE** - All 4 phases successfully implemente... | analytics, multi-agent, dashboard | current |
| [JustNews Systemd Deployment - Action Plan & Status](markdown_docs/development_reports/action_systemd_prod.md) | Documentation for JustNews Systemd Deployment - Action Plan & Status... | monitoring, deployment, architecture | current |
| [JustNews Systemd Deployment - Shortfalls Analysis](markdown_docs/development_reports/shortfalls_analysis.md) | Documentation for JustNews Systemd Deployment - Shortfalls Analysis... | deployment, architecture | current |
| [JustNews V4 Quality Management System - Final Status](markdown_docs/development_reports/QUALITY_SYSTEM_STATUS.md) | ## 🎯 Quality Achievement
- **Target**: >90% Quality Score
- **Achieved**: ✅ 100.0/100 (Perfect Score... | version-specific, monitoring | current |
| [JustNews V4 Systemd Deployment Status Report](markdown_docs/development_reports/deployment_status.md) | Documentation for JustNews V4 Systemd Deployment Status Report... | deployment, version-specific, architecture | current |
| [JustNews V4 Workspace Organization Summary](markdown_docs/development_reports/WORKSPACE_ORGANIZATION_SUMMARY.md) | ### ✅ **COMPLETE WORKSPACE ORGANIZATION ACCOMPLISHED**...... | deployment, mcp, api | current |
| [JustNewsAgent Crawling and Ingestion System Analysis](markdown_docs/development_reports/Crawler_and_Ingestion_Dev_Plan.md) | Documentation for JustNewsAgent Crawling and Ingestion System Analysis... | scout, gpu, memory | current |
| [JustNewsAgent Documentation Catalogue](markdown_docs/development_reports/DOCUMENTATION_CATALOGUE.md) | **Version:** 2.0
**Last Updated:** 2025-09-07
**Total Documents:** 140
**Categories:** 14...... | security, knowledge-graph, gpu | current |
| [JustNewsAgent Status Report](markdown_docs/development_reports/PROJECT_STATUS.md) | Detailed report documenting justnewsagent status report with analysis, findings, and recommendations... | security, gpu, version-specific | current |
| [JustNewsAgent V4 - Current Development Status Summary](markdown_docs/development_reports/CURRENT_DEVELOPMENT_STATUS.md) | **Last Updated**: August 31, 2025
**Status**: ✅ RTX3090 GPU Production Readiness Achieved - FULLY OP... | security, gpu, version-specific | current |
| [JustNewsAgent V4 - Strategic Assessment and Implementation Plan (Version 2)](markdown_docs/development_reports/strategic_assessment_and_implementation_plan.md) | **Date:** September 13, 2025
**Author:** GitHub Copilot
**Status:** Version 2 - Revised based on cla... | production, performance, scout | current |
| [JustNewsAgentic — Implementation Plan for Evidence, KG, Fact‑Checking & Conservative Generation](markdown_docs/development_reports/IMPLEMENTATION_PLAN.md) | Date: 2025-09-07  
Branch: dev/agent_review
Status: ✅ **PHASE 2 COMPLETE - PRODUCTION READY**...... | optimization, compliance, security | current |
| [MCP Bus Architecture Cleanup - August 2, 2025](markdown_docs/development_reports/mcp_bus_architecture_cleanup.md) | Documentation for MCP Bus Architecture Cleanup - August 2, 2025 providing essential information and ... | mcp, api, archive | current |
| [Newsreader Training Integration Success](markdown_docs/development_reports/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | Documentation for Newsreader Training Integration Success providing essential information and techni... | training, performance, synthesizer | current |
| [Practical NewsReader Solution - File Organization Complete ✅](markdown_docs/development_reports/practical_newsreader_solution_organization.md) | Documentation for Practical NewsReader Solution - File Organization Complete ✅ providing essential i... | optimization, deployment, mcp | current |
| [Production BBC Crawler - Duplicate Resolution Complete ✅](markdown_docs/development_reports/production_bbc_crawler_duplicate_resolution.md) | Documentation for Production BBC Crawler - Duplicate Resolution Complete ✅ providing essential infor... | mcp, archive, multi-agent | current |
| [Robust loading with meta tensor handling](markdown_docs/development_reports/META_TENSOR_RESOLUTION_SUCCESS.md) | ### 🎯 **Issue Analysis: System-Wide Meta Tensor Problem**...... | deployment, training, performance | current |
| [Scout Agent Production Crawler Integration - COMPLETED ✅](markdown_docs/development_reports/scout_production_crawler_integration_complete.md) | Complete documentation for Scout Agent Production Crawler Integration - COMPLETED ✅ including implem... | mcp, api, performance | current |
| [Synthesizer V2 Dependencies & Training Integration - SUCCESS REPORT](markdown_docs/development_reports/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ **COMPLETE SUCCESS**  
**Task**: Fix Synthesizer dependenci... | optimization, training, tensorrt | current |
| [System Assessment and Improvement Plan — 2025-08-09](markdown_docs/development_reports/System_Assessment_2025-08-09.md) | This document captures a focused assessment of the JustNewsAgentic V4 system and proposes prioritize... | security, gpu, version-specific | current |
| [System Startup Scripts - Restored and Enhanced ✅](markdown_docs/development_reports/system_startup_scripts_restored.md) | Documentation for System Startup Scripts - Restored and Enhanced ✅ providing essential information a... | gpu, version-specific, synthesizer | current |
| [Systemd Orchestrator Incident Report — Sept 13, 2025](markdown_docs/development_reports/systemd_operational_incident_report_2025-09-13.md) | Documentation for Systemd Orchestrator Incident Report — Sept 13, 2025... | scout, gpu, ai-agents | current |
| [Testing & Dependency Upgrade: Paused (2025-08-24)](markdown_docs/development_reports/TESTING_PAUSED.md) | Summary
-------
This document records the dependency-testing work performed and the reason we paused... | production, api | current |
| [The Definitive User Guide: JustNews Agentic System (V4)](markdown_docs/development_reports/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide: JustNews Agentic System (V4)... | gpu, version-specific, cuda | current |
| [Tomotopy: Online LDA with Dynamic Topic Capabilities](markdown_docs/development_reports/Tomotopy_Online_LDA_Detailed_Analysis.md) | **Date:** September 15, 2025  
**Topic:** Advanced Topic Modeling for JustNewsAgent  
**Focus:** Tom... | production, performance, synthesizer | current |
| [Using The GPU Correctly - Complete Configuration Guide](markdown_docs/development_reports/Using-The-GPU-Correctly.md) | **Date**: August 13, 2025  
**Status**: Production-Validated Configuration  
**GPU**: NVIDIA GeForce... | optimization, performance, gpu | current |
| [🎉 JustNews V4 Quality Management System - Implementation Complete](markdown_docs/development_reports/IMPLEMENTATION_COMPLETE.md) | Documentation for 🎉 JustNews V4 Quality Management System - Implementation Complete providing essent... | training, version-specific, production | current |
| [📊 **DOCUMENTATION COVERAGE ANALYSIS REPORT**](markdown_docs/development_reports/DOCUMENTATION_COVERAGE_ANALYSIS.md) | **Analysis Date:** September 7, 2025  
**Codebase Size:** 221 Python files  
**Current Documentation... | security, gpu, synthesizer | current |
| [📊 Documentation Quality Report](markdown_docs/development_reports/weekly_report_20250907.md) | Comprehensive documentation covering 📊 documentation quality report with detailed technical informat... | optimization, compliance, deployment | current |
| [📋 Documentation Change Report](markdown_docs/development_reports/version_report_20250907.md) | Comprehensive documentation covering 📋 documentation change report with detailed technical informati... | documentation, reports | current |
| [🚀 JustNews V4 Documentation Quality Management - Team Training Guide](markdown_docs/development_reports/TEAM_TRAINING_GUIDE.md) | Comprehensive documentation covering 🚀 justnews v4 documentation quality management - team training ... | optimization, compliance, deployment | current |

---

## Scripts Tools

**Category ID:** scripts_tools
**Priority:** medium
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Build engine scaffold](tools/build_engine/README.md) | This folder contains a host-native scaffold for building TensorRT engines This comprehensive guide p... | pytorch, cuda, gpu | current |
| [Deprecate Dialogpt Readme](scripts/DEPRECATE_DIALOGPT_README.md) | Utility scripts and tools documentation covering automation, deployment helpers, model management, a... | agents, multi-agent, ai-agents | current |
| [If you omit --target, the script will use the DATA_DRIVE_TARGET env var or fall back to the](scripts/README_MIRROR.md) | Documentation for If you omit --target, the script will use the DATA_DRIVE_TARGET env var or fall ba... | synthesizer, multi-agent, agents | current |
| [Readme Bootstrap Models](scripts/README_BOOTSTRAP_MODELS.md) | Utility scripts and tools documentation covering automation, deployment helpers, model management, a... | models | current |

---

## Deployment System

**Category ID:** deployment_system
**Priority:** medium
**Documents:** 11

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Action Systemd Prod](markdown_docs/development_reports/action_systemd_prod.md) | Documentation for Action Systemd Prod... |  | current |
| [Comprehensive systemd guide](markdown_docs/agent_documentation/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md) | Documentation for Comprehensive systemd guide... | scout, gpu, cuda | current |
| [Deployment Status](markdown_docs/development_reports/deployment_status.md) | Documentation for Deployment Status... | deployment | current |
| [Examples for systemd native deployment](markdown_docs/agent_documentation/systemd/examples/README.md) | Files in this directory are examples and helpers to install the JustNews systemd units. | mcp, ai-agents, scout | current |
| [JustNews native deployment (systemd)](markdown_docs/agent_documentation/systemd/DEPLOYMENT.md) | Documentation for JustNews native deployment (systemd)... | mcp, models, logging | current |
| [Phase5 Advanced Monitoring Plan](markdown_docs/agent_documentation/systemd/phase5_advanced_monitoring_plan.md) | Documentation for Phase5 Advanced Monitoring Plan... | monitoring | current |
| [PostgreSQL integration](markdown_docs/agent_documentation/systemd/postgresql_integration.md) | Documentation for PostgreSQL integration... | memory | current |
| [Quick Reference – systemd](markdown_docs/agent_documentation/systemd/QUICK_REFERENCE.md) | Documentation for Quick Reference – systemd... | reasoning, scout, archive | current |
| [Readme](markdown_docs/agent_documentation/monitoring/README.md) | Documentation for Readme... |  | current |
| [Shortfalls Analysis](markdown_docs/development_reports/shortfalls_analysis.md) | Documentation for Shortfalls Analysis... |  | current |
| [Systemd scaffold for JustNews](markdown_docs/agent_documentation/systemd/README.md) | This folder contains a native deployment scaffold:, covering system design, component interactions, ... | mcp, memory, reasoning | current |

---

## General Documentation

**Category ID:** general_documentation
**Priority:** medium
**Documents:** 9

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Upgrade Plan - JustNewsAgent V4](markdown_docs/agent_upgrade_plan.md) | Comprehensive documentation covering agent upgrade plan - justnewsagent v4 with detailed technical i... | logging, ai-agents, scout | current |
| [Canonical list of all files that can come into use](markdown_docs/IN_USE_FILES_FULL_LIST.md) | Documentation for Canonical list of all files that can come into use This comprehensive guide provid... | ai-agents, scout, security | current |
| [GPU Management Implementation - Complete Documentation](markdown_docs/GPU_IMPLEMENTATION_COMPLETE.md) | **Date:** August 31, 2025
**Status:** ✅ **FULLY IMPLEMENTED & PRODUCTION READY**
**Version:** v2.0.0... | logging, ai-agents, scout | current |
| [JustNews Agentic - Development Context](markdown_docs/DEVELOPMENT_CONTEXT.md) | **Last Updated**: September 2, 2025  
**Branch**: `dev/gpu_implementation`  
**Status**: Production-... | ai-agents, optimization, production | current |
| [JustNews V4 Documentation Index](markdown_docs/README.md) | This directory contains organized documentation for the JustNews V4 project. Files are categorized f... | version-specific, memory, models | current |
| [JustNews V4 — In‑Use Files Inventory](markdown_docs/IN_USE_FILES.md) | Comprehensive documentation covering justnews v4 — in‑use files inventory with detailed technical in... | analyst, dashboard, version-specific | current |
| [JustNewsAgent V4 - Technical Architecture](markdown_docs/TECHNICAL_ARCHITECTURE.md) | This document provides comprehensive technical details about the JustNewsAgent V4 system architectur... | logging, ai-agents, scout | current |
| [Workspace Cleanup Summary - August 8, 2025](markdown_docs/WORKSPACE_CLEANUP_SUMMARY_20250808.md) | Comprehensive documentation covering workspace cleanup summary - august 8, 2025 with detailed techni... | dashboard, multi-agent, archive | current |
| [🎉 HOUSEKEEPING COMPLETE - Ready for Manual Push](markdown_docs/HOUSEKEEPING_COMPLETE_SUMMARY.md) | ## ✅ **WORKSPACE CLEANUP SUCCESSFULLY COMPLETED** This comprehensive guide provides detailed informa... | training, memory, multi-agent | current |

---

## Performance Optimization

**Category ID:** performance_optimization
**Priority:** medium
**Documents:** 3

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [JustNews Agent - Knowledge Graph Documentation](markdown_docs/optimization_reports/PHASE3_KNOWLEDGE_GRAPH.md) | ## Phase 3 Sprint 3-4: Advanced Knowledge Graph Features...... | knowledge-graph, api, archive | current |
| [NewsReader V2 Optimization Complete - Component Redundancy Analysis](markdown_docs/optimization_reports/NEWSREADER_V2_OPTIMIZATION_COMPLETE.md) | Success report documenting achievements, implementation details, and validation results for newsread... | memory, mcp, models | current |
| [OCR Redundancy Analysis - NewsReader V2 Engine](markdown_docs/optimization_reports/OCR_REDUNDANCY_ANALYSIS.md) | ## Executive Summary
**Recommendation**: 🟡 **OCR is LIKELY REDUNDANT** but low-risk to maintain This... | training, memory, models | current |

---

## Architecture & Design Reports

**Category ID:** development_reports_architecture
**Priority:** high
**Documents:** 0


---

## Implementation Reports

**Category ID:** development_reports_implementation
**Priority:** high
**Documents:** 0


---

## Performance & Optimization Reports

**Category ID:** development_reports_performance
**Priority:** high
**Documents:** 0


---

## Testing & Quality Assurance Reports

**Category ID:** development_reports_testing
**Priority:** medium
**Documents:** 0


---

## Deployment & Operations Reports

**Category ID:** development_reports_deployment
**Priority:** medium
**Documents:** 0


---

## Training & Learning Reports

**Category ID:** development_reports_training
**Priority:** high
**Documents:** 0


---

## Integration & Workflow Reports

**Category ID:** development_reports_integration
**Priority:** medium
**Documents:** 0


---

## Maintenance & Housekeeping Reports

**Category ID:** development_reports_maintenance
**Priority:** low
**Documents:** 0


---

## Core Agent Documentation

**Category ID:** agent_documentation_core_agents
**Priority:** high
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Scout Agent V2 - AI-First Architecture](markdown_docs/agent_documentation/SCOUT_AGENT_V2_DOCUMENTATION.md) | Complete documentation for the 5-model AI-first Scout Agent with RTX3090 GPU acceleration Details AI... | scout, ai-first, 5-models | production_ready |

---

## Specialized Agent Documentation

**Category ID:** agent_documentation_specialized_agents
**Priority:** high
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Balancer Agent V1](markdown_docs/agent_documentation/BALANCER_AGENT_V1.md) | News neutralization and balancing agent with MCP integration and GPU acceleration Details AI agent c... | balancer, neutralization, mcp-integration | production_ready |
| [Reasoning Agent - Nucleoid Integration](markdown_docs/agent_documentation/REASONING_AGENT_COMPLETE_IMPLEMENTATION.md) | Complete Nucleoid-based symbolic reasoning agent with GPU memory optimization Details AI agent capab... | reasoning, nucleoid, symbolic-logic | production_ready |

---

## Deprecated Agent Documentation

**Category ID:** agent_documentation_deprecated_agents
**Priority:** low
**Documents:** 0


---

## Agent Management & Tools

**Category ID:** agent_documentation_agent_management
**Priority:** medium
**Documents:** 0


---

## Model Integration Documentation

**Category ID:** agent_documentation_model_integration
**Priority:** high
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Model Map](markdown_docs/agent_documentation/AGENT_MODEL_MAP.md) | Complete mapping of agents to models, resources, and performance characteristics Details AI agent ca... | agents, models, mapping | current |

---

## Crawling & Data Collection

**Category ID:** agent_documentation_crawling_systems
**Priority:** medium
**Documents:** 0


---

## Search Index Summary

**Available Tags:** 82
**Indexed Keywords:** 100

## Maintenance Information

**Last Catalogue Update:** 2025-09-15
**Next Review Date:** 2025-10-07
