# JustNewsAgent Documentation Catalogue

**Version:** 2.0
**Last Updated:** 2025-09-07
**Total Documents:** 265
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
**Documents:** 53

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [AGENT_MODEL_MAP — Definitive Agent → Model mapping](markdown_docs/agent_documentation/AGENT_MODEL_MAP.md) | This document lists the authoritative mapping of agents to their external model dependencies as defi... | models, reasoning, multi-agent | current |
| [API Documentation](markdown_docs/agent_documentation/api_documentation.md) | ## Overview...... | gpu, version-specific, synthesizer | current |
| [Agent Communication Protocols Documentation](markdown_docs/agent_documentation/agent_communication_protocols.md) | ## Overview...... | optimization, deployment, mcp | current |
| [Agent Documentation index](markdown_docs/agent_documentation/README.md) | This folder contains agent-specific documentation used by operators and
developers. Key documents:..... | ai-agents, multi-agent, agents | current |
| [Balancer Agent V1 - Integration & Debugging Guide](markdown_docs/agent_documentation/BALANCER_AGENT_INTEGRATION_GUIDE.md) | ## Overview
The Balancer Agent is a production-grade component of the JustNews V4 system, designed t... | compliance, mcp, api | current |
| [Balancer Agent V1 Documentation](markdown_docs/agent_documentation/BALANCER_AGENT_V1.md) | ## Overview
The Balancer Agent is a production-ready component of the JustNews V4 system, designed t... | optimization, compliance, training | current |
| [Configuration Management Documentation](markdown_docs/agent_documentation/configuration_management_guide.md) | ## Overview...... | security, gpu, version-specific | current |
| [Crawl4AI vs Playwright — feature-by-feature comparison](markdown_docs/agent_documentation/Crawl4AI_vs_Playwright_Comparison.md) | Generated: 2025-08-27...... | compliance, deployment, api | current |
| [Crawler Consolidation Plan — JustNewsAgent](markdown_docs/agent_documentation/Crawler_Consolidation_Plan.md) | Date: 2025-08-27
Author: Consolidation plan generated from interactive session...... | api, archive, models | current |
| [Data Pipeline Documentation](markdown_docs/agent_documentation/data_pipeline_documentation.md) | ## Overview...... | gpu, version-specific, cuda | current |
| [Database Schema & Operations Documentation](markdown_docs/agent_documentation/database_schema_operations.md) | ## Overview...... | optimization, security, training | current |
| [Deployment Procedures Documentation](markdown_docs/agent_documentation/deployment_procedures_guide.md) | ## Overview...... | security, gpu, version-specific | current |
| [Embedding Helper](markdown_docs/agent_documentation/EMBEDDING_HELPER.md) | Documentation for Embedding Helper... | models, gpu, multi-agent | current |
| [GPU Acceleration Documentation](markdown_docs/agent_documentation/gpu_acceleration_guide.md) | ## Overview...... | gpu, version-specific, cuda | current |
| [Hugging Face model caching and pre-download for Memory Agent](markdown_docs/agent_documentation/HF_MODEL_CACHING.md) | This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cach... | deployment, api, models | current |
| [Implementation Summary](agents/newsreader/IMPLEMENTATION_SUMMARY.md) | Documentation for Implementation Summary... |  | current |
| [JustNews Agent - API Documentation](markdown_docs/agent_documentation/PHASE3_API_DOCUMENTATION.md) | ## Phase 3 Sprint 3-4: Advanced Knowledge Graph APIs...... | security, gpu, synthesizer | current |
| [JustNewsAgent Canonical Port Mapping](markdown_docs/agent_documentation/canonical_port_mapping.md) | ## 📋 Complete Port Usage Analysis...... | analytics, mcp, api | current |
| [LLaVA NewsReader Agent Implementation Summary](agents/newsreader/documentation/IMPLEMENTATION_SUMMARY.md) | ## ✅ Completed Implementation...... | optimization, pytorch, mcp | current |
| [Later: resume](markdown_docs/agent_documentation/Crawl4AI_API_SUMMARY.md) | This short reference summarises the Crawl4AI programmatic APIs, dispatcher classes, REST endpoints, ... | deployment, logging, api | current |
| [Lifespan Migration](agents/newsreader/LIFESPAN_MIGRATION.md) | Documentation for Lifespan Migration... |  | current |
| [Lifespan Migration](agents/newsreader/documentation/LIFESPAN_MIGRATION.md) | ### Changes Made...... | mcp, api, gpu | current |
| [MCP Bus Architecture Documentation](markdown_docs/agent_documentation/mcp_bus_architecture.md) | ## Overview...... | deployment, mcp, security | current |
| [Model Usage](markdown_docs/agent_documentation/MODEL_USAGE.md) | Documentation for Model Usage... | deployment, mcp, api | current |
| [Model store guidelines](markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md) | This document explains the canonical model-store layout and safe update patterns for
per-agent model... | deployment, training, security | current |
| [Monitoring and Observability Documentation](markdown_docs/agent_documentation/monitoring_observability_guide.md) | ## Overview...... | security, gpu, version-specific | current |
| [Native TensorRT Analyst Agent - Production Ready](agents/analyst/NATIVE_TENSORRT_README.md) | ## 🏆 **Production Status: VALIDATED & DEPLOYED**...... | optimization, deployment, mcp | current |
| [Native TensorRT Analyst Agent - Quick Start Guide](agents/analyst/NATIVE_AGENT_README.md) | ## 🏆 **Production Status: OPERATIONAL**...... | deployment, mcp, api | current |
| [New Blueprint Agents](markdown_docs/agent_documentation/New_Blueprint_Agents.md) | Documentation for New Blueprint Agents... | security, knowledge-graph, gpu | current |
| [News Outlets Loader & Backfill Runbook](markdown_docs/agent_documentation/NEWS_OUTLETS_RUNBOOK.md) | This runbook explains how to safely run the canonical sources loader (`scripts/news_outlets.py`) and... | performance, multi-agent, ai-agents | current |
| [NewsReader Agent - Production-Validated Configuration](agents/newsreader/README.md) | ## 🚨 **CRITICAL UPDATE: GPU Crash Resolution - August 13, 2025**...... | optimization, deployment, api | current |
| [NewsReader V2 Vision-Language Model Fallback Logic](markdown_docs/agent_documentation/NEWSREADER_V2_MODEL_FALLBACK.md) | ## Overview
The NewsReader V2 agent now implements robust fallback logic for vision-language model i... | optimization, mcp, gpu | current |
| [Next-Generation AI-First Scout Agent V2 Documentation](markdown_docs/agent_documentation/SCOUT_AGENT_V2_DOCUMENTATION.md) | *Last Updated: August 7, 2025*...... | gpu, cuda, agents | current |
| [Performance Optimization Documentation](markdown_docs/agent_documentation/performance_optimization_documentation.md) | ## Overview...... | optimization, pytorch, training | current |
| [Potential Development Paths](markdown_docs/agent_documentation/potential_development_paths.md) | This document captures a compact summary of recent analysis and recommendations about the project's ... | analytics, mcp, api | current |
| [Potential News Sources](markdown_docs/agent_documentation/potential_news_sources.md) | Documentation for Potential News Sources... | security | current |
| [Product Modalities Comparison](markdown_docs/agent_documentation/product_modalities_comparison.md) | This document compares three high-level product modalities the JustNews system can pursue, aligns ea... | gpu, synthesizer, fact-checker | current |
| [Reasoning Agent](agents/reasoning/README.md) | This package contains the reasoning agent (Nucleoid) for JustNews....... | mcp, api, reasoning | current |
| [Reasoning Agent Complete Implementation Documentation](markdown_docs/agent_documentation/REASONING_AGENT_COMPLETE_IMPLEMENTATION.md) | ## Overview...... | optimization, deployment, mcp | current |
| [Scout Agent - Enhanced Deep Crawl Documentation](markdown_docs/agent_documentation/SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md) | **JustNews V4 Scout Agent with Native Crawl4AI Integration**...... | optimization, analytics, deployment | current |
| [Scout Agent V2 - Next-Generation AI-First Content Analysis System](agents/scout/README.md) | ## 🎯 **Agent Overview**...... | optimization, deployment, training | current |
| [Scout → Memory Pipeline Success Summary](markdown_docs/agent_documentation/SCOUT_MEMORY_PIPELINE_SUCCESS.md) | **Date**: January 29, 2025  
**Milestone**: Core JustNews V4 pipeline operational with native deploy... | deployment, training, mcp | current |
| [Security Implementation Documentation](markdown_docs/agent_documentation/security_implementation_documentation.md) | ## Overview...... | optimization, compliance, security | current |
| [Security Implementation Guide](markdown_docs/agent_documentation/security_implementation_guide.md) | ## Overview...... | compliance, deployment, mcp | current |
| [Sources Schema and Workflow](markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md) | This document specifies the `sources` schema, provenance mapping (`article_source_map`), ingestion w... | analytics, deployment, security | current |
| [System Decisions](markdown_docs/agent_documentation/system_decisions.md) | Documentation for System Decisions... | models | current |
| [TensorRT Quickstart (safe, no-GPU stub)](agents/analyst/TENSORRT_QUICKSTART.md) | This file explains how to run a safe, developer-friendly stub for the TensorRT engine build process.... | tensorrt, gpu, multi-agent | current |
| [The Definitive User Guide](markdown_docs/agent_documentation/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide... |  | current |
| [Training System API Documentation](markdown_docs/agent_documentation/training_system_api.md) | ## Overview...... | training, api, performance | current |
| [Use the main RAPIDS environment](markdown_docs/agent_documentation/gpu_runner_README.md) | Documentation for Use the main RAPIDS environment... | pytorch, analytics, api | current |
| [Why INT8 Quantization Should Be Implemented Immediately](agents/newsreader/documentation/INT8_QUANTIZATION_RATIONALE.md) | ## You're Absolutely Right! Here's Why:...... | optimization, deployment, tensorrt | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/README.md) | Documentation for all-mpnet-base-v2... | training, api, models | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md) | Documentation for all-mpnet-base-v2... | training, api, models | current |

---

## GPU Setup & Configuration

**Category ID:** gpu_configuration
**Priority:** high
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [GPU Environment Setup Guide](docs/gpu_runner_README.md) | Complete guide for RTX3090 GPU environment with PyTorch 2.6.0+cu124, CUDA 12.4, and RAPIDS 25.04 Thi... | gpu, setup, rtx3090 | production_ready |
| [GPU Model Store Assessment](docs/GPU_ModelStore_Assessment.md) | Model performance analysis and GPU resource optimization assessment This comprehensive guide provide... | gpu, models, assessment | current |
| [GPU Usage Audit Report](docs/GPU_Audit_Report.md) | Comprehensive GPU usage audit with performance metrics and optimization recommendations, providing c... | gpu, audit, performance | completed |
| [RAPIDS Integration Guide](docs/RAPIDS_USAGE_GUIDE.md) | GPU-accelerated data science and machine learning with RAPIDS 25.04 This comprehensive guide provide... | rapids, gpu-acceleration, data-science | current |

---

## Production & Deployment

**Category ID:** production_deployment
**Priority:** high
**Documents:** 17

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Canonical Port Mapping](docs/canonical_port_mapping.md) | Complete service port allocation reference with status and configuration details This comprehensive ... | ports, services, configuration | current |
| [Fact Checker Fixes Success](markdown_docs/production_status/FACT_CHECKER_FIXES_SUCCESS.md) | Production deployment and operational documentation including service management, configuration, sca... | operational, deployment, production | current |
| [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) | Detailed implementation roadmap with phase breakdowns and success criteria, including detailed imple... | implementation, roadmap, phases | current |
| [JustNews V4 Memory Optimization - Mission Accomplished](markdown_docs/production_status/MEMORY_OPTIMIZATION_SUCCESS_SUMMARY.md) | Production deployment and operational documentation including service management, configuration, sca... | analyst, version-specific, memory | current |
| [JustNewsAgentic System Assessment Summary](markdown_docs/production_status/SYSTEM_OVERLAP_ANALYSIS.md) | **Assessment Date**: 7th August 2025 
**System Version**: V4 Hybrid Architecture  
**Lead Assessment... | analyst, version-specific, training | current |
| [Legal Compliance Framework Documentation](markdown_docs/production_status/LEGAL_COMPLIANCE_FRAMEWORK.md) | ## Overview...... | optimization, compliance, deployment | current |
| [Meta Tensor Resolution Success](markdown_docs/production_status/META_TENSOR_RESOLUTION_SUCCESS.md) | Production deployment and operational documentation including service management, configuration, sca... | operational, deployment, production | current |
| [Newsreader Training Integration Success](markdown_docs/production_status/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | Documentation for Newsreader Training Integration Success Implements continuous learning algorithms ... | training | current |
| [Package Management & Environment Optimization - PRODUCTION READY](markdown_docs/production_status/PACKAGE_MANAGEMENT_SUCCESS.md) | **Date**: September 2, 2025
**Status**: ✅ COMPLETE - All core packages installed, tested, and produc... | analyst, dashboard, version-specific | current |
| [Production Deployment Status](markdown_docs/production_status/PRODUCTION_DEPLOYMENT_STATUS.md) | Current operational status with RTX3090 GPU utilization and performance metrics Defines production d... | production, deployment, operational | current |
| [Project Status Report](docs/PROJECT_STATUS.md) | Current development status, milestones, and roadmap with version tracking, providing comprehensive a... | status, milestones, roadmap | current |
| [RAPIDS Integration Guide](markdown_docs/production_status/RAPIDS_USAGE_GUIDE.md) | ## Overview...... | optimization, pytorch, analytics | current |
| [Synthesizer Training Integration Success](markdown_docs/production_status/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | Documentation for Synthesizer Training Integration Success Implements continuous learning algorithms... | synthesizer, training | current |
| [Synthesizer V3 Production Success Summary](markdown_docs/production_status/SYNTHESIZER_V3_PRODUCTION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ PRODUCTION READY  
**Version**: V4.16.0 This comprehensive ... | version-specific, training, memory | current |
| [Workspace Organization Summary](markdown_docs/production_status/WORKSPACE_ORGANIZATION_SUMMARY.md) | Production deployment and operational documentation including service management, configuration, sca... | operational, deployment, production | current |
| [🎉 JustNews V4 Memory Optimization - DEPLOYMENT SUCCESS](markdown_docs/production_status/DEPLOYMENT_SUCCESS_SUMMARY.md) | ## 🏆 Mission Accomplished - Memory Crisis Resolved! Defines production deployment procedures, monito... | version-specific, memory, models | current |
| [🎯 **USER INSIGHT VALIDATION: COMPLETE SUCCESS**](markdown_docs/production_status/USER_INSIGHT_VALIDATION_SUCCESS.md) | ## **✅ Key Achievement: Your INT8 Quantization Approach Works!** This comprehensive guide provides d... | memory, multi-agent, ai-agents | current |

---

## API & Integration

**Category ID:** api_integration
**Priority:** medium
**Documents:** 3

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Phase 3 API Documentation](docs/PHASE3_API_DOCUMENTATION.md) | RESTful and GraphQL API specifications for archive access and knowledge graph queries This comprehen... | api, rest, graphql | current |
| [Phase 3 Archive Integration Plan](docs/phase3_archive_integration_plan.md) | Research-scale archiving infrastructure with provenance tracking and legal compliance This comprehen... | archive, research, provenance | planning |
| [Phase 3 Knowledge Graph](docs/PHASE3_KNOWLEDGE_GRAPH.md) | Entity extraction, disambiguation, clustering, and relationship analysis documentation This comprehe... | knowledge-graph, entities, relationships | current |

---

## Training & Learning

**Category ID:** training_learning
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Online Learning Architecture](markdown_docs/development_reports/ONLINE_LEARNING_ARCHITECTURE.md) | Real-time model improvement system with active learning and feedback loops This comprehensive guide ... | online-learning, active-learning, feedback-loops | current |
| [Training System Documentation](markdown_docs/development_reports/TRAINING_SYSTEM_DOCUMENTATION.md) | Complete training system architecture with GPU-accelerated continuous learning Covers complete syste... | training, continuous-learning, gpu-acceleration | operational |

---

## Monitoring & Analytics

**Category ID:** monitoring_analytics
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Analytics Dashboard Fixes](docs/ANALYTICS_DASHBOARD_FIXES_SUMMARY.md) | Dashboard fixes, enhancements, and user experience improvements This comprehensive guide provides de... | analytics, dashboard, fixes | completed |
| [Centralized Logging Migration](docs/LOGGING_MIGRATION.md) | Centralized logging system with structured JSON logging and performance tracking This comprehensive ... | logging, centralized, structured | current |

---

## Compliance & Security

**Category ID:** compliance_security
**Priority:** high
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Legal Compliance Framework](docs/LEGAL_COMPLIANCE_FRAMEWORK.md) | GDPR and CCPA compliance framework with data minimization and consent management This comprehensive ... | gdpr, ccpa, compliance | current |

---

## Development Reports

**Category ID:** development_reports
**Priority:** medium
**Documents:** 69

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Action Plan Implementation Status (Code/Tests Evidence Only)](markdown_docs/development_reports/action_plan_implementation_status.md) | This document maps the actions listed in the action plan to their current implementation status in t... | mcp, api, tensorrt | current |
| [Action Plan: JustNews V4 RTX-Accelerated Development](markdown_docs/development_reports/action_plan.md) | **Current Status**: Enhanced Scout Agent + TensorRT-LLM Integration Complete - Ready for Multi-Agent... | gpu, version-specific, cuda | current |
| [Added CPU fallback for meta tensor issues](markdown_docs/development_reports/FACT_CHECKER_FIXES_SUCCESS.md) | ### 🎯 **Issues Fixed Successfully**...... | optimization, training, performance | current |
| [Agent Assessment — 2025-08-18](markdown_docs/development_reports/agent_assessment_2025-08-18.md) | This document summarizes an inspection of the `agents/` directory and how each agent maps to the Jus... | gpu, version-specific, synthesizer | current |
| [Analysis Nucleoid Potential](markdown_docs/development_reports/analysis_nucleoid_potential.md) | Documentation for Analysis Nucleoid Potential... |  | current |
| [Analytics Dashboard Fixes Summary](markdown_docs/development_reports/ANALYTICS_DASHBOARD_FIXES_SUMMARY.md) | ## Overview
This document summarizes all the fixes and improvements made to the JustNewsAgent Analyt... | optimization, analytics, api | current |
| [Architectural Changes Summary](markdown_docs/development_reports/ARCHITECTURAL_CHANGES_SUMMARY.md) | Documentation for Architectural Changes Summary... |  | current |
| [Architectural Review Findings](markdown_docs/development_reports/architectural_review_findings.md) | Documentation for Architectural Review Findings... |  | current |
| [Architectural Review Summary](markdown_docs/development_reports/ARCHITECTURAL_REVIEW_SUMMARY.md) | Documentation for Architectural Review Summary... |  | current |
| [BBC Crawler Duplicates - Complete Resolution ✅](markdown_docs/development_reports/bbc_crawler_duplicates_complete_resolution.md) | ## 🎯 **Duplicate Resolution Summary**...... | mcp, performance, archive | current |
| [Centralized Logging Migration Guide](markdown_docs/development_reports/LOGGING_MIGRATION.md) | ## Overview
JustNewsAgent now has a centralized logging system that provides:
- Structured JSON logg... | multi-agent, production, monitoring | current |
| [Complete V2 Upgrade Assessment](markdown_docs/development_reports/COMPLETE_V2_UPGRADE_ASSESSMENT.md) | Documentation for Complete V2 Upgrade Assessment... |  | current |
| [Corrected Scout Analysis](markdown_docs/development_reports/CORRECTED_SCOUT_ANALYSIS.md) | Documentation for Corrected Scout Analysis... | scout | current |
| [Cross Reference Repair Report](markdown_docs/development_reports/CROSS_REFERENCE_REPAIR_REPORT.md) | Documentation for Cross Reference Repair Report... | security, gpu, version-specific | current |
| [Development Reports Reorganization Plan](markdown_docs/development_reports/DEVELOPMENT_REPORTS_REORGANIZATION_PLAN.md) | Documentation for Development Reports Reorganization Plan... | optimization, deployment, training | current |
| [Docker Deprecation Notice](markdown_docs/development_reports/DOCKER_DEPRECATION_NOTICE.md) | Documentation for Docker Deprecation Notice... |  | current |
| [Enhanced Reasoning Architecture](markdown_docs/development_reports/enhanced_reasoning_architecture.md) | Documentation for Enhanced Reasoning Architecture... | reasoning, architecture | current |
| [Entrypoints and Orchestration Flows — 2025-08-18](markdown_docs/development_reports/entrypoints_assessment_2025-08-18.md) | This document lists entry points into the JustNewsAgentic system that accept a URL or "news topic as... | mcp, api, performance | current |
| [Full GPU Implementation Action Plan](markdown_docs/development_reports/full_gpu_implementation_action_plan.md) | Goal: take JustNewsAgent from the current hybrid/partial TensorRT implementation to a robust, reprod... | pytorch, deployment, security | current |
| [GPU Crash Investigation - Final Report](markdown_docs/development_reports/GPU-Crash-Investigation-Final-Report.md) | **Investigation Period**: August 13, 2025  
**Status**: ✅ **RESOLVED - Production Validated**  
**Im... | optimization, deployment, training | current |
| [GPU Model Store Assessment](markdown_docs/development_reports/GPU_ModelStore_Assessment.md) | **Assessment Date:** September 7, 2025
**Last Updated:** September 7, 2025
**System:** JustNewsAgent... | gpu, version-specific, cuda | current |
| [GPU Usage Audit Report - JustNewsAgent](markdown_docs/development_reports/GPU_Audit_Report.md) | ## Executive Summary...... | gpu, version-specific, cuda | current |
| [GitHub Copilot Instructions Update Summary - August 2, 2025](markdown_docs/development_reports/COPILOT_INSTRUCTIONS_UPDATE_SUMMARY.md) | ## 🎯 **Key Updates Made to `.github/copilot-instructions.md`**...... | deployment, tensorrt, archive | current |
| [Housekeeping Complete Summary](markdown_docs/development_reports/HOUSEKEEPING_COMPLETE_SUMMARY.md) | Documentation for Housekeeping Complete Summary... |  | current |
| [Immediate Overlap Elimination Summary](markdown_docs/development_reports/IMMEDIATE_OVERLAP_ELIMINATION_SUMMARY.md) | Documentation for Immediate Overlap Elimination Summary... |  | current |
| [JustNews Documentation Maintenance Action Plan - COMPLETED](markdown_docs/development_reports/MAINTENANCE_COMPLETION_REPORT.md) | **Date**: September 7, 2025  
**Status**: ✅ **100% COMPLETE** - All 4 phases successfully implemente... | analytics, multi-agent, dashboard | current |
| [JustNews V4 Quality Management System - Final Status](markdown_docs/development_reports/QUALITY_SYSTEM_STATUS.md) | ## 🎯 Quality Achievement
- **Target**: >90% Quality Score
- **Achieved**: ✅ 100.0/100 (Perfect Score... | version-specific, monitoring | current |
| [JustNews V4 Workspace Organization Summary](markdown_docs/development_reports/WORKSPACE_ORGANIZATION_SUMMARY.md) | ### ✅ **COMPLETE WORKSPACE ORGANIZATION ACCOMPLISHED**...... | deployment, mcp, api | current |
| [JustNewsAgent Documentation Catalogue](markdown_docs/development_reports/DOCUMENTATION_CATALOGUE.md) | **Version:** 2.0
**Last Updated:** 2025-09-07
**Total Documents:** 140
**Categories:** 14...... | security, knowledge-graph, gpu | current |
| [JustNewsAgent Status Report](markdown_docs/development_reports/PROJECT_STATUS.md) | ## Executive Summary...... | security, gpu, version-specific | current |
| [JustNewsAgent V4 - Current Development Status Summary](markdown_docs/development_reports/CURRENT_DEVELOPMENT_STATUS.md) | **Last Updated**: August 31, 2025
**Status**: ✅ RTX3090 GPU Production Readiness Achieved - FULLY OP... | security, gpu, version-specific | current |
| [JustNewsAgentic — Implementation Plan for Evidence, KG, Fact‑Checking & Conservative Generation](markdown_docs/development_reports/IMPLEMENTATION_PLAN.md) | Date: 2025-09-07  
Branch: dev/agent_review
Status: ✅ **PHASE 2 COMPLETE - PRODUCTION READY**...... | optimization, compliance, security | current |
| [Kiss Architecture Redesign](markdown_docs/development_reports/kiss_architecture_redesign.md) | Documentation for Kiss Architecture Redesign... | architecture | current |
| [Local Model Training Plan](markdown_docs/development_reports/LOCAL_MODEL_TRAINING_PLAN.md) | Documentation for Local Model Training Plan... | training | current |
| [MCP Bus Architecture Cleanup - August 2, 2025](markdown_docs/development_reports/mcp_bus_architecture_cleanup.md) | ## 🎯 Issue Identified...... | mcp, api, archive | current |
| [Needed-For-Live-Run](markdown_docs/development_reports/Needed-for-live-run.md) | Documentation for Needed-For-Live-Run... |  | current |
| [Neural Vs Rules Strategic Analysis](markdown_docs/development_reports/NEURAL_VS_RULES_STRATEGIC_ANALYSIS.md) | Documentation for Neural Vs Rules Strategic Analysis... |  | current |
| [Newsreader Training Integration Success](markdown_docs/development_reports/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | ### 🎯 **Integration Completed Successfully**...... | training, performance, synthesizer | current |
| [Newsreader V2 Optimization Complete](markdown_docs/development_reports/NEWSREADER_V2_OPTIMIZATION_COMPLETE.md) | Documentation for Newsreader V2 Optimization Complete... | optimization | current |
| [Next Steps 2025-08-10 1436](markdown_docs/development_reports/NEXT_STEPS_2025-08-10_1436.md) | Documentation for Next Steps 2025-08-10 1436... |  | current |
| [Ocr Redundancy Analysis](markdown_docs/development_reports/OCR_REDUNDANCY_ANALYSIS.md) | Documentation for Ocr Redundancy Analysis... |  | current |
| [Online Learning Architecture](markdown_docs/development_reports/ONLINE_LEARNING_ARCHITECTURE.md) | Documentation for Online Learning Architecture... | architecture | current |
| [Online Training Integration Summary](markdown_docs/development_reports/ONLINE_TRAINING_INTEGRATION_SUMMARY.md) | Documentation for Online Training Integration Summary... | training | current |
| [Optimal Agent Separation](markdown_docs/development_reports/optimal_agent_separation.md) | Documentation for Optimal Agent Separation... |  | current |
| [Practical NewsReader Solution - File Organization Complete ✅](markdown_docs/development_reports/practical_newsreader_solution_organization.md) | ## 🎯 File Relocation Summary...... | optimization, deployment, mcp | current |
| [Production BBC Crawler - Duplicate Resolution Complete ✅](markdown_docs/development_reports/production_bbc_crawler_duplicate_resolution.md) | ## 🎯 Issue Identified & Resolved...... | mcp, archive, multi-agent | current |
| [Production Deployment Guide](markdown_docs/development_reports/PRODUCTION_DEPLOYMENT_GUIDE.md) | Documentation for Production Deployment Guide... | deployment, production | current |
| [Production Validation Summary](markdown_docs/development_reports/PRODUCTION_VALIDATION_SUMMARY.md) | Documentation for Production Validation Summary... | production | current |
| [Readme Bootstrap Models](markdown_docs/development_reports/README_BOOTSTRAP_MODELS.md) | Documentation for Readme Bootstrap Models... | models | current |
| [Readme Live Smoke](markdown_docs/development_reports/README_LIVE_SMOKE.md) | Documentation for Readme Live Smoke... |  | current |
| [Robust loading with meta tensor handling](markdown_docs/development_reports/META_TENSOR_RESOLUTION_SUCCESS.md) | ### 🎯 **Issue Analysis: System-Wide Meta Tensor Problem**...... | deployment, training, performance | current |
| [Scout Agent Production Crawler Integration - COMPLETED ✅](markdown_docs/development_reports/scout_production_crawler_integration_complete.md) | ## 🎯 Integration Summary...... | mcp, api, performance | current |
| [Synthesizer V2 Dependencies & Training Integration - SUCCESS REPORT](markdown_docs/development_reports/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ **COMPLETE SUCCESS**  
**Task**: Fix Synthesizer dependenci... | optimization, training, tensorrt | current |
| [System Architecture Assessment](markdown_docs/development_reports/SYSTEM_ARCHITECTURE_ASSESSMENT.md) | Documentation for System Architecture Assessment... | architecture | current |
| [System Assessment and Improvement Plan — 2025-08-09](markdown_docs/development_reports/System_Assessment_2025-08-09.md) | This document captures a focused assessment of the JustNewsAgentic V4 system and proposes prioritize... | security, gpu, version-specific | current |
| [System Startup Scripts - Restored and Enhanced ✅](markdown_docs/development_reports/system_startup_scripts_restored.md) | ## 🎯 **Script Recovery & Enhancement**...... | gpu, version-specific, synthesizer | current |
| [System V2 Upgrade Plan](markdown_docs/development_reports/SYSTEM_V2_UPGRADE_PLAN.md) | Documentation for System V2 Upgrade Plan... |  | current |
| [Testing & Dependency Upgrade: Paused (2025-08-24)](markdown_docs/development_reports/TESTING_PAUSED.md) | Summary
-------
This document records the dependency-testing work performed and the reason we paused... | production, api | current |
| [The Definitive User Guide: JustNews Agentic System (V4)](markdown_docs/development_reports/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide: JustNews Agentic System (V4)... | gpu, version-specific, cuda | current |
| [Training System Documentation](markdown_docs/development_reports/TRAINING_SYSTEM_DOCUMENTATION.md) | Documentation for Training System Documentation... | training | current |
| [Training System Organization Summary](markdown_docs/development_reports/TRAINING_SYSTEM_ORGANIZATION_SUMMARY.md) | Documentation for Training System Organization Summary... | training | current |
| [Using The GPU Correctly - Complete Configuration Guide](markdown_docs/development_reports/Using-The-GPU-Correctly.md) | **Date**: August 13, 2025  
**Status**: Production-Validated Configuration  
**GPU**: NVIDIA GeForce... | optimization, performance, gpu | current |
| [V2 Complete Ecosystem Action Plan](markdown_docs/development_reports/V2_COMPLETE_ECOSYSTEM_ACTION_PLAN.md) | Documentation for V2 Complete Ecosystem Action Plan... |  | current |
| [Workspace Cleanup Summary 20250808](markdown_docs/development_reports/WORKSPACE_CLEANUP_SUMMARY_20250808.md) | Documentation for Workspace Cleanup Summary 20250808... |  | current |
| [🎉 JustNews V4 Quality Management System - Implementation Complete](markdown_docs/development_reports/IMPLEMENTATION_COMPLETE.md) | ## ✅ NEXT STEPS COMPLETED SUCCESSFULLY...... | training, version-specific, production | current |
| [📊 **DOCUMENTATION COVERAGE ANALYSIS REPORT**](markdown_docs/development_reports/DOCUMENTATION_COVERAGE_ANALYSIS.md) | **Analysis Date:** September 7, 2025  
**Codebase Size:** 221 Python files  
**Current Documentation... | security, gpu, synthesizer | current |
| [📊 Documentation Quality Report](markdown_docs/development_reports/weekly_report_20250907.md) | Documentation for 📊 Documentation Quality Report... | optimization, compliance, deployment | current |
| [📋 Documentation Change Report](markdown_docs/development_reports/version_report_20250907.md) | Documentation for 📋 Documentation Change Report... |  | current |
| [🚀 JustNews V4 Documentation Quality Management - Team Training Guide](markdown_docs/development_reports/TEAM_TRAINING_GUIDE.md) | ## 📅 Training Session: September 7, 2025...... | optimization, compliance, deployment | current |

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
**Documents:** 3

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Examples for systemd native deployment](deploy/systemd/examples/README.md) | Files in this directory are examples and helpers to install the JustNews systemd units, covering sys... | mcp, ai-agents, scout | current |
| [JustNews native deployment (systemd)](deploy/systemd/DEPLOYMENT.md) | This scaffold lets you run the MCP Bus and all agents natively on Ubuntu using
systemd units and sim... | mcp, models, logging | current |
| [Systemd scaffold for JustNews](deploy/systemd/README.md) | This folder contains a native deployment scaffold:, covering system design, component interactions, ... | mcp, memory, reasoning | current |

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
**Documents:** 17

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Assessment — 2025-08-18](markdown_docs/development_reports/agent_assessment_2025-08-18.md) | This document summarizes an inspection of the `agents/` directory and how each agent maps to the Jus... | gpu, api, architecture | current |
| [BBC Crawler Duplicates - Complete Resolution ✅](markdown_docs/development_reports/bbc_crawler_duplicates_complete_resolution.md) | Technical architecture documentation covering system design, component interactions, performance cha... | scout, production, architecture | current |
| [Enhanced Reasoning Architecture](markdown_docs/development_reports/enhanced_reasoning_architecture.md) | Technical architecture documentation covering system design, component interactions, performance cha... | architecture, reasoning | current |
| [GPU Crash Investigation - Final Report](markdown_docs/development_reports/GPU-Crash-Investigation-Final-Report.md) | **Investigation Period**: August 13, 2025  
**Status**: ✅ **RESOLVED - Production Validated**  
**Im... | gpu, cuda, production | current |
| [JustNewsAgent V4 - Current Development Status Summary](markdown_docs/development_reports/CURRENT_DEVELOPMENT_STATUS.md) | **Last Updated**: August 31, 2025
**Status**: ✅ RTX3090 GPU Production Readiness Achieved - FULLY OP... | dashboard, gpu, compliance | current |
| [Kiss Architecture Redesign](markdown_docs/development_reports/kiss_architecture_redesign.md) | Technical architecture documentation covering system design, component interactions, performance cha... | architecture | current |
| [MCP Bus Architecture Cleanup - August 2, 2025](markdown_docs/development_reports/mcp_bus_architecture_cleanup.md) | Technical architecture documentation covering system design, component interactions, performance cha... | scout, api, architecture | current |
| [Newsreader Training Integration Success](markdown_docs/development_reports/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | Technical architecture documentation covering system design, component interactions, performance cha... | gpu, scout, production | current |
| [Online Learning Architecture](markdown_docs/development_reports/ONLINE_LEARNING_ARCHITECTURE.md) | Technical architecture documentation covering system design, component interactions, performance cha... | architecture | current |
| [Production BBC Crawler - Duplicate Resolution Complete ✅](markdown_docs/development_reports/production_bbc_crawler_duplicate_resolution.md) | Technical architecture documentation covering system design, component interactions, performance cha... | scout, architecture, production | current |
| [Scout Agent Production Crawler Integration - COMPLETED ✅](markdown_docs/development_reports/scout_production_crawler_integration_complete.md) | Technical architecture documentation covering system design, component interactions, performance cha... | dashboard, scout, api | current |
| [Synthesizer V2 Dependencies & Training Integration - SUCCESS REPORT](markdown_docs/development_reports/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ **COMPLETE SUCCESS**  
**Task**: Fix Synthesizer dependenci... | tensorrt, gpu, scout | current |
| [System Architecture Assessment](markdown_docs/development_reports/SYSTEM_ARCHITECTURE_ASSESSMENT.md) | Technical architecture documentation covering system design, component interactions, performance cha... | architecture | current |
| [System Assessment and Improvement Plan — 2025-08-09](markdown_docs/development_reports/System_Assessment_2025-08-09.md) | This document captures a focused assessment of the JustNewsAgentic V4 system and proposes prioritize... | dashboard, gpu, compliance | current |
| [System Startup Scripts - Restored and Enhanced ✅](markdown_docs/development_reports/system_startup_scripts_restored.md) | Technical architecture documentation covering system design, component interactions, performance cha... | dashboard, gpu, api | current |
| [The Definitive User Guide: JustNews Agentic System (V4)](markdown_docs/development_reports/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide: JustNews Agentic System (V4), covering system design, c... | gpu, api, architecture | current |
| [Using The GPU Correctly - Complete Configuration Guide](markdown_docs/development_reports/Using-The-GPU-Correctly.md) | **Date**: August 13, 2025  
**Status**: Production-Validated Configuration  
**GPU**: NVIDIA GeForce... | gpu, cuda, production | current |

---

## Implementation Reports

**Category ID:** development_reports_implementation
**Priority:** high
**Documents:** 20

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Action Plan Implementation Status (Code/Tests Evidence Only)](markdown_docs/development_reports/action_plan_implementation_status.md) | This document maps the actions listed in the action plan to their current implementation status in t... | tensorrt, gpu, scout | current |
| [Action Plan: JustNews V4 RTX-Accelerated Development](markdown_docs/development_reports/action_plan.md) | **Current Status**: Enhanced Scout Agent + TensorRT-LLM Integration Complete - Ready for Multi-Agent... | gpu, api, multi-agent | current |
| [Analysis Nucleoid Potential](markdown_docs/development_reports/analysis_nucleoid_potential.md) | Comprehensive analysis and assessment documentation covering analysis nucleoid potential with detail... | analysis | current |
| [Architectural Changes Summary](markdown_docs/development_reports/ARCHITECTURAL_CHANGES_SUMMARY.md) | Comprehensive documentation covering architectural changes summary with detailed technical informati... | analysis, development, report | current |
| [Architectural Review Findings](markdown_docs/development_reports/architectural_review_findings.md) | Comprehensive documentation covering architectural review findings with detailed technical informati... | development, implementation | current |
| [Architectural Review Summary](markdown_docs/development_reports/ARCHITECTURAL_REVIEW_SUMMARY.md) | Comprehensive documentation covering architectural review summary with detailed technical informatio... | analysis, development, report | current |
| [Complete V2 Upgrade Assessment](markdown_docs/development_reports/COMPLETE_V2_UPGRADE_ASSESSMENT.md) | Success report documenting achievements, implementation details, and validation results for complete... | success | current |
| [Corrected Scout Analysis](markdown_docs/development_reports/CORRECTED_SCOUT_ANALYSIS.md) | Comprehensive analysis and assessment documentation covering corrected scout analysis with detailed ... | scout | current |
| [Full GPU Implementation Action Plan](markdown_docs/development_reports/full_gpu_implementation_action_plan.md) | Goal: take JustNewsAgent from the current hybrid/partial TensorRT implementation to a robust, reprod... | tensorrt, gpu, api | current |
| [Immediate Overlap Elimination Summary](markdown_docs/development_reports/IMMEDIATE_OVERLAP_ELIMINATION_SUMMARY.md) | Documentation for Immediate Overlap Elimination Summary This comprehensive guide provides detailed i... | analysis, development, report | current |
| [JustNews V4 Workspace Organization Summary](markdown_docs/development_reports/WORKSPACE_ORGANIZATION_SUMMARY.md) | ### ✅ **COMPLETE WORKSPACE ORGANIZATION ACCOMPLISHED** This comprehensive guide provides detailed in... | tensorrt, gpu, scout | current |
| [Needed-For-Live-Run](markdown_docs/development_reports/Needed-for-live-run.md) | Comprehensive documentation covering needed-for-live-run with detailed technical information, implem... | development, implementation | current |
| [Neural Vs Rules Strategic Analysis](markdown_docs/development_reports/NEURAL_VS_RULES_STRATEGIC_ANALYSIS.md) | Documentation for Neural Vs Rules Strategic Analysis This comprehensive guide provides detailed info... | analysis | current |
| [Newsreader V2 Optimization Complete](markdown_docs/development_reports/NEWSREADER_V2_OPTIMIZATION_COMPLETE.md) | Documentation for Newsreader V2 Optimization Complete This comprehensive guide provides detailed inf... | optimization | current |
| [Next Steps 2025-08-10 1436](markdown_docs/development_reports/NEXT_STEPS_2025-08-10_1436.md) | Comprehensive documentation covering next steps 2025-08-10 1436 with detailed technical information,... | development, implementation | current |
| [Ocr Redundancy Analysis](markdown_docs/development_reports/OCR_REDUNDANCY_ANALYSIS.md) | Comprehensive analysis and assessment documentation covering ocr redundancy analysis with detailed f... | analysis | current |
| [Optimal Agent Separation](markdown_docs/development_reports/optimal_agent_separation.md) | Comprehensive documentation covering optimal agent separation with detailed technical information, i... | ai, development, implementation | current |
| [Readme Live Smoke](markdown_docs/development_reports/README_LIVE_SMOKE.md) | Complete guide and reference documentation for readme live smoke including setup, configuration, and... | guide | current |
| [System V2 Upgrade Plan](markdown_docs/development_reports/SYSTEM_V2_UPGRADE_PLAN.md) | Comprehensive documentation covering system v2 upgrade plan with detailed technical information, imp... | planning, implementation, architecture | current |
| [V2 Complete Ecosystem Action Plan](markdown_docs/development_reports/V2_COMPLETE_ECOSYSTEM_ACTION_PLAN.md) | Documentation for V2 Complete Ecosystem Action Plan, covering system design, component interactions,... | success | current |

---

## Performance & Optimization Reports

**Category ID:** development_reports_performance
**Priority:** high
**Documents:** 5

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Added CPU fallback for meta tensor issues](markdown_docs/development_reports/FACT_CHECKER_FIXES_SUCCESS.md) | Comprehensive documentation covering added cpu fallback for meta tensor issues with detailed technic... | gpu, production, optimization | current |
| [Entrypoints and Orchestration Flows — 2025-08-18](markdown_docs/development_reports/entrypoints_assessment_2025-08-18.md) | This document lists entry points into the JustNewsAgentic system that accept a URL or "news topic as... | gpu, scout, api | current |
| [GitHub Copilot Instructions Update Summary - August 2, 2025](markdown_docs/development_reports/COPILOT_INSTRUCTIONS_UPDATE_SUMMARY.md) | ## 🎯 **Key Updates Made to `.github/copilot-instructions.md`** This comprehensive guide provides det... | tensorrt, scout, production | current |
| [Practical NewsReader Solution - File Organization Complete ✅](markdown_docs/development_reports/practical_newsreader_solution_organization.md) | Success report documenting achievements, implementation details, and validation results for practica... | api, production, multi-agent | current |
| [Robust loading with meta tensor handling](markdown_docs/development_reports/META_TENSOR_RESOLUTION_SUCCESS.md) | ### 🎯 **Issue Analysis: System-Wide Meta Tensor Problem** This comprehensive guide provides detailed... | gpu, scout, cuda | current |

---

## Testing & Quality Assurance Reports

**Category ID:** development_reports_testing
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Production Validation Summary](markdown_docs/development_reports/PRODUCTION_VALIDATION_SUMMARY.md) | Comprehensive documentation covering production validation summary with detailed technical informati... | production | current |
| [Testing & Dependency Upgrade: Paused (2025-08-24)](markdown_docs/development_reports/TESTING_PAUSED.md) | Summary
-------
This document records the dependency-testing work performed and the reason we paused... | production, api | current |

---

## Deployment & Operations Reports

**Category ID:** development_reports_deployment
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Docker Deprecation Notice](markdown_docs/development_reports/DOCKER_DEPRECATION_NOTICE.md) | Production deployment and operational documentation including service management, configuration, sca... | deployment, operations, infrastructure | current |
| [Production Deployment Guide](markdown_docs/development_reports/PRODUCTION_DEPLOYMENT_GUIDE.md) | Production deployment and operational documentation including service management, configuration, sca... | production, deployment | current |

---

## Training & Learning Reports

**Category ID:** development_reports_training
**Priority:** high
**Documents:** 5

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Local Model Training Plan](markdown_docs/development_reports/LOCAL_MODEL_TRAINING_PLAN.md) | Comprehensive documentation of the continuous learning system, including EWC-based training, active ... | training | current |
| [Online Training Integration Summary](markdown_docs/development_reports/ONLINE_TRAINING_INTEGRATION_SUMMARY.md) | Documentation for Online Training Integration Summary Implements continuous learning algorithms with... | training | current |
| [Readme Bootstrap Models](markdown_docs/development_reports/README_BOOTSTRAP_MODELS.md) | Comprehensive documentation of the continuous learning system, including EWC-based training, active ... | models | current |
| [Training System Documentation](markdown_docs/development_reports/TRAINING_SYSTEM_DOCUMENTATION.md) | Comprehensive documentation of the continuous learning system, including EWC-based training, active ... | training | current |
| [Training System Organization Summary](markdown_docs/development_reports/TRAINING_SYSTEM_ORGANIZATION_SUMMARY.md) | Documentation for Training System Organization Summary, covering system design, component interactio... | training | current |

---

## Integration & Workflow Reports

**Category ID:** development_reports_integration
**Priority:** medium
**Documents:** 0


---

## Maintenance & Housekeeping Reports

**Category ID:** development_reports_maintenance
**Priority:** low
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Housekeeping Complete Summary](markdown_docs/development_reports/HOUSEKEEPING_COMPLETE_SUMMARY.md) | Success report documenting achievements, implementation details, and validation results for housekee... | success | current |
| [Workspace Cleanup Summary 20250808](markdown_docs/development_reports/WORKSPACE_CLEANUP_SUMMARY_20250808.md) | Documentation for Workspace Cleanup Summary 20250808 This comprehensive guide provides detailed info... | maintenance, analysis, report | current |

---

## Core Agent Documentation

**Category ID:** agent_documentation_core_agents
**Priority:** high
**Documents:** 10

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Implementation Summary](agents/newsreader/IMPLEMENTATION_SUMMARY.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | implementation | current |
| [Lifespan Migration](agents/newsreader/LIFESPAN_MIGRATION.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | ai, agent | current |
| [Native TensorRT Analyst Agent - Production Ready](agents/analyst/NATIVE_TENSORRT_README.md) | ## 🏆 **Production Status: VALIDATED & DEPLOYED** Details AI agent capabilities, communication protoc... | version-specific, analyst, logging | current |
| [Native TensorRT Analyst Agent - Quick Start Guide](agents/analyst/NATIVE_AGENT_README.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | version-specific, analyst, logging | current |
| [Potential News Sources](markdown_docs/agent_documentation/potential_news_sources.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | security | current |
| [Scout Agent - Enhanced Deep Crawl Documentation](markdown_docs/agent_documentation/SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md) | **JustNews V4 Scout Agent with Native Crawl4AI Integration** Details AI agent capabilities, communic... | version-specific, multi-agent, tensorrt | current |
| [Scout Agent V2 - AI-First Architecture](markdown_docs/agent_documentation/SCOUT_AGENT_V2_DOCUMENTATION.md) | Complete documentation for the 5-model AI-first Scout Agent with RTX3090 GPU acceleration Details AI... | scout, ai-first, 5-models | production_ready |
| [Scout Agent V2 - Next-Generation AI-First Content Analysis System](agents/scout/README.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | agents, scout, multi-agent | current |
| [Scout → Memory Pipeline Success Summary](markdown_docs/agent_documentation/SCOUT_MEMORY_PIPELINE_SUCCESS.md) | **Date**: January 29, 2025  
**Milestone**: Core JustNews V4 pipeline operational with native deploy... | version-specific, multi-agent, tensorrt | current |
| [The Definitive User Guide](markdown_docs/agent_documentation/The_Definitive_User_Guide.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | guide | current |

---

## Specialized Agent Documentation

**Category ID:** agent_documentation_specialized_agents
**Priority:** high
**Documents:** 8

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Balancer Agent V1](markdown_docs/agent_documentation/BALANCER_AGENT_V1.md) | News neutralization and balancing agent with MCP integration and GPU acceleration Details AI agent c... | balancer, neutralization, mcp-integration | production_ready |
| [Balancer Agent V1 - Integration & Debugging Guide](markdown_docs/agent_documentation/BALANCER_AGENT_INTEGRATION_GUIDE.md) | ## Overview
The Balancer Agent is a production-grade component of the JustNews V4 system, designed t... | version-specific, analyst, logging | current |
| [Hugging Face model caching and pre-download for Memory Agent](markdown_docs/agent_documentation/HF_MODEL_CACHING.md) | This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cach... | multi-agent, ai-agents, api | current |
| [LLaVA NewsReader Agent Implementation Summary](agents/newsreader/documentation/IMPLEMENTATION_SUMMARY.md) | Detailed documentation covering agent implementation, configuration, capabilities, and integration p... | version-specific, agents, pytorch | current |
| [NewsReader Agent - Production-Validated Configuration](agents/newsreader/README.md) | ## 🚨 **CRITICAL UPDATE: GPU Crash Resolution - August 13, 2025** Details AI agent capabilities, comm... | multi-agent, ai-agents, api | current |
| [NewsReader V2 Vision-Language Model Fallback Logic](markdown_docs/agent_documentation/NEWSREADER_V2_MODEL_FALLBACK.md) | ## Overview
The NewsReader V2 agent now implements robust fallback logic for vision-language model i... | multi-agent, ai-agents, mcp | current |
| [Reasoning Agent](agents/reasoning/README.md) | This package contains the reasoning agent (Nucleoid) for JustNews Details AI agent capabilities, com... | multi-agent, mcp, architecture | current |
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
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Sources Schema and Workflow](markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md) | This document specifies the `sources` schema, provenance mapping (`article_source_map`), ingestion w... | security, api, performance | current |

---

## Model Integration Documentation

**Category ID:** agent_documentation_model_integration
**Priority:** high
**Documents:** 17

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Documentation index](markdown_docs/agent_documentation/README.md) | This folder contains agent-specific documentation used by operators and
developers. Key documents: D... | multi-agent, agents, models | current |
| [Agent Model Map](markdown_docs/agent_documentation/AGENT_MODEL_MAP.md) | Complete mapping of agents to models, resources, and performance characteristics Details AI agent ca... | agents, models, mapping | current |
| [Crawl4AI vs Playwright — feature-by-feature comparison](markdown_docs/agent_documentation/Crawl4AI_vs_Playwright_Comparison.md) | Comprehensive documentation covering crawl4ai vs playwright — feature-by-feature comparison with det... | logging, agents, scout | current |
| [Crawler Consolidation Plan — JustNewsAgent](markdown_docs/agent_documentation/Crawler_Consolidation_Plan.md) | Date: 2025-08-27
Author: Consolidation plan generated from interactive session Details AI agent capa... | multi-agent, scout, ai-agents | current |
| [Embedding Helper](markdown_docs/agent_documentation/EMBEDDING_HELPER.md) | Comprehensive documentation covering embedding helper with detailed technical information, implement... | agents, multi-agent, ai-agents | current |
| [Later: resume](markdown_docs/agent_documentation/Crawl4AI_API_SUMMARY.md) | This short reference summarises the Crawl4AI programmatic APIs, dispatcher classes, REST endpoints, ... | deployment, logging, memory | current |
| [Lifespan Migration](agents/newsreader/documentation/LIFESPAN_MIGRATION.md) | Comprehensive documentation covering lifespan migration with detailed technical information, impleme... | multi-agent, mcp, api | current |
| [Model Usage](markdown_docs/agent_documentation/MODEL_USAGE.md) | Comprehensive documentation covering model usage with detailed technical information, implementation... | agents, multi-agent, ai-agents | current |
| [Model store guidelines](markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md) | This document explains the canonical model-store layout and safe update patterns for
per-agent model... | multi-agent, scout, security | current |
| [News Outlets Loader & Backfill Runbook](markdown_docs/agent_documentation/NEWS_OUTLETS_RUNBOOK.md) | This runbook explains how to safely run the canonical sources loader (`scripts/news_outlets.py`) and... | ai-agents, performance, multi-agent | current |
| [Potential Development Paths](markdown_docs/agent_documentation/potential_development_paths.md) | This document captures a compact summary of recent analysis and recommendations about the project's ... | version-specific, analyst, multi-agent | current |
| [Product Modalities Comparison](markdown_docs/agent_documentation/product_modalities_comparison.md) | This document compares three high-level product modalities the JustNews system can pursue, aligns ea... | training, fact-checker, archive | current |
| [System Decisions](markdown_docs/agent_documentation/system_decisions.md) | Comprehensive documentation covering system decisions with detailed technical information, implement... | models | current |
| [TensorRT Quickstart (safe, no-GPU stub)](agents/analyst/TENSORRT_QUICKSTART.md) | This file explains how to run a safe, developer-friendly stub for the TensorRT engine build process ... | analyst, agents, tensorrt | current |
| [Why INT8 Quantization Should Be Implemented Immediately](agents/newsreader/documentation/INT8_QUANTIZATION_RATIONALE.md) | Comprehensive documentation covering why int8 quantization should be implemented immediately with de... | analyst, multi-agent, tensorrt | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/README.md) | Comprehensive documentation covering all-mpnet-base-v2 with detailed technical information, implemen... | version-specific, training, api | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md) | Comprehensive documentation covering all-mpnet-base-v2 with detailed technical information, implemen... | version-specific, training, api | current |

---

## Crawling & Data Collection

**Category ID:** agent_documentation_crawling_systems
**Priority:** medium
**Documents:** 0


---

## Search Index Summary

**Available Tags:** 103
**Indexed Keywords:** 100

## Maintenance Information

**Last Catalogue Update:** 2025-09-07
**Next Review Date:** 2025-10-07
