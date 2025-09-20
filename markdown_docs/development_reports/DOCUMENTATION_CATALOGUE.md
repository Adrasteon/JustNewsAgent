---
title: JustNewsAgent Documentation Catalogue
description: Auto-generated description for JustNewsAgent Documentation Catalogue
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNewsAgent Documentation Catalogue

**Version:** 2.1
**Last Updated:** 2025-09-20
**Total Documents:** 141
**Categories:** 14

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

---

## Main Documentation

**Category ID:** main_documentation
**Priority:** critical
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Main Project Documentation](README.md) | Complete system overview, installation, usage, and deployment guide with RTX3090 GPU support... | overview, installation, deployment | production_ready |
| [Version History & Changelog](CHANGELOG.md) | Detailed changelog including PyTorch 2.6.0+cu124 upgrade and GPU optimization achievements... | versions, history, releases | current |

---

## Architecture & Design

**Category ID:** architecture_design
**Priority:** critical
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [JustNews V4 Architecture Proposal](docs/JustNews_Proposal_V4.md) | Hybrid architecture proposal with specialized models and continuous learning... | proposal, hybrid-architecture, continuous-learning | current |
| [JustNews V4 Implementation Plan](docs/JustNews_Plan_V4.md) | Native GPU-accelerated architecture migration plan with specialized models... | planning, migration, specialized-models | current |
| [MCP Bus Architecture](markdown_docs/mcp_bus_architecture_cleanup.md) | Central communication hub design and implementation for agent coordination... | mcp, communication, agents | current |
| [Technical Architecture Overview](markdown_docs/TECHNICAL_ARCHITECTURE.md) | Complete system architecture with RTX3090 GPU allocation and PyTorch 2.6.0+cu124 integration... | architecture, gpu, pytorch | current |

---

## Agent Documentation

**Category ID:** agent_documentation
**Priority:** high
**Documents:** 36

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Documentation index](markdown_docs/agent_documentation/README.md) | This folder contains agent-specific documentation used by operators and
developers. Key documents:..... | multi-agent, agents, models | current |
| [Agent Model Map](markdown_docs/agent_documentation/AGENT_MODEL_MAP.md) | Complete mapping of agents to models, resources, and performance characteristics... | agents, models, mapping | current |
| [Balancer Agent V1](markdown_docs/agent_documentation/BALANCER_AGENT_V1.md) | News neutralization and balancing agent with MCP integration and GPU acceleration... | balancer, neutralization, mcp-integration | production_ready |
| [Balancer Agent V1 - Integration & Debugging Guide](markdown_docs/agent_documentation/BALANCER_AGENT_INTEGRATION_GUIDE.md) | ## Overview
The Balancer Agent is a production-grade component of the JustNews V4 system, designed t... | version-specific, analyst, logging | current |
| [Crawl4AI vs Playwright — feature-by-feature comparison](markdown_docs/agent_documentation/Crawl4AI_vs_Playwright_Comparison.md) | Generated: 2025-08-27...... | logging, agents, scout | current |
| [Crawler Consolidation Plan — JustNewsAgent](markdown_docs/agent_documentation/Crawler_Consolidation_Plan.md) | Date: 2025-08-27
Author: Consolidation plan generated from interactive session...... | multi-agent, scout, ai-agents | current |
| [Embedding Helper](markdown_docs/agent_documentation/EMBEDDING_HELPER.md) | Documentation for Embedding Helper... | agents, multi-agent, ai-agents | current |
| [Hugging Face model caching and pre-download for Memory Agent](markdown_docs/agent_documentation/HF_MODEL_CACHING.md) | This document explains how to avoid Hugging Face rate limits (HTTP 429) and how to pre-download/cach... | multi-agent, ai-agents, api | current |
| [Implementation Summary](agents/newsreader/IMPLEMENTATION_SUMMARY.md) | Documentation for Implementation Summary... |  | current |
| [LLaVA NewsReader Agent Implementation Summary](agents/newsreader/documentation/IMPLEMENTATION_SUMMARY.md) | ## ✅ Completed Implementation...... | version-specific, agents, pytorch | current |
| [Later: resume](markdown_docs/agent_documentation/Crawl4AI_API_SUMMARY.md) | This short reference summarises the Crawl4AI programmatic APIs, dispatcher classes, REST endpoints, ... | deployment, logging, memory | current |
| [Lifespan Migration](agents/newsreader/LIFESPAN_MIGRATION.md) | Documentation for Lifespan Migration... |  | current |
| [Lifespan Migration](agents/newsreader/documentation/LIFESPAN_MIGRATION.md) | ### Changes Made...... | multi-agent, mcp, api | current |
| [Model Usage](markdown_docs/agent_documentation/MODEL_USAGE.md) | Documentation for Model Usage... | agents, multi-agent, ai-agents | current |
| [Model store guidelines](markdown_docs/agent_documentation/MODEL_STORE_GUIDELINES.md) | This document explains the canonical model-store layout and safe update patterns for
per-agent model... | multi-agent, scout, security | current |
| [Native TensorRT Analyst Agent - Production Ready](agents/analyst/NATIVE_TENSORRT_README.md) | ## 🏆 **Production Status: VALIDATED & DEPLOYED**...... | version-specific, analyst, logging | current |
| [Native TensorRT Analyst Agent - Quick Start Guide](agents/analyst/NATIVE_AGENT_README.md) | ## 🏆 **Production Status: OPERATIONAL**...... | version-specific, analyst, logging | current |
| [News Outlets Loader & Backfill Runbook](markdown_docs/agent_documentation/NEWS_OUTLETS_RUNBOOK.md) | This runbook explains how to safely run the canonical sources loader (`scripts/news_outlets.py`) and... | ai-agents, performance, multi-agent | current |
| [NewsReader Agent - Production-Validated Configuration](agents/newsreader/README.md) | ## 🚨 **CRITICAL UPDATE: GPU Crash Resolution - August 13, 2025**...... | multi-agent, ai-agents, api | current |
| [NewsReader V2 Vision-Language Model Fallback Logic](markdown_docs/agent_documentation/NEWSREADER_V2_MODEL_FALLBACK.md) | ## Overview
The NewsReader V2 agent now implements robust fallback logic for vision-language model i... | multi-agent, ai-agents, mcp | current |
| [Potential Development Paths](markdown_docs/agent_documentation/potential_development_paths.md) | This document captures a compact summary of recent analysis and recommendations about the project's ... | version-specific, analyst, multi-agent | current |
| [Potential News Sources](markdown_docs/agent_documentation/potential_news_sources.md) | Documentation for Potential News Sources... | security | current |
| [Product Modalities Comparison](markdown_docs/agent_documentation/product_modalities_comparison.md) | This document compares three high-level product modalities the JustNews system can pursue, aligns ea... | training, fact-checker, archive | current |
| [Reasoning Agent](agents/reasoning/README.md) | This package contains the reasoning agent (Nucleoid) for JustNews....... | multi-agent, mcp, architecture | current |
| [Reasoning Agent - Nucleoid Integration](markdown_docs/agent_documentation/REASONING_AGENT_COMPLETE_IMPLEMENTATION.md) | Complete Nucleoid-based symbolic reasoning agent with GPU memory optimization... | reasoning, nucleoid, symbolic-logic | production_ready |
| [Scout Agent - Enhanced Deep Crawl Documentation](markdown_docs/agent_documentation/SCOUT_ENHANCED_DEEP_CRAWL_DOCUMENTATION.md) | **JustNews V4 Scout Agent with Native Crawl4AI Integration**...... | version-specific, multi-agent, tensorrt | current |
| [Scout Agent V2 - AI-First Architecture](markdown_docs/agent_documentation/SCOUT_AGENT_V2_DOCUMENTATION.md) | Complete documentation for the 5-model AI-first Scout Agent with RTX3090 GPU acceleration... | scout, ai-first, 5-models | production_ready |
| [Scout Agent V2 - Next-Generation AI-First Content Analysis System](agents/scout/README.md) | ## 🎯 **Agent Overview**...... | agents, scout, multi-agent | current |
| [Scout → Memory Pipeline Success Summary](markdown_docs/agent_documentation/SCOUT_MEMORY_PIPELINE_SUCCESS.md) | **Date**: January 29, 2025  
**Milestone**: Core JustNews V4 pipeline operational with native deploy... | version-specific, multi-agent, tensorrt | current |
| [Sources Schema and Workflow](markdown_docs/agent_documentation/SOURCES_SCHEMA_AND_WORKFLOW.md) | This document specifies the `sources` schema, provenance mapping (`article_source_map`), ingestion w... | security, api, performance | current |
| [System Decisions](markdown_docs/agent_documentation/system_decisions.md) | Documentation for System Decisions... | models | current |
| [TensorRT Quickstart (safe, no-GPU stub)](agents/analyst/TENSORRT_QUICKSTART.md) | This file explains how to run a safe, developer-friendly stub for the TensorRT engine build process.... | analyst, agents, tensorrt | current |
| [The Definitive User Guide](markdown_docs/agent_documentation/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide... |  | current |
| [Why INT8 Quantization Should Be Implemented Immediately](agents/newsreader/documentation/INT8_QUANTIZATION_RATIONALE.md) | ## You're Absolutely Right! Here's Why:...... | analyst, multi-agent, tensorrt | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/README.md) | Documentation for all-mpnet-base-v2... | version-specific, training, api | current |
| [all-mpnet-base-v2](agents/fact_checker/models/sentence-transformers_all-mpnet-base-v2/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130/README.md) | Documentation for all-mpnet-base-v2... | version-specific, training, api | current |

---

## GPU Setup & Configuration

**Category ID:** gpu_configuration
**Priority:** high
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [GPU Environment Setup Guide](docs/gpu_runner_README.md) | Complete guide for RTX3090 GPU environment with PyTorch 2.6.0+cu124, CUDA 12.4, and RAPIDS 25.04... | gpu, setup, rtx3090 | production_ready |
| [GPU Model Store Assessment](docs/GPU_ModelStore_Assessment.md) | Model performance analysis and GPU resource optimization assessment... | gpu, models, assessment | current |
| [GPU Usage Audit Report](docs/GPU_Audit_Report.md) | Comprehensive GPU usage audit with performance metrics and optimization recommendations... | gpu, audit, performance | completed |
| [RAPIDS Integration Guide](docs/RAPIDS_USAGE_GUIDE.md) | GPU-accelerated data science and machine learning with RAPIDS 25.04... | rapids, gpu-acceleration, data-science | current |

---

## Production & Deployment

**Category ID:** production_deployment
**Priority:** high
**Documents:** 15

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Canonical Port Mapping](docs/canonical_port_mapping.md) | Complete service port allocation reference with status and configuration details... | ports, services, configuration | current |
| [Fact Checker Fixes Success](markdown_docs/production_status/FACT_CHECKER_FIXES_SUCCESS.md) | Documentation for Fact Checker Fixes Success... |  | current |
| [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) | Detailed implementation roadmap with phase breakdowns and success criteria... | implementation, roadmap, phases | current |
| [JustNews V4 Memory Optimization - Mission Accomplished](markdown_docs/production_status/MEMORY_OPTIMIZATION_SUCCESS_SUMMARY.md) | ## 🎯 Problem Resolution Summary...... | analyst, version-specific, memory | current |
| [JustNewsAgentic System Assessment Summary](markdown_docs/production_status/SYSTEM_OVERLAP_ANALYSIS.md) | **Assessment Date**: 7th August 2025 
**System Version**: V4 Hybrid Architecture  
**Lead Assessment... | analyst, version-specific, training | current |
| [Meta Tensor Resolution Success](markdown_docs/production_status/META_TENSOR_RESOLUTION_SUCCESS.md) | Documentation for Meta Tensor Resolution Success... |  | current |
| [Newsreader Training Integration Success](markdown_docs/production_status/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | Documentation for Newsreader Training Integration Success... | training | current |
| [Package Management & Environment Optimization - PRODUCTION READY](markdown_docs/production_status/PACKAGE_MANAGEMENT_SUCCESS.md) | **Date**: September 2, 2025
**Status**: ✅ COMPLETE - All core packages installed, tested, and produc... | analyst, dashboard, version-specific | current |
| [Production Deployment Status](markdown_docs/production_status/PRODUCTION_DEPLOYMENT_STATUS.md) | Current operational status with RTX3090 GPU utilization and performance metrics... | production, deployment, operational | current |
| [Project Status Report](docs/PROJECT_STATUS.md) | Current development status, milestones, and roadmap with version tracking... | status, milestones, roadmap | current |
| [Synthesizer Training Integration Success](markdown_docs/production_status/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | Documentation for Synthesizer Training Integration Success... | synthesizer, training | current |
| [Synthesizer V3 Production Success Summary](markdown_docs/production_status/SYNTHESIZER_V3_PRODUCTION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ PRODUCTION READY  
**Version**: V4.16.0...... | version-specific, training, memory | current |
| [Workspace Organization Summary](markdown_docs/production_status/WORKSPACE_ORGANIZATION_SUMMARY.md) | Documentation for Workspace Organization Summary... |  | current |
| [🎉 JustNews V4 Memory Optimization - DEPLOYMENT SUCCESS](markdown_docs/production_status/DEPLOYMENT_SUCCESS_SUMMARY.md) | ## 🏆 Mission Accomplished - Memory Crisis Resolved!...... | version-specific, memory, models | current |
| [🎯 **USER INSIGHT VALIDATION: COMPLETE SUCCESS**](markdown_docs/production_status/USER_INSIGHT_VALIDATION_SUCCESS.md) | ## **✅ Key Achievement: Your INT8 Quantization Approach Works!**...... | memory, multi-agent, ai-agents | current |

---

## API & Integration

**Category ID:** api_integration
**Priority:** high
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Public API Documentation](markdown_docs/agent_documentation/PUBLIC_API_DOCUMENTATION.md) | Complete documentation for the production-ready public API with authentication, rate limiting, and real-time data access... | api, security, authentication, public | production_ready |
| [Phase 3 API Documentation](docs/PHASE3_API_DOCUMENTATION.md) | RESTful and GraphQL API specifications for archive access and knowledge graph queries... | api, rest, graphql | current |
| [Phase 3 Archive Integration Plan](docs/phase3_archive_integration_plan.md) | Research-scale archiving infrastructure with provenance tracking and legal compliance... | archive, research, provenance | planning |
| [Phase 3 Knowledge Graph](docs/PHASE3_KNOWLEDGE_GRAPH.md) | Entity extraction, disambiguation, clustering, and relationship analysis documentation... | knowledge-graph, entities, relationships | current |

---

## Training & Learning

**Category ID:** training_learning
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Online Learning Architecture](markdown_docs/development_reports/ONLINE_LEARNING_ARCHITECTURE.md) | Real-time model improvement system with active learning and feedback loops... | online-learning, active-learning, feedback-loops | current |
| [Training System Documentation](markdown_docs/development_reports/TRAINING_SYSTEM_DOCUMENTATION.md) | Complete training system architecture with GPU-accelerated continuous learning... | training, continuous-learning, gpu-acceleration | operational |

---

## Monitoring & Analytics

**Category ID:** monitoring_analytics
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Analytics Dashboard Fixes](docs/ANALYTICS_DASHBOARD_FIXES_SUMMARY.md) | Dashboard fixes, enhancements, and user experience improvements... | analytics, dashboard, fixes | completed |
| [Centralized Logging Migration](docs/LOGGING_MIGRATION.md) | Centralized logging system with structured JSON logging and performance tracking... | logging, centralized, structured | current |

---

## Compliance & Security

**Category ID:** compliance_security
**Priority:** high
**Documents:** 1

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Legal Compliance Framework](docs/LEGAL_COMPLIANCE_FRAMEWORK.md) | GDPR and CCPA compliance framework with data minimization and consent management... | gdpr, ccpa, compliance | current |

---

## Development Reports

**Category ID:** development_reports
**Priority:** medium
**Documents:** 53

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Action Plan Implementation Status (Code/Tests Evidence Only)](markdown_docs/development_reports/action_plan_implementation_status.md) | This document maps the actions listed in the action plan to their current implementation status in t... | tensorrt, gpu, scout | current |
| [Action Plan: JustNews V4 RTX-Accelerated Development](markdown_docs/development_reports/action_plan.md) | **Current Status**: Enhanced Scout Agent + TensorRT-LLM Integration Complete - Ready for Multi-Agent... | gpu, api, multi-agent | current |
| [Added CPU fallback for meta tensor issues](markdown_docs/development_reports/FACT_CHECKER_FIXES_SUCCESS.md) | ### 🎯 **Issues Fixed Successfully**...... | gpu, production, optimization | current |
| [Agent Assessment — 2025-08-18](markdown_docs/development_reports/agent_assessment_2025-08-18.md) | This document summarizes an inspection of the `agents/` directory and how each agent maps to the Jus... | gpu, api, architecture | current |
| [Analysis Nucleoid Potential](markdown_docs/development_reports/analysis_nucleoid_potential.md) | Documentation for Analysis Nucleoid Potential... |  | current |
| [Architectural Changes Summary](markdown_docs/development_reports/ARCHITECTURAL_CHANGES_SUMMARY.md) | Documentation for Architectural Changes Summary... |  | current |
| [Architectural Review Findings](markdown_docs/development_reports/architectural_review_findings.md) | Documentation for Architectural Review Findings... |  | current |
| [Architectural Review Summary](markdown_docs/development_reports/ARCHITECTURAL_REVIEW_SUMMARY.md) | Documentation for Architectural Review Summary... |  | current |
| [BBC Crawler Duplicates - Complete Resolution ✅](markdown_docs/development_reports/bbc_crawler_duplicates_complete_resolution.md) | ## 🎯 **Duplicate Resolution Summary**...... | scout, production, architecture | current |
| [Complete V2 Upgrade Assessment](markdown_docs/development_reports/COMPLETE_V2_UPGRADE_ASSESSMENT.md) | Documentation for Complete V2 Upgrade Assessment... |  | current |
| [Corrected Scout Analysis](markdown_docs/development_reports/CORRECTED_SCOUT_ANALYSIS.md) | Documentation for Corrected Scout Analysis... | scout | current |
| [Docker Deprecation Notice](markdown_docs/development_reports/DOCKER_DEPRECATION_NOTICE.md) | Documentation for Docker Deprecation Notice... |  | current |
| [Enhanced Reasoning Architecture](markdown_docs/development_reports/enhanced_reasoning_architecture.md) | Documentation for Enhanced Reasoning Architecture... | architecture, reasoning | current |
| [Entrypoints and Orchestration Flows — 2025-08-18](markdown_docs/development_reports/entrypoints_assessment_2025-08-18.md) | This document lists entry points into the JustNewsAgentic system that accept a URL or "news topic as... | gpu, scout, api | current |
| [Full GPU Implementation Action Plan](markdown_docs/development_reports/full_gpu_implementation_action_plan.md) | Goal: take JustNewsAgent from the current hybrid/partial TensorRT implementation to a robust, reprod... | tensorrt, gpu, api | current |
| [GPU Crash Investigation - Final Report](markdown_docs/development_reports/GPU-Crash-Investigation-Final-Report.md) | **Investigation Period**: August 13, 2025  
**Status**: ✅ **RESOLVED - Production Validated**  
**Im... | gpu, cuda, production | current |
| [GitHub Copilot Instructions Update Summary - August 2, 2025](markdown_docs/development_reports/COPILOT_INSTRUCTIONS_UPDATE_SUMMARY.md) | ## 🎯 **Key Updates Made to `.github/copilot-instructions.md`**...... | tensorrt, scout, production | current |
| [Housekeeping Complete Summary](markdown_docs/development_reports/HOUSEKEEPING_COMPLETE_SUMMARY.md) | Documentation for Housekeeping Complete Summary... |  | current |
| [Immediate Overlap Elimination Summary](markdown_docs/development_reports/IMMEDIATE_OVERLAP_ELIMINATION_SUMMARY.md) | Documentation for Immediate Overlap Elimination Summary... |  | current |
| [JustNews V4 Workspace Organization Summary](markdown_docs/development_reports/WORKSPACE_ORGANIZATION_SUMMARY.md) | ### ✅ **COMPLETE WORKSPACE ORGANIZATION ACCOMPLISHED**...... | tensorrt, gpu, scout | current |
| [JustNewsAgent V4 - Current Development Status Summary](markdown_docs/development_reports/CURRENT_DEVELOPMENT_STATUS.md) | **Last Updated**: August 31, 2025
**Status**: ✅ RTX3090 GPU Production Readiness Achieved - FULLY OP... | dashboard, gpu, compliance | current |
| [Kiss Architecture Redesign](markdown_docs/development_reports/kiss_architecture_redesign.md) | Documentation for Kiss Architecture Redesign... | architecture | current |
| [Local Model Training Plan](markdown_docs/development_reports/LOCAL_MODEL_TRAINING_PLAN.md) | Documentation for Local Model Training Plan... | training | current |
| [MCP Bus Architecture Cleanup - August 2, 2025](markdown_docs/development_reports/mcp_bus_architecture_cleanup.md) | ## 🎯 Issue Identified...... | scout, api, architecture | current |
| [Needed-For-Live-Run](markdown_docs/development_reports/Needed-for-live-run.md) | Documentation for Needed-For-Live-Run... |  | current |
| [Neural Vs Rules Strategic Analysis](markdown_docs/development_reports/NEURAL_VS_RULES_STRATEGIC_ANALYSIS.md) | Documentation for Neural Vs Rules Strategic Analysis... |  | current |
| [Newsreader Training Integration Success](markdown_docs/development_reports/NEWSREADER_TRAINING_INTEGRATION_SUCCESS.md) | ### 🎯 **Integration Completed Successfully**...... | gpu, scout, production | current |
| [Newsreader V2 Optimization Complete](markdown_docs/development_reports/NEWSREADER_V2_OPTIMIZATION_COMPLETE.md) | Documentation for Newsreader V2 Optimization Complete... | optimization | current |
| [Next Steps 2025-08-10 1436](markdown_docs/development_reports/NEXT_STEPS_2025-08-10_1436.md) | Documentation for Next Steps 2025-08-10 1436... |  | current |
| [Ocr Redundancy Analysis](markdown_docs/development_reports/OCR_REDUNDANCY_ANALYSIS.md) | Documentation for Ocr Redundancy Analysis... |  | current |
| [Online Learning Architecture](markdown_docs/development_reports/ONLINE_LEARNING_ARCHITECTURE.md) | Documentation for Online Learning Architecture... | architecture | current |
| [Online Training Integration Summary](markdown_docs/development_reports/ONLINE_TRAINING_INTEGRATION_SUMMARY.md) | Documentation for Online Training Integration Summary... | training | current |
| [Optimal Agent Separation](markdown_docs/development_reports/optimal_agent_separation.md) | Documentation for Optimal Agent Separation... |  | current |
| [Practical NewsReader Solution - File Organization Complete ✅](markdown_docs/development_reports/practical_newsreader_solution_organization.md) | ## 🎯 File Relocation Summary...... | api, production, multi-agent | current |
| [Production BBC Crawler - Duplicate Resolution Complete ✅](markdown_docs/development_reports/production_bbc_crawler_duplicate_resolution.md) | ## 🎯 Issue Identified & Resolved...... | scout, architecture, production | current |
| [Production Deployment Guide](markdown_docs/development_reports/PRODUCTION_DEPLOYMENT_GUIDE.md) | Documentation for Production Deployment Guide... | production, deployment | current |
| [Production Validation Summary](markdown_docs/development_reports/PRODUCTION_VALIDATION_SUMMARY.md) | Documentation for Production Validation Summary... | production | current |
| [Readme Bootstrap Models](markdown_docs/development_reports/README_BOOTSTRAP_MODELS.md) | Documentation for Readme Bootstrap Models... | models | current |
| [Readme Live Smoke](markdown_docs/development_reports/README_LIVE_SMOKE.md) | Documentation for Readme Live Smoke... |  | current |
| [Robust loading with meta tensor handling](markdown_docs/development_reports/META_TENSOR_RESOLUTION_SUCCESS.md) | ### 🎯 **Issue Analysis: System-Wide Meta Tensor Problem**...... | gpu, scout, cuda | current |
| [Scout Agent Production Crawler Integration - COMPLETED ✅](markdown_docs/development_reports/scout_production_crawler_integration_complete.md) | ## 🎯 Integration Summary...... | dashboard, scout, api | current |
| [Synthesizer V2 Dependencies & Training Integration - SUCCESS REPORT](markdown_docs/development_reports/SYNTHESIZER_TRAINING_INTEGRATION_SUCCESS.md) | **Date**: August 9, 2025  
**Status**: ✅ **COMPLETE SUCCESS**  
**Task**: Fix Synthesizer dependenci... | tensorrt, gpu, scout | current |
| [System Architecture Assessment](markdown_docs/development_reports/SYSTEM_ARCHITECTURE_ASSESSMENT.md) | Documentation for System Architecture Assessment... | architecture | current |
| [System Assessment and Improvement Plan — 2025-08-09](markdown_docs/development_reports/System_Assessment_2025-08-09.md) | This document captures a focused assessment of the JustNewsAgentic V4 system and proposes prioritize... | dashboard, gpu, compliance | current |
| [System Startup Scripts - Restored and Enhanced ✅](markdown_docs/development_reports/system_startup_scripts_restored.md) | ## 🎯 **Script Recovery & Enhancement**...... | dashboard, gpu, api | current |
| [System V2 Upgrade Plan](markdown_docs/development_reports/SYSTEM_V2_UPGRADE_PLAN.md) | Documentation for System V2 Upgrade Plan... |  | current |
| [Testing & Dependency Upgrade: Paused (2025-08-24)](markdown_docs/development_reports/TESTING_PAUSED.md) | Summary
-------
This document records the dependency-testing work performed and the reason we paused... | production, api | current |
| [The Definitive User Guide: JustNews Agentic System (V4)](markdown_docs/development_reports/The_Definitive_User_Guide.md) | Documentation for The Definitive User Guide: JustNews Agentic System (V4)... | gpu, api, architecture | current |
| [Training System Documentation](markdown_docs/development_reports/TRAINING_SYSTEM_DOCUMENTATION.md) | Documentation for Training System Documentation... | training | current |
| [Training System Organization Summary](markdown_docs/development_reports/TRAINING_SYSTEM_ORGANIZATION_SUMMARY.md) | Documentation for Training System Organization Summary... | training | current |
| [Using The GPU Correctly - Complete Configuration Guide](markdown_docs/development_reports/Using-The-GPU-Correctly.md) | **Date**: August 13, 2025  
**Status**: Production-Validated Configuration  
**GPU**: NVIDIA GeForce... | gpu, cuda, production | current |
| [V2 Complete Ecosystem Action Plan](markdown_docs/development_reports/V2_COMPLETE_ECOSYSTEM_ACTION_PLAN.md) | Documentation for V2 Complete Ecosystem Action Plan... |  | current |
| [Workspace Cleanup Summary 20250808](markdown_docs/development_reports/WORKSPACE_CLEANUP_SUMMARY_20250808.md) | Documentation for Workspace Cleanup Summary 20250808... |  | current |

---

## Scripts Tools

**Category ID:** scripts_tools
**Priority:** medium
**Documents:** 4

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Build engine scaffold](tools/build_engine/README.md) | This folder contains a host-native scaffold for building TensorRT engines....... | pytorch, cuda, gpu | current |
| [Deprecate Dialogpt Readme](scripts/DEPRECATE_DIALOGPT_README.md) | Documentation for Deprecate Dialogpt Readme... | agents, multi-agent, ai-agents | current |
| [If you omit --target, the script will use the DATA_DRIVE_TARGET env var or fall back to the](scripts/README_MIRROR.md) | Documentation for If you omit --target, the script will use the DATA_DRIVE_TARGET env var or fall ba... | synthesizer, multi-agent, agents | current |
| [Readme Bootstrap Models](scripts/README_BOOTSTRAP_MODELS.md) | Documentation for Readme Bootstrap Models... | models | current |

---

## Deployment System

**Category ID:** deployment_system
**Priority:** medium
**Documents:** 3

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Examples for systemd native deployment](../agent_documentation/systemd/examples/README.md) | Files in this directory are examples and helpers to install the JustNews systemd units....... | mcp, ai-agents, scout | current |
| [JustNews native deployment (systemd)](../agent_documentation/systemd/DEPLOYMENT.md) | This scaffold lets you run the MCP Bus and all agents natively on Ubuntu using
systemd units and sim... | mcp, models, logging | current |
| [Systemd scaffold for JustNews](../agent_documentation/systemd/README.md) | This folder contains a native deployment scaffold:...... | mcp, memory, reasoning | current |

---

## General Documentation

**Category ID:** general_documentation
**Priority:** medium
**Documents:** 9

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [Agent Upgrade Plan - JustNewsAgent V4](markdown_docs/agent_upgrade_plan.md) | ## Executive Summary...... | logging, ai-agents, scout | current |
| [Canonical list of all files that can come into use](markdown_docs/IN_USE_FILES_FULL_LIST.md) | Documentation for Canonical list of all files that can come into use... | ai-agents, scout, security | current |
| [GPU Management Implementation - Complete Documentation](markdown_docs/GPU_IMPLEMENTATION_COMPLETE.md) | **Date:** August 31, 2025
**Status:** ✅ **FULLY IMPLEMENTED & PRODUCTION READY**
**Version:** v2.0.0... | logging, ai-agents, scout | current |
| [JustNews Agentic - Development Context](markdown_docs/DEVELOPMENT_CONTEXT.md) | **Last Updated**: September 2, 2025  
**Branch**: `dev/gpu_implementation`  
**Status**: Production-... | ai-agents, optimization, production | current |
| [JustNews V4 Documentation Index](markdown_docs/README.md) | This directory contains organized documentation for the JustNews V4 project. Files are categorized f... | version-specific, memory, models | current |
| [JustNews V4 — In‑Use Files Inventory](markdown_docs/IN_USE_FILES.md) | Generated: 2025-08-23...... | analyst, dashboard, version-specific | current |
| [JustNewsAgent V4 - Technical Architecture](markdown_docs/TECHNICAL_ARCHITECTURE.md) | This document provides comprehensive technical details about the JustNewsAgent V4 system architectur... | logging, ai-agents, scout | current |
| [Workspace Cleanup Summary - August 8, 2025](markdown_docs/WORKSPACE_CLEANUP_SUMMARY_20250808.md) | ## Housekeeping Actions Completed ✅...... | dashboard, multi-agent, archive | current |
| [🎉 HOUSEKEEPING COMPLETE - Ready for Manual Push](markdown_docs/HOUSEKEEPING_COMPLETE_SUMMARY.md) | ## ✅ **WORKSPACE CLEANUP SUCCESSFULLY COMPLETED**...... | training, memory, multi-agent | current |

---

## Performance Optimization

**Category ID:** performance_optimization
**Priority:** medium
**Documents:** 2

| Document | Description | Tags | Status |
|----------|-------------|------|--------|
| [NewsReader V2 Optimization Complete - Component Redundancy Analysis](markdown_docs/optimization_reports/NEWSREADER_V2_OPTIMIZATION_COMPLETE.md) | ## Executive Summary ✅...... | memory, mcp, models | current |
| [OCR Redundancy Analysis - NewsReader V2 Engine](markdown_docs/optimization_reports/OCR_REDUNDANCY_ANALYSIS.md) | ## Executive Summary
**Recommendation**: 🟡 **OCR is LIKELY REDUNDANT** but low-risk to maintain...... | training, memory, models | current |

---

## Search Index Summary

**Available Tags:** 93
**Indexed Keywords:** 100

## Maintenance Information

**Last Catalogue Update:** 2025-09-20
**Next Review Date:** 2025-10-20

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

