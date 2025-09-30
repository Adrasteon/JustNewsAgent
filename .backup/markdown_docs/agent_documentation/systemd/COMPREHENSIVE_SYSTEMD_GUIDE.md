---
title: JustNews V4 Systemd Implementation Guide
description: Auto-generated description for JustNews V4 Systemd Implementation Guide
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews V4 Systemd Implementation Guide

## Overview

JustNews V4 uses a native systemd-based deployment system for production-ready operation. This guide provides comprehensive documentation for implementing, using, maintaining, troubleshooting, and testing the systemd deployment.

## Table of Contents

1. Architecture Overview
2. System Requirements
3. Installation and Setup
4. Service Management
5. Configuration Management
6. Monitoring and Health Checks
7. Maintenance Procedures
8. Troubleshooting Guide
9. Testing Procedures
10. Performance Tuning
11. Backup and Recovery
12. Security Considerations

## Architecture Overview

### Core Components

The systemd deployment consists of:

- Systemd Unit Template: `justnews@.service` - Instanced service template
- Environment Configuration: Global and per-service environment files
- Management Scripts: Automated service lifecycle management
- Health Monitoring: Comprehensive service health checking
- Dependency Management: Proper service startup ordering

### Service Architecture

```
┌───────────────────┐    ┌───────────────────┐
│   MCP Bus       │◄──►│  Chief Editor   │
│   (Port 8000)   │    │  (Port 8001)    │
└───────────────────┘    └───────────────────┘
         ▲                       ▲
         │                       │
    ┌────┴───────────┐            ┌────┴───────────┐
    │ Agents   │            │ Agents   │
    │ 8002-8013│            │ 8002-8013│
    └───────────────┘            └───────────────┘
```

### Key Features

- Native Systemd Integration
- Dependency Management
- Resource Isolation
- Centralized Logging
- Health Monitoring
- Rolling Updates

## System Requirements

### Hardware

- CPU: 8+ cores (16+ for production)
- RAM: 32GB+ (64GB+ with GPU)
- Storage: 500GB+ SSD
- GPU: NVIDIA 8GB+ VRAM

### Software

- OS: Ubuntu 20.04+ or RHEL/CentOS 8+
- Python: 3.12+
- PostgreSQL: 16.9+ with pgvector
- CUDA: 11.8+ (if GPU)

## Installation and Setup

1. Prepare system directories and permissions
2. Copy env files to `/etc/justnews`
3. Install `justnews@.service` to `/etc/systemd/system/`
4. Reload systemd and start services

## Service Management

- Start/Stop/Restart and logs via systemd
- Bulk management with `enable_all.sh`

## Monitoring and Health Checks

- `health_check.sh` and service `/health` endpoints
- Journal and metrics integration

## Maintenance and Troubleshooting

- Daily/Weekly/Monthly tasks
- Diagnostic tools and common issues

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md
