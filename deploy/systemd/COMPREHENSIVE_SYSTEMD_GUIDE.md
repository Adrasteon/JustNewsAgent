# JustNews V4 Systemd Implementation Guide

## Overview

JustNews V4 uses a native systemd-based deployment system for production-ready operation. This guide provides comprehensive documentation for implementing, using, maintaining, troubleshooting, and testing the systemd deployment.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Requirements](#system-requirements)
3. [Installation and Setup](#installation-and-setup)
4. [Service Management](#service-management)
5. [Configuration Management](#configuration-management)
6. [Monitoring and Health Checks](#monitoring-and-health-checks)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Testing Procedures](#testing-procedures)
10. [Performance Tuning](#performance-tuning)
11. [Backup and Recovery](#backup-and-recovery)
12. [Security Considerations](#security-considerations)

## Architecture Overview

### Core Components

The systemd deployment consists of:

- **Systemd Unit Template**: `justnews@.service` - Instanced service template
- **Environment Configuration**: Global and per-service environment files
- **Management Scripts**: Automated service lifecycle management
- **Health Monitoring**: Comprehensive service health checking
- **Dependency Management**: Proper service startup ordering

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   MCP Bus       │◄──►│  Chief Editor   │
│   (Port 8000)   │    │  (Port 8001)    │
└─────────────────┘    └─────────────────┘
         ▲                       ▲
         │                       │
    ┌────┴─────┐            ┌────┴─────┐
    │ Agents   │            │ Agents   │
    │ 8002-8013│            │ 8002-8013│
    └──────────┘            └──────────┘
```

### Key Features

- **Native Systemd Integration**: Uses systemd's native capabilities
- **Dependency Management**: Automatic service ordering and health checks
- **Resource Isolation**: Per-service GPU and resource allocation
- **Centralized Logging**: Integrated with systemd journal
- **Health Monitoring**: Built-in health check endpoints
- **Rolling Updates**: Zero-downtime service updates

## System Requirements

### Hardware Requirements

- **CPU**: 8+ cores recommended (16+ cores for production)
- **RAM**: 32GB+ (64GB+ for production with GPU)
- **Storage**: 500GB+ SSD for models and data
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 30-series or better)

### Software Requirements

- **OS**: Ubuntu 20.04+ or RHEL/CentOS 8+
- **Python**: 3.12+ with conda environment
- **PostgreSQL**: 16.9+ with pgvector extension
- **Redis**: 6.0+ (optional, for caching)
- **CUDA**: 11.8+ (if using GPU acceleration)

### Network Requirements

- **Ports**: 8000-8013 for services
- **Database**: PostgreSQL on port 5432
- **Redis**: Port 6379 (optional)
- **Monitoring**: Port 9090 for metrics

## Installation and Setup

### 1. Initial System Preparation

```bash
# Create required directories
sudo mkdir -p /etc/justnews
sudo mkdir -p /var/log/justnews
sudo mkdir -p /opt/justnews/models

# Set proper permissions
sudo chown -R adra:adra /etc/justnews
sudo chown -R adra:adra /var/log/justnews
sudo chown -R adra:adra /opt/justnews

# Create justnews group for shared access
sudo groupadd -f justnews
sudo usermod -a -G justnews adra
```

### 2. Environment Configuration

```bash
# Copy environment files
sudo cp deploy/systemd/env/global.env /etc/justnews/
sudo cp deploy/systemd/env/*.env /etc/justnews/

# Edit global configuration
sudo nano /etc/justnews/global.env
```

### 3. Systemd Unit Installation

```bash
# Install systemd unit template
sudo cp deploy/systemd/units/justnews@.service /etc/systemd/system/
sudo cp deploy/systemd/units/postgresql.service /etc/systemd/system/

# Install management scripts
sudo cp deploy/systemd/scripts/justnews-start-agent.sh /usr/local/bin/
sudo cp deploy/systemd/scripts/wait_for_mcp.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/justnews-start-agent.sh
sudo chmod +x /usr/local/bin/wait_for_mcp.sh

# Reload systemd
sudo systemctl daemon-reload
```

### 4. PostgreSQL Setup

```bash
# Run PostgreSQL setup script
sudo ./deploy/systemd/complete_postgresql.sh

# Verify database setup
psql -h localhost -U justnews_user -d justnews -c "SELECT version();"
```

### 5. Service Deployment

```bash
# Enable and start all services
sudo ./deploy/systemd/enable_all.sh fresh

# Verify deployment
./deploy/systemd/health_check.sh
```

## Service Management

### Service Instances

The system uses instanced systemd services:

| Service | Port | Description |
|---------|------|-------------|
| mcp_bus | 8000 | Central communication hub |
| chief_editor | 8001 | Workflow orchestration |
| scout | 8002 | Content discovery |
| fact_checker | 8003 | Fact verification |
| analyst | 8004 | Sentiment analysis |
| synthesizer | 8005 | Content synthesis |
| critic | 8006 | Quality assessment |
| memory | 8007 | Knowledge storage |
| reasoning | 8008 | Logical reasoning |
| newsreader | 8009 | Content processing |
| balancer | 8010 | Load balancing |
| dashboard | 8011 | Web interface |
| analytics | 8012 | Performance analytics |
| archive | 8013 | Content archiving |

### Basic Service Commands

```bash
# Start specific service
sudo systemctl start justnews@mcp_bus

# Stop specific service
sudo systemctl stop justnews@mcp_bus

# Restart service
sudo systemctl restart justnews@mcp_bus

# Check service status
sudo systemctl status justnews@mcp_bus

# View service logs
journalctl -u justnews@mcp_bus -f

# Enable service at boot
sudo systemctl enable justnews@mcp_bus

# Disable service at boot
sudo systemctl disable justnews@mcp_bus
```

### Bulk Service Management

```bash
# Start all services
sudo ./deploy/systemd/enable_all.sh start

# Stop all services
sudo ./deploy/systemd/enable_all.sh stop

# Restart all services
sudo ./deploy/systemd/enable_all.sh restart

# Fresh start (stop → disable → enable → start)
sudo ./deploy/systemd/enable_all.sh fresh

# Show status of all services
sudo ./deploy/systemd/enable_all.sh status
```

### Service Dependencies

Services start in the following order:
1. **mcp_bus** (critical dependency)
2. **chief_editor** (orchestration)
3. **scout, fact_checker, analyst, synthesizer, critic, memory, reasoning, newsreader, balancer, dashboard, analytics, archive**

## Configuration Management

### Environment File Structure

```
/etc/justnews/
├── global.env          # Global settings
├── mcp_bus.env         # MCP Bus specific
├── chief_editor.env    # Chief Editor specific
├── scout.env           # Scout agent specific
└── ...                 # Other service configs
```

### Global Configuration

Key global settings in `/etc/justnews/global.env`:

```bash
# Project Configuration
PROJECT_ROOT=/home/adra/justnewsagent/JustNewsAgent
PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Database Configuration
DATABASE_URL=postgresql://justnews_user:password123@localhost:5432/justnews

# GPU Configuration
USE_GPU=false
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.8

# Performance Configuration
MAX_WORKERS=4
BATCH_SIZE=16

# Monitoring
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### Service-Specific Configuration

Example `/etc/justnews/mcp_bus.env`:

```bash
# MCP Bus specific settings
MCP_BUS_HOST=0.0.0.0
MCP_BUS_PORT=8000
MCP_BUS_WORKERS=4

# Agent management
MAX_AGENTS=20
AGENT_TIMEOUT=300
HEARTBEAT_INTERVAL=30
```

### GPU Assignment

Assign specific GPUs to services:

```bash
# In /etc/justnews/analyst.env
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9

# In /etc/justnews/synthesizer.env
CUDA_VISIBLE_DEVICES=1
GPU_MEMORY_FRACTION=0.8
```

### Configuration Validation

```bash
# Validate environment files
./deploy/systemd/preflight.sh

# Check specific service configuration
sudo systemctl show justnews@mcp_bus -p EnvironmentFiles
```

## Monitoring and Health Checks

### Health Check System

The system includes comprehensive health monitoring:

```bash
# Check all services
./deploy/systemd/health_check.sh

# Check specific services
./deploy/systemd/health_check.sh mcp_bus chief_editor

# Verbose output
./deploy/systemd/health_check.sh --verbose

# Custom timeout
./deploy/systemd/health_check.sh --timeout 10
```

### Health Check Output

```
SERVICE         STATUS   SYSTEMD      PORT       HTTP
---------------------------------------------------------------
mcp_bus         healthy  active       listening  healthy
chief_editor    healthy  active       listening  healthy
scout           healthy  active       listening  healthy
fact_checker    stopped  inactive     not_listening unknown
...
```

### Service Health Endpoints

Each service exposes health endpoints:

- **/health**: Basic health check
- **/ready**: Readiness probe
- **/metrics**: Prometheus metrics (if enabled)

### Monitoring Integration

```bash
# View systemd journal
journalctl -u justnews@mcp_bus -f

# Follow all JustNews logs
journalctl -t justnews -f

# Export logs to file
journalctl -u justnews@mcp_bus --since "1 hour ago" > mcp_bus.log
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Tasks
```bash
# Check service health
./deploy/systemd/health_check.sh

# Monitor disk usage
df -h /opt/justnews /var/log/justnews

# Check database connections
psql -h localhost -U justnews_user -d justnews -c "SELECT count(*) FROM pg_stat_activity;"

# Review recent errors
journalctl -t justnews --since "1 day ago" -p err
```

#### Weekly Tasks
```bash
# Clean old logs
sudo journalctl --vacuum-time=7d

# Update model cache
find /opt/justnews/models -name "*.cache" -mtime +7 -delete

# Database maintenance
psql -h localhost -U justnews_user -d justnews -c "VACUUM ANALYZE;"

# Check for security updates
sudo apt update && sudo apt list --upgradable
```

#### Monthly Tasks
```bash
# Full system backup
./deploy/systemd/backup.sh full

# Performance analysis
./deploy/systemd/performance_analysis.sh

# Model cache optimization
./deploy/systemd/clean_model_cache.sh
```

### Log Management

```bash
# View logs by priority
journalctl -t justnews -p info

# Search for specific errors
journalctl -t justnews | grep -i error

# Export logs for analysis
journalctl -t justnews --since "2024-01-01" --until "2024-01-31" > monthly_logs.txt

# Monitor log size
journalctl --disk-usage
```

### Database Maintenance

```bash
# Database health check
psql -h localhost -U justnews_user -d justnews -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Vacuum and analyze
psql -h localhost -U justnews_user -d justnews -c "VACUUM ANALYZE;"

# Reindex if needed
psql -h localhost -U justnews_user -d justnews -c "REINDEX DATABASE justnews;"

# Monitor connections
psql -h localhost -U justnews_user -d justnews -c "
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state 
ORDER BY count(*) DESC;"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Service Won't Start

**Symptoms**: Service shows as "failed" or won't start

**Troubleshooting Steps**:
```bash
# Check service status
sudo systemctl status justnews@mcp_bus

# View detailed logs
journalctl -u justnews@mcp_bus -n 50

# Check environment configuration
sudo systemctl show justnews@mcp_bus -p EnvironmentFiles

# Test manual startup
sudo -u adra /usr/local/bin/justnews-start-agent.sh mcp_bus
```

**Common Causes**:
- Missing environment files
- Incorrect Python path
- Database connection issues
- Port conflicts

#### Port Already in Use

**Symptoms**: Service fails with "Address already in use"

**Solution**:
```bash
# Find process using the port
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or use the port cleanup script
./deploy/systemd/scripts/free_ports.sh
```

#### Database Connection Issues

**Symptoms**: Services fail with database errors

**Troubleshooting**:
```bash
# Test database connection
psql -h localhost -U justnews_user -d justnews -c "SELECT 1;"

# Check PostgreSQL status
sudo systemctl status postgresql

# View database logs
sudo tail -f /var/log/postgresql/postgresql-16-main.log

# Reset database connection pool
sudo systemctl restart postgresql
```

#### GPU Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
```bash
# Check GPU memory usage
nvidia-smi

# Restart GPU-intensive services
sudo systemctl restart justnews@analyst justnews@synthesizer

# Adjust GPU memory fraction in environment files
echo "GPU_MEMORY_FRACTION=0.7" | sudo tee -a /etc/justnews/analyst.env
```

#### High CPU/Memory Usage

**Symptoms**: System becomes unresponsive

**Troubleshooting**:
```bash
# Check system resources
top -p $(pgrep -f justnews)

# Monitor service resource usage
sudo systemctl status justnews@mcp_bus --no-pager -l

# Restart problematic service
sudo systemctl restart justnews@<service_name>

# Check for memory leaks
ps aux --sort=-%mem | head -10
```

### Diagnostic Tools

#### System Diagnostics
```bash
# Run preflight checks
./deploy/systemd/preflight.sh

# Comprehensive system check
./deploy/systemd/diagnostics.sh

# Network connectivity test
./deploy/systemd/network_test.sh
```

#### Service Diagnostics
```bash
# Test individual service health
curl -s http://localhost:8000/health

# Check service dependencies
systemctl list-dependencies justnews@mcp_bus

# Monitor service performance
./deploy/systemd/monitor_service.sh mcp_bus
```

#### Log Analysis
```bash
# Search for error patterns
journalctl -t justnews | grep -E "(ERROR|CRITICAL|FATAL)"

# Analyze log frequency
journalctl -t justnews --since "1 hour ago" | awk '{print $1}' | sort | uniq -c | sort -nr

# Export logs for external analysis
journalctl -t justnews --since "2024-01-01" > full_logs.txt
```

## Testing Procedures

### Unit Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_mcp_bus.py -v

# Run with coverage
python -m pytest tests/ --cov=agents --cov-report=html
```

### Integration Testing

```bash
# Test MCP Bus communication
python -m pytest tests/integration/test_mcp_integration.py -v

# Test database integration
python -m pytest tests/integration/test_database_integration.py -v

# Test GPU operations
python -m pytest tests/integration/test_gpu_operations.py -v
```

### Performance Testing

```bash
# Load testing
./deploy/systemd/load_test.sh --duration 300 --concurrency 10

# GPU performance test
./deploy/systemd/gpu_benchmark.sh

# Database performance test
./deploy/systemd/db_benchmark.sh
```

### Health Check Testing

```bash
# Test health check script
./deploy/systemd/health_check.sh --verbose

# Test individual endpoints
curl -s http://localhost:8000/health | jq .
curl -s http://localhost:8001/health | jq .

# Test service dependencies
./deploy/systemd/test_dependencies.sh
```

### Automated Testing

```bash
# Run full test suite
./deploy/systemd/run_tests.sh

# Run smoke tests (quick validation)
./deploy/systemd/smoke_test.sh

# Run performance regression tests
./deploy/systemd/performance_test.sh
```

## Performance Tuning

### GPU Optimization

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Adjust batch sizes based on GPU memory
# In /etc/justnews/analyst.env
BATCH_SIZE=8
MAX_SEQUENCE_LENGTH=256

# Enable GPU memory pooling
# In /etc/justnews/global.env
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Database Optimization

```bash
# Optimize PostgreSQL configuration
sudo nano /etc/postgresql/16/main/postgresql.conf

# Key settings for JustNews:
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Service Optimization

```bash
# Adjust worker counts based on CPU cores
# In /etc/justnews/mcp_bus.env
MCP_BUS_WORKERS=8

# Optimize batch processing
# In /etc/justnews/analyst.env
BATCH_SIZE=16
MAX_WORKERS=4

# Configure connection pooling
# In /etc/justnews/global.env
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

### Network Optimization

```bash
# Increase system limits
sudo nano /etc/security/limits.conf
# Add:
* soft nofile 65536
* hard nofile 65536

# Configure systemd service limits
sudo nano /etc/systemd/system/justnews@.service
# Add to [Service] section:
LimitNOFILE=65536
LimitNPROC=4096
```

## Backup and Recovery

### Database Backup

```bash
# Full database backup
pg_dump -h localhost -U justnews_user -d justnews > justnews_backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump -h localhost -U justnews_user -d justnews | gzip > justnews_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Automated backup script
./deploy/systemd/backup_database.sh
```

### Configuration Backup

```bash
# Backup all environment files
tar -czf justnews_config_backup_$(date +%Y%m%d).tar.gz /etc/justnews/

# Backup systemd units
sudo tar -czf systemd_backup_$(date +%Y%m%d).tar.gz /etc/systemd/system/justnews*

# Backup model cache
tar -czf model_cache_backup_$(date +%Y%m%d).tar.gz /opt/justnews/models/
```

### Full System Backup

```bash
# Complete backup script
./deploy/systemd/full_backup.sh

# This includes:
# - Database dump
# - Configuration files
# - Model cache
# - Log files
# - Systemd configuration
```

### Recovery Procedures

#### Database Recovery
```bash
# Stop all services
sudo ./deploy/systemd/enable_all.sh stop

# Restore database
psql -h localhost -U justnews_user -d postgres -c "DROP DATABASE IF EXISTS justnews;"
psql -h localhost -U justnews_user -d postgres -c "CREATE DATABASE justnews;"
psql -h localhost -U justnews_user -d justnews < justnews_backup.sql

# Restart services
sudo ./deploy/systemd/enable_all.sh start
```

#### Configuration Recovery
```bash
# Restore configuration files
sudo tar -xzf justnews_config_backup.tar.gz -C /

# Reload systemd
sudo systemctl daemon-reload

# Restart services
sudo ./deploy/systemd/enable_all.sh restart
```

#### Emergency Recovery
```bash
# Quick recovery script
./deploy/systemd/emergency_recovery.sh

# This performs:
# - Service restart
# - Database connection reset
# - Log rotation
# - Health check validation
```

## Security Considerations

### Service Isolation

```bash
# Run services as separate users
sudo useradd -r -s /bin/false justnews_mcp
sudo useradd -r -s /bin/false justnews_analyst

# Update service files
sudo nano /etc/systemd/system/justnews@.service
# Change User=adra to User=justnews_%i
```

### Network Security

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 8000:8013/tcp  # JustNews services
sudo ufw allow 5432/tcp      # PostgreSQL
sudo ufw allow 22/tcp        # SSH

# Use internal networking
# In environment files, change 0.0.0.0 to 127.0.0.1 for internal services
```

### API Security

```bash
# Enable authentication
# In /etc/justnews/mcp_bus.env
ENABLE_AUTH=true
API_KEYS=your-secure-api-keys

# Configure TLS/SSL
# In /etc/justnews/global.env
SSL_CERT_PATH=/etc/ssl/certs/justnews.crt
SSL_KEY_PATH=/etc/ssl/private/justnews.key
```

### Log Security

```bash
# Secure log files
sudo chmod 640 /var/log/justnews/*.log
sudo chown root:justnews /var/log/justnews/*.log

# Configure log rotation
sudo nano /etc/logrotate.d/justnews
/var/log/justnews/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 640 root justnews
    postrotate
        systemctl reload justnews@*.service
    endscript
}
```

### Monitoring Security

```bash
# Secure metrics endpoint
# In /etc/justnews/global.env
METRICS_USERNAME=admin
METRICS_PASSWORD=secure-password

# Configure monitoring access
sudo ufw allow from 192.168.1.0/24 to any port 9090  # Monitoring subnet only
```

---

## Quick Reference

### Emergency Commands

```bash
# Stop everything immediately
sudo ./deploy/systemd/enable_all.sh stop

# Emergency restart
sudo ./deploy/systemd/enable_all.sh fresh

# Check system health
./deploy/systemd/health_check.sh

# View critical errors
journalctl -t justnews -p err -n 20
```

### Daily Monitoring

```bash
# Service status
sudo ./deploy/systemd/enable_all.sh status

# System resources
top -p $(pgrep -f justnews | tr '\n' ',' | sed 's/,$//')

# Database health
psql -h localhost -U justnews_user -d justnews -c "SELECT version();"

# GPU status
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

### Performance Metrics

- **Normal CPU Usage**: < 70% across all cores
- **Normal Memory Usage**: < 80% of available RAM
- **Normal GPU Memory**: < 90% of allocated GPU memory
- **Normal Response Time**: < 2 seconds for health checks
- **Normal Database Connections**: < 50 active connections

This comprehensive guide covers all aspects of the JustNews V4 systemd implementation. For additional support or questions, refer to the project documentation or create an issue in the repository.</content>
<parameter name="filePath">/home/adra/justnewsagent/JustNewsAgent/deploy/systemd/COMPREHENSIVE_SYSTEMD_GUIDE.md
