---
title: JustNews PostgreSQL Integration Guide
description: Auto-generated description for JustNews PostgreSQL Integration Guide
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# JustNews PostgreSQL Integration Guide

**Date**: September 8, 2025
**System**: JustNews V4 Native Ubuntu Deployment
**Database**: PostgreSQL 15 (Native, not containerized)

## Executive Summary

For a complete systemd setup, PostgreSQL databases should be **run natively** on Ubuntu, not in containers. This aligns with JustNews V4's "Native Ubuntu Deployment" philosophy that eliminates Docker containerization overhead while providing better performance, monitoring, and integration with systemd services.

## Architecture Decision: Native PostgreSQL

### ✅ **Recommended: Native PostgreSQL**
- **Performance**: No container overhead, direct hardware access
- **Integration**: Seamless systemd service management and monitoring
- **Reliability**: Native Ubuntu package management and updates
- **Monitoring**: Direct integration with system logging and metrics
- **Consistency**: Aligns with JustNews "native Ubuntu" deployment philosophy

### ❌ **Not Recommended: Containerized PostgreSQL**
- **Overhead**: Additional container layer reduces performance
- **Complexity**: Adds Docker dependency to "native" deployment
- **Isolation**: Unnecessary for single-host deployment
- **Maintenance**: Additional container management complexity

## PostgreSQL Setup for JustNews

### Database Architecture

JustNews uses **two PostgreSQL databases**:

```
PostgreSQL Instance (localhost:5432)
├── justnews (Main Database)
│   ├── Core application data
│   ├── User sessions and authentication
│   ├── Analytics and reporting data
│   └── System configuration
│
└── justnews_memory (Memory/Vector Database)
    ├── Vector embeddings for semantic search
    ├── Article content storage
    ├── Similarity search indexes
    └── pgvector extension for AI operations
```

### Required Extensions

```sql
-- Main Database Extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- Query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_trgm;            -- Text similarity search

-- Memory Database Extensions
CREATE EXTENSION IF NOT EXISTS vector;             -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS pg_trgm;            -- Text similarity search
```

## Installation and Configuration

### Automated Setup Script

Use the provided setup script for complete PostgreSQL configuration:

```bash
# Run as root
sudo ./deploy/systemd/setup_postgresql.sh
```

This script will:
- Install PostgreSQL 15 and pgvector extension
- Create JustNews databases and user
- Configure production settings
- Setup automated backups
- Enable monitoring and logging

### Manual Installation (Alternative)

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15 pgvector

# Start and enable service
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Create JustNews user and databases
sudo -u postgres psql
```

```sql
-- Execute in PostgreSQL shell
CREATE USER justnews WITH PASSWORD 'justnews_password';
ALTER USER justnews CREATEDB;

CREATE DATABASE justnews OWNER justnews;
CREATE DATABASE justnews_memory OWNER justnews;

GRANT ALL PRIVILEGES ON DATABASE justnews TO justnews;
GRANT ALL PRIVILEGES ON DATABASE justnews_memory TO justnews;

-- Enable extensions
\c justnews_memory
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\c justnews
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## Environment Configuration

### Global Environment (/etc/justnews/global.env)

```bash
# Database Configuration (PostgreSQL)
DATABASE_URL=postgresql://justnews:justnews_password@localhost:5432/justnews
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# Redis Configuration (for caching and queues)
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10
```

### Memory Agent Environment (/etc/justnews/memory.env)

```bash
# Database Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://justnews:justnews_password@localhost:5432/justnews_memory
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Vector Store Configuration
VECTOR_STORE_TYPE=pgvector
VECTOR_DIMENSION=768
INDEX_TYPE=ivfflat
INDEX_LISTS=1000
```

## Systemd Integration

### PostgreSQL Service

PostgreSQL runs as a native systemd service:

```bash
# Status
sudo systemctl status postgresql

# Logs
sudo journalctl -u postgresql -f

# Restart
sudo systemctl restart postgresql
```

### Service Dependencies

JustNews agents depend on PostgreSQL:

```bash
# PostgreSQL must start before JustNews agents
sudo systemctl enable postgresql

# JustNews services depend on PostgreSQL
# (configured in justnews@.service unit file)
Wants=postgresql.service
After=postgresql.service
```

## Performance Optimization

### Production Configuration

The setup script configures PostgreSQL for production:

```ini
# /etc/postgresql/15/main/postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
max_connections = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

### Memory Tuning

Adjust based on your system RAM:

```bash
# For systems with 8GB RAM
shared_buffers = 512MB
effective_cache_size = 2GB

# For systems with 16GB+ RAM
shared_buffers = 1GB
effective_cache_size = 4GB
```

## Backup and Recovery

### Automated Backups

The setup script creates automated daily backups:

```bash
# Backup location
/var/backups/postgresql/

# Manual backup
/usr/local/bin/justnews-postgres-backup.sh

# Backup contents
justnews_20250908_120000.sql.gz
justnews_memory_20250908_120000.sql.gz
```

### Backup Schedule

```bash
# Daily cron job
/etc/cron.daily/justnews-postgres-backup

# Retention: 7 days
# Compression: gzip
# Location: /var/backups/postgresql/
```

### Recovery Procedure

```bash
# Stop JustNews services
sudo ./deploy/systemd/enable_all.sh stop

# Restore databases
gunzip justnews_20250908_120000.sql.gz
gunzip justnews_memory_20250908_120000.sql.gz

# Restore main database
sudo -u postgres psql -d justnews < justnews_20250908_120000.sql

# Restore memory database
sudo -u postgres psql -d justnews_memory < justnews_memory_20250908_120000.sql

# Restart services
sudo ./deploy/systemd/enable_all.sh start
```

## Monitoring and Maintenance

### Health Checks

PostgreSQL health is monitored via:

```bash
# Service status
sudo systemctl status postgresql

# Database connectivity
PGPASSWORD=justnews_password psql -U justnews -h localhost -d justnews -c "SELECT 1;"

# Extension status
PGPASSWORD=justnews_password psql -U justnews -h localhost -d justnews_memory -c "SELECT * FROM pg_extension;"
```

### Performance Monitoring

```sql
-- Query performance
SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;

-- Database size
SELECT pg_size_pretty(pg_database_size('justnews'));
SELECT pg_size_pretty(pg_database_size('justnews_memory'));

-- Connection count
SELECT count(*) FROM pg_stat_activity;
```

### Log Monitoring

```bash
# PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# JustNews backup logs
sudo tail -f /var/log/justnews/postgres_backup.log
```

## Security Configuration

### Access Control

```ini
# /etc/postgresql/15/main/pg_hba.conf
local   justnews        justnews                          md5
local   justnews_memory  justnews                          md5
host    justnews        justnews     127.0.0.1/32        md5
host    justnews_memory  justnews     127.0.0.1/32        md5
```

### Password Security

```bash
# Change default password after setup
sudo -u postgres psql
ALTER USER justnews PASSWORD 'your_secure_password';
```

## Integration with JustNews Agents

### Agent Dependencies

```
PostgreSQL Service
├── Memory Agent (vector storage, semantic search)
├── Analytics Agent (reporting data)
├── Archive Agent (historical data)
├── Chief Editor (workflow data)
└── All agents (configuration and session data)
```

### Connection Pooling

Each agent uses connection pooling:

```python
# Example from memory agent
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

### Vector Operations

Memory agent uses pgvector for AI operations:

```python
# Semantic search with embeddings
SELECT content_id, 1 - (embedding <=> $1) as similarity
FROM article_embeddings
ORDER BY embedding <=> $1
LIMIT 50;
```

## Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check listening ports
sudo netstat -tlnp | grep 5432

# Check logs
sudo journalctl -u postgresql -n 50
```

#### Authentication Failed
```bash
# Verify user exists
sudo -u postgres psql -c "SELECT * FROM pg_user WHERE usename = 'justnews';"

# Reset password
sudo -u postgres psql
ALTER USER justnews PASSWORD 'new_password';
```

#### Extension Missing
```bash
# Install pgvector if missing
sudo apt install postgresql-15-pgvector

# Enable extension
sudo -u postgres psql -d justnews_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Migration from Development

### From SQLite to PostgreSQL

If migrating from development SQLite databases:

```bash
# Export SQLite data
sqlite3 development.db .dump > development.sql

# Convert and import to PostgreSQL
# (Manual conversion required for schema differences)

# Update environment files
# DATABASE_URL=postgresql://justnews:justnews_password@localhost:5432/justnews
```

## Performance Benchmarks

### Expected Performance

```
Database Operations:
├── Connection time: < 10ms
├── Simple query: < 5ms
├── Vector similarity search: < 50ms
└── Bulk insert (1000 rows): < 200ms

Resource Usage (8GB RAM system):
├── Memory: 256MB shared_buffers + 1GB cache
├── CPU: < 5% average load
└── Disk: < 1GB/day growth (with compression)
```

## Conclusion

Native PostgreSQL deployment provides:

- ✅ **Optimal Performance**: No container overhead
- ✅ **Seamless Integration**: Native systemd management
- ✅ **Production Ready**: Enterprise-grade reliability
- ✅ **Easy Maintenance**: Standard Ubuntu package management
- ✅ **Comprehensive Monitoring**: Full system integration

**Recommendation**: Use native PostgreSQL for all JustNews deployments to maintain consistency with the "Native Ubuntu Deployment" architecture and maximize performance.

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md
