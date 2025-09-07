# Deployment Procedures Documentation

## Overview

The JustNews V4 system supports multiple deployment strategies optimized for different environments and use cases. The primary deployment approach uses native Ubuntu with systemd for production stability, while development and testing environments can use the daemon startup script. This documentation covers all deployment procedures, from development setup to production deployment and scaling.

## Deployment Strategies

### Primary Deployment: Native Ubuntu with systemd

The recommended production deployment uses native Ubuntu packages with systemd service management, providing:

- **Stability**: Direct OS integration with proper process management
- **Performance**: No container overhead for GPU and I/O operations
- **Observability**: Native systemd logging and monitoring integration
- **Security**: Standard Linux security practices and permissions
- **Maintenance**: Standard Ubuntu package management and updates

### Alternative Deployments

#### Development/Testing Deployment
- Uses `start_services_daemon.sh` script
- Conda environment activation
- Local logging and health monitoring
- Graceful shutdown and port management

#### Container Deployment (Deprecated)
- Docker containers available but deprecated
- Maintained for legacy compatibility
- Not recommended for production GPU workloads

## System Requirements

### Hardware Requirements

#### Minimum Requirements
```yaml
CPU: 4 cores (Intel/AMD x64)
RAM: 16GB
Storage: 100GB SSD
Network: 1Gbps
GPU: Optional (NVIDIA with CUDA support)
```

#### Recommended Production Requirements
```yaml
CPU: 8+ cores (Intel/AMD x64)
RAM: 32GB+
Storage: 500GB+ NVMe SSD
Network: 10Gbps
GPU: NVIDIA RTX 30-series or A-series (24GB+ VRAM)
```

#### High-Performance Requirements
```yaml
CPU: 16+ cores (Intel/AMD x64)
RAM: 64GB+
Storage: 1TB+ NVMe SSD (RAID 1/10)
Network: 25Gbps+
GPU: NVIDIA RTX 40-series or H100 (48GB+ VRAM)
```

### Software Requirements

#### Ubuntu Server 22.04 LTS (Recommended)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
  python3.12 \
  python3.12-venv \
  python3-pip \
  postgresql-14 \
  postgresql-contrib-14 \
  redis-server \
  nginx \
  curl \
  wget \
  git \
  htop \
  iotop \
  ncdu \
  tmux \
  ufw \
  fail2ban \
  unattended-upgrades
```

#### GPU Support (Optional)
```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Install cuDNN
# Download and install from NVIDIA website
```

#### Conda Environment
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create JustNews environment
conda create -n justnews-v2-py312 python=3.12 -y
conda activate justnews-v2-py312

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure Setup

### Production Directory Structure
```bash
# Application directory
/opt/justnews/
├── JustNewsAgent/          # Application code
├── model_store/            # Shared model storage
├── logs/                   # Application logs
└── data/                   # Persistent data

# Configuration directory
/etc/justnews/
├── global.env              # Global environment variables
├── mcp_bus.env            # MCP Bus configuration
├── synthesizer.env        # Synthesizer configuration
└── ...                    # Per-agent configurations

# System directories
/var/log/justnews/          # Systemd logs
/var/lib/justnews/          # Application data
```

### Directory Creation
```bash
# Create application directories
sudo mkdir -p /opt/justnews/{JustNewsAgent,model_store,logs,data}
sudo mkdir -p /var/log/justnews
sudo mkdir -p /var/lib/justnews

# Create configuration directory
sudo mkdir -p /etc/justnews

# Set ownership
sudo chown -R adra:adra /opt/justnews
sudo chown -R adra:adra /var/log/justnews
sudo chown -R adra:adra /var/lib/justnews
```

## Configuration Setup

### Global Environment Configuration

Create `/etc/justnews/global.env`:

```bash
# Global JustNews Configuration
SERVICE_USER=adra
SERVICE_GROUP=adra
SERVICE_DIR=/opt/justnews/JustNewsAgent
PYTHON_BIN=/opt/conda/envs/justnews-v2-py312/bin/python

# Model Store Configuration
MODEL_STORE_ROOT=/opt/justnews/model_store

# Logging Configuration
LOG_DIR=/var/log/justnews

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=justnews
POSTGRES_USER=justnews_user
POSTGRES_PASSWORD=secure_password_here

# MCP Bus Configuration
MCP_BUS_URL=http://localhost:8000

# GPU Configuration
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# Security Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Per-Agent Configuration

#### MCP Bus Configuration (`/etc/justnews/mcp_bus.env`)
```bash
# MCP Bus Service Configuration
MCP_BUS_PORT=8000
LOG_LEVEL=INFO
WORKERS=4
```

#### Synthesizer Configuration (`/etc/justnews/synthesizer.env`)
```bash
# Synthesizer Agent Configuration
SYNTHESIZER_AGENT_PORT=8005
GPU_MEMORY_LIMIT=8GB
BATCH_SIZE=16
LOG_LEVEL=INFO
```

#### Scout Configuration (`/etc/justnews/scout.env`)
```bash
# Scout Agent Configuration
SCOUT_AGENT_PORT=8002
CRAWL_TIMEOUT=30
MAX_CONCURRENT_SITES=5
LOG_LEVEL=INFO
```

## Systemd Deployment

### Systemd Unit Template

The system uses a parameterized systemd unit template for all agents:

```ini
# /etc/systemd/system/justnews@.service
[Unit]
Description=JustNews Service (%i)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
EnvironmentFile=-/etc/justnews/global.env
EnvironmentFile=-/etc/justnews/%i.env
User=adra
Group=adra
WorkingDirectory=/opt/justnews/JustNewsAgent
ExecStartPre=/usr/local/bin/wait_for_mcp.sh
ExecStart=/usr/local/bin/justnews-start-agent.sh %i
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Service Startup Scripts

#### MCP Bus Wait Script (`/usr/local/bin/wait_for_mcp.sh`)
```bash
#!/bin/bash
# Wait for MCP Bus to be ready

TIMEOUT=30
HOST=localhost
PORT=8000

echo "Waiting for MCP Bus on $HOST:$PORT..."

for i in $(seq 1 $TIMEOUT); do
  if nc -z $HOST $PORT 2>/dev/null; then
    echo "MCP Bus is ready!"
    exit 0
  fi
  echo "Attempt $i/$TIMEOUT: MCP Bus not ready yet..."
  sleep 1
done

echo "MCP Bus failed to start within $TIMEOUT seconds"
exit 1
```

#### Agent Startup Script (`/usr/local/bin/justnews-start-agent.sh`)
```bash
#!/bin/bash
# Start JustNews agent

AGENT_NAME=$1
SERVICE_DIR=${SERVICE_DIR:-/opt/justnews/JustNewsAgent}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$SERVICE_DIR" || exit 1

echo "Starting JustNews agent: $AGENT_NAME"

case $AGENT_NAME in
  mcp_bus)
    exec $PYTHON_BIN -m agents.mcp_bus.main
    ;;
  synthesizer)
    exec $PYTHON_BIN -m agents.synthesizer.main
    ;;
  scout)
    exec $PYTHON_BIN -m agents.scout.main
    ;;
  *)
    echo "Unknown agent: $AGENT_NAME"
    exit 1
    ;;
esac
```

### Service Installation

```bash
# Copy unit template
sudo cp deploy/systemd/units/justnews@.service /etc/systemd/system/

# Copy environment files
sudo cp deploy/systemd/env/*.env /etc/justnews/

# Set permissions
sudo chmod 644 /etc/systemd/system/justnews@.service
sudo chmod 600 /etc/justnews/*.env
sudo chown root:root /etc/justnews/*.env

# Reload systemd
sudo systemctl daemon-reload
```

### Service Management

#### Enable and Start Services
```bash
# Enable MCP Bus (starts first)
sudo systemctl enable --now justnews@mcp_bus

# Enable core agents
sudo systemctl enable --now justnews@synthesizer
sudo systemctl enable --now justnews@scout
sudo systemctl enable --now justnews@analyst

# Enable all agents at once
sudo ./deploy/systemd/enable_all.sh
```

#### Service Status and Logs
```bash
# Check service status
sudo systemctl status justnews@mcp_bus

# View service logs
sudo journalctl -u justnews@mcp_bus -f

# Follow all JustNews logs
sudo journalctl -u "justnews@*" -f
```

#### Service Control
```bash
# Stop specific service
sudo systemctl stop justnews@synthesizer

# Restart service
sudo systemctl restart justnews@scout

# Disable service
sudo systemctl disable justnews@analyst
```

## Development Deployment

### Using the Daemon Script

The `start_services_daemon.sh` script provides a simple way to start all services for development and testing:

```bash
# Start all services in background
./start_services_daemon.sh

# Start services and keep them running (for testing)
./start_services_daemon.sh --no-detach

# Custom health check timeout
./start_services_daemon.sh --health-timeout 20
```

### Script Features

#### Automatic Port Management
```bash
# Pre-flight port check
echo "Checking ports 8000..8013 for running agents..."

# Graceful shutdown attempt
attempt_shutdown_port() {
  local port="$1"
  local url="http://localhost:${port}/shutdown"
  curl -X POST --max-time 3 "$url"
}

# Force kill if graceful shutdown fails
free_port_force() {
  local port="$1"
  pids=$(lsof -ti tcp:"$port")
  kill -TERM $pids
}
```

#### Health Monitoring
```bash
# Health check with timeout
wait_for_health() {
  local name="$1" port="$2"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  local url="http://localhost:$port/health"

  while [ $(date +%s) -le $deadline ]; do
    if curl -s --max-time 2 "$url" >/dev/null; then
      echo "✅ $name is healthy"
      return 0
    fi
    sleep 1
  done

  echo "⚠️ $name failed health check"
  return 1
}
```

#### Environment Setup
```bash
# Model store configuration
export MODEL_STORE_ROOT="${MODEL_STORE_ROOT:-$DEFAULT_BASE_MODELS_DIR/model_store}"
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-$DEFAULT_BASE_MODELS_DIR/agents}"

# Database defaults
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_DB="${POSTGRES_DB:-justnews}"
export POSTGRES_USER="${POSTGRES_USER:-justnews_user}"

# Per-agent cache directories
export SYNTHESIZER_MODEL_CACHE="${SYNTHESIZER_MODEL_CACHE:-$DEFAULT_BASE_MODELS_DIR/agents/synthesizer/models}"
```

## Database Setup

### PostgreSQL Installation and Configuration

```bash
# Install PostgreSQL
sudo apt install -y postgresql-14 postgresql-contrib-14

# Create database and user
sudo -u postgres psql

CREATE DATABASE justnews;
CREATE USER justnews_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE justnews TO justnews_user;
ALTER USER justnews_user CREATEDB;

# Exit psql
\q
```

### Database Initialization

```bash
# Run database initialization scripts
cd /opt/justnews/JustNewsAgent
python scripts/init_database.py

# Run migrations
python scripts/db_operations.py migrate

# Verify connection
python scripts/db_operations.py test_connection
```

### Connection Pooling Configuration

```postgresql
# /etc/postgresql/14/main/postgresql.conf
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

## GPU Configuration

### GPU Driver and CUDA Setup

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Verify installation
nvidia-smi

# Install CUDA (if not using conda)
# Download from NVIDIA website and install

# Verify CUDA installation
nvcc --version
```

### GPU Memory Management

```bash
# Set GPU memory limits per agent
export CUDA_VISIBLE_DEVICES=0
export GPU_MEMORY_LIMIT=8GB

# For multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
export GPU_0_MEMORY_LIMIT=12GB
export GPU_1_MEMORY_LIMIT=12GB
```

### GPU Health Monitoring

```bash
# Monitor GPU status
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

# Watch GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Networking and Security

### Firewall Configuration

```bash
# Configure UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow JustNews ports
sudo ufw allow 8000:8013/tcp

# Allow PostgreSQL (local only)
sudo ufw allow from 127.0.0.1 to any port 5432

# Check status
sudo ufw status
```

### SSL/TLS Configuration

```bash
# Install certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Configure nginx for SSL termination
sudo nano /etc/nginx/sites-available/justnews

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/justnews
upstream justnews_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
    # Add more backend servers as needed
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://justnews_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for real-time features
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Static file serving
    location /static/ {
        alias /opt/justnews/JustNewsAgent/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Monitoring and Logging

### System Monitoring Setup

```bash
# Install monitoring tools
sudo apt install -y prometheus prometheus-node-exporter grafana

# Configure node exporter
sudo systemctl enable --now prometheus-node-exporter

# Install application monitoring
pip install prometheus-client
```

### Log Aggregation

```bash
# Install rsyslog configuration
sudo nano /etc/rsyslog.d/justnews.conf

# JustNews log aggregation
:programname, startswith, "justnews" /var/log/justnews/aggregated.log
& stop

# Restart rsyslog
sudo systemctl restart rsyslog
```

### Log Rotation

```bash
# Configure logrotate
sudo nano /etc/logrotate.d/justnews

/var/log/justnews/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 adra adra
    postrotate
        systemctl reload justnews@*
    endscript
}
```

## Backup and Recovery

### Database Backup

```bash
# Create backup script
sudo nano /usr/local/bin/justnews-backup.sh

#!/bin/bash
BACKUP_DIR="/opt/justnews/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
pg_dump -U justnews_user -h localhost justnews > $BACKUP_DIR/justnews_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/justnews_$DATE.sql

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/justnews_$DATE.sql.gz"
```

### Automated Backups

```bash
# Add to cron for daily backups
sudo crontab -e

# Daily backup at 2 AM
0 2 * * * /usr/local/bin/justnews-backup.sh
```

### Recovery Procedures

```bash
# Stop all services
sudo ./deploy/systemd/rollback_native.sh --all

# Restore database
gunzip /opt/justnews/backups/justnews_20231201_020000.sql.gz
psql -U justnews_user -h localhost justnews < /opt/justnews/backups/justnews_20231201_020000.sql

# Restore model store
rsync -av /opt/justnews/backups/model_store/ /opt/justnews/model_store/

# Restart services
sudo ./deploy/systemd/enable_all.sh
```

## Scaling and High Availability

### Horizontal Scaling

#### Load Balancer Configuration

```nginx
# /etc/nginx/sites-available/justnews-lb
upstream justnews_cluster {
    server 192.168.1.10:8000;
    server 192.168.1.11:8000;
    server 192.168.1.12:8000;
}

server {
    listen 80;
    server_name cluster.justnews.com;

    location / {
        proxy_pass http://justnews_cluster;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Database Read Replicas

```bash
# Configure PostgreSQL streaming replication
sudo nano /etc/postgresql/14/main/postgresql.conf

# Master configuration
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64

# Replica configuration
hot_standby = on
```

### Vertical Scaling

#### GPU Scaling
```bash
# Multiple GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPU_MEMORY_LIMIT=12GB

# GPU affinity per agent
# Agent 1: GPU 0
export CUDA_VISIBLE_DEVICES=0
# Agent 2: GPU 1
export CUDA_VISIBLE_DEVICES=1
```

#### Memory Optimization
```bash
# Increase system memory limits
echo "adra soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "adra hard memlock unlimited" | sudo tee -a /etc/security/limits.conf

# Configure huge pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
```

## Troubleshooting Deployment

### Common Issues and Solutions

#### Service Startup Failures
```bash
# Check service status
sudo systemctl status justnews@mcp_bus

# View detailed logs
sudo journalctl -u justnews@mcp_bus -n 50

# Check environment files
sudo cat /etc/justnews/global.env
sudo cat /etc/justnews/mcp_bus.env
```

#### Port Conflicts
```bash
# Check port usage
sudo netstat -tlnp | grep :8000

# Kill conflicting processes
sudo fuser -k 8000/tcp

# Change port configuration
sudo nano /etc/justnews/mcp_bus.env
# MCP_BUS_PORT=8001
```

#### Database Connection Issues
```bash
# Test database connection
psql -U justnews_user -h localhost -d justnews

# Check PostgreSQL status
sudo systemctl status postgresql

# View PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

#### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Check CUDA installation
nvcc --version

# Test GPU with PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Check system memory
free -h

# Check process memory usage
ps aux --sort=-%mem | head

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Health Checks and Diagnostics

#### Automated Health Monitoring
```bash
# Run health checks
./deploy/systemd/health_check.sh

# Check all services
curl -s http://localhost:8000/health
curl -s http://localhost:8005/health
curl -s http://localhost:8002/health
```

#### Performance Diagnostics
```bash
# System performance
top -b -n 1

# Disk I/O
iotop -b -n 1

# Network connections
ss -tlnp

# GPU performance
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
conda activate justnews-v2-py312
pip install --upgrade -r requirements.txt

# Restart services
sudo systemctl restart justnews@*
```

#### Log Rotation
```bash
# Manual log rotation
sudo logrotate -f /etc/logrotate.d/justnews

# Check log sizes
du -sh /var/log/justnews/*
```

#### Database Maintenance
```bash
# Vacuum database
psql -U justnews_user -d justnews -c "VACUUM ANALYZE;"

# Reindex database
psql -U justnews_user -d justnews -c "REINDEX DATABASE justnews;"

# Check database size
psql -U justnews_user -d justnews -c "SELECT pg_size_pretty(pg_database_size('justnews'));"
```

### Emergency Procedures

#### Service Recovery
```bash
# Quick service restart
sudo systemctl restart justnews@mcp_bus

# Full system restart
sudo ./deploy/systemd/enable_all.sh --fresh

# Emergency stop
sudo ./deploy/systemd/rollback_native.sh --all --force
```

#### Data Recovery
```bash
# Restore from latest backup
LATEST_BACKUP=$(ls -t /opt/justnews/backups/*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | psql -U justnews_user -d justnews

# Point-in-time recovery (if WAL archiving enabled)
# Configure PostgreSQL for PITR recovery
```

## Security Hardening

### System Security

#### User and Permissions
```bash
# Create dedicated service user
sudo useradd -r -s /bin/false justnews

# Set proper permissions
sudo chown -R justnews:justnews /opt/justnews
sudo chmod -R 755 /opt/justnews
sudo chmod 600 /etc/justnews/*.env
```

#### Network Security
```bash
# Configure fail2ban
sudo apt install -y fail2ban

# Enable SSH hardening
sudo nano /etc/ssh/sshd_config
# PermitRootLogin no
# PasswordAuthentication no

sudo systemctl restart ssh
```

#### Application Security
```bash
# Enable security features in configuration
export LOG_LEVEL=WARNING
export DEBUG_MODE=false
export SECURITY_HEADERS=true
export RATE_LIMITING=true
```

## Performance Tuning

### System Optimization

#### Kernel Parameters
```bash
# Network optimization
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=1024

# Memory management
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.dirty_ratio=60
sudo sysctl -w vm.dirty_background_ratio=2
```

#### Service Optimization
```bash
# Increase file descriptors
echo "adra soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "adra hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize systemd
sudo mkdir -p /etc/systemd/system/justnews@.service.d
sudo nano /etc/systemd/system/justnews@.service.d/overrides.conf

[Service]
LimitNOFILE=65536
```

### Database Optimization

#### PostgreSQL Tuning
```sql
-- Performance configuration
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
```

#### Connection Pooling
```python
# Database connection pool configuration
pool = ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    host='localhost',
    database='justnews',
    user='justnews_user',
    password='password'
)
```

## Migration and Upgrades

### Version Upgrades

#### Application Upgrade
```bash
# Stop services
sudo ./deploy/systemd/rollback_native.sh --all

# Backup current version
cp -r /opt/justnews/JustNewsAgent /opt/justnews/JustNewsAgent.backup

# Update code
cd /opt/justnews
git pull origin main

# Update dependencies
conda activate justnews-v2-py312
pip install -r requirements.txt

# Run database migrations
python scripts/db_operations.py migrate

# Start services
sudo ./deploy/systemd/enable_all.sh
```

#### System Upgrade
```bash
# Update Ubuntu
sudo apt update && sudo apt upgrade -y

# Update NVIDIA drivers (if needed)
sudo apt install -y nvidia-driver-535

# Reboot if kernel updated
sudo reboot
```

### Rollback Procedures

#### Application Rollback
```bash
# Stop services
sudo ./deploy/systemd/rollback_native.sh --all

# Restore backup
rm -rf /opt/justnews/JustNewsAgent
cp -r /opt/justnews/JustNewsAgent.backup /opt/justnews/JustNewsAgent

# Start services
sudo ./deploy/systemd/enable_all.sh
```

#### Database Rollback
```bash
# Restore from backup
gunzip -c /opt/justnews/backups/justnews_20231201_020000.sql.gz | psql -U justnews_user -d justnews

# Rollback migrations (if using migration system)
python scripts/db_operations.py rollback
```

## Monitoring and Alerting

### Production Monitoring Setup

#### Prometheus Configuration
```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'justnews'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8005', 'localhost:8002']
    metrics_path: '/metrics'

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

#### Grafana Dashboards

Create dashboards for:
- Service health and uptime
- Response times and throughput
- Resource utilization (CPU, memory, disk, GPU)
- Error rates and logs
- Database performance
- Network traffic

#### Alert Rules
```yaml
# /etc/prometheus/alert_rules.yml
groups:
  - name: justnews
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "JustNews service is down"

      - alert: HighCPUUsage
        expr: cpu_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
```

## Compliance and Auditing

### Security Compliance

#### Data Protection
```json
{
  "data_minimization": {
    "enabled": true,
    "retention_days": {
      "articles": 365,
      "logs": 90,
      "metrics": 30
    },
    "anonymization": {
      "enabled": true,
      "fields": ["ip_address", "user_agent"]
    }
  }
}
```

#### Audit Logging
```python
# Enable comprehensive audit logging
import logging
from logging.handlers import RotatingFileHandler

audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    '/var/log/justnews/audit.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=5
)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
audit_logger.addHandler(handler)

# Log security events
audit_logger.info(f"User {user} accessed {resource}")
```

### Performance Auditing

#### Automated Performance Testing
```bash
# Performance benchmark script
#!/bin/bash
echo "Running JustNews performance audit..."

# Test API response times
ab -n 1000 -c 10 http://localhost:8000/health

# Test database performance
pgbench -U justnews_user -d justnews -c 10 -t 100

# Test GPU performance
python -c "
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

start = time.time()
for _ in range(100):
    z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f'GPU performance: {(end - start) * 1000:.2f}ms per operation')
"
```

---

*This comprehensive deployment documentation covers all aspects of deploying and managing JustNews V4 in production environments. For specific configuration examples and troubleshooting guides, refer to the individual deployment scripts and configuration files.*
