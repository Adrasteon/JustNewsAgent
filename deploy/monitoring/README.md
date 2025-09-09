# JustNews Advanced Monitoring Setup

## Overview

This directory contains the complete monitoring infrastructure for JustNews V4, featuring:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and notifications
- **Node Exporter**: System metrics
- **cAdvisor**: Container metrics

## Quick Start

### 1. Prerequisites

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Start Monitoring Stack

```bash
cd deploy/monitoring
./manage-monitoring.sh start
```

### 3. Access Monitoring Interfaces

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

## Architecture

```
JustNews Agents
    ├── /metrics endpoints (Prometheus format)
    └── Custom business metrics

Prometheus Server
    ├── Scrapes metrics from all agents
    ├── Stores time-series data
    └── Evaluates alert rules

Grafana
    ├── Visualizes metrics from Prometheus
    ├── Custom dashboards for JustNews
    └── Alert integration

AlertManager
    ├── Receives alerts from Prometheus
    ├── Routes alerts to appropriate channels
    └── Handles alert grouping and silencing
```

## Configuration Files

### Prometheus Configuration
- `prometheus.yml`: Main Prometheus configuration with all JustNews agents
- `alert_rules.yml`: Alert rules for critical events and warnings

### Grafana Configuration
- `grafana/provisioning/datasources/datasources.yml`: Data source configuration
- `grafana/provisioning/dashboards/dashboards.yml`: Dashboard provisioning
- `grafana/dashboards/justnews-system-overview.json`: Main system dashboard

### AlertManager Configuration
- `alertmanager.yml`: Alert routing and notification configuration

## Available Metrics

### Standard HTTP Metrics
- `justnews_requests_total`: Total requests by method, endpoint, status
- `justnews_request_duration_seconds`: Request duration histograms
- `justnews_active_connections`: Current active connections

### System Metrics
- `justnews_memory_usage_bytes`: Memory usage (RSS/VMS)
- `justnews_cpu_usage_percent`: CPU utilization percentage
- `justnews_gpu_memory_used_bytes`: GPU memory usage
- `justnews_gpu_utilization_percent`: GPU utilization percentage

### Business Metrics
- `justnews_processing_queue_size`: Processing queue depth
- `justnews_quality_score`: Content quality scores
- `justnews_errors_total`: Error counts by type

## Dashboards

### System Overview Dashboard
- Service health status
- Active alerts summary
- Request rates by service
- Error rates and response times
- Resource utilization (CPU, Memory, GPU)
- Processing queue depths
- Quality score trends

## Alert Rules

### Critical Alerts
- Service down/unhealthy
- High error rates (>5%)
- Critical queue depth (>1000)
- Resource exhaustion

### Warning Alerts
- High CPU/memory usage
- Slow response times
- Low quality scores
- Performance degradation

### Business Alerts
- Content processing stalled
- High rejection rates
- Source reliability issues

## Management Commands

```bash
# Start monitoring stack
./manage-monitoring.sh start

# Stop monitoring stack
./manage-monitoring.sh stop

# Restart monitoring stack
./manage-monitoring.sh restart

# Show status
./manage-monitoring.sh status

# View logs for specific service
./manage-monitoring.sh logs grafana

# Clean up everything (removes data)
./manage-monitoring.sh cleanup
```

## Agent Integration

### 1. Add Metrics to Agent

```python
from common.metrics import JustNewsMetrics

# Initialize metrics in your FastAPI app
metrics = JustNewsMetrics("your_agent_name")

# Add middleware for automatic request metrics
app.middleware("http")(metrics.request_middleware)

# Add /metrics endpoint
@app.get("/metrics")
def get_metrics():
    return Response(
        content=metrics.get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )
```

### 2. Record Custom Metrics

```python
# Record processing time
with metrics.measure_time("content_processing"):
    # Your processing logic here
    pass

# Record quality score
metrics.record_quality_score("content_quality", 0.85)

# Update queue size
metrics.update_queue_size("processing_queue", current_size)

# Record errors
metrics.record_error("processing_failed", "/process")
```

### 3. Update System Metrics

```python
# Update system metrics periodically
@app.on_event("startup")
async def startup_event():
    # Update system metrics every 30 seconds
    while True:
        metrics.update_system_metrics()
        await asyncio.sleep(30)
```

## Production Deployment

### 1. Environment Variables

```bash
# Grafana
GF_SECURITY_ADMIN_PASSWORD=your_secure_password
GF_USERS_ALLOW_SIGN_UP=false

# AlertManager SMTP
SMTP_USER=alerts@yourdomain.com
SMTP_PASSWORD=your_smtp_password
```

### 2. Persistent Storage

```bash
# Create persistent volumes
docker volume create prometheus_data
docker volume create grafana_data
docker volume create alertmanager_data
```

### 3. Security

```bash
# Use reverse proxy for production
# Configure TLS certificates
# Set up authentication
# Restrict network access
```

## Troubleshooting

### Common Issues

#### Grafana Not Accessible
```bash
# Check if Grafana is running
./manage-monitoring.sh status

# View Grafana logs
./manage-monitoring.sh logs grafana

# Restart Grafana
docker-compose restart grafana
```

#### Prometheus Not Scraping Metrics
```bash
# Check Prometheus targets
curl http://localhost:9090/targets

# Verify agent /metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus configuration
docker-compose logs prometheus
```

#### Alerts Not Working
```bash
# Check AlertManager status
curl http://localhost:9093/-/healthy

# View alert rules
curl http://localhost:9090/alerts

# Check AlertManager logs
./manage-monitoring.sh logs alertmanager
```

## Performance Tuning

### Prometheus
- **Retention**: 200h (configurable in docker-compose.yml)
- **Scrape Interval**: 15s (configurable per target)
- **Storage**: SSD recommended for better performance

### Grafana
- **Query Timeout**: 60s for complex queries
- **Dashboard Refresh**: 30s default
- **Panel Limits**: Monitor query performance

### Resource Requirements
- **Prometheus**: 2GB RAM, 50GB storage
- **Grafana**: 1GB RAM, 10GB storage
- **AlertManager**: 512MB RAM, 10GB storage

## Backup and Recovery

### Automated Backups
```bash
# Backup monitoring data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus_backup.tar.gz -C /data .

# Backup Grafana dashboards
docker run --rm -v grafana_data:/data -v $(pwd):/backup alpine tar czf /backup/grafana_backup.tar.gz -C /data .
```

### Recovery
```bash
# Restore from backup
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine tar xzf /backup/prometheus_backup.tar.gz -C /data
```

## Integration with JustNews Agents

### Automatic Metrics Registration
The monitoring system is designed to automatically discover and monitor all JustNews agents that expose `/metrics` endpoints. No additional configuration is required when adding new agents.

### Service Discovery
Prometheus is configured to scrape metrics from all standard JustNews agent ports (8000-8013). New agents will be automatically discovered if they follow the standard port allocation.

## Support

For issues with the monitoring setup:

1. Check the logs: `./manage-monitoring.sh logs <service>`
2. Verify configuration files
3. Ensure Docker networking is working
4. Check firewall rules and port availability

## Roadmap

### Phase 5.2: Enhanced Dashboards
- Agent-specific dashboards
- Performance analytics
- Business intelligence metrics

### Phase 5.3: Advanced Alerting
- Machine learning-based anomaly detection
- Predictive alerting
- Automated incident response

### Phase 5.4: High Availability
- Prometheus federation
- Grafana clustering
- AlertManager clustering
