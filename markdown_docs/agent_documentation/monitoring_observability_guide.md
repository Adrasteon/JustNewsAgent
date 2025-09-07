# Monitoring and Observability Documentation

## Overview

The JustNews V4 system implements a comprehensive monitoring and observability framework designed for production environments. This documentation covers the centralized logging system, performance monitoring, health checks, metrics collection, and alerting mechanisms that ensure system reliability and operational visibility.

## Architecture Overview

### Monitoring Components

1. **Centralized Logging System** - Structured logging with file rotation and environment-specific configuration
2. **Health Monitoring** - Service health checks and readiness probes
3. **Performance Monitoring** - GPU, CPU, memory, and I/O metrics collection
4. **Metrics Collection** - Application and system metrics aggregation
5. **Alerting System** - Automated alerts for critical conditions
6. **Distributed Tracing** - Request tracing across agent boundaries

### Observability Stack

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Centralized   │───▶│   Log          │
│   Services      │    │   Logging       │    │   Aggregation  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Health        │    │   Metrics       │    │   Alerting      │
│   Checks        │    │   Collection    │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboards    │    │   Incident      │    │   Automated     │
│   &             │    │   Response      │    │   Remediation   │
│   Visualization │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Centralized Logging System

### Core Logging Architecture

The system uses a centralized logging framework with the following features:

- **Structured JSON Logging** - Machine-readable log format for production
- **Environment-Specific Configuration** - Different log levels and formats per environment
- **File Rotation** - Automatic log rotation with configurable retention
- **Multi-Level Logging** - Separate logs for different components and severity levels
- **Performance Logging** - Specialized logging for performance metrics

### Logger Configuration

#### Basic Setup
```python
from common.observability import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Log messages at different levels
logger.debug("Detailed debugging information")
logger.info("General information about operation")
logger.warning("Warning about potential issues")
logger.error("Error that doesn't stop operation")
logger.critical("Critical error requiring immediate attention")
```

#### Structured Logging
```python
# Add structured data to log entries
logger.info("Processing completed", extra={
    'request_id': request_id,
    'processing_time': 1.23,
    'articles_processed': 150,
    'method': 'gpu_accelerated'
})

# Performance logging
from common.observability import log_performance
log_performance("article_clustering", 2.34, articles_count=150, method="semantic")
```

### Log Configuration

#### Environment Variables
```bash
# Logging configuration
export LOG_LEVEL=INFO
export LOG_FORMAT=structured  # or 'readable'
export LOG_DIR=/var/log/justnews
export LOG_MAX_BYTES=10485760  # 10MB
export LOG_BACKUP_COUNT=5
```

#### Configuration File
```json
{
  "logging": {
    "level": "INFO",
    "format": "structured",
    "directory": "/var/log/justnews",
    "max_file_size": "10MB",
    "retention_days": 30,
    "compression": true
  }
}
```

### Log Files Structure

```
/var/log/justnews/
├── justnews.log              # Main application log
├── justnews_error.log        # Error-only log
├── scout.log                 # Scout agent logs
├── synthesizer.log           # Synthesizer agent logs
├── analyst.log               # Analyst agent logs
├── fact_checker.log          # Fact checker agent logs
├── critic.log                # Critic agent logs
├── chief_editor.log          # Chief editor agent logs
├── memory.log                # Memory agent logs
├── reasoning.log             # Reasoning agent logs
└── performance.log           # Performance metrics log
```

### Log Format Examples

#### Structured JSON Format (Production)
```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "INFO",
  "logger": "agents.synthesizer.main",
  "message": "GPU synthesis completed successfully",
  "module": "main",
  "function": "synthesize_articles",
  "line": 245,
  "extra_fields": {
    "request_id": "req_12345",
    "processing_time_ms": 1234.56,
    "articles_processed": 150,
    "gpu_memory_used_mb": 2048
  }
}
```

#### Readable Format (Development)
```
2024-01-15 10:30:45,123 - agents.synthesizer.main - INFO - GPU synthesis completed successfully
  request_id: req_12345
  processing_time: 1.23s
  articles_processed: 150
  gpu_memory_used: 2048MB
```

## Health Monitoring

### Health Check Endpoints

All agents implement standardized health check endpoints:

#### Basic Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45Z",
  "version": "4.0.0",
  "uptime_seconds": 3600
}
```

#### Readiness Check
```http
GET /ready
```

Response:
```json
{
  "ready": true,
  "dependencies": {
    "database": "connected",
    "mcp_bus": "connected",
    "gpu": "available"
  }
}
```

#### Detailed Health Check
```http
GET /health/detailed
```

Response:
```json
{
  "status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "latency_ms": 5.2,
      "connections_active": 3,
      "connections_idle": 7
    },
    "gpu": {
      "status": "healthy",
      "devices_available": 2,
      "memory_free_mb": 8192,
      "temperature_c": 65
    },
    "mcp_bus": {
      "status": "healthy",
      "agents_registered": 8,
      "message_queue_size": 0
    }
  }
}
```

### Health Check Implementation

```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time

router = APIRouter()

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            # Database health check logic
            latency = await self.measure_db_latency()
            return {
                "status": "healthy",
                "latency_ms": latency,
                "connections": self.get_connection_stats()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and health"""
        try:
            # GPU health check logic
            return {
                "status": "healthy",
                "devices": self.get_gpu_devices(),
                "memory": self.get_gpu_memory_stats()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        return {
            "status": "healthy",
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": time.time()
        }

health_checker = HealthChecker()

@router.get("/health")
async def health():
    """Basic health check"""
    health_data = health_checker.get_system_health()
    if health_data["status"] != "healthy":
        raise HTTPException(status_code=503, detail="Service unhealthy")
    return health_data

@router.get("/ready")
async def ready():
    """Readiness check"""
    # Check all dependencies
    db_health = await health_checker.check_database()
    gpu_health = await health_checker.check_gpu()

    ready = all([
        db_health["status"] == "healthy",
        gpu_health["status"] == "healthy"
    ])

    return {"ready": ready}
```

## Performance Monitoring

### GPU Monitoring

#### GPU Monitor Tool

The system includes a dedicated GPU monitoring tool:

```bash
# Start GPU monitoring
python tools/gpu_monitor.py --interval 2 --duration 300

# Monitor indefinitely
python tools/gpu_monitor.py --interval 5
```

#### GPU Metrics Collection

```python
import json
import time
from tools.gpu_monitor import sample_once

def collect_gpu_metrics():
    """Collect GPU metrics for monitoring"""
    metrics = sample_once()

    # Log metrics
    logger.info("GPU metrics collected", extra={
        'gpu_devices': len(metrics.get('nvidia_smi', [])),
        'gpu_utilization': metrics.get('nvidia_smi', [{}])[0].get('utilization_gpu_pct', 0),
        'gpu_memory_used': metrics.get('nvidia_smi', [{}])[0].get('memory_used_mb', 0),
        'gpu_temperature': metrics.get('nvidia_smi', [{}])[0].get('temperature_c', 0)
    })

    return metrics
```

#### GPU Health Alerts

```python
def check_gpu_health(metrics):
    """Check GPU health and generate alerts"""
    alerts = []

    for gpu in metrics.get('nvidia_smi', []):
        # Temperature check
        if gpu['temperature_c'] > 85:
            alerts.append({
                'level': 'critical',
                'message': f"GPU {gpu['index']} temperature too high: {gpu['temperature_c']}°C"
            })
        elif gpu['temperature_c'] > 75:
            alerts.append({
                'level': 'warning',
                'message': f"GPU {gpu['index']} temperature high: {gpu['temperature_c']}°C"
            })

        # Memory usage check
        memory_pct = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
        if memory_pct > 95:
            alerts.append({
                'level': 'critical',
                'message': f"GPU {gpu['index']} memory usage critical: {memory_pct:.1f}%"
            })
        elif memory_pct > 85:
            alerts.append({
                'level': 'warning',
                'message': f"GPU {gpu['index']} memory usage high: {memory_pct:.1f}%"
            })

    return alerts
```

### System Performance Monitoring

#### CPU and Memory Monitoring

```python
import psutil
import time

def collect_system_metrics():
    """Collect system performance metrics"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_percent': psutil.disk_usage('/').percent,
        'network_connections': len(psutil.net_connections()),
        'load_average': psutil.getloadavg()
    }

def monitor_system_performance(interval=60):
    """Monitor system performance continuously"""
    while True:
        metrics = collect_system_metrics()

        logger.info("System performance metrics", extra=metrics)

        # Check for alerts
        if metrics['cpu_percent'] > 90:
            logger.warning("High CPU usage detected", extra={'cpu_percent': metrics['cpu_percent']})

        if metrics['memory_percent'] > 85:
            logger.warning("High memory usage detected", extra={'memory_percent': metrics['memory_percent']})

        time.sleep(interval)
```

#### Database Performance Monitoring

```python
import asyncpg
from typing import Dict, Any

async def collect_database_metrics(pool) -> Dict[str, Any]:
    """Collect database performance metrics"""
    async with pool.acquire() as conn:
        # Active connections
        active_connections = await conn.fetchval("""
            SELECT count(*) FROM pg_stat_activity
            WHERE state = 'active'
        """)

        # Database size
        db_size = await conn.fetchval("""
            SELECT pg_size_pretty(pg_database_size(current_database()))
        """)

        # Slow queries
        slow_queries = await conn.fetch("""
            SELECT query, total_time, calls
            FROM pg_stat_statements
            WHERE total_time > 1000
            ORDER BY total_time DESC
            LIMIT 10
        """)

        return {
            'active_connections': active_connections,
            'database_size': db_size,
            'slow_queries_count': len(slow_queries),
            'slow_queries': [dict(q) for q in slow_queries]
        }
```

## Metrics Collection

### Application Metrics

#### Request Metrics

```python
from fastapi import Request, Response
import time

async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    # Log request metrics
    logger.info("Request processed", extra={
        'method': request.method,
        'url': str(request.url),
        'status_code': response.status_code,
        'process_time_ms': round(process_time * 1000, 2),
        'client_ip': request.client.host if request.client else None
    })

    return response
```

#### Business Metrics

```python
def collect_business_metrics():
    """Collect business-specific metrics"""
    return {
        'articles_processed_today': get_articles_processed_count(),
        'average_processing_time': get_average_processing_time(),
        'error_rate_percent': get_error_rate(),
        'gpu_utilization_average': get_gpu_utilization_average(),
        'memory_usage_peak': get_memory_usage_peak(),
        'active_users': get_active_user_count()
    }
```

### Prometheus Integration

#### Metrics Exposition

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
REQUEST_COUNT = Counter('justnews_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('justnews_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
GPU_MEMORY_USAGE = Gauge('justnews_gpu_memory_usage_mb', 'GPU memory usage in MB', ['gpu_id'])
ARTICLES_PROCESSED = Counter('justnews_articles_processed_total', 'Total articles processed')

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(time.time() - start_time)

    return response

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")
```

#### Prometheus Configuration

```yaml
# prometheus.yml
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

## Alerting System

### Alert Configuration

#### Alert Rules

```python
class AlertManager:
    def __init__(self):
        self.alerts = []

    def add_alert(self, alert_type: str, severity: str, message: str, **kwargs):
        """Add an alert"""
        alert = {
            'type': alert_type,
            'severity': severity,
            'message': message,
            'timestamp': time.time(),
            'metadata': kwargs
        }

        self.alerts.append(alert)

        # Log alert
        logger.warning(f"Alert triggered: {alert_type}", extra=alert)

        # Send notification if critical
        if severity == 'critical':
            self.send_notification(alert)

    def check_thresholds(self, metrics):
        """Check metrics against thresholds"""
        # CPU usage alert
        if metrics.get('cpu_percent', 0) > 90:
            self.add_alert(
                'high_cpu_usage',
                'warning',
                f"CPU usage is {metrics['cpu_percent']:.1f}%",
                cpu_percent=metrics['cpu_percent']
            )

        # Memory usage alert
        if metrics.get('memory_percent', 0) > 85:
            self.add_alert(
                'high_memory_usage',
                'warning',
                f"Memory usage is {metrics['memory_percent']:.1f}%",
                memory_percent=metrics['memory_percent']
            )

        # GPU temperature alert
        gpu_temp = metrics.get('gpu_temperature', 0)
        if gpu_temp > 85:
            self.add_alert(
                'gpu_overheating',
                'critical',
                f"GPU temperature is {gpu_temp}°C",
                temperature=gpu_temp
            )

    def send_notification(self, alert):
        """Send alert notification"""
        # Email notification
        send_email_alert(alert)

        # Slack notification
        send_slack_alert(alert)

        # PagerDuty notification for critical alerts
        if alert['severity'] == 'critical':
            trigger_pagerduty_alert(alert)
```

#### Alert Channels

```python
def send_email_alert(alert):
    """Send email alert"""
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(f"""
    Alert: {alert['type']}
    Severity: {alert['severity']}
    Message: {alert['message']}
    Time: {time.ctime(alert['timestamp'])}
    """)

    msg['Subject'] = f"JustNews Alert: {alert['type']}"
    msg['From'] = 'alerts@justnews.com'
    msg['To'] = 'ops@justnews.com'

    # Send email logic here
    pass

def send_slack_alert(alert):
    """Send Slack alert"""
    import requests

    payload = {
        "text": f"🚨 *JustNews Alert*\n*{alert['type']}*\n{alert['message']}\nSeverity: {alert['severity']}"
    }

    # Send to Slack webhook
    requests.post(SLACK_WEBHOOK_URL, json=payload)

def trigger_pagerduty_alert(alert):
    """Trigger PagerDuty alert"""
    import requests

    payload = {
        "routing_key": PAGERDUTY_ROUTING_KEY,
        "event_action": "trigger",
        "payload": {
            "summary": alert['message'],
            "severity": alert['severity'],
            "source": "justnews-monitoring"
        }
    }

    requests.post("https://events.pagerduty.com/v2/enqueue", json=payload)
```

## Distributed Tracing

### Request Tracing Implementation

```python
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for request ID
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class TracingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Generate or extract request ID
        request_id = self.get_or_create_request_id(scope)

        # Set in context
        request_id_context.set(request_id)

        # Add to response headers
        async def send_with_trace(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append([b"x-request-id", request_id.encode()])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)

    def get_or_create_request_id(self, scope) -> str:
        """Get request ID from headers or create new one"""
        headers = dict(scope.get("headers", []))
        request_id_header = headers.get(b"x-request-id")

        if request_id_header:
            return request_id_header.decode()

        return str(uuid.uuid4())

def get_current_request_id() -> Optional[str]:
    """Get current request ID from context"""
    return request_id_context.get()

def log_with_request_id(message: str, **kwargs):
    """Log message with current request ID"""
    request_id = get_current_request_id()
    if request_id:
        logger.info(message, extra={'request_id': request_id, **kwargs})
    else:
        logger.info(message, extra=kwargs)
```

### Cross-Agent Tracing

```python
import requests

def make_traced_request(url: str, method: str = "GET", **kwargs):
    """Make HTTP request with tracing headers"""
    request_id = get_current_request_id()

    headers = kwargs.get('headers', {})
    if request_id:
        headers['x-request-id'] = request_id

    kwargs['headers'] = headers

    # Log outgoing request
    log_with_request_id(f"Making {method} request to {url}")

    response = requests.request(method, url, **kwargs)

    # Log response
    log_with_request_id(
        f"Received response from {url}",
        status_code=response.status_code,
        response_time=response.elapsed.total_seconds()
    )

    return response
```

## Monitoring Dashboards

### Grafana Dashboard Configuration

#### System Overview Dashboard

```json
{
  "dashboard": {
    "title": "JustNews System Overview",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU Usage %"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_percent",
            "legendFormat": "Memory Usage %"
          }
        ]
      },
      {
        "title": "GPU Temperature",
        "type": "graph",
        "targets": [
          {
            "expr": "gpu_temperature_celsius",
            "legendFormat": "GPU {{gpu_id}} Temperature °C"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(justnews_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      }
    ]
  }
}
```

#### Agent-Specific Dashboard

```json
{
  "dashboard": {
    "title": "Synthesizer Agent Metrics",
    "panels": [
      {
        "title": "Articles Processed",
        "type": "stat",
        "targets": [
          {
            "expr": "justnews_articles_processed_total",
            "legendFormat": "Articles Processed"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(justnews_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile processing time"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "justnews_gpu_memory_usage_mb",
            "legendFormat": "GPU Memory Usage MB"
          }
        ]
      }
    ]
  }
}
```

## Log Analysis and Search

### Log Aggregation

#### ELK Stack Integration

```python
from elasticsearch import Elasticsearch
import json

class LogAggregator:
    def __init__(self, es_host: str = "localhost:9200"):
        self.es = Elasticsearch([es_host])

    def index_log_entry(self, log_entry: dict):
        """Index log entry in Elasticsearch"""
        self.es.index(
            index="justnews-logs",
            document=log_entry
        )

    def search_logs(self, query: str, size: int = 100):
        """Search logs using Elasticsearch query"""
        return self.es.search(
            index="justnews-logs",
            query={"query_string": {"query": query}},
            size=size
        )
```

#### Logstash Configuration

```conf
# logstash.conf
input {
  file {
    path => "/var/log/justnews/*.log"
    start_position => "beginning"
    sincedb_path => "/var/lib/logstash/sincedb"
  }
}

filter {
  json {
    source => "message"
  }

  date {
    match => ["timestamp", "ISO8601"]
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "justnews-logs-%{+YYYY.MM.dd}"
  }
}
```

### Log Analysis Queries

```python
# Search for errors in last hour
error_query = {
    "query": {
        "bool": {
            "must": [
                {"term": {"level": "ERROR"}},
                {"range": {"timestamp": {"gte": "now-1h"}}}
            ]
        }
    }
}

# Find slow requests
slow_query = {
    "query": {
        "bool": {
            "must": [
                {"range": {"extra_fields.processing_time_ms": {"gte": 5000}}},
                {"range": {"timestamp": {"gte": "now-24h"}}}
            ]
        }
    }
}

# GPU performance analysis
gpu_query = {
    "query": {
        "bool": {
            "must": [
                {"exists": {"field": "extra_fields.gpu_utilization"}},
                {"range": {"timestamp": {"gte": "now-1h"}}}
            ]
        }
    },
    "aggs": {
        "avg_gpu_utilization": {
            "avg": {"field": "extra_fields.gpu_utilization"}
        }
    }
}
```

## Automated Remediation

### Self-Healing Mechanisms

```python
class AutoRemediation:
    def __init__(self):
        self.remediation_actions = {
            'high_memory_usage': self.handle_high_memory,
            'gpu_overheating': self.handle_gpu_overheating,
            'service_down': self.handle_service_down
        }

    async def handle_high_memory(self, alert):
        """Handle high memory usage"""
        # Try garbage collection first
        import gc
        gc.collect()

        # Check memory after GC
        memory_after = psutil.virtual_memory().percent

        if memory_after > 90:
            # Restart problematic service
            await self.restart_service(alert.get('service_name'))

    async def handle_gpu_overheating(self, alert):
        """Handle GPU overheating"""
        gpu_id = alert.get('gpu_id', 0)

        # Reduce GPU workload
        await self.throttle_gpu_workload(gpu_id)

        # Check temperature after throttling
        temperature = await self.get_gpu_temperature(gpu_id)

        if temperature > 90:
            # Emergency shutdown
            await self.emergency_gpu_shutdown(gpu_id)

    async def handle_service_down(self, alert):
        """Handle service down situation"""
        service_name = alert.get('service_name')

        # Attempt restart
        success = await self.restart_service(service_name)

        if not success:
            # Escalate to human operators
            await self.escalate_to_ops(alert)

    async def restart_service(self, service_name):
        """Restart a systemd service"""
        process = await asyncio.create_subprocess_exec(
            'sudo', 'systemctl', 'restart', f'justnews@{service_name}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.wait()
        return process.returncode == 0
```

## Performance Profiling

### Application Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_function(func):
    """Decorator to profile function execution"""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        logger.info("Function profile", extra={
            'function': func.__name__,
            'profile': s.getvalue()
        })

        return result
    return wrapper

# Usage
@profile_function
def process_articles(articles):
    # Article processing logic
    pass
```

### Memory Profiling

```python
from memory_profiler import profile
import tracemalloc

@profile
def memory_intensive_function():
    """Function with memory profiling"""
    # Memory intensive operations
    pass

def profile_memory():
    """Profile memory usage"""
    tracemalloc.start()

    # Run operations
    result = memory_intensive_function()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info("Memory profile", extra={
        'current_memory_mb': current / 1024 / 1024,
        'peak_memory_mb': peak / 1024 / 1024
    })

    return result
```

## Compliance and Security Monitoring

### Audit Logging

```python
class AuditLogger:
    def __init__(self):
        self.audit_log = logging.getLogger('audit')
        self.audit_log.setLevel(logging.INFO)

        handler = logging.handlers.RotatingFileHandler(
            '/var/log/justnews/audit.log',
            maxBytes=100*1024*1024,
            backupCount=12
        )

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_log.addHandler(handler)

    def log_access(self, user: str, resource: str, action: str, **kwargs):
        """Log access to resources"""
        self.audit_log.info(f"Access: {user} {action} {resource}", extra=kwargs)

    def log_security_event(self, event_type: str, details: dict):
        """Log security events"""
        self.audit_log.warning(f"Security: {event_type}", extra=details)

    def log_data_access(self, user: str, data_type: str, record_count: int):
        """Log data access events"""
        self.audit_log.info(f"Data access: {user} accessed {record_count} {data_type} records")

# Global audit logger
audit_logger = AuditLogger()
```

### Compliance Monitoring

```python
def check_compliance():
    """Check system compliance with requirements"""
    compliance_checks = {
        'data_retention': check_data_retention_compliance(),
        'access_controls': check_access_control_compliance(),
        'encryption': check_encryption_compliance(),
        'audit_logging': check_audit_logging_compliance()
    }

    for check_name, result in compliance_checks.items():
        if not result['compliant']:
            logger.warning(f"Compliance violation: {check_name}", extra=result)

    return compliance_checks
```

## Troubleshooting Guide

### Common Monitoring Issues

#### Logs Not Appearing
```bash
# Check log directory permissions
ls -la /var/log/justnews/

# Check logger configuration
python -c "from common.observability import get_logger; print(get_logger('test').handlers)"

# Verify log file creation
touch /var/log/justnews/test.log
```

#### Metrics Not Collecting
```bash
# Check Prometheus endpoint
curl http://localhost:8000/metrics

# Verify metrics configuration
python -c "from prometheus_client import generate_latest; print(generate_latest().decode())"

# Check Prometheus configuration
cat /etc/prometheus/prometheus.yml
```

#### Alerts Not Triggering
```bash
# Check alert rules
cat /etc/prometheus/alert_rules.yml

# Verify alert manager configuration
cat /etc/prometheus/alertmanager.yml

# Check alert manager logs
sudo journalctl -u alertmanager -f
```

#### GPU Monitoring Issues
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify GPU monitor permissions
python tools/gpu_monitor.py --interval 1 --duration 5

# Check GPU monitoring logs
tail -f logs/gpu_monitor.jsonl
```

### Performance Troubleshooting

#### High CPU Usage
```bash
# Find high CPU processes
ps aux --sort=-%cpu | head

# Profile Python application
python -m cProfile -s cumulative your_script.py

# Check system load
uptime
cat /proc/loadavg
```

#### Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Profile memory usage
python -c "import tracemalloc; tracemalloc.start(); # your code; print(tracemalloc.get_traced_memory())"

# Check for memory leaks
python -c "import gc; gc.set_debug(gc.DEBUG_LEAK); # your code"
```

#### Database Performance
```bash
# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Find slow queries
psql -c "SELECT query, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check database locks
psql -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

### Network Troubleshooting

```bash
# Check network connections
netstat -tlnp | grep :8000

# Test service connectivity
curl -v http://localhost:8000/health

# Check firewall rules
sudo ufw status
sudo iptables -L
```

---

*This comprehensive monitoring and observability documentation covers all aspects of the JustNews V4 monitoring system. For specific implementation details, refer to the individual monitoring components and configuration files.*
