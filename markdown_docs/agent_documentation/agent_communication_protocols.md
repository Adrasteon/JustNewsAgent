# Agent Communication Protocols Documentation

## Overview

The JustNews V4 system implements a sophisticated multi-agent communication architecture centered around the MCP (Message Control Protocol) Bus. This document provides comprehensive documentation of the communication protocols, API patterns, and inter-agent interaction mechanisms that enable seamless coordination between specialized agents.

## Architecture Overview

### Core Components

1. **MCP Bus** - Central communication hub and service registry
2. **Agent Registry** - Dynamic agent discovery and registration system
3. **Circuit Breaker Pattern** - Fault tolerance and resilience mechanisms
4. **Tool Call Protocol** - Standardized inter-agent function invocation
5. **Health Monitoring** - Distributed system health and readiness checks

### Communication Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│   MCP Bus   │───▶│   Agent     │
│             │    │             │    │             │
│  REST API   │◀───│  Tool Call  │◀───│ FastAPI App │
└─────────────┘    └─────────────┘    └─────────────┘
```

## MCP Bus Architecture

### Core Functionality

The MCP Bus serves as the central nervous system for the JustNews agent ecosystem:

- **Agent Registration**: Dynamic registration of agents and their capabilities
- **Tool Routing**: Intelligent routing of tool calls to appropriate agents
- **Circuit Breaker**: Automatic failure detection and recovery
- **Health Monitoring**: Real-time status tracking of all registered agents

### API Endpoints

#### Agent Registration
```http
POST /register
Content-Type: application/json

{
  "name": "synthesizer",
  "address": "http://localhost:8005"
}
```

#### Tool Invocation
```http
POST /call
Content-Type: application/json

{
  "agent": "synthesizer",
  "tool": "cluster_articles",
  "args": [["article1", "article2"]],
  "kwargs": {"method": "semantic"}
}
```

#### Agent Discovery
```http
GET /agents
```

Returns registered agents and their addresses.

### Circuit Breaker Implementation

The MCP Bus implements a sophisticated circuit breaker pattern:

- **Failure Threshold**: 3 consecutive failures trigger circuit opening
- **Cooldown Period**: 10 seconds before attempting recovery
- **Automatic Recovery**: Gradual restoration of service after cooldown
- **State Tracking**: Per-agent failure counting and recovery monitoring

## Agent Communication Patterns

### Standard Agent Structure

All agents follow a consistent FastAPI-based architecture:

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

class MCPBusClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        # Registration logic

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Agent startup and MCP Bus registration
    mcp_client = MCPBusClient()
    mcp_client.register_agent(
        agent_name="agent_name",
        agent_address=f"http://localhost:{PORT}",
        tools=["tool1", "tool2", "tool3"]
    )
    yield
    # Cleanup logic

app = FastAPI(lifespan=lifespan)
```

### Dual API Pattern

Agents implement both MCP Bus compatible and direct REST API endpoints:

#### MCP Bus Compatible Endpoints
```python
class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/cluster_articles")
def cluster_articles_endpoint(call: ToolCall):
    from .tools import cluster_articles
    return cluster_articles(*call.args, **call.kwargs)
```

#### Direct REST API Endpoints
```python
class ArticleRequest(BaseModel):
    articles: List[str]
    method: str = "semantic"

@app.post("/api/cluster")
def cluster_articles_api(request: ArticleRequest):
    from .tools import cluster_articles
    return cluster_articles(request.articles, method=request.method)
```

## Agent-Specific Communication Protocols

### Synthesizer Agent (Port 8005)

**Registered Tools:**
- `cluster_articles` - Semantic article clustering
- `neutralize_text` - Bias neutralization
- `aggregate_cluster` - Cluster aggregation
- `synthesize_news_articles_gpu` - GPU-accelerated synthesis
- `get_synthesizer_performance` - Performance monitoring

**Key Endpoints:**
```http
POST /synthesize_news_articles_gpu
POST /cluster_articles
POST /neutralize_text
POST /aggregate_cluster
GET /health
GET /ready
```

### NewsReader Agent (Port 8009)

**Registered Tools:**
- `extract_news_content` - Multi-modal content extraction
- `capture_screenshot` - Webpage screenshot capture
- `analyze_screenshot` - LLaVA-based image analysis
- `analyze_content` - Content structure analysis
- `extract_structure` - Document structure parsing
- `extract_multimedia` - Multimedia content extraction

**Key Endpoints:**
```http
POST /extract_news
POST /analyze_content
POST /capture_screenshot
POST /analyze_image
POST /extract_structure
POST /extract_multimedia
```

### Scout Agent (Port 8002)

**Registered Tools:**
- `discover_sources` - News source discovery
- `crawl_url` - Single URL crawling
- `deep_crawl_site` - Comprehensive site crawling
- `enhanced_deep_crawl_site` - AI-enhanced crawling
- `intelligent_source_discovery` - ML-powered source finding
- `intelligent_content_crawl` - Smart content extraction
- `intelligent_batch_analysis` - Batch processing
- `enhanced_newsreader_crawl` - Integrated news extraction
- `production_crawl_ultra_fast` - High-performance crawling
- `get_production_crawler_info` - Crawler status and metrics

**Key Endpoints:**
```http
POST /crawl_url
POST /deep_crawl_site
POST /enhanced_deep_crawl_site
POST /production_crawl_ultra_fast
POST /intelligent_source_discovery
```

## Security and Middleware

### Common Security Patterns

All agents implement consistent security middleware:

```python
# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # Rate limiting
    if not rate_limit(request):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Input validation
    if SECURITY_AVAILABLE:
        validate_input(request)

    # Logging
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)
    return response
```

### Input Validation

- **URL Validation**: Domain whitelisting and pattern matching
- **Content Sanitization**: XSS prevention and content filtering
- **Rate Limiting**: Per-IP and per-endpoint request throttling
- **Request Size Limits**: Prevention of oversized payloads

### Authentication and Authorization

- **API Key Authentication**: For external service access
- **Internal Token Validation**: For inter-agent communication
- **Role-Based Access**: Different permission levels for different operations

## Error Handling and Resilience

### Standardized Error Responses

All agents follow consistent error response patterns:

```json
{
  "error": "Detailed error message",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_12345"
}
```

### Graceful Degradation

- **Fallback Mechanisms**: CPU fallbacks for GPU operations
- **Circuit Breaker Integration**: Automatic failure isolation
- **Retry Logic**: Exponential backoff for transient failures
- **Timeout Handling**: Configurable timeouts with sensible defaults

### Monitoring and Observability

- **Health Endpoints**: `/health` and `/ready` for load balancer integration
- **Performance Metrics**: Response times, throughput, error rates
- **Logging Integration**: Structured logging with correlation IDs
- **Distributed Tracing**: Request tracing across agent boundaries

## Configuration Management

### Environment Variables

Standard environment variable patterns across all agents:

```bash
# Agent-specific configuration
AGENT_NAME_PORT=8005
AGENT_NAME_LOG_LEVEL=INFO

# MCP Bus configuration
MCP_BUS_URL=http://localhost:8000

# Security configuration
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Performance tuning
MAX_WORKERS=4
TIMEOUT_SECONDS=30
```

### Dynamic Configuration

- **Runtime Reconfiguration**: Hot reloading of certain settings
- **Environment-Specific Configs**: Different settings for dev/staging/prod
- **Feature Flags**: Enable/disable features without restart
- **Resource Limits**: Dynamic adjustment based on system load

## Deployment and Scaling

### Container Orchestration

```yaml
# Docker Compose example
version: '3.8'
services:
  mcp-bus:
    image: justnews/mcp-bus:latest
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO

  synthesizer:
    image: justnews/synthesizer:latest
    ports:
      - "8005:8005"
    environment:
      - MCP_BUS_URL=http://mcp-bus:8000
    depends_on:
      - mcp-bus
```

### Load Balancing

- **Round Robin**: Basic load distribution
- **Least Connections**: Intelligent routing based on current load
- **Health-Check Based**: Automatic removal of unhealthy instances
- **Geographic Distribution**: Multi-region deployment support

### Auto-Scaling

- **CPU/Memory Based**: Horizontal scaling triggers
- **Queue Depth Monitoring**: Scale based on request backlog
- **Predictive Scaling**: ML-based scaling predictions
- **Cooldown Periods**: Prevent scaling thrashing

## Performance Optimization

### Connection Pooling

```python
# Database connection pooling
from psycopg2.pool import ThreadedConnectionPool

pool = ThreadedConnectionPool(
    minconn=2,
    maxconn=20,
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
```

### Caching Strategies

- **Redis Integration**: Distributed caching for shared data
- **In-Memory Caching**: Local caching for frequently accessed data
- **Cache Invalidation**: TTL-based and event-driven cache clearing
- **Cache Warming**: Pre-population of critical cache entries

### Async Processing

```python
@app.post("/process_async")
async def process_async_endpoint(request: Request):
    # Queue async task
    task_id = await task_queue.add_task(process_data, request.data)

    return {"task_id": task_id, "status": "queued"}
```

## Monitoring and Alerting

### Key Metrics

- **Response Times**: P95, P99 latency measurements
- **Error Rates**: Per-endpoint error percentages
- **Throughput**: Requests per second
- **Resource Usage**: CPU, memory, disk, network
- **Circuit Breaker Status**: Open/closed state tracking

### Alert Conditions

- **High Error Rate**: >5% errors over 5-minute window
- **High Latency**: P99 > 2 seconds
- **Circuit Breaker Open**: Automatic notification
- **Resource Exhaustion**: >90% resource utilization

### Logging Integration

```python
# Structured logging
logger.info("Processing completed", extra={
    "request_id": request_id,
    "processing_time": processing_time,
    "articles_processed": len(articles),
    "method": "gpu_accelerated"
})
```

## Troubleshooting Guide

### Common Issues

#### Agent Registration Failures
```bash
# Check MCP Bus connectivity
curl http://localhost:8000/health

# Verify agent configuration
echo $MCP_BUS_URL

# Check agent logs
tail -f /var/log/justnews/agent.log
```

#### Circuit Breaker Trips
```bash
# Check agent health
curl http://localhost:8005/health

# Monitor circuit breaker state
curl http://localhost:8000/agents

# Reset circuit breaker (if needed)
# Automatic recovery after cooldown period
```

#### Performance Degradation
```bash
# Check system resources
top -p $(pgrep -f "uvicorn.*main:app")

# Monitor GPU usage (if applicable)
nvidia-smi

# Check database connections
psql -c "SELECT count(*) FROM pg_stat_activity;"
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export LOG_LEVEL=DEBUG
export AGENT_DEBUG=true
```

## Migration and Compatibility

### Version Compatibility

- **API Versioning**: Semantic versioning for all endpoints
- **Backward Compatibility**: Graceful handling of deprecated features
- **Migration Scripts**: Automated upgrade procedures
- **Rollback Procedures**: Safe rollback mechanisms

### Legacy System Integration

- **Adapter Patterns**: Wrappers for legacy system integration
- **Protocol Translation**: Convert between different communication protocols
- **Data Format Conversion**: Handle different data serialization formats
- **Gradual Migration**: Phased rollout with feature flags

## Future Enhancements

### Planned Improvements

1. **gRPC Integration**: High-performance binary protocol support
2. **Event-Driven Architecture**: Asynchronous event processing
3. **Service Mesh**: Advanced service discovery and routing
4. **Distributed Tracing**: End-to-end request tracing
5. **Auto-Discovery**: Zero-configuration service registration

### Research Areas

- **Machine Learning Integration**: AI-powered routing decisions
- **Quantum-Safe Cryptography**: Future-proof security
- **Edge Computing**: Distributed agent deployment
- **Blockchain Integration**: Decentralized trust mechanisms

---

*This documentation covers the comprehensive communication protocols implemented in JustNews V4. For specific agent implementations, refer to individual agent documentation files.*
