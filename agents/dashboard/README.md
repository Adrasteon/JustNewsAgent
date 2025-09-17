# JustNews Dashboard Agent

Web interface for controlling crawls and monitoring system metrics.

## Features

- **Crawl Control**: Start and monitor news crawling jobs with configurable parameters
- **Metrics Dashboards**: Real-time charts for crawler, analyst, and memory performance
- **System Health**: Monitor the status of all agents in the system
- **GPU Monitoring**: Integrated GPU usage tracking and management

## Quick Start

### Main Dashboard (Port 8013)
1. **Start the Dashboard Agent**:
   ```bash
   cd agents/dashboard
   ./start_dashboard.sh
   ```

2. **Access the Web Interface**:
   Open your browser to `http://localhost:8013`

### Crawler Control Web Interface (Port 8016)
1. **Start Crawler Control**:
   ```bash
   cd agents/dashboard/web_interface
   ./start_crawler_control.sh
   ```

2. **Access the Web Interface**:
   Open your browser to `http://localhost:8016`

## Configuration

### Main Dashboard Environment Variables:
- `DASHBOARD_HOST`: Host to bind to (default: 0.0.0.0)
- `DASHBOARD_PORT`: Port to listen on (default: 8013)
- `MCP_BUS_URL`: MCP bus URL (default: http://localhost:8000)

### Crawler Control Environment Variables:
- `CRAWLER_CONTROL_HOST`: Host to bind to (default: 0.0.0.0)
- `CRAWLER_CONTROL_PORT`: Port to listen on (default: 8016)
- `CRAWLER_AGENT_URL`: Crawler agent URL (default: http://localhost:8015)
- `ANALYST_AGENT_URL`: Analyst agent URL (default: http://localhost:8004)
- `MEMORY_AGENT_URL`: Memory agent URL (default: http://localhost:8007)
- `MCP_BUS_URL`: MCP bus URL (default: http://localhost:8000)

## API Endpoints

### Crawler Control
- `POST /api/crawl/start`: Start a new crawl job
- `GET /api/crawl/status`: Get current crawl job statuses

### Metrics
- `GET /api/metrics/crawler`: Get crawler performance metrics
- `GET /api/metrics/analyst`: Get analyst metrics
- `GET /api/metrics/memory`: Get memory usage metrics

### System Health
- `GET /api/health`: Get overall system health status

### GPU Monitoring (inherited from main dashboard)
- `GET /gpu/info`: Current GPU information
- `GET /gpu/history`: GPU usage history
- `GET /gpu/agents`: Per-agent GPU usage
- And many more GPU-related endpoints...

## Crawl Configuration Options

When starting a crawl, you can configure:

- **Domains**: Comma-separated list of domains to crawl
- **Max Sites**: Maximum number of sites to process (default: 5)
- **Max Articles per Site**: Articles to extract per site (default: 10)
- **Concurrent Sites**: Number of sites to crawl simultaneously (default: 3)
- **Strategy**: Crawling strategy - "auto", "ultra_fast", "ai_enhanced", "generic"
- **Enable AI**: Whether to use AI analysis during crawling (default: true)
- **Timeout**: Request timeout in seconds (default: 300)
- **User Agent**: Custom user agent string

## Dependencies

- fastapi
- uvicorn
- requests
- pydantic

Install with:
```bash
pip install -r web_interface/requirements.txt
```

## Integration

The dashboard integrates with the MCP bus to communicate with other agents:
- **Crawler Agent**: For starting and monitoring crawls
- **Analyst Agent**: For AI analysis metrics
- **Memory Agent**: For storage and retrieval metrics
- **GPU Manager**: For GPU monitoring and allocation

## Development

The web interface is located in `web_interface/index.html` and uses:
- Bootstrap 5 for styling
- Chart.js for metrics visualization
- Vanilla JavaScript for API calls

To modify the interface, edit `web_interface/index.html` and restart the dashboard agent.
