# Crawler Agent

The **Crawler Agent** is a specialized JustNewsAgent service that provides unified production crawling capabilities. It runs as an independent systemd service and communicates bi-directionally with other agents via the MCP Bus.

## Overview

The Crawler Agent implements the unified production crawler that combines:
- **🎯 Intelligent Strategy Selection**: Automatically chooses optimal crawling strategy per site
- **🤖 Comprehensive AI Analysis**: Integrated LLaVA, BERTopic, NewsReader, and Tomotopy analysis
- **⚡ High Performance**: Ultra-fast crawling (8.14+ articles/sec) with AI-enhanced quality (0.86+ articles/sec)
- **🔄 Multi-Site Orchestration**: Concurrent processing across multiple news sources
- **📊 Performance Monitoring**: Real-time metrics and optimization recommendations
- **🛡️ Ethical Compliance**: Robots.txt checking, rate limiting, and modal dismissal
- **🔧 Production Ready**: Systemd integration and comprehensive error handling

## Architecture

### Agent-Based Design

The Crawler Agent is designed as a separate agent in the JustNewsAgent ecosystem:

- **Independent Service**: Runs in its own systemd container
- **MCP Bus Integration**: Registers with the MCP Bus for inter-agent communication
- **Scout Agent Tool**: Provides crawling tools that the Scout agent exposes to the system
- **Bi-directional Communication**: Can call other agents and be called by them

### Crawling Strategies

1. **Ultra-Fast Mode** (8.14+ articles/sec)
   - Optimized for high-volume sites (BBC, CNN, Reuters)
   - Minimal AI analysis for maximum throughput
   - Best for: Breaking news, high-frequency updates

2. **AI-Enhanced Mode** (0.86+ articles/sec)
   - Full AI analysis pipeline with NewsReader integration
   - Quality-focused content extraction
   - Best for: In-depth analysis, premium content

3. **Generic Mode** (Variable performance)
   - Crawl4AI-first strategy with Playwright fallback
   - Supports any news source dynamically
   - Best for: New sources, custom configurations

### AI Analysis Pipeline

- **LLaVA-OneVision**: Visual content analysis from screenshots
- **BERTopic**: Dynamic topic modeling and clustering
- **Tomotopy Online LDA**: Streaming topic evolution
- **NewsReader**: Multi-modal content understanding
- **Sentiment Analysis**: RoBERTa-based sentiment detection

## Quick Start

### Agent Startup

```bash
# Start the Crawler Agent
sudo systemctl start justnews-crawler-agent@production

# Check status
sudo systemctl status justnews-crawler-agent@production

# View logs
journalctl -u justnews-crawler-agent@production -f
```

### API Usage

The Crawler Agent exposes FastAPI endpoints that can be called directly or via MCP Bus:

```python
import requests

# Unified production crawl
response = requests.post("http://localhost:8009/unified_production_crawl", json={
    "args": [],
    "kwargs": {
        "domains": ["bbc.com", "cnn.com"],
        "max_articles_per_site": 25,
        "concurrent_sites": 3
    }
})

results = response.json()
print(f"Crawled {results['total_articles']} articles")
```

### Scout Agent Integration

The Crawler Agent integrates with the Scout Agent through the MCP Bus:

```python
# Call via Scout Agent (recommended)
response = requests.post("http://localhost:8002/production_crawl_dynamic", json={
    "args": [],
    "kwargs": {
        "domains": ["bbc.com", "reuters.com"],
        "articles_per_site": 25
    }
})
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRAWLER_AGENT_PORT` | `8009` | Agent listening port |
| `UNIFIED_CRAWLER_MODE` | `production` | Operating mode |
| `UNIFIED_CRAWLER_MAX_ARTICLES_PER_SITE` | `25` | Articles per site |
| `UNIFIED_CRAWLER_CONCURRENT_SITES` | `3` | Concurrent site processing |
| `UNIFIED_CRAWLER_AI_ANALYSIS` | `true` | Enable AI analysis |
| `UNIFIED_CRAWLER_PERFORMANCE_MONITORING` | `true` | Enable performance monitoring |

### Strategy Selection Rules

The crawler automatically selects strategies based on domain patterns:

- **Ultra-Fast**: `bbc.com`, `bbc.co.uk`, `cnn.com`, `reuters.com`, `apnews.com`, `npr.org`, `nytimes.com`, `washingtonpost.com`
- **AI-Enhanced**: `wsj.com`, `ft.com`, `economist.com`, `newyorker.com`, `theatlantic.com`, `foreignaffairs.com`
- **Generic**: All other domains

## API Endpoints

### Core Endpoints

- `POST /unified_production_crawl`: Main crawling endpoint
  - Parameters: `domains`, `max_articles_per_site`, `concurrent_sites`, `max_total_articles`
  - Returns: Comprehensive crawl results with articles and metrics

- `POST /get_crawler_info`: Get crawler capabilities
  - Returns: Supported strategies, capabilities, and performance targets

- `POST /get_performance_metrics`: Get current performance metrics
  - Returns: Real-time performance statistics and recommendations

### Health Endpoints

- `GET /health`: Basic health check
- `GET /ready`: Readiness check for MCP Bus registration

## Systemd Deployment

### Service Installation

```bash
# Install the Crawler Agent service
sudo cp deploy/systemd/units/justnews-crawler-agent@.service /etc/systemd/system/
sudo cp deploy/systemd/env/crawler-agent.env /etc/justnews/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable justnews-crawler-agent@production
```

### Service Management

```bash
# Start the agent
sudo systemctl start justnews-crawler-agent@production

# Check status
sudo systemctl status justnews-crawler-agent@production

# View logs
journalctl -u justnews-crawler-agent@production -f

# Restart
sudo systemctl restart justnews-crawler-agent@production
```

## MCP Bus Integration

### Agent Registration

The Crawler Agent automatically registers with the MCP Bus on startup:

```json
{
  "name": "crawler",
  "address": "http://localhost:8009",
  "tools": [
    "unified_production_crawl",
    "get_crawler_info", 
    "get_performance_metrics"
  ]
}
```

### Bi-directional Communication

- **Receiving Calls**: Other agents can call Crawler tools via MCP Bus
- **Making Calls**: Crawler can call other agents (e.g., NewsReader for visual analysis)

### Scout Agent Integration

The Scout Agent acts as the primary interface for crawling operations:

1. **Tool Exposure**: Scout exposes Crawler tools via its FastAPI endpoints
2. **Request Routing**: Scout forwards crawling requests to Crawler Agent
3. **Result Processing**: Scout processes and returns results to callers

## Performance Monitoring

### Real-Time Metrics

The agent provides comprehensive performance monitoring:

```python
# Get current performance metrics
response = requests.post("http://localhost:8009/get_performance_metrics", json={"args": [], "kwargs": {}})
metrics = response.json()

# Key metrics include:
# - articles_per_second: Overall throughput
# - success_rate: Content extraction success
# - strategy_usage: Strategy effectiveness
# - site_performance: Per-site metrics
```

### Performance Reports

Performance data includes:
- Articles processed per second
- Strategy effectiveness
- Site-specific performance
- Error rates and patterns
- Optimization recommendations

## Testing

### Run Test Suite

```bash
# Run complete test suite
cd agents/crawler
python -m pytest test_unified_crawler.py -v

# Run unit tests only
python -m pytest test_unified_crawler.py::TestUnifiedCrawler::test_strategy_selection -v

# Run integration tests
python -m pytest test_unified_crawler.py::TestUnifiedCrawler::test_full_crawl_pipeline -v
```

### Test Coverage

The test suite covers:

- ✅ Strategy selection logic
- ✅ AI model loading and availability
- ✅ AI analysis pipeline functionality
- ✅ Performance monitoring system
- ✅ Database integration
- ✅ Unified crawl orchestration
- ✅ MCP Bus communication
- ✅ FastAPI endpoint functionality

## Database Integration

### Source Management

The crawler integrates with PostgreSQL for source management:

```sql
-- Active sources table
SELECT id, domain, name, last_verified
FROM public.sources
WHERE last_verified > now() - interval '30 days'
ORDER BY last_verified DESC;

-- Crawling performance tracking
SELECT source_id, strategy_used, articles_per_second, timestamp
FROM public.crawling_performance
WHERE timestamp > now() - interval '24 hours'
ORDER BY articles_per_second DESC;
```

### Performance Tracking

Crawling performance is automatically tracked:
- Articles found/successful per crawl
- Processing time and throughput
- Strategy effectiveness
- Error rates and patterns

## Troubleshooting

### Common Issues

**Agent Not Starting**
```
Cause: MCP Bus not available or port conflict
Solution: Check MCP Bus status and ensure port 8009 is available
```

**Low Throughput**
```
Cause: Too many concurrent sites or AI analysis enabled
Solution: Reduce concurrent_sites or disable AI analysis for speed
```

**High Error Rate**
```
Cause: Site blocking or content structure changes
Solution: Check robots.txt compliance and update selectors
```

**MCP Bus Communication Issues**
```
Cause: Network issues or agent registration failure
Solution: Check MCP Bus logs and agent registration status
```

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH=/home/adra/justnewsagent/JustNewsAgent
export LOG_LEVEL=DEBUG

# Run agent with debug output
python3 -m agents.crawler.main
```

## Development

### Local Development

```bash
# Install dependencies
cd agents/crawler
pip install -r requirements.txt

# Run the agent
python -m agents.crawler.main

# Test endpoints
curl -X POST http://localhost:8009/health
```

### Adding New Features

1. **New Crawling Strategies**: Add to `UnifiedProductionCrawler._determine_optimal_strategy()`
2. **New AI Models**: Integrate in `_load_ai_models()` and `_apply_ai_analysis()`
3. **New Endpoints**: Add to `main.py` and register in MCP Bus tools
4. **Database Changes**: Update schema and crawler_utils.py

## Performance Benchmarks

### Strategy Performance

| Strategy | Articles/sec | Use Case | Quality |
|----------|-------------|----------|---------|
| Ultra-Fast | 8.14+ | High-volume news | Basic |
| AI-Enhanced | 0.86+ | Premium content | High |
| Generic | 1.5-3.0 | Any source | Variable |

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for AI models
- **GPU**: Optional, 4GB+ VRAM for LLaVA
- **Storage**: 10GB+ for models and data
- **Network**: Stable broadband connection

## Changelog

### v3.0.0 (Current)
- ✅ Agent-based architecture with MCP Bus integration
- ✅ Unified multi-strategy crawling
- ✅ Comprehensive AI analysis pipeline
- ✅ Performance monitoring and optimization
- ✅ Systemd production deployment
- ✅ Database-driven source management
- ✅ Comprehensive testing framework
- ✅ Bi-directional agent communication

### Future Enhancements
- 🔄 Dynamic strategy learning
- 🔄 Multi-language support
- 🔄 Advanced content deduplication
- 🔄 Real-time performance dashboard
- 🔄 Distributed crawling across multiple agents

## Architecture

### Crawling Strategies

1. **Ultra-Fast Mode** (8.14+ articles/sec)
   - Optimized for high-volume sites (BBC, CNN, Reuters)
   - Minimal AI analysis for maximum throughput
   - Best for: Breaking news, high-frequency updates

2. **AI-Enhanced Mode** (0.86+ articles/sec)
   - Full AI analysis pipeline with NewsReader integration
   - Quality-focused content extraction
   - Best for: In-depth analysis, premium content

3. **Generic Mode** (Variable performance)
   - Crawl4AI-first strategy with Playwright fallback
   - Supports any news source dynamically
   - Best for: New sources, custom configurations

### AI Analysis Pipeline

- **LLaVA-OneVision**: Visual content analysis from screenshots
- **BERTopic**: Dynamic topic modeling and clustering
- **Tomotopy Online LDA**: Streaming topic evolution
- **NewsReader**: Multi-modal content understanding
- **Sentiment Analysis**: RoBERTa-based sentiment detection

## Quick Start

### Basic Usage

```python
from agents.scout.production_crawlers.unified_production_crawler import UnifiedProductionCrawler

async def main():
    crawler = UnifiedProductionCrawler()

    # Crawl all active sources
    results = await crawler.run_unified_crawl()

    print(f"Crawled {results['sites_crawled']} sites")
    print(f"Collected {results['total_articles']} articles")
    print(".2f")

# Run the crawler
asyncio.run(main())
```

### Advanced Usage

```python
# Crawl specific domains with custom settings
results = await crawler.run_unified_crawl(
    domains=['bbc.com', 'cnn.com', 'reuters.com'],
    max_articles_per_site=50,
    concurrent_sites=5
)

# Get performance metrics
metrics = crawler.performance_monitor.get_current_metrics()
print(f"Success Rate: {metrics['success_rate']:.1%}")

# Get optimization recommendations
recommendations = crawler.performance_optimizer.get_optimization_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIFIED_CRAWLER_MODE` | `production` | Operating mode |
| `UNIFIED_CRAWLER_MAX_ARTICLES_PER_SITE` | `25` | Articles per site |
| `UNIFIED_CRAWLER_CONCURRENT_SITES` | `3` | Concurrent site processing |
| `UNIFIED_CRAWLER_AI_ANALYSIS` | `true` | Enable AI analysis |
| `UNIFIED_CRAWLER_PERFORMANCE_MONITORING` | `true` | Enable performance monitoring |

### Strategy Selection Rules

The crawler automatically selects strategies based on domain patterns:

- **Ultra-Fast**: `bbc.com`, `bbc.co.uk`, `cnn.com`, `reuters.com`, `apnews.com`, `npr.org`, `nytimes.com`, `washingtonpost.com`
- **AI-Enhanced**: `wsj.com`, `ft.com`, `economist.com`, `newyorker.com`, `theatlantic.com`, `foreignaffairs.com`
- **Generic**: All other domains

## Systemd Deployment

### Service Installation

```bash
# Install the unified crawler service
sudo cp deploy/systemd/units/justnews-unified-crawler@.service /etc/systemd/system/
sudo cp deploy/systemd/env/unified-crawler.env /etc/justnews/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable justnews-unified-crawler@production
```

### Service Management

```bash
# Start the crawler
sudo systemctl start justnews-unified-crawler@production

# Check status
sudo systemctl status justnews-unified-crawler@production

# View logs
journalctl -u justnews-unified-crawler@production -f

# Restart
sudo systemctl restart justnews-unified-crawler@production
```

### Crawler Management Script

Use the specialized crawler management script:

```bash
# Enable crawler services
sudo ./deploy/systemd/manage_crawlers.sh enable

# Start crawlers
sudo ./deploy/systemd/manage_crawlers.sh start

# Check status
sudo ./deploy/systemd/manage_crawlers.sh status

# View performance
sudo ./deploy/systemd/manage_crawlers.sh performance
```

## Performance Monitoring

### Real-Time Metrics

The crawler provides comprehensive performance monitoring:

```python
# Get current performance metrics
metrics = crawler.performance_monitor.get_current_metrics()

# Key metrics include:
# - articles_per_second: Overall throughput
# - success_rate: Content extraction success
# - strategy_usage: Strategy effectiveness
# - site_performance: Per-site metrics
```

### Performance Reports

```python
# Export detailed performance report
export_performance_metrics("crawler_performance.json")

# Get optimization recommendations
recommendations = crawler.performance_optimizer.get_optimization_recommendations()

# Get configuration suggestions
suggestions = crawler.performance_optimizer.suggest_configuration_changes()
```

### Monitoring Dashboard

Performance metrics are logged every 60 seconds:

```
📊 Performance: 2.34 articles/sec, Success: 94.2%, Sites: 5
🎯 Best performing strategy: ultra_fast (3.45 articles/sec)
```

## Testing

### Run Test Suite

```bash
# Run complete test suite
./agents/scout/production_crawlers/run_tests.sh full

# Run unit tests only
./agents/scout/production_crawlers/run_tests.sh unit

# Run integration tests
./agents/scout/production_crawlers/run_tests.sh integration

# Generate test report
./agents/scout/production_crawlers/run_tests.sh report
```

### Test Coverage

The test suite covers:

- ✅ Strategy selection logic
- ✅ AI model loading and availability
- ✅ AI analysis pipeline functionality
- ✅ Performance monitoring system
- ✅ Database integration
- ✅ Unified crawl orchestration

## Database Integration

### Source Management

The crawler integrates with PostgreSQL for source management:

```sql
-- Active sources table
SELECT id, domain, name, last_verified
FROM public.sources
WHERE last_verified > now() - interval '30 days'
ORDER BY last_verified DESC;

-- Crawling performance tracking
SELECT source_id, strategy_used, articles_per_second, timestamp
FROM public.crawling_performance
WHERE timestamp > now() - interval '24 hours'
ORDER BY articles_per_second DESC;
```

### Performance Tracking

Crawling performance is automatically tracked:

- Articles found/successful per crawl
- Processing time and throughput
- Strategy effectiveness
- Error rates and patterns

## Troubleshooting

### Common Issues

**Low Throughput**
```
Cause: Too many concurrent sites or AI analysis enabled
Solution: Reduce concurrent_sites or disable AI analysis for speed
```

**High Error Rate**
```
Cause: Site blocking or content structure changes
Solution: Check robots.txt compliance and update selectors
```

**Memory Issues**
```
Cause: Large AI models loaded simultaneously
Solution: Reduce batch sizes or disable heavy models
```

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH=/home/adra/justnewsagent/JustNewsAgent
export LOG_LEVEL=DEBUG

# Run with debug output
python3 -c "
import asyncio
from agents.scout.production_crawlers.unified_production_crawler import UnifiedProductionCrawler

async def debug_run():
    crawler = UnifiedProductionCrawler()
    results = await crawler.run_unified_crawl(domains=['bbc.com'], max_articles_per_site=5)
    print('Debug results:', results)

asyncio.run(debug_run())
"
```

## API Reference

### UnifiedProductionCrawler

#### Methods

- `run_unified_crawl(domains=None, max_articles_per_site=25, concurrent_sites=3)`
  - Main crawling entry point
  - Returns comprehensive results dictionary

- `crawl_site(site_config, max_articles=25)`
  - Crawl individual site with optimal strategy
  - Returns list of article dictionaries

- `get_performance_report()`
  - Get current performance metrics
  - Returns metrics dictionary

#### Properties

- `performance_monitor`: Real-time performance tracking
- `performance_optimizer`: Optimization recommendations
- `site_strategies`: Cached strategy assignments

## Contributing

### Adding New Strategies

1. Implement strategy method in `UnifiedProductionCrawler`
2. Add strategy to `_determine_optimal_strategy()`
3. Update performance tracking
4. Add tests in `test_unified_crawler.py`

### Adding New AI Models

1. Add model loading in `_load_ai_models()`
2. Implement analysis in `_apply_ai_analysis()`
3. Update performance tracking
4. Add model availability checks

## Performance Benchmarks

### Strategy Performance

| Strategy | Articles/sec | Use Case | Quality |
|----------|-------------|----------|---------|
| Ultra-Fast | 8.14+ | High-volume news | Basic |
| AI-Enhanced | 0.86+ | Premium content | High |
| Generic | 1.5-3.0 | Any source | Variable |

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ for AI models
- **GPU**: Optional, 4GB+ VRAM for LLaVA
- **Storage**: 10GB+ for models and data
- **Network**: Stable broadband connection

## Changelog

### v3.0.0 (Current)
- ✅ Unified multi-strategy crawling
- ✅ Comprehensive AI analysis pipeline
- ✅ Performance monitoring and optimization
- ✅ Systemd production deployment
- ✅ Database-driven source management
- ✅ Comprehensive testing framework

### Future Enhancements
- 🔄 Dynamic strategy learning
- 🔄 Multi-language support
- 🔄 Advanced content deduplication
- 🔄 Real-time performance dashboard
