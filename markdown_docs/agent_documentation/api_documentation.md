---
title: API Documentation
description: Auto-generated description for API Documentation
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# API Documentation

## Overview

The JustNews V4 system provides a comprehensive REST API through multiple specialized agents, each exposing domain-specific functionality. This documentation covers all API endpoints, their parameters, responses, authentication, and usage patterns for integration and development.

## API Architecture

### Agent-Based Architecture

The system uses a distributed agent architecture where each agent provides specialized functionality:

- **MCP Bus** (Port 8000): Central communication hub and service registry
- **Synthesizer** (Port 8005): News synthesis and clustering with GPU acceleration
- **Scout** (Port 8002): Web crawling and content discovery
- **NewsReader** (Port 8009): Multi-modal content extraction and analysis
- **Analyst** (Port 8004): Sentiment analysis and bias detection
- **Fact Checker** (Port 8003): Source verification and fact-checking
- **Critic** (Port 8006): Quality assessment and review
- **Chief Editor** (Port 8001): Workflow orchestration
- **Memory** (Port 8007): Knowledge storage and retrieval
- **Reasoning** (Port 8008): Logical reasoning and inference

### Communication Patterns

#### Direct API Calls
```http
POST /agent_endpoint
Content-Type: application/json

{
  "parameter1": "value1",
  "parameter2": "value2"
}
```

#### MCP Bus Routing
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

## MCP Bus API

The MCP Bus serves as the central communication hub for all agents.

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Register Agent
Register a new agent with the MCP Bus.

```http
POST /register
Content-Type: application/json

{
  "name": "agent_name",
  "address": "http://localhost:port"
}
```

**Response:**
```json
{
  "status": "ok"
}
```

#### Call Agent Tool
Execute a tool on a registered agent.

```http
POST /call
Content-Type: application/json

{
  "agent": "agent_name",
  "tool": "tool_name",
  "args": ["arg1", "arg2"],
  "kwargs": {"param1": "value1"}
}
```

**Response:**
```json
{
  "status": "success",
  "data": {...}
}
```

**Error Response:**
```json
{
  "detail": "Agent not found: agent_name"
}
```

#### List Agents
Get list of all registered agents.

```http
GET /agents
```

**Response:**
```json
{
  "synthesizer": "http://localhost:8005",
  "scout": "http://localhost:8002",
  "newsreader": "http://localhost:8009"
}
```

#### Health Check
Basic health check endpoint.

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

#### Readiness Check
Check if MCP Bus is ready to accept requests.

```http
GET /ready
```

**Response:**
```json
{
  "ready": true
}
```

## Synthesizer Agent API

The Synthesizer agent provides news synthesis, clustering, and GPU-accelerated processing.

### Base URL
```
http://localhost:8005
```

### Endpoints

#### Cluster Articles
Group similar news articles into thematic clusters.

```http
POST /cluster_articles
Content-Type: application/json

{
  "args": [["Article 1 content...", "Article 2 content..."]],
  "kwargs": {
    "method": "semantic",
    "num_clusters": 5,
    "min_cluster_size": 2
  }
}
```

**Parameters:**
- `args[0]` (list): List of article texts to cluster
- `method` (string): Clustering method ("semantic", "keyword", "temporal")
- `num_clusters` (int): Number of clusters to create (default: auto)
- `min_cluster_size` (int): Minimum articles per cluster (default: 2)

**Response:**
```json
{
  "success": true,
  "clusters": [
    {
      "cluster_id": 0,
      "theme": "Technology Innovation",
      "articles": ["Article 1", "Article 3"],
      "confidence": 0.85
    }
  ],
  "performance": {
    "processing_time": 1.23,
    "articles_per_sec": 8.13
  }
}
```

#### Neutralize Text
Remove bias and neutralize language in news content.

```http
POST /neutralize_text
Content-Type: application/json

{
  "args": ["Biased article content..."],
  "kwargs": {
    "intensity": "moderate",
    "preserve_facts": true
  }
}
```

**Parameters:**
- `args[0]` (string): Text content to neutralize
- `intensity` (string): Neutralization intensity ("light", "moderate", "strong")
- `preserve_facts` (boolean): Preserve factual accuracy (default: true)

**Response:**
```json
{
  "success": true,
  "original_text": "Original biased content...",
  "neutralized_text": "Neutralized content...",
  "changes_made": [
    "Removed loaded language",
    "Balanced perspective"
  ],
  "confidence": 0.92
}
```

#### Aggregate Cluster
Create synthesized summary from article cluster.

```http
POST /aggregate_cluster
Content-Type: application/json

{
  "args": [[{"title": "Article 1", "content": "..."}]],
  "kwargs": {
    "summary_length": "medium",
    "include_quotes": true
  }
}
```

**Parameters:**
- `args[0]` (list): List of article objects with title/content
- `summary_length` (string): Summary length ("short", "medium", "long")
- `include_quotes` (boolean): Include direct quotes (default: true)

**Response:**
```json
{
  "success": true,
  "summary": "Comprehensive summary of the article cluster...",
  "key_points": [
    "Main point 1",
    "Main point 2"
  ],
  "quotes": [
    {"text": "Quote text", "source": "Article title"}
  ],
  "themes": ["Technology", "Innovation"]
}
```

#### GPU Synthesis
High-performance GPU-accelerated news synthesis.

```http
POST /synthesize_news_articles_gpu
Content-Type: application/json

{
  "args": [[{"title": "Article 1", "content": "..."}]],
  "kwargs": {
    "batch_size": 16,
    "use_gpu": true,
    "model": "bart-large-cnn"
  }
}
```

**Parameters:**
- `args[0]` (list): List of article objects
- `batch_size` (int): Processing batch size (default: 16)
- `use_gpu` (boolean): Force GPU usage (default: auto-detect)
- `model` (string): Synthesis model to use

**Response:**
```json
{
  "success": true,
  "synthesis": "GPU-accelerated synthesis result...",
  "performance": {
    "articles_per_sec": 45.2,
    "gpu_memory_used_mb": 2048,
    "processing_time": 0.89
  },
  "model_used": "bart-large-cnn",
  "gpu_used": true
}
```

#### Performance Statistics
Get synthesizer performance metrics.

```http
POST /get_synthesizer_performance
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "time_range": "24h",
    "include_gpu_stats": true
  }
}
```

**Response:**
```json
{
  "total_processed": 15420,
  "gpu_processed": 14850,
  "cpu_processed": 570,
  "average_processing_time": 1.23,
  "gpu_utilization_percent": 78.5,
  "memory_usage_mb": 3072,
  "models_loaded": ["bart-large-cnn", "sentence-transformers"],
  "uptime_hours": 24.5
}
```

## Scout Agent API

The Scout agent handles web crawling, content discovery, and source intelligence.

### Base URL
```
http://localhost:8002
```

### Endpoints

#### Discover Sources
Discover news sources based on topics and criteria.

```http
POST /discover_sources
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "topics": ["technology", "politics"],
    "regions": ["us", "eu"],
    "max_sources": 50,
    "min_credibility": 0.7
  }
}
```

**Parameters:**
- `topics` (list): List of topics to search for
- `regions` (list): Geographic regions to focus on
- `max_sources` (int): Maximum sources to return
- `min_credibility` (float): Minimum credibility score

**Response:**
```json
{
  "success": true,
  "sources": [
    {
      "name": "TechCrunch",
      "url": "https://techcrunch.com",
      "credibility_score": 0.89,
      "topics": ["technology", "startups"],
      "region": "us",
      "article_count": 2450
    }
  ],
  "total_found": 150,
  "search_criteria": {...}
}
```

#### Crawl URL
Crawl a single URL for content.

```http
POST /crawl_url
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "url": "https://example.com/article",
    "depth": 1,
    "follow_links": false,
    "extract_images": true
  }
}
```

**Parameters:**
- `url` (string): URL to crawl (required)
- `depth` (int): Crawl depth (default: 1)
- `follow_links` (boolean): Follow internal links (default: false)
- `extract_images` (boolean): Extract images (default: true)

**Response:**
```json
{
  "success": true,
  "url": "https://example.com/article",
  "title": "Article Title",
  "content": "Full article content...",
  "metadata": {
    "author": "John Doe",
    "published_date": "2024-01-15",
    "word_count": 850
  },
  "images": [
    {
      "url": "https://example.com/image.jpg",
      "alt": "Image description",
      "caption": "Image caption"
    }
  ],
  "links": ["https://example.com/related"],
  "processing_time": 2.34
}
```

#### Deep Crawl Site
Perform comprehensive site crawling.

```http
POST /deep_crawl_site
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "site": "https://example.com",
    "max_pages": 100,
    "respect_robots": true,
    "delay_between_requests": 1.0
  }
}
```

**Parameters:**
- `site` (string): Base site URL (required)
- `max_pages` (int): Maximum pages to crawl
- `respect_robots` (boolean): Respect robots.txt (default: true)
- `delay_between_requests` (float): Delay between requests in seconds

**Response:**
```json
{
  "success": true,
  "site": "https://example.com",
  "pages_crawled": 87,
  "articles_found": 23,
  "images_found": 156,
  "processing_time": 45.67,
  "articles": [
    {
      "url": "https://example.com/article1",
      "title": "Article 1",
      "published_date": "2024-01-15"
    }
  ]
}
```

#### Enhanced Deep Crawl
AI-enhanced site crawling with content analysis.

```http
POST /enhanced_deep_crawl_site
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "site": "https://example.com",
    "content_filter": "news",
    "quality_threshold": 0.8,
    "max_articles": 50
  }
}
```

**Parameters:**
- `site` (string): Site URL (required)
- `content_filter` (string): Content type filter ("news", "blog", "all")
- `quality_threshold` (float): Minimum quality score
- `max_articles` (int): Maximum articles to extract

**Response:**
```json
{
  "success": true,
  "site": "https://example.com",
  "articles_extracted": 34,
  "quality_scores": {
    "average": 0.82,
    "min": 0.65,
    "max": 0.95
  },
  "content_types": {
    "news": 28,
    "blog": 6
  },
  "processing_time": 67.89
}
```

#### Intelligent Source Discovery
AI-powered source discovery and ranking.

```http
POST /intelligent_source_discovery
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "query": "artificial intelligence",
    "max_sources": 25,
    "diversity_factor": 0.7,
    "freshness_days": 7
  }
}
```

**Parameters:**
- `query` (string): Search query for source discovery
- `max_sources` (int): Maximum sources to return
- `diversity_factor` (float): Source diversity (0-1)
- `freshness_days` (int): Maximum age of sources in days

**Response:**
```json
{
  "success": true,
  "query": "artificial intelligence",
  "sources": [
    {
      "name": "AI News Daily",
      "url": "https://ai-news-daily.com",
      "relevance_score": 0.95,
      "credibility_score": 0.87,
      "freshness_score": 0.92,
      "article_velocity": 12.5
    }
  ],
  "diversity_metrics": {
    "topic_diversity": 0.78,
    "geographic_diversity": 0.65
  }
}
```

#### Production Crawler (Ultra Fast)
High-performance production crawling.

```http
POST /production_crawl_ultra_fast
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "site": "https://news-site.com",
    "batch_size": 50,
    "concurrent_requests": 10,
    "timeout_seconds": 30
  }
}
```

**Parameters:**
- `site` (string): Site to crawl (required)
- `batch_size` (int): Articles per batch
- `concurrent_requests` (int): Concurrent request limit
- `timeout_seconds` (int): Request timeout

**Response:**
```json
{
  "success": true,
  "site": "https://news-site.com",
  "articles_crawled": 234,
  "processing_rate": 8.14,
  "total_time": 28.76,
  "errors": 2,
  "performance_metrics": {
    "avg_response_time": 1.23,
    "success_rate": 0.991,
    "bandwidth_used_mb": 45.6
  }
}
```

#### Production Crawler (AI Enhanced)
AI-enhanced production crawling with content analysis.

```http
POST /production_crawl_ai_enhanced
Content-Type: application/json

{
  "args": [],
  "kwargs": {
    "site": "https://news-site.com",
    "content_quality_threshold": 0.8,
    "relevance_filter": "technology",
    "max_articles": 100
  }
}
```

**Parameters:**
- `site` (string): Site to crawl (required)
- `content_quality_threshold` (float): Minimum quality score
- `relevance_filter` (string): Content relevance filter
- `max_articles` (int): Maximum articles to extract

**Response:**
```json
{
  "success": true,
  "site": "https://news-site.com",
  "articles_extracted": 87,
  "quality_filtered": 23,
  "relevance_filtered": 12,
  "final_count": 52,
  "ai_processing_time": 45.23,
  "quality_distribution": {
    "excellent": 15,
    "good": 28,
    "average": 9
  }
}
```

## NewsReader Agent API

The NewsReader agent provides multi-modal content extraction and analysis.

### Base URL
```
http://localhost:8009
```

### Endpoints

#### Root Information
Get agent information and capabilities.

```http
GET /
```

**Response:**
```json
{
  "agent": "NewsReader (Unified)",
  "version": "0.8.0",
  "capabilities": [
    "news_extraction",
    "visual_analysis",
    "multi_modal_processing",
    "structure_analysis",
    "multimedia_extraction",
    "screenshot_capture",
    "llava_analysis"
  ],
  "endpoints": [
    "/extract_news",
    "/analyze_content",
    "/extract_structure",
    "/extract_multimedia",
    "/capture_screenshot",
    "/analyze_image",
    "/health",
    "/ready"
  ]
}
```

#### Extract News
Extract news content from URL with visual analysis.

```http
POST /extract_news
Content-Type: application/json

{
  "url": "https://example.com/article",
  "screenshot_path": "/tmp/screenshot.png"
}
```

**Parameters:**
- `url` (string): URL to extract from (required)
- `screenshot_path` (string): Path to save screenshot

**Response:**
```json
{
  "success": true,
  "url": "https://example.com/article",
  "title": "Article Title",
  "content": "Full article content...",
  "summary": "Article summary...",
  "images": [
    {
      "url": "https://example.com/image.jpg",
      "description": "AI-generated description",
      "relevance_score": 0.89
    }
  ],
  "metadata": {
    "author": "John Doe",
    "published_date": "2024-01-15T10:30:00Z",
    "word_count": 850,
    "reading_time_minutes": 4
  },
  "visual_analysis": {
    "layout_complexity": 0.72,
    "content_density": 0.85,
    "image_count": 3,
    "text_blocks": 12
  },
  "processing_time": 3.45,
  "method": "llava_screenshot"
}
```

#### Analyze Content
Multi-modal content analysis.

```http
POST /analyze_content
Content-Type: application/json

{
  "content": "Article content...",
  "content_type": "article",
  "processing_mode": "comprehensive",
  "include_visual_analysis": true,
  "include_layout_analysis": true
}
```

**Parameters:**
- `content` (string/object): Content to analyze (required)
- `content_type` (string): Content type ("article", "image", "pdf", "webpage")
- `processing_mode` (string): Processing mode ("comprehensive", "fast", "basic")
- `include_visual_analysis` (boolean): Include visual analysis
- `include_layout_analysis` (boolean): Include layout analysis

**Response:**
```json
{
  "status": "success",
  "content_type": "article",
  "processing_mode": "comprehensive",
  "analysis": {
    "sentiment": {
      "polarity": 0.15,
      "subjectivity": 0.62,
      "confidence": 0.89
    },
    "readability": {
      "flesch_score": 45.2,
      "grade_level": "12th grade",
      "reading_ease": "difficult"
    },
    "structure": {
      "paragraphs": 8,
      "sentences": 24,
      "avg_words_per_sentence": 18.5
    }
  },
  "visual_analysis": {
    "layout_score": 0.78,
    "content_density": 0.82,
    "image_analysis": [...]
  },
  "processing_time": 2.34
}
```

#### Extract Structure
Extract and analyze content structure.

```http
POST /extract_structure
Content-Type: application/json

{
  "content": "Article content...",
  "analysis_depth": "comprehensive"
}
```

**Parameters:**
- `content` (string): Content to analyze (required)
- `analysis_depth` (string): Analysis depth ("comprehensive", "standard", "basic")

**Response:**
```json
{
  "success": true,
  "structure": {
    "title": "Article Title",
    "headline": "Breaking News Headline",
    "byline": "By John Doe",
    "dateline": "New York, NY",
    "lead": "Lead paragraph...",
    "body": [
      {"type": "paragraph", "content": "Body paragraph 1..."},
      {"type": "quote", "content": "Quote text...", "attribution": "Source"}
    ],
    "images": [...],
    "links": [...]
  },
  "metadata": {
    "word_count": 850,
    "sentence_count": 24,
    "paragraph_count": 8,
    "complexity_score": 0.72
  }
}
```

#### Extract Multimedia
Extract multimedia content from various sources.

```http
POST /extract_multimedia
Content-Type: application/json

{
  "content": "Article content or URL...",
  "extraction_types": ["images", "text", "layout", "metadata"]
}
```

**Parameters:**
- `content` (string/bytes/object): Source content (required)
- `extraction_types` (list): Types to extract

**Response:**
```json
{
  "success": true,
  "extraction_types": ["images", "text", "layout", "metadata"],
  "images": [
    {
      "url": "https://example.com/image.jpg",
      "local_path": "/tmp/extracted/image1.jpg",
      "description": "AI-generated description",
      "dimensions": {"width": 800, "height": 600},
      "format": "JPEG",
      "size_bytes": 245760
    }
  ],
  "text": {
    "full_text": "Complete extracted text...",
    "language": "en",
    "confidence": 0.95
  },
  "layout": {
    "blocks": [
      {"type": "title", "bbox": [10, 10, 790, 50]},
      {"type": "paragraph", "bbox": [10, 60, 790, 120]}
    ]
  },
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "created_date": "2024-01-15",
    "modified_date": "2024-01-15"
  }
}
```

#### Capture Screenshot
Capture webpage screenshot.

```http
POST /capture_screenshot
Content-Type: application/json

{
  "url": "https://example.com",
  "screenshot_path": "/tmp/screenshot.png"
}
```

**Parameters:**
- `url` (string): URL to capture (required)
- `screenshot_path` (string): Output path (optional)

**Response:**
```json
{
  "success": true,
  "url": "https://example.com",
  "screenshot_path": "/tmp/screenshot.png",
  "dimensions": {"width": 1920, "height": 1080},
  "file_size_bytes": 524288,
  "capture_time": 2.34,
  "quality": "high"
}
```

#### Analyze Image
Analyze image content with LLaVA model.

```http
POST /analyze_image
Content-Type: application/json

{
  "image_path": "/path/to/image.jpg"
}
```

**Parameters:**
- `image_path` (string): Path to image file (required)

**Response:**
```json
{
  "success": true,
  "image_path": "/path/to/image.jpg",
  "description": "Detailed AI-generated description of the image content...",
  "objects": [
    {"name": "person", "confidence": 0.95, "bbox": [100, 200, 300, 400]},
    {"name": "building", "confidence": 0.87, "bbox": [50, 100, 750, 600]}
  ],
  "text_content": "Extracted text from image...",
  "sentiment": "neutral",
  "relevance_score": 0.78,
  "processing_time": 1.23
}
```

## Common API Patterns

### Error Handling

All endpoints follow consistent error response patterns:

#### Validation Error
```json
{
  "detail": "Invalid URL"
}
```

#### Server Error
```json
{
  "detail": "Internal server error"
}
```

#### MCP Bus Error
```json
{
  "detail": "Agent not found: unknown_agent"
}
```

### Authentication

#### API Key Authentication
```http
POST /endpoint
Authorization: Bearer your-api-key
Content-Type: application/json

{...}
```

#### MCP Bus Authentication
```http
POST /call
Authorization: Bearer your-mcp-token
Content-Type: application/json

{
  "agent": "synthesizer",
  "tool": "cluster_articles",
  "args": [...],
  "kwargs": {...}
}
```

### Rate Limiting

All endpoints implement rate limiting:

#### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

#### Rate Limit Exceeded
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

### Pagination

For endpoints returning large datasets:

```http
GET /list_articles?page=1&per_page=50&sort=published_date&order=desc
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 1250,
    "total_pages": 25,
    "has_next": true,
    "has_prev": false
  }
}
```

## SDK and Client Libraries

### Python Client

```python
from justnews_client import JustNewsClient

# Initialize client
client = JustNewsClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Use synthesizer
result = client.synthesizer.cluster_articles([
    "Article 1 content...",
    "Article 2 content..."
])

# Use scout
sources = client.scout.discover_sources(topics=["technology"])

# Use newsreader
content = client.newsreader.extract_news(
    url="https://example.com/article"
)
```

### JavaScript Client

```javascript
import { JustNewsClient } from 'justnews-client';

const client = new JustNewsClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Async/await usage
const result = await client.synthesizer.clusterArticles([
  'Article 1 content...',
  'Article 2 content...'
]);
```

### cURL Examples

#### Basic API Call
```bash
curl -X POST http://localhost:8005/cluster_articles \
  -H "Content-Type: application/json" \
  -d '{
    "args": [["Article 1", "Article 2"]],
    "kwargs": {"method": "semantic"}
  }'
```

#### MCP Bus Call
```bash
curl -X POST http://localhost:8000/call \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "synthesizer",
    "tool": "cluster_articles",
    "args": [["Article 1", "Article 2"]],
    "kwargs": {"method": "semantic"}
  }'
```

## API Versioning

### Version Headers

```http
Accept: application/vnd.justnews.v4+json
X-API-Version: v4
```

### Version Response

```json
{
  "api_version": "v4",
  "agent_version": "0.8.0",
  "compatibility": ["v3", "v4"]
}
```

## Monitoring and Metrics

### API Metrics

```http
GET /metrics
```

**Response:**
```json
{
  "requests_total": 15420,
  "requests_by_endpoint": {
    "/cluster_articles": 4520,
    "/extract_news": 3890,
    "/crawl_url": 3210
  },
  "response_times": {
    "average_ms": 1234,
    "p95_ms": 2345,
    "p99_ms": 3456
  },
  "error_rate_percent": 0.05,
  "uptime_seconds": 86400
}
```

### Health Endpoints

#### Individual Agent Health
```http
GET /health
```

#### System Health
```http
GET /health/system
```

#### Dependencies Health
```http
GET /health/dependencies
```

## Troubleshooting

### Common Issues

#### Connection Refused
```bash
# Check if service is running
sudo systemctl status justnews@mcp_bus

# Check port availability
netstat -tlnp | grep :8000

# Restart service
sudo systemctl restart justnews@mcp_bus
```

#### Timeout Errors
```bash
# Increase timeout in configuration
# config/system_config.json
{
  "mcp_bus": {
    "timeout_seconds": 60
  }
}

# Restart service
sudo systemctl restart justnews@mcp_bus
```

#### Authentication Errors
```bash
# Verify API key
curl -H "Authorization: Bearer your-api-key" \
  http://localhost:8000/health

# Check API key configuration
cat /etc/justnews/global.env
```

#### Rate Limiting
```bash
# Check rate limit status
curl -v http://localhost:8005/cluster_articles

# Wait for reset
sleep 60
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Enable debug mode
export LOG_LEVEL=DEBUG
export DEBUG_MODE=true

# Restart services
sudo systemctl restart justnews@*

# Check logs
sudo journalctl -u justnews@mcp_bus -f
```

### Performance Optimization

#### Connection Pooling
```python
# Configure connection pooling
import aiohttp

connector = aiohttp.TCPConnector(
    limit=100,  # Max connections
    limit_per_host=10,  # Per host limit
    ttl_dns_cache=300  # DNS cache TTL
)

async with aiohttp.ClientSession(connector=connector) as session:
    # Use session for requests
    pass
```

#### Batch Processing
```python
# Batch API calls
batch_data = [
    {"url": "https://example1.com"},
    {"url": "https://example2.com"},
    {"url": "https://example3.com"}
]

results = await client.batch_process(
    endpoint="/extract_news",
    data=batch_data,
    batch_size=10
)
```

---

*This comprehensive API documentation covers all endpoints, parameters, responses, and usage patterns for the JustNews V4 system. For specific implementation details, refer to the individual agent source code and configuration files.*

## See also

- Technical Architecture: markdown_docs/TECHNICAL_ARCHITECTURE.md
- Documentation Catalogue: docs/DOCUMENTATION_CATALOGUE.md

