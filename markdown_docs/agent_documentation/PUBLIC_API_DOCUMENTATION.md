---
title: Public API Documentation
description: Comprehensive documentation for the JustNews public API with authentication, security, and data access features
tags: [documentation, api, public, security]
status: current
last_updated: 2025-09-20
---

# JustNews Public API Documentation

## Overview

The JustNews Public API provides secure, authenticated access to news analysis data, statistics, and research tools. Built on FastAPI with comprehensive security features, the API serves both public users and researchers with different access levels and rate limits.

**Version**: v1.0.0
**Base URL**: `https://api.justnews.ai/v1` (production) / `http://localhost:8000/v1` (development)
**Authentication**: HTTP Bearer tokens for research endpoints
**Rate Limiting**: 1000 req/hr (public), 100 req/hr (research)

## Architecture

### API Structure
- **Public Endpoints**: Statistics, articles, trends, fact-checks (no authentication required)
- **Research Endpoints**: Data export, detailed metrics (API key required)
- **Security**: Rate limiting, authentication, input validation
- **Caching**: 5-minute TTL for frequently accessed data
- **Data Sources**: Real-time integration with JustNews agents via MCP bus

### Authentication & Security

#### API Key Authentication
Research endpoints require valid API keys via HTTP Bearer authentication:

```http
Authorization: Bearer your-api-key-here
```

**Getting an API Key:**
- Contact: admin@justnews.ai
- Purpose: Academic research, journalism, data analysis
- Rate Limit: 100 requests per hour

#### Rate Limiting
- **Public Endpoints**: 1000 requests per hour per IP
- **Research Endpoints**: 100 requests per hour per API key
- **Headers**: Rate limit status included in responses
- **Reset**: Automatic hourly reset

#### Security Features
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- Request logging and monitoring

## API Endpoints

### Public Statistics

#### Get Public Statistics
Get overview statistics for the JustNews system.

```http
GET /stats
```

**Response:**
```json
{
  "total_articles": 125000,
  "sources_tracked": 2500,
  "accuracy_rate": "95.2%",
  "daily_updates": 650,
  "active_sources": 1800,
  "fact_checks_performed": 45000,
  "average_credibility_score": 78.5,
  "last_updated": "2025-09-20T10:30:00Z"
}
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Article Access

#### Get Articles List
Retrieve paginated list of articles with optional filtering.

```http
GET /articles?page=1&limit=20&search=technology&sentiment=positive
```

**Parameters:**
- `page` (int): Page number (default: 1, max: 1000)
- `limit` (int): Items per page (default: 20, max: 100)
- `search` (string): Text search in title/content
- `topic` (string): Filter by topic
- `source` (string): Filter by news source
- `credibility_min` (int): Minimum credibility score (0-100)
- `credibility_max` (int): Maximum credibility score (0-100)
- `sentiment` (string): Filter by sentiment (positive/negative/neutral)
- `date_from` (string): Start date (ISO format)
- `date_to` (string): End date (ISO format)
- `sort` (string): Sort order (newest/oldest/credibility/relevance)

**Response:**
```json
{
  "articles": [
    {
      "id": "article-123",
      "title": "Breakthrough in Quantum Computing",
      "summary": "Scientists achieve major advancement...",
      "url": "https://example.com/article-123",
      "source": "Tech News Daily",
      "source_credibility": 92,
      "published_date": "2025-09-20T08:15:00Z",
      "sentiment_score": 0.75,
      "fact_check_score": 88,
      "bias_score": 0.05,
      "topics": ["Technology", "Science", "Innovation"],
      "word_count": 1200,
      "reading_time": 6
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total_articles": 1250,
    "total_pages": 63,
    "has_more": true
  },
  "filters_applied": {
    "search": "technology",
    "sentiment": "positive",
    "sort": "newest"
  }
}
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

#### Get Article Details
Get detailed information for a specific article.

```http
GET /article/{article_id}
```

**Parameters:**
- `article_id` (string): Unique article identifier

**Response:**
```json
{
  "id": "article-123",
  "title": "Breakthrough in Quantum Computing",
  "content": "Full article content...",
  "summary": "Article summary...",
  "url": "https://example.com/article-123",
  "source": "Tech News Daily",
  "source_credibility": 92,
  "published_date": "2025-09-20T08:15:00Z",
  "author": "Jane Smith",
  "sentiment_score": 0.75,
  "fact_check_score": 88,
  "bias_score": 0.05,
  "emotional_tone": "Optimistic",
  "readability_score": "College",
  "topics": ["Technology", "Science", "Innovation"],
  "word_count": 1200,
  "reading_time": 6,
  "detailed_analysis": {
    "sentiment_breakdown": {
      "positive_words": ["breakthrough", "advancement", "innovative"],
      "negative_words": [],
      "neutral_words": ["research", "system", "development"]
    },
    "bias_analysis": {
      "political_bias": 0.02,
      "sensationalism": 0.08,
      "objectivity_score": 94
    },
    "fact_check_details": {
      "claims_verified": 8,
      "claims_questioned": 0,
      "claims_debunked": 0,
      "sources_cited": 12,
      "expert_reviews": 3
    },
    "readability_metrics": {
      "flesch_score": 48.2,
      "grade_level": "College",
      "complex_words": 28,
      "avg_sentence_length": 19.5
    },
    "engagement_metrics": {
      "shares": 245,
      "bookmarks": 156,
      "comments": 67,
      "avg_reading_time": 5.2
    }
  }
}
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Trending Topics

#### Get Trending Topics
Get currently trending news topics.

```http
GET /trending-topics?limit=10
```

**Parameters:**
- `limit` (int): Number of topics to return (default: 10, max: 50)

**Response:**
```json
[
  {
    "name": "Artificial Intelligence",
    "count": 245,
    "change": "+12%",
    "trend": "rising"
  },
  {
    "name": "Climate Change",
    "count": 189,
    "change": "+8%",
    "trend": "stable"
  }
]
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Source Credibility

#### Get Source Credibility Rankings
Get credibility rankings for news sources.

```http
GET /source-credibility?limit=20
```

**Parameters:**
- `limit` (int): Number of sources to return (default: 20, max: 100)

**Response:**
```json
[
  {
    "name": "Reuters",
    "score": 95,
    "articles": 1250,
    "reliability": "high",
    "country": "Global",
    "last_updated": "2025-09-20T10:00:00Z"
  },
  {
    "name": "BBC News",
    "score": 92,
    "articles": 980,
    "reliability": "high",
    "country": "UK",
    "last_updated": "2025-09-20T09:45:00Z"
  }
]
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Fact Checks

#### Get Recent Fact Checks
Get recent fact-checking corrections.

```http
GET /fact-checks?limit=10
```

**Parameters:**
- `limit` (int): Number of fact checks to return (default: 10, max: 50)

**Response:**
```json
[
  {
    "id": "fc-001",
    "claim": "COVID-19 vaccines contain microchips",
    "verdict": "False",
    "confidence": 98,
    "date": "2025-09-19T14:30:00Z",
    "source": "FactCheck.org",
    "article_url": "https://factcheck.org/claim-001",
    "explanation": "No evidence supports this claim...",
    "tags": ["health", "misinformation"]
  }
]
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Temporal Analysis

#### Get Temporal Analysis
Get trend analysis over time periods.

```http
GET /temporal-analysis?topic=technology&date_from=2025-09-01&date_to=2025-09-20&interval=day
```

**Parameters:**
- `topic` (string): Topic to analyze (optional)
- `date_from` (string): Start date (ISO format)
- `date_to` (string): End date (ISO format)
- `interval` (string): Time interval (hour/day/week/month)

**Response:**
```json
{
  "topic": "technology",
  "date_range": "2025-09-01 to 2025-09-20",
  "interval": "day",
  "data_points": [
    {
      "date": "2025-09-01",
      "article_count": 45,
      "average_sentiment": 0.12,
      "average_credibility": 82.5,
      "top_topics": ["AI", "Quantum Computing"]
    }
  ],
  "summary": {
    "total_articles": 920,
    "avg_sentiment": 0.08,
    "avg_credibility": 81.2,
    "trend_direction": "increasing"
  }
}
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Search Suggestions

#### Get Search Suggestions
Get autocomplete suggestions for search queries.

```http
GET /search/suggestions?query=artificial
```

**Parameters:**
- `query` (string): Partial search query (min 2 characters)

**Response:**
```json
{
  "query": "artificial",
  "suggestions": [
    "artificial intelligence news",
    "artificial intelligence analysis",
    "latest artificial intelligence",
    "artificial intelligence trends"
  ],
  "article_suggestions": [
    "The Future of Artificial Intelligence in Healthcare",
    "Artificial Intelligence Breakthroughs in 2025"
  ],
  "categories": ["news", "analysis", "fact-checks", "sources"],
  "popular_searches": ["AI", "climate change", "politics", "technology"]
}
```

**Rate Limit**: 1000 req/hr
**Caching**: 5 minutes

### Research Endpoints (API Key Required)

#### Export Articles
Export article data for research purposes.

```http
GET /export/articles?format=json&date_from=2025-09-01&topic=technology&min_credibility=80
Authorization: Bearer your-api-key
```

**Parameters:**
- `format` (string): Export format (json/csv/xml)
- `date_from` (string): Start date filter
- `date_to` (string): End date filter
- `topic` (string): Topic filter
- `min_credibility` (int): Minimum credibility score (0-100)

**Response:**
```json
{
  "export_info": {
    "format": "json",
    "total_articles": 1250,
    "generated_at": "2025-09-20T10:30:00Z",
    "filters_applied": {
      "date_from": "2025-09-01",
      "topic": "technology",
      "min_credibility": 80
    },
    "api_key_hash": "a1b2c3d4"
  },
  "articles": [...]
}
```

**Rate Limit**: 100 req/hr
**Authentication**: Required

#### Get Research Metrics
Get detailed research analytics and system metrics.

```http
GET /research/metrics?date_from=2025-09-01&granularity=day
Authorization: Bearer your-api-key
```

**Parameters:**
- `date_from` (string): Start date for metrics
- `date_to` (string): End date for metrics
- `granularity` (string): Time granularity (hour/day/week/month)

**Response:**
```json
{
  "time_range": "2025-09-01 to 2025-09-20",
  "granularity": "day",
  "metrics": {
    "total_articles_analyzed": 125000,
    "unique_sources": 2500,
    "fact_checks_performed": 45000,
    "sentiment_distribution": {
      "positive": 0.35,
      "negative": 0.25,
      "neutral": 0.40
    },
    "credibility_distribution": {
      "high_credibility": 0.45,
      "medium_credibility": 0.35,
      "low_credibility": 0.20
    },
    "topic_coverage": {
      "politics": 0.25,
      "technology": 0.20,
      "business": 0.15,
      "health": 0.12,
      "science": 0.10
    },
    "geographic_distribution": {
      "north_america": 0.40,
      "europe": 0.30,
      "asia": 0.20,
      "other": 0.10
    },
    "temporal_trends": {
      "sentiment_volatility": 0.15,
      "credibility_trend": "increasing",
      "article_volume_trend": "stable"
    }
  },
  "data_quality": {
    "analysis_accuracy": 0.952,
    "source_verification_rate": 0.987,
    "fact_check_coverage": 0.823,
    "update_frequency": "real-time"
  },
  "api_usage": {
    "total_requests": 1250000,
    "unique_researchers": 850,
    "institutional_users": 120,
    "most_requested_topics": ["AI", "Climate Change", "Politics"]
  },
  "api_key_hash": "a1b2c3d4"
}
```

**Rate Limit**: 100 req/hr
**Authentication**: Required

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid/missing API key)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found (resource doesn't exist)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "retry_after": 60  // for rate limiting
}
```

### Common Errors

#### Rate Limit Exceeded
```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 3600
}
```

#### Authentication Required
```json
{
  "detail": "Valid API key required for research endpoints. Contact admin@justnews.ai for access.",
  "error_code": "AUTHENTICATION_REQUIRED"
}
```

#### Invalid API Key
```json
{
  "detail": "Invalid API key provided",
  "error_code": "INVALID_API_KEY"
}
```

## Rate Limiting Details

### Headers
All responses include rate limiting headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1638360000
X-RateLimit-Retry-After: 3600
```

### Rate Limit Logic
- **Public**: 1000 requests per hour per IP address
- **Research**: 100 requests per hour per API key
- **Reset**: Hourly at the top of each hour
- **Tracking**: In-memory with automatic cleanup

## Client Libraries

### Python Client

```python
import requests

class JustNewsClient:
    def __init__(self, base_url="https://api.justnews.ai/v1", api_key=None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get_stats(self):
        response = self.session.get(f"{self.base_url}/stats")
        return response.json()

    def get_articles(self, **params):
        response = self.session.get(f"{self.base_url}/articles", params=params)
        return response.json()

    def export_articles(self, **params):
        if not self.api_key:
            raise ValueError("API key required for research endpoints")
        response = self.session.get(
            f"{self.base_url}/export/articles",
            params=params,
            headers=self._get_headers()
        )
        return response.json()

# Usage
client = JustNewsClient()
stats = client.get_stats()

# Research usage
research_client = JustNewsClient(api_key="your-api-key")
export_data = research_client.export_articles(format="json", topic="technology")
```

### JavaScript Client

```javascript
class JustNewsClient {
    constructor(baseURL = 'https://api.justnews.ai/v1', apiKey = null) {
        this.baseURL = baseURL;
        this.apiKey = apiKey;
    }

    getHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        return headers;
    }

    async getStats() {
        const response = await fetch(`${this.baseURL}/stats`);
        return response.json();
    }

    async getArticles(params = {}) {
        const url = new URL(`${this.baseURL}/articles`);
        Object.keys(params).forEach(key =>
            url.searchParams.append(key, params[key])
        );
        const response = await fetch(url);
        return response.json();
    }

    async exportArticles(params = {}) {
        if (!this.apiKey) {
            throw new Error('API key required for research endpoints');
        }
        const url = new URL(`${this.baseURL}/export/articles`);
        Object.keys(params).forEach(key =>
            url.searchParams.append(key, params[key])
        );
        const response = await fetch(url, {
            headers: this.getHeaders()
        });
        return response.json();
    }
}

// Usage
const client = new JustNewsClient();
const stats = await client.getStats();

// Research usage
const researchClient = new JustNewsClient('https://api.justnews.ai/v1', 'your-api-key');
const exportData = await researchClient.exportArticles({
    format: 'json',
    topic: 'technology'
});
```

## Usage Examples

### Basic Article Search

```bash
# Get recent technology articles
curl "https://api.justnews.ai/v1/articles?topic=technology&limit=10&sort=newest"

# Search for specific content
curl "https://api.justnews.ai/v1/articles?search=artificial+intelligence&sentiment=positive"

# Get article details
curl "https://api.justnews.ai/v1/article/article-123"
```

### Research Data Export

```bash
# Export technology articles from 2025
curl -H "Authorization: Bearer your-api-key" \
  "https://api.justnews.ai/v1/export/articles?format=json&topic=technology&date_from=2025-01-01&min_credibility=80"
```

### Analytics and Trends

```bash
# Get trending topics
curl "https://api.justnews.ai/v1/trending-topics?limit=5"

# Get temporal analysis
curl "https://api.justnews.ai/v1/temporal-analysis?topic=politics&interval=week"

# Get research metrics
curl -H "Authorization: Bearer your-api-key" \
  "https://api.justnews.ai/v1/research/metrics?granularity=month"
```

## Data Schema

### Article Object
```json
{
  "id": "string",
  "title": "string",
  "summary": "string",
  "content": "string",
  "url": "string",
  "source": "string",
  "source_credibility": "integer",
  "published_date": "ISO8601 datetime",
  "author": "string",
  "sentiment_score": "float",
  "fact_check_score": "integer",
  "bias_score": "float",
  "emotional_tone": "string",
  "readability_score": "string",
  "topics": ["string"],
  "word_count": "integer",
  "reading_time": "integer",
  "detailed_analysis": {
    "sentiment_breakdown": {...},
    "bias_analysis": {...},
    "fact_check_details": {...},
    "readability_metrics": {...},
    "engagement_metrics": {...}
  }
}
```

## Support and Contact

### Getting Help
- **Documentation**: https://docs.justnews.ai
- **API Status**: https://status.justnews.ai
- **Support**: support@justnews.ai
- **API Key Requests**: admin@justnews.ai

### Rate Limit Increases
For research projects requiring higher rate limits, contact admin@justnews.ai with:
- Research institution/organization
- Project description and goals
- Expected API usage volume
- Data usage justification

### Bug Reports and Feature Requests
- **GitHub Issues**: https://github.com/Adrasteon/JustNewsAgent/issues
- **API Feedback**: api-feedback@justnews.ai

## Changelog

### v1.0.0 (2025-09-20)
- Initial public API release
- Authentication and rate limiting implementation
- Real-time data integration via MCP bus
- Comprehensive article search and filtering
- Research data export capabilities
- Temporal analysis and trending topics
- Source credibility rankings
- Fact-check database access

---

*This documentation covers the JustNews Public API v1.0.0. The API is designed for both public access and academic research with appropriate security and rate limiting measures.*</content>
<parameter name="filePath">/home/adra/justnewsagent/JustNewsAgent/markdown_docs/agent_documentation/PUBLIC_API_DOCUMENTATION.md