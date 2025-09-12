---
title: Scout Agent - Enhanced Deep Crawl Documentation
description: Auto-generated description for Scout Agent - Enhanced Deep Crawl Documentation
tags: [documentation]
status: current
last_updated: 2025-09-12
---

# Scout Agent - Enhanced Deep Crawl Documentation

**JustNews V4 Scout Agent with Native Crawl4AI Integration**

*Last Updated: July 29, 2025*  
*Status: ✅ Production Ready - Integration Testing Completed Successfully*

---

## 🌐 Overview

The Scout Agent has been enhanced with native Crawl4AI integration featuring BestFirstCrawlingStrategy for advanced web crawling capabilities. This implementation combines intelligent crawling strategies with Scout Intelligence analysis to deliver high-quality, filtered content discovery.

## 🚀 Key Features

### Native Crawl4AI Integration
- **Version**: Crawl4AI 0.7.2 with BestFirstCrawlingStrategy
- **Advanced Crawling**: Intelligent content prioritization and discovery
- **Filter Chain**: ContentTypeFilter and DomainFilter for focused crawling
- **Performance Optimized**: Asynchronous processing with batch optimization

### Scout Intelligence Engine
- **GPU-Accelerated Analysis**: LLaMA-3-8B model for content quality assessment
- **Comprehensive Analysis**: News classification, bias detection, quality metrics
- **Quality Scoring**: Dynamic threshold-based content selection
- **Recommendation System**: AI-powered content recommendation and filtering

### User-Configurable Parameters
- **max_depth**: Maximum crawl depth (default: 3, user requested)
- **max_pages**: Maximum pages to crawl (default: 100, user requested)
---
title: Scout — Enhanced Deep Crawl with Crawl4AI
description: Production guide for the Scout agent’s native Crawl4AI integration and best-first crawling with quality filtering.
tags: [agent, scout, crawl4ai, crawling, quality, gpu]
status: active
last_updated: 2025-09-12
---

# Scout Agent - Enhanced Deep Crawl Documentation

JustNews V4 Scout Agent with Native Crawl4AI Integration

Last Updated: July 29, 2025  
Status: Production Ready — Integration Testing Completed Successfully

## 🌐 Overview

The Scout Agent has been enhanced with native Crawl4AI integration featuring BestFirstCrawlingStrategy for advanced web crawling capabilities. This implementation combines intelligent crawling strategies with Scout Intelligence analysis to deliver high-quality, filtered content discovery.

## 🚀 Key Features

### Native Crawl4AI Integration
- Version: Crawl4AI 0.7.2 with BestFirstCrawlingStrategy
- Advanced Crawling: Intelligent content prioritization and discovery
- Filter Chain: ContentTypeFilter and DomainFilter for focused crawling
- Performance Optimized: Asynchronous processing with batch optimization

### Scout Intelligence Engine
- GPU-Accelerated Analysis: LLaMA-3-8B model for content quality assessment
- Comprehensive Analysis: News classification, bias detection, quality metrics
- Quality Scoring: Dynamic threshold-based content selection
- Recommendation System: AI-powered content recommendation and filtering

### User-Configurable Parameters
- max_depth: Maximum crawl depth (default: 3)
- max_pages: Maximum pages to crawl (default: 100)
- word_count_threshold: Minimum word count for content inclusion (default: 500)
- quality_threshold: Scout intelligence quality score threshold (configurable: 0.05-0.8)
- analyze_content: Enable/disable Scout Intelligence analysis (default: True)

## �️ Technical Implementation

### Core Function: enhanced_deep_crawl_site()

```python
async def enhanced_deep_crawl_site(
    url: str,
    max_depth: int = 3,
    max_pages: int = 100,
    word_count_threshold: int = 500,
    quality_threshold: float = 0.6,
    analyze_content: bool = True
) -> List[Dict]
```

Parameters:
- url: Target website URL for crawling
- max_depth: Maximum crawl depth
- max_pages: Maximum number of pages to crawl
- word_count_threshold: Minimum word count for content inclusion
- quality_threshold: Scout intelligence quality score threshold
- analyze_content: Enable Scout Intelligence analysis

Returns:
- List of dictionaries containing crawled content with Scout analysis

### BestFirstCrawlingStrategy Configuration

```python
strategy = BestFirstCrawlingStrategy(
    max_depth=max_depth,
    max_pages=max_pages,
    filter_chain=FilterChain([
        ContentTypeFilter(["text/html"]),
        DomainFilter(allowed_domains=[domain])
    ]),
    word_count_threshold=word_count_threshold
)
```

### Scout Intelligence Analysis

```python
analysis = scout_engine.comprehensive_content_analysis(content, url)
scout_score = analysis.get("scout_score", 0.0)

# Quality filtering
if scout_score >= quality_threshold:
    result["scout_analysis"] = analysis
    result["scout_score"] = scout_score
    result["recommendation"] = analysis.get("recommendation", "")
    result["is_news"] = analysis.get("news_classification", {}).get("is_news", False)
    result["quality_metrics"] = analysis.get("quality_assessment", {})
    result["bias_analysis"] = analysis.get("bias_analysis", {})
```

## 🎯 Production Performance

### Integration Test Results
- Test Target: https://news.sky.com
- Content Volume: 148,000 characters crawled
- Processing Time: 1.3 seconds
- Scout Intelligence Score: 0.10 (quality assessment)
- Quality Filtering: Operational with configurable thresholds

### System Performance
- Crawling Speed: Native async processing with BestFirstCrawlingStrategy
- Analysis Speed: GPU-accelerated LLaMA-3-8B content analysis
- Memory Efficiency: Optimized GPU utilization with intelligent batching
- Reliability: Automatic Docker fallback for enhanced system stability

## 🔄 MCP Bus Integration

### Agent Registration
The enhanced Scout agent automatically registers with the MCP Bus at startup:

```python
def register_with_mcp_bus():
    try:
        response = requests.post(f"{MCP_BUS_URL}/register", json={
            "agent_name": "scout",
            "agent_url": "http://localhost:8002",
            "tools": [
                "discover_sources", "crawl_url", "deep_crawl_site", "enhanced_deep_crawl_site",
                "search_web", "verify_url", "analyze_webpage", "get_page_text",
                "extract_links", "check_robots_txt", "get_site_structure"
            ]
        })
        if response.status_code == 200:
            logger.info("Scout agent registered with MCP Bus successfully")
        else:
            logger.warning(f"Scout agent registration failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not register with MCP Bus: {e}")
```

### Tool Endpoint
```python
@app.post("/enhanced_deep_crawl_site")
async def enhanced_deep_crawl_site_endpoint(call: ToolCall):
    try:
        from tools import enhanced_deep_crawl_site
        logger.info(f"Calling enhanced_deep_crawl_site with args: {call.args} and kwargs: {call.kwargs}")
        return await enhanced_deep_crawl_site(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in enhanced_deep_crawl_site: {e}")
        return {"error": str(e)}
```

## 🧪 Testing Framework

### Integration Testing
Complete test suite available in test_enhanced_deepcrawl_integration.py:

- MCP Bus Testing: Validates agent registration and tool calling via bus
- Direct API Testing: Tests Scout agent endpoints directly
- Performance Validation: Measures crawling speed and analysis quality
- Quality Assessment: Validates Scout Intelligence scoring and filtering

### Test Execution
```bash
# Run integration tests
python test_enhanced_deepcrawl_integration.py

# Expected output: Enhanced deep crawl SUCCESS with performance metrics
```

## 📦 Dependencies

### Core Requirements
```text
crawl4ai>=0.7.0
asyncio
aiohttp
requests
fastapi
uvicorn
torch
transformers
```

### Environment Setup
```bash
# Activate rapids environment
conda activate rapids-25.06

# Install Crawl4AI
pip install crawl4ai>=0.7.0

# Verify installation
python -c "from crawl4ai import AsyncWebCrawler, BestFirstCrawlingStrategy; print('Crawl4AI ready')"
```

## 🚀 Deployment

### Native Scout Agent Startup
```bash
cd /home/adra/JustNewsAgentic/agents/scout
python start_enhanced_scout.py
```

### Service Health Check
```bash
curl -s http://localhost:8002/health
# Expected: {"status":"ok"}
```

### MCP Bus Integration Check
```bash
curl -s http://localhost:8000/agents
# Expected: scout agent listed in registered agents
```

## 🔧 Configuration Options

### Quality Threshold Settings
- High Quality (0.6-0.8): Strict filtering for premium content
- Medium Quality (0.3-0.6): Balanced filtering for general use
- Low Quality (0.05-0.3): Permissive filtering for maximum coverage
- Development (0.05): Testing threshold for validation

### Crawling Parameters
- Depth Control: max_depth parameter controls crawling depth
- Volume Control: max_pages parameter limits total pages crawled
- Content Filtering: word_count_threshold ensures substantial content
- Domain Focus: BestFirstCrawlingStrategy prioritizes relevant domains

## � Quality Metrics

### Scout Intelligence Analysis
- News Classification: Identifies genuine news content vs. opinion/blog posts
- Bias Detection: Analyzes political and ideological bias in content
- Quality Assessment: Evaluates content quality, credibility, and newsworthiness
- Recommendation: Provides AI-powered content recommendations

### Performance Indicators
- Scout Score: Composite quality score (0.0-1.0)
- Processing Speed: Content analysis time per article
- Filtering Efficiency: Ratio of high-quality to total content discovered
- System Reliability: Uptime and error rate metrics

## 🛠️ Troubleshooting

### Common Issues
1. Crawl4AI Import Error: Ensure rapids-25.06 environment is activated and Crawl4AI is installed
2. Scout Intelligence Unavailable: GPU Scout engine initialization may fail — system will operate in web-crawling only mode
3. MCP Bus Registration Failed: Check that MCP Bus is running on port 8000
4. Quality Threshold Too High: Adjust quality_threshold parameter for more permissive filtering

### Debug Commands
```bash
# Check Crawl4AI installation
python -c "import crawl4ai; print(f'Crawl4AI version: {crawl4ai.__version__}')"

# Verify Scout agent service
curl -s http://localhost:8002/health

# Test enhanced deep crawl directly
python -c "
import asyncio
from agents.scout.tools import enhanced_deep_crawl_site
result = asyncio.run(enhanced_deep_crawl_site('https://news.sky.com', max_pages=5, quality_threshold=0.05))
print(f'Results: {len(result)} pages found')
"
```

## � Future Enhancements

### Planned Improvements
- Multi-Domain Crawling: Support for crawling multiple domains simultaneously
- Advanced Filtering: Enhanced filter chains with custom content filters
- Caching System: Intelligent content caching for improved performance
- Analytics Dashboard: Real-time crawling and analysis metrics visualization

### Integration Roadmap
- TensorRT Optimization: Migrate Scout Intelligence to native TensorRT for enhanced performance
- Distributed Crawling: Multi-agent crawling coordination for large-scale content discovery
- ML Pipeline Integration: Enhanced integration with downstream analysis agents

## See also

- OPERATIONS_QUICK_REFERENCE.md
- preflight_runbook.md
- MCP_BUS_OPERATIONS.md
- GPU_ORCHESTRATOR_OPERATIONS.md
- OPERATOR_GUIDE_SYSTEMD.md
