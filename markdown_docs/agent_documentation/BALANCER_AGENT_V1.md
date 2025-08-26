# Balancer Agent V1 Documentation

## Overview
The Balancer Agent is a production-ready component of the JustNews V4 system, designed to neutralize news articles, integrate counter-balancing statements, and ensure objective reporting. It leverages existing JustNews models for sentiment, bias, fact-checking, summarization, and semantic embeddings, and introduces a quote extraction model for enhanced content integration.

## API Endpoints
- `/register`: Agent registration with MCP bus
- `/health`: Health check endpoint
- `/endpoints`: Service discovery
- `/balance_article`: Balance an article using alternative sources and quotes
- `/extract_quotes`: Extract quoted statements from an article
- `/analyze_article`: Analyze sentiment, bias, and fact-checking for an article
- `/chief_editor/balance_article`: Chief Editor workflow integration

## Multi-Agent Interoperability
- Delegates sentiment, bias, fact-checking, and summarization tasks to Analyst, Fact Checker, and Synthesizer agents via MCP bus
- Fallback to local models if remote agents are unavailable

## Performance Optimization
- GPU usage and batch processing monitoring
- Configurable batch size for scalable operation
- Structured logging for all major operations

## Testing & Validation
- Comprehensive unit and integration tests in `test_balancer_agent.py`
- Validates endpoint responses, error handling, and output neutrality

## Integration Steps
1. Register agent with MCP bus
2. Expose tool endpoints via FastAPI
3. Integrate with Chief Editor and multi-agent workflows
4. Enable multi-agent delegation and fallback
5. Optimize GPU and batch processing
6. Document API and workflow integration
7. Connect to training system for continuous learning

## Usage Example
See `balancer.py` for implementation details and endpoint usage.

## Change Log
- V1: Initial release with full JustNews compliance

---
For further details, see the technical architecture and development reports in `markdown_docs/`.
