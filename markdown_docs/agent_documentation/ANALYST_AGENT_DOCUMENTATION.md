---
title: Analyst Agent Documentation
description: Complete documentation for the JustNews Analyst Agent including GPU acceleration, troubleshooting, and operations
tags: [analyst, gpu, sentiment, bias, documentation, troubleshooting]
status: current
last_updated: 2025-09-25
version: v4
---

# Analyst Agent Documentation

## Overview

The Analyst Agent is a specialized AI-powered component of the JustNewsAgent system responsible for quantitative content analysis, focusing on sentiment analysis and bias detection. The agent features GPU-accelerated processing using TensorRT-optimized models for high-performance analysis of news content.

**Key Features:**
- GPU-accelerated sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest)
- GPU-accelerated bias detection (unitary/toxic-bert)
- Automatic fallback to heuristic methods when GPU unavailable
- Real-time analysis with comprehensive confidence scoring
- Integrated entity extraction and statistical analysis

## Architecture

### Core Components

#### GPU Analyst (`gpu_analyst.py`)
- **Purpose**: GPU-accelerated analysis implementation
- **Models**: RoBERTa-based sentiment model, BERT-based bias detection
- **Memory Management**: MPS allocation with circuit breaker protection
- **Fallback Logic**: Automatic CPU fallback when GPU unavailable

#### Analysis Tools (`tools.py`)
- **Purpose**: Main analysis orchestration functions
- **Functions**:
  - `analyze_sentiment_and_bias()`: Combined sentiment and bias analysis
  - `analyze_sentiment()`: Sentiment analysis with GPU-first logic
  - `detect_bias()`: Bias detection with GPU-first logic
- **Response Structure**: Nested dictionaries with method indicators

#### FastAPI Server (`main.py`)
- **Port**: 8004
- **Endpoints**:
  - `POST /analyze_sentiment_and_bias`: Combined analysis endpoint
  - `POST /analyze_sentiment`: Sentiment-only analysis
  - `POST /detect_bias`: Bias-only detection
- **MCP Integration**: Registered with MCP bus for inter-agent communication

### GPU Acceleration Architecture

#### MPS Resource Management
```python
# GPU Memory Allocation Structure
{
  "mps_resource_allocation": {
    "agent_allocations": {
      "analyst": {
        "mps_memory_limit_gb": 2.3,
        "gpu_device": "cuda:0",
        "allocation_timestamp": "2025-09-25T18:29:00Z"
      }
    }
  }
}
```

#### Model Loading Strategy
- **Initialization**: Models loaded on GPU at startup if orchestrator allows
- **Memory Protection**: Circuit breaker prevents memory exhaustion
- **Health Checks**: Continuous GPU availability monitoring
- **Fallback**: Keyword-based heuristic analysis when GPU unavailable

## GPU Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Articles Analyzed with Heuristic Methods Instead of GPU

**Symptoms:**
- Database shows `method: "heuristic"` instead of `method: "gpu_accelerated"`
- GPU orchestrator reports healthy status
- Individual function calls work with GPU

**Root Causes:**
1. MCP bus communication failures
2. GPU orchestrator timing windows
3. Validation logic bugs
4. Service initialization inconsistencies

**Diagnostic Steps:**
```bash
# 1. Check GPU orchestrator health
curl http://localhost:8014/health

# 2. Check GPU info directly
curl http://localhost:8014/gpu_info

# 3. Test analyst endpoint directly
curl -X POST http://localhost:8004/analyze_sentiment_and_bias \
  -H "Content-Type: application/json" \
  -d '{"text": "Test article text", "article_id": "test-123"}'

# 4. Verify MPS allocation
curl http://localhost:8014/mps_allocation
```

**Resolution Steps:**

1. **Fix MCP Communication Issues**
   ```python
   # Use direct HTTP calls instead of MCP bus
   response = requests.get(f"{gpu_orchestrator_url}/gpu_info")
   ```

2. **Correct Validation Logic**
   ```python
   # Use correct JSON path for MPS allocation
   analyst_allocation = mps_allocation.get("mps_resource_allocation", {})
     .get("agent_allocations", {}).get("analyst", {})
   ```

3. **Ensure Proper Service Initialization**
   ```bash
   # Kill manual processes
   pkill -f "uvicorn.*analyst.*main"
   lsof -ti:8004 | xargs kill -9 2>/dev/null || true

   # Start via systemd for consistent initialization
   sudo systemctl start justnews@analyst
   ```

#### Issue: GPU Models Not Loading

**Symptoms:**
- GPU analyst reports `models_loaded: false`
- GPU orchestrator allows GPU usage
- CUDA errors in logs

**Resolution:**
```python
# Check GPU availability
from agents.analyst.gpu_analyst import get_gpu_analyst
gpu_analyst = get_gpu_analyst()
print(f"GPU available: {gpu_analyst.gpu_available}")
print(f"Models loaded: {gpu_analyst.models_loaded}")

# Force model reload
gpu_analyst.load_models()
```

#### Issue: MPS Memory Allocation Errors

**Symptoms:**
- MPS allocation fails with memory errors
- GPU utilization spikes unexpectedly

**Resolution:**
```bash
# Check current MPS status
curl http://localhost:8014/mps_status

# Reset MPS allocation
curl -X POST http://localhost:8014/reset_mps

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Performance Monitoring

#### Key Metrics to Monitor
- **Analysis Method Distribution**: GPU vs heuristic usage percentage
- **Processing Performance**: Articles/second processing rate
- **GPU Memory Usage**: MPS allocation per agent
- **Model Loading Success**: GPU model initialization status

#### Health Check Endpoints
```bash
# Analyst agent health
curl http://localhost:8004/health

# GPU orchestrator health
curl http://localhost:8014/health

# MCP bus connectivity
curl http://localhost:8000/agents
```

### Validation Patterns

#### GPU Usage Validation
```python
def validate_gpu_analysis(response_data):
    """Comprehensive GPU usage validation"""

    # Check method fields
    sentiment_method = response_data.get('sentiment_analysis', {}).get('method')
    bias_method = response_data.get('bias_analysis', {}).get('method')

    if sentiment_method == 'gpu_accelerated' and bias_method == 'gpu_accelerated':
        return True, "GPU accelerated"

    # Check for heuristic fallback
    if sentiment_method == 'heuristic' or bias_method == 'heuristic':
        return False, "Heuristic fallback detected"

    return False, "Unknown analysis method"
```

#### MPS Allocation Validation
```python
def validate_mps_allocation(mps_data):
    """Validate MPS memory allocation for analyst"""

    analyst_allocation = mps_data.get("mps_resource_allocation", {})
        .get("agent_allocations", {}).get("analyst", {})

    memory_limit = analyst_allocation.get("mps_memory_limit_gb", 0)
    if memory_limit >= 2.0:  # Minimum required for analyst
        return True, f"MPS allocated: {memory_limit}GB"

    return False, f"Insufficient MPS allocation: {memory_limit}GB"
```

## API Reference

### Combined Analysis Endpoint

**Endpoint:** `POST /analyze_sentiment_and_bias`

**Request:**
```json
{
  "text": "Article content to analyze",
  "article_id": "unique-article-identifier",
  "metadata": {
    "source": "news-source",
    "timestamp": "2025-09-25T18:29:00Z"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "sentiment_analysis": {
      "dominant_sentiment": "negative",
      "confidence": 0.95,
      "intensity": "strong",
      "sentiment_scores": {
        "positive": 0.025,
        "negative": 0.975,
        "neutral": 0.5
      },
      "method": "gpu_accelerated",
      "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
      "analysis_timestamp": "2025-09-25T18:29:06.360952"
    },
    "bias_analysis": {
      "has_bias": false,
      "bias_score": 0.001,
      "bias_level": "minimal",
      "confidence": 0.301,
      "political_bias": 0.001,
      "emotional_bias": 0.001,
      "factual_bias": 0.001,
      "method": "gpu_accelerated",
      "model_used": "unitary/toxic-bert",
      "timestamp": "2025-09-25T18:29:06.374302"
    },
    "combined_assessment": {
      "overall_reliability": 0.903,
      "content_quality_score": 0.819,
      "recommendations": ["Content shows strong negative sentiment - consider fact-checking"]
    },
    "analysis_timestamp": "2025-09-25T18:29:06.374589",
    "text_length": 3850,
    "method": "analyst_combined_analysis"
  }
}
```

### Individual Analysis Endpoints

#### Sentiment Analysis
**Endpoint:** `POST /analyze_sentiment`

#### Bias Detection
**Endpoint:** `POST /detect_bias`

## Configuration

### Environment Variables
```bash
# Agent Configuration
ANALYST_AGENT_PORT=8004
ANALYST_LOG_LEVEL=INFO

# GPU Configuration
GPU_ORCHESTRATOR_URL=http://localhost:8014
MPS_MEMORY_LIMIT_GB=2.3

# MCP Bus Configuration
MCP_BUS_URL=http://localhost:8000
AGENT_NAME=analyst
```

### GPU Orchestrator Integration
- **Registration**: Automatic registration with GPU orchestrator on startup
- **Resource Allocation**: Dynamic MPS memory allocation based on workload
- **Health Monitoring**: Continuous GPU availability and memory monitoring
- **Circuit Breaker**: Automatic fallback when GPU resources exhausted

## Deployment

### Systemd Service Configuration
```ini
[Unit]
Description=JustNews Analyst Agent
After=network.target gpu-orchestrator.service
Requires=gpu-orchestrator.service

[Service]
Type=simple
User=justnews
EnvironmentFile=/etc/justnews/global.env
ExecStart=/usr/bin/python3 /home/adra/justnewsagent/JustNewsAgent/agents/analyst/main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker Deployment
```dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04

# GPU and Python dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Copy agent code
COPY agents/analyst/ /app/
WORKDIR /app

# Start command
CMD ["python3", "main.py"]
```

## Testing

### Unit Tests
```bash
# Run analyst-specific tests
pytest tests/test_analyst_*

# GPU integration tests
pytest tests/test_gpu_analyst*

# Performance tests
pytest tests/test_analyst_segmented.py
```

### Integration Tests
```bash
# Full pipeline test
python3 test_analyst_segmented.py --articles 100 --force-reanalyze

# GPU validation test
python3 test_gpu_analyst_debug.py
```

### Performance Benchmarks
- **Target Performance**: 20+ articles/second with GPU acceleration
- **Memory Usage**: < 3GB MPS allocation
- **Accuracy**: > 90% confidence on sentiment analysis
- **Reliability**: 99.9% uptime with automatic fallback

## Monitoring and Observability

### Key Metrics
- Analysis method distribution (GPU vs heuristic)
- Processing latency and throughput
- GPU memory utilization
- Model loading success rate
- Error rates by analysis type

### Logging
```python
# Structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analyst_agent.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
- **Endpoint**: `GET /health`
- **Checks**: GPU availability, model loading status, MCP connectivity
- **Response**: JSON with component health status

## Troubleshooting Checklist

### Quick Diagnosis
1. **Check GPU orchestrator health**
   ```bash
   curl http://localhost:8014/health
   ```

2. **Verify analyst agent status**
   ```bash
   curl http://localhost:8004/health
   ```

3. **Test direct analysis**
   ```bash
   curl -X POST http://localhost:8004/analyze_sentiment_and_bias \
     -H "Content-Type: application/json" \
     -d '{"text": "Test", "article_id": "test"}'
   ```

4. **Check MPS allocation**
   ```bash
   curl http://localhost:8014/mps_allocation
   ```

### Common Issues Resolution

| Issue | Symptom | Solution |
|-------|---------|----------|
| MCP Bus Failure | 502 errors | Use direct HTTP calls |
| GPU Not Available | Heuristic fallback | Check orchestrator safe mode |
| Memory Exhaustion | MPS allocation errors | Reset MPS, check memory usage |
| Model Loading Failure | CUDA errors | Restart agent, check GPU status |
| Validation Errors | False GPU detection | Fix JSON path access |

## Future Enhancements

### Planned Improvements
- **Model Optimization**: TensorRT engine caching for faster startup
- **Batch Processing**: Multi-article GPU batch analysis
- **Advanced Metrics**: Confidence scoring improvements
- **Real-time Learning**: Online model adaptation

### Research Areas
- **Multi-modal Analysis**: Image and video content analysis
- **Cross-lingual Models**: Support for non-English content
- **Explainability**: Analysis reasoning transparency
- **Bias Detection**: Enhanced political bias classification

## References

### Related Documentation
- [GPU Orchestrator Operations](../agent_documentation/GPU_ORCHESTRATOR_OPERATIONS.md)
- [MCP Bus Operations](../agent_documentation/MCP_BUS_OPERATIONS.md)
- [GPU Analysis Resolution Report](../development_reports/GPU_ANALYSIS_RESOLUTION_REPORT.md)

### External Resources
- [HuggingFace Model Documentation](https://huggingface.co/docs)
- [PyTorch GPU Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/)

---

**Version**: v4 (GPU-Accelerated)
**Last Updated**: September 25, 2025
**Status**: Production Ready
**GPU Acceleration**: ✅ Enabled
**Performance**: 20.52 articles/second
**Reliability**: 99.9% with automatic fallback