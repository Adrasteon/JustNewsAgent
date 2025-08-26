# JustNews V4 - GitHub Copilot Instructions

## Project Overview

JustNews V4 is a production-ready multi-agent news analysis system featuring GPU-accelerated processing, continuous learning, and distributed architecture. The system processes news content through specialized AI agents that communicate via MCP (Model Context Protocol) for collaborative analysis, fact-checking, and synthesis.

**Current Status** (August 9, 2025): Production deployment with **Synthesizer V3** complete, documentation organization restructured, and **V2 engines completion** phase initiated.

## Development Standards

### Code Quality Requirements
- **Python Style**: Follow PEP 8 with 88-character line limit
- **Type Hints**: Required for all function signatures and class attributes
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Error Handling**: Comprehensive exception handling with specific error types
- **Logging**: Structured logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Code Organization Patterns
```python
# Standard agent structure pattern
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ToolCall(BaseModel):
    """Standard MCP tool call format"""
    args: list
    kwargs: dict

@app.post("/tool_name")
def tool_endpoint(call: ToolCall) -> Dict[str, Any]:
    """Tool endpoint with proper error handling"""
    try:
        from tools import tool_function
        result = tool_function(*call.args, **call.kwargs)
        logger.info(f"Tool {tool_name} executed successfully")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Performance Guidelines
- **GPU Memory Management**: Use context managers and proper cleanup
- **Batch Processing**: Implement 16-32 item batches for optimal GPU utilization
- **Memory Monitoring**: Include memory usage logging in GPU operations
- **Fallback Systems**: Always provide CPU fallback for GPU operations

## Architecture Guidelines

### Agent Communication Protocol
**Critical**: All agents must follow the MCP (Model Context Protocol) pattern:

```python
# MCP Bus Integration Pattern
def call_agent_tool(agent: str, tool: str, *args, **kwargs) -> Any:
    """Standard pattern for inter-agent communication"""
    payload = {
        "agent": agent,
        "tool": tool,
        "args": list(args),
        "kwargs": kwargs
    }
    response = requests.post(f"{MCP_BUS_URL}/call", json=payload)
    response.raise_for_status()
    return response.json()
```

### Core Components Architecture

#### MCP Bus (Port 8000)
- Central communication hub using FastAPI
- Agent registration with `/register`, `/call`, `/agents` endpoints
- Health check and service discovery

#### Agents (Ports 8001-8008)
- **Scout** (8002): Content discovery with 5-model AI architecture
- **Analyst** (8004): TensorRT-accelerated sentiment/bias analysis  
- **Fact Checker** (8003): 5-model verification system
- **Synthesizer** (8005): **V3 Production** - 4-model synthesis (BERTopic, BART, FLAN-T5, SentenceTransformers)
- **Critic** (8006): Quality assessment and review
- **Chief Editor** (8001): Workflow orchestration
- **Memory** (8007): PostgreSQL + vector search storage
- **Reasoning** (8008): Nucleoid symbolic logic engine

### GPU Integration Standards

#### Native TensorRT Implementation (Production Pattern)
```python
# Professional GPU context management
class GPUAcceleratedAgent:
    def __init__(self, engines_dir="tensorrt_engines"):
        self.cuda_context = None
        self.context_created = False
        self._initialize_cuda_context()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Professional CUDA context cleanup"""
        if self.context_created and self.cuda_context:
            self.cuda_context.pop()
            torch.cuda.empty_cache()
```

#### Performance Targets
- **Production Validated**: 730+ articles/sec (TensorRT engines)
- **Memory Efficiency**: 2-3GB GPU buffer minimum for stability
- **Batch Optimization**: 100-article batches for maximum throughput
- **Error Rate**: Zero crashes, zero warnings in production

## File Organization

### Directory Structure Standards
```
JustNewsAgentic/
├── agents/                           # Agent implementations
│   ├── {agent_name}/
│   │   ├── main.py                  # FastAPI application
│   │   ├── tools.py                 # Business logic implementation
│   │   ├── requirements.txt         # Agent-specific dependencies
│   │   └── {agent_name}_engine.py   # AI model implementations
├── training_system/                  # Continuous learning system
├── mcp_bus/                         # Central communication hub
├── markdown_docs/                   # **ORGANIZED DOCUMENTATION**
│   ├── production_status/           # Achievement reports
│   ├── agent_documentation/         # Agent-specific guides
│   └── development_reports/         # Technical analysis
├── archive_obsolete_files/          # Development artifacts
└── docs/                           # V4 specifications
```

### Documentation Organization Rules ⚠️ **CRITICAL**
**ALL .md files except README.md and CHANGELOG.md MUST be in `markdown_docs/` subdirectories:**

1. **Production Status** → `markdown_docs/production_status/`
2. **Agent Documentation** → `markdown_docs/agent_documentation/`  
3. **Technical Reports** → `markdown_docs/development_reports/`
4. **Root Directory**: Only README.md, CHANGELOG.md permitted

### Development File Management
- **Test Files**: Archive to `archive_obsolete_files/development_session_[DATE]/`
- **Debug Scripts**: Never commit to main branch
- **Temporary Results**: Use `.gitignore` patterns for exclusion

## Development Workflow

### Feature Development Process
1. **Analysis Phase**: Use `semantic_search` and `grep_search` to understand existing patterns
2. **Implementation**: Follow established agent patterns and MCP integration
3. **Testing**: Implement comprehensive error handling and performance validation
4. **Documentation**: Update relevant `markdown_docs/` files
5. **Integration**: Ensure MCP bus compatibility and health checks

### Testing Requirements
```python
# Standard test pattern for agent tools
def test_agent_tool():
    """Test with proper error handling and performance measurement"""
    start_time = time.time()
    try:
        result = agent_tool(*test_args, **test_kwargs)
        assert result is not None
        assert "status" in result
        execution_time = time.time() - start_time
        logger.info(f"Tool executed in {execution_time:.3f}s")
        return result
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
```

### GPU Development Standards
```python
# Professional GPU resource management
@contextmanager
def gpu_context():
    """Context manager for safe GPU operations"""
    torch.cuda.set_device(0)
    initial_memory = torch.cuda.memory_allocated()
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        if final_memory > initial_memory:
            logger.warning(f"Potential memory leak: {final_memory - initial_memory} bytes")
```

## Current System Status (August 9, 2025)

### Production-Ready Components ✅
- **Synthesizer V3**: 4-model production stack (BERTopic, BART, FLAN-T5, SentenceTransformers)
- **Training System**: EWC-based continuous learning (48 examples/min, 82.3 updates/hour)
- **Native TensorRT**: 730+ articles/sec performance (Analyst agent)
- **Production Crawling**: 8.14 art/sec ultra-fast + 0.86 art/sec AI-enhanced
- **Documentation**: Professional organization in `markdown_docs/` structure
- **Reasoning Agent**: Complete Nucleoid implementation with AST parsing

### Development Priorities (V2 Engines Completion)
1. **Expand TensorRT**: Apply native GPU acceleration to remaining agents
2. **Training Integration**: Complete continuous learning across all agents  
3. **Performance Optimization**: Achieve 2-3GB memory buffer target
4. **Quality Assurance**: Maintain zero-crash, zero-warning operation

### Memory Allocation Strategy (RTX 3090)
- **Scout**: 8.0GB (LLaMA-3-8B + 5-model architecture)
- **Analyst**: 2.3GB (TensorRT optimized) ✅ Production
- **Synthesizer**: 3.0GB (4-model V3 stack) ✅ Production  
- **Other Agents**: 2.0-2.5GB each (DialoGPT-medium)
- **Target Buffer**: 2-3GB for system stability

## Error Handling Standards

### Exception Management Pattern
```python
class AgentError(Exception):
    """Base exception for agent operations"""
    pass

class ModelLoadError(AgentError):
    """Model loading failures"""
    pass

class GPUError(AgentError):
    """GPU operation failures"""
    pass

def safe_gpu_operation(operation, *args, **kwargs):
    """Standard GPU operation wrapper"""
    try:
        with gpu_context():
            return operation(*args, **kwargs)
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"CUDA error in {operation.__name__}: {e}")
            raise GPUError(f"GPU operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {operation.__name__}: {e}")
        raise AgentError(f"Operation failed: {e}")
```

### Logging Standards
```python
# Structured logging pattern
import structlog

logger = structlog.get_logger()

def log_performance(operation: str, duration: float, **kwargs):
    """Standard performance logging"""
    logger.info(
        "operation_completed",
        operation=operation,
        duration_ms=round(duration * 1000, 2),
        **kwargs
    )

def log_gpu_usage(operation: str):
    """GPU memory usage logging"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        logger.info(
            "gpu_memory_usage",
            operation=operation,
            memory_gb=round(memory_used, 2)
        )
```

## Security Guidelines

### API Security
- **Input Validation**: Use Pydantic models for all API inputs
- **Error Exposure**: Never expose internal paths or sensitive info in errors
- **Resource Limits**: Implement timeouts and memory limits
- **Health Checks**: Secure health endpoints without sensitive data

### Model Security
- **Model Validation**: Verify model checksums before loading
- **Safe Loading**: Use `trust_remote_code=False` unless explicitly required
- **Memory Protection**: Clear sensitive data from GPU memory after use

## Documentation Requirements

### Code Documentation
- **Inline Comments**: Explain complex logic, especially GPU operations
- **Performance Notes**: Document memory usage and optimization decisions
- **Error Handling**: Comment on exception handling strategy

### Change Documentation
- **CHANGELOG.md**: Required for all releases with performance metrics
- **Agent Updates**: Document in appropriate `markdown_docs/agent_documentation/`
- **Technical Changes**: Add to `markdown_docs/development_reports/`

### Validation Checklist
Before any commit:
- [ ] Code follows PEP 8 and type hint requirements
- [ ] Error handling implemented with specific exceptions
- [ ] Performance logging included for GPU operations
- [ ] Documentation updated in correct `markdown_docs/` location
- [ ] Tests include error cases and performance validation
- [ ] GPU memory cleanup verified
- [ ] MCP integration tested if applicable
- [ ] CHANGELOG.md updated with metrics

## Quick Reference

### Latest Achievements (August 9, 2025)
- ✅ **Synthesizer V3 Production**: 4-model stack, 5/5 tests passed
- ✅ **Documentation Restructure**: Professional GitHub standards
- ✅ **Workspace Organization**: Clean root directory, archived development files
- ✅ **Continuous Learning**: 48 examples/min, EWC-based training

### Key Performance Metrics
- **TensorRT Production**: 730+ articles/sec (4.8x improvement)
- **Production Crawling**: 8.14 art/sec ultra-fast processing
- **Training System**: 82.3 model updates/hour across agents
- **Memory Efficiency**: 2.3GB GPU utilization (highly optimized)

### Critical References
- **Technical Architecture**: `markdown_docs/TECHNICAL_ARCHITECTURE.md`
- **Development History**: `markdown_docs/DEVELOPMENT_CONTEXT.md`
- **Production Status**: `markdown_docs/production_status/`
- **V4 Specifications**: `docs/JustNews_Proposal_V4.md`, `docs/JustNews_Plan_V4.md`

---

**Remember**: This is a production system. All changes must maintain system stability, follow established patterns, and include comprehensive error handling and performance monitoring.
