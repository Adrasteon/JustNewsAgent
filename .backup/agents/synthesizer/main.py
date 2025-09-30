"""
Main file for the Synthesizer Agent.
"""
# main.py for Synthesizer Agent
import os
from contextlib import asynccontextmanager
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.observability import get_logger
from common.metrics import JustNewsMetrics

# Configure centralized logging
logger = get_logger(__name__)

ready = False

# Environment variables
SYNTHESIZER_AGENT_PORT = int(os.environ.get("SYNTHESIZER_AGENT_PORT", 8005))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(2, 5))
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Synthesizer agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="synthesizer",
            agent_address=f"http://localhost:{SYNTHESIZER_AGENT_PORT}",
            tools=["cluster_articles", "neutralize_text", "aggregate_cluster",
                   "synthesize_news_articles_gpu", "get_synthesizer_performance"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")

    # Note: Models will be downloaded automatically by HuggingFace transformers when first used

    global ready
    ready = True
    yield
    logger.info("Synthesizer agent is shutting down.")

app = FastAPI(lifespan=lifespan)

# Initialize metrics
metrics = JustNewsMetrics("synthesizer")
app.middleware("http")(metrics.request_middleware)

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for synthesizer")

# Register reload endpoint if available
try:
    from agents.common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for synthesizer")

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

# Metrics endpoint
@app.get("/metrics")
def get_metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(content=metrics.get_metrics(), media_type="text/plain")

@app.post("/log_feedback")
def log_feedback(call: ToolCall):
    try:
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "feedback": call.kwargs.get("feedback")
        }
        logger.info(f"Logging feedback: {feedback_data}")
        return feedback_data
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/aggregate_cluster")
def aggregate_cluster_endpoint(call: ToolCall):
    try:
        # Use relative import to resolve the agent-local tools module
        from .tools import aggregate_cluster as _aggregate_cluster
        logger.info(f"Calling aggregate_cluster with args: {call.args} and kwargs: {call.kwargs}")
        return _aggregate_cluster(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in aggregate_cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster_articles")
def cluster_articles_endpoint(call: ToolCall):
    try:
        # Use relative import to resolve the agent-local tools module
        from .tools import cluster_articles as _cluster_articles
        logger.info(f"Calling cluster_articles with args: {call.args} and kwargs: {call.kwargs}")
        return _cluster_articles(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in cluster_articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/neutralize_text")
def neutralize_text_endpoint(call: ToolCall):
    try:
        # Use relative import to resolve the agent-local tools module
        from .tools import neutralize_text as _neutralize_text
        logger.info(f"Calling neutralize_text with args: {call.args} and kwargs: {call.kwargs}")
        return _neutralize_text(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in neutralize_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# GPU-accelerated endpoints (V4 performance implementation)
@app.post("/synthesize_news_articles_gpu")
def synthesize_news_articles_gpu_endpoint(call: ToolCall):
    """GPU-accelerated news article synthesis endpoint"""
    try:
        from .gpu_tools import synthesize_news_articles_gpu
        # Normalize input: support args[0] or kwargs['articles']
        articles = []
        if call.args and len(call.args) > 0 and isinstance(call.args[0], list):
            articles = call.args[0]
        elif isinstance(call.kwargs.get("articles"), list):
            articles = call.kwargs.get("articles", [])

        logger.info(f"Calling GPU synthesize with {len(articles)} articles")
        # Only pass supported parameter to the tool to avoid unexpected kwargs errors
        result = synthesize_news_articles_gpu(articles)

        # Log performance for monitoring
        if result.get('success') and 'performance' in result:
            perf = result['performance']
            logger.info(f"✅ GPU synthesis: {perf['articles_per_sec']:.1f} articles/sec")

        return result
    except Exception as e:
        logger.error(f"❌ GPU synthesis error: {e}")
        # Graceful fallback to CPU implementation
        try:
            # Use relative import to resolve the agent-local tools module
            from .tools import aggregate_cluster as _aggregate_cluster
            from .tools import cluster_articles as _cluster_articles
            logger.info("🔄 Falling back to CPU synthesis")
            # Simple fallback implementation
            articles = []
            if call.args and len(call.args) > 0 and isinstance(call.args[0], list):
                articles = call.args[0]
            elif isinstance(call.kwargs.get("articles"), list):
                articles = call.kwargs.get("articles", [])
            clusters = _cluster_articles(articles)
            synthesis = _aggregate_cluster(clusters)
            return {
                "success": True,
                "themes": [{"theme_name": "General News", "articles": articles}],
                "synthesis": synthesis,
                "performance": {"articles_per_sec": 1.0, "gpu_used": False}
            }
        except Exception as fallback_error:
            logger.error(f"❌ CPU fallback failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"GPU synthesis failed: {e}, CPU fallback failed: {fallback_error}")

@app.post("/get_synthesizer_performance")
def get_synthesizer_performance_endpoint(call: ToolCall):
    """Get synthesizer performance statistics"""
    try:
        from .gpu_tools import get_synthesizer_performance
        logger.info("Retrieving synthesizer performance stats")
        return get_synthesizer_performance(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"❌ Performance stats error: {e}")
        # Return basic stats if GPU tools unavailable
        return {
            "total_processed": 0,
            "gpu_processed": 0,
            "cpu_processed": 0,
            "gpu_allocated": False,
            "models_loaded": False,
            "error": str(e)
        }

# MCP compatibility alias: older clients may call /synthesize_content
@app.post("/synthesize_content")
def synthesize_content_alias(call: ToolCall):
    """Alias for compatibility with existing E2E tests; delegates to GPU implementation."""
    try:
        # Reuse GPU pathway for best performance; args/kwargs are identical
        return synthesize_news_articles_gpu_endpoint(call)
    except Exception as e:
        logger.error(f"❌ synthesize_content alias failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os

    host = os.environ.get("SYNTHESIZER_HOST", "0.0.0.0")
    port = int(os.environ.get("SYNTHESIZER_PORT", 8005))

    logger.info(f"Starting Synthesizer Agent on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
