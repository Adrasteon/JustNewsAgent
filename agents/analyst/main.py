"""
Main file for the Analyst Agent.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from .tools import (
    identify_entities,
    analyze_text_statistics,
    extract_key_metrics,
    analyze_content_trends,
    log_feedback,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Readiness flag
ready = False

# Environment variables
ANALYST_AGENT_PORT = int(os.environ.get("ANALYST_AGENT_PORT", 8004))
MODEL_PATH = os.environ.get("MISTRAL_7B_PATH", "./models/mistral-7b-instruct-v0.2")
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Pydantic models
class ToolCall(BaseModel):
    args: list
    kwargs: dict

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
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("🔍 Analyst Agent V2 - Specialized Quantitative Analysis")
    logger.info("📊 Focus: Entity extraction, statistical analysis, numerical metrics")
    logger.info("🎯 Specialization: Text statistics, trends, financial/temporal data")
    logger.info("🤝 Integration: Works with Scout V2 for comprehensive content analysis")
    
    logger.info("Specialized analysis modules loaded and ready")

    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="analyst",
            agent_address=f"http://localhost:{ANALYST_AGENT_PORT}",
            tools=[
                "identify_entities",
                "analyze_text_statistics", 
                "extract_key_metrics",
                "analyze_content_trends"
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    # Mark ready after successful startup tasks
    global ready
    ready = True
    yield
    
    # Cleanup on shutdown
    logger.info("✅ Analyst agent shutdown completed.")

    logger.info("Analyst agent is shutting down.")

app = FastAPI(lifespan=lifespan)

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for analyst")

# Register reload endpoint if available
try:
    from agents.common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for analyst")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    """Readiness endpoint for startup gating."""
    return {"ready": ready}

# REMOVED ENDPOINTS - Sentiment and bias analysis centralized in Scout V2 Agent
# Use Scout V2 for all sentiment and bias analysis:
# - POST /comprehensive_content_analysis (includes sentiment + bias)  
# - POST /analyze_sentiment (dedicated sentiment analysis)
# - POST /detect_bias (dedicated bias detection)

# @app.post("/score_bias") - REMOVED
# @app.post("/score_sentiment") - REMOVED
# @app.post("/analyze_sentiment_and_bias") - REMOVED

@app.post("/identify_entities")
def identify_entities_endpoint(call: ToolCall):
    """Identifies entities in a given text."""
    try:
        return identify_entities(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in identify_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_feedback")
def log_feedback_endpoint(call: ToolCall):
    """Logs feedback."""
    try:
        feedback = call.kwargs.get("feedback", {})
        log_feedback("log_feedback", feedback)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# REMOVED ENDPOINTS - All sentiment and bias analysis centralized in Scout V2 Agent
# Use Scout V2 for all sentiment and bias analysis (including batch operations):
# - POST /comprehensive_content_analysis (includes sentiment + bias)  
# - POST /analyze_sentiment (dedicated sentiment analysis)
# - POST /detect_bias (dedicated bias detection)

# The following TensorRT batch endpoints have been removed from Analyst:
# - POST /score_bias_batch - REMOVED (use Scout V2 batch analysis)
# - POST /score_sentiment_batch - REMOVED (use Scout V2 batch analysis) 
# - POST /analyze_article - REMOVED (use Scout V2 comprehensive analysis)
# - POST /analyze_articles_batch - REMOVED (use Scout V2 batch analysis)

# Engine information endpoint
@app.post("/analyze_text_statistics")
def analyze_text_statistics_endpoint(call: ToolCall):
    """Analyzes text statistics including readability and complexity."""
    try:
        return analyze_text_statistics(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in analyze_text_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_key_metrics")
def extract_key_metrics_endpoint(call: ToolCall):
    """Extracts key numerical and statistical metrics from text."""
    try:
        return extract_key_metrics(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in extract_key_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_content_trends")
def analyze_content_trends_endpoint(call: ToolCall):
    """Analyzes trends across multiple content pieces."""
    try:
        return analyze_content_trends(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in analyze_content_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# REMOVED ENDPOINT - Combined sentiment and bias analysis centralized in Scout V2 Agent
# Use Scout V2 /comprehensive_content_analysis endpoint for combined analysis

# @app.post("/analyze_sentiment_and_bias") - REMOVED