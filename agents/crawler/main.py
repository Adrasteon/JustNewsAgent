"""
Main file for the Crawler Agent.
"""
# main.py for Crawler Agent

import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from common.observability import get_logger
from common.metrics import JustNewsMetrics

# Configure logging
logger = get_logger(__name__)

ready = False

# Environment variables
CRAWLER_AGENT_PORT = int(os.environ.get("CRAWLER_AGENT_PORT", 8015))  # Updated to 8015 per canonical port mapping
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Security configuration
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(1, 2))
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Crawler agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="crawler",
            agent_address=f"http://localhost:{CRAWLER_AGENT_PORT}",
            tools=[
                "unified_production_crawl",
                "get_crawler_info",
                "get_performance_metrics"
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    global ready
    ready = True
    yield
    logger.info("Crawler agent is shutting down.")

app = FastAPI(lifespan=lifespan, title="Crawler Agent", description="Unified production crawling agent")

# Initialize metrics
metrics = JustNewsMetrics("crawler")

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Metrics middleware (must be added after CORS middleware)
app.middleware("http")(metrics.request_middleware)

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.post("/unified_production_crawl")
async def unified_production_crawl_endpoint(call: ToolCall):
    try:
        from agents.crawler.unified_production_crawler import unified_production_crawl
        logger.info(f"Calling unified_production_crawl with args: {call.args} and kwargs: {call.kwargs}")
        return await unified_production_crawl(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in unified_production_crawl: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_crawler_info")
def get_crawler_info_endpoint(call: ToolCall):
    try:
        from agents.crawler.unified_production_crawler import get_crawler_info
        logger.info(f"Calling get_crawler_info with args: {call.args} and kwargs: {call.kwargs}")
        return get_crawler_info(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in get_crawler_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_performance_metrics")
def get_performance_metrics_endpoint(call: ToolCall):
    try:
        from agents.crawler.performance_monitoring import get_performance_metrics
        logger.info(f"Calling get_performance_metrics with args: {call.args} and kwargs: {call.kwargs}")
        return get_performance_metrics(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in get_performance_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(metrics.get_metrics(), media_type="text/plain; charset=utf-8")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Crawler Agent on port {CRAWLER_AGENT_PORT}")
    uvicorn.run(
        "agents.crawler.main:app",
        host="0.0.0.0",
        port=CRAWLER_AGENT_PORT,
        reload=False,
        log_level="info"
    )
