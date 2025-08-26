"""
MCP Bus Integration for LLaVA NewsReader Agent
Integrates with JustNews V4 MCP Bus system
"""

from fastapi import FastAPI
import uvicorn
import requests
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ready = False

# Environment variables
NEWSREADER_AGENT_PORT = int(os.environ.get("NEWSREADER_AGENT_PORT", 8009))
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

# Ensure the current package directory is on sys.path so sibling modules can be imported
# This makes `from newsreader_agent import PracticalNewsReader` work when running the FastAPI app.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from newsreader_agent import PracticalNewsReader

# Global agent instance
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    global agent
    logger.info("NewsReader agent is starting up.")
    agent = PracticalNewsReader()
    logger.info("NewsReader Agent initialized")
    
    # Register with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="newsreader",
            agent_address=f"http://localhost:{NEWSREADER_AGENT_PORT}",
            tools=["extract_news_content", "capture_screenshot", "analyze_image"]
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    
    global ready
    ready = True
    yield
    
    # Shutdown
    logger.info("NewsReader Agent shutdown")
    # Cleanup if needed
    logger.info("NewsReader Agent shutdown complete")

app = FastAPI(
    title="NewsReader Agent", 
    description="LLaVA-based news content extraction",
    lifespan=lifespan
)

# Register common shutdown endpoint
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for newsreader")

class ToolCall(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]

@app.post("/extract_news_content")
async def extract_news_content_endpoint(call: ToolCall):
    """Extract news content from URL - MCP Bus compatible"""
    return await extract_news_content(*call.args, **call.kwargs)

@app.post("/capture_screenshot")
async def capture_screenshot(call: ToolCall):
    """Capture webpage screenshot - MCP Bus compatible"""
    if not agent:
        return {"error": "Agent not initialized"}
    
    url = call.args[0] if call.args else call.kwargs.get("url")
    screenshot_path = call.args[1] if len(call.args) > 1 else call.kwargs.get("screenshot_path", "page_llava.png")
    
    try:
        result_path = await agent.capture_screenshot(url, screenshot_path)
        return {"screenshot_path": result_path, "success": True}
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/analyze_screenshot")
async def analyze_screenshot(call: ToolCall):
    """Analyze screenshot with LLaVA - MCP Bus compatible"""
    if not agent:
        return {"error": "Agent not initialized"}
    
    image_path = call.args[0] if call.args else call.kwargs.get("image_path")
    
    try:
        result = agent.extract_content_with_llava(image_path)
        return result
    except Exception as e:
        return {"error": str(e), "success": False}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "agent": "newsreader",
        "model": "llava-v1.6-mistral-7b-optimized",
        "environment": "rapids-25.06",
        "lifespan": "modern"
    }

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

# Tool functions for direct import
async def extract_news_content(url: str, screenshot_path: str = None) -> Dict[str, Any]:
    """Extract news content from URL"""
    global agent
    if not agent:
        agent = PracticalNewsReader()
    
    result = await agent.process_news_url(url, screenshot_path)
    # Use model_dump() for Pydantic v2 compatibility; fall back to dict() when unavailable
    return (result.model_dump() if hasattr(result, "model_dump") else result.dict())

async def capture_webpage_screenshot(url: str, output_path: str = "page_llava.png") -> str:
    """Capture webpage screenshot"""
    global agent
    if not agent:
        agent = PracticalNewsReader()
    
    return await agent.capture_screenshot(url, output_path)

def analyze_image_content(image_path: str) -> Dict[str, str]:
    """Analyze image content with LLaVA"""
    global agent
    if not agent:
        agent = PracticalNewsReader()
    
    return agent.extract_content_with_llava(image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
