"""
Main file for the Dashboard Agent.
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from contextlib import asynccontextmanager
import sys
import os

# Ensure the current package directory is on sys.path so sibling modules can be imported
# This makes `from config import load_config` work when running the FastAPI app directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, save_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_agent")

# Load configuration
config = load_config()
# Default dashboard port set to 8011 to avoid conflicts with other agents (e.g., balancer at 8010)
DASHBOARD_AGENT_PORT = config.get("dashboard_port", 8011)
MCP_BUS_URL = config.get("mcp_bus_url", "http://localhost:8000")

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="dashboard",
            agent_address=f"http://localhost:{DASHBOARD_AGENT_PORT}",
            tools=["get_status", "send_command", "receive_logs"],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    global ready
    ready = True
    yield
    logger.info("Dashboard agent is shutting down.")
    save_config(config)

app = FastAPI(lifespan=lifespan)

ready = False

# Register shutdown endpoint
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for dashboard")

class ToolCall(BaseModel):
    args: list
    kwargs: dict

@app.get("/get_status")
def get_status():
    """Fetch the status of all agents."""
    try:
        response = requests.get(f"{MCP_BUS_URL}/agents")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred while fetching agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "agent": "dashboard"}


@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

@app.post("/send_command")
def send_command(call: ToolCall):
    """Send a command to another agent."""
    try:
        # Use model_dump() for Pydantic v2 compatibility; fall back to dict() when unavailable
        response = requests.post(
            f"{MCP_BUS_URL}/call",
            json=(call.model_dump() if hasattr(call, "model_dump") else call.dict()),
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"An error occurred while sending a command: {e}")
        raise HTTPException(status_code=500, detail=str(e))
