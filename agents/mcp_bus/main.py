"""
Main file for the MCP Bus.
"""
# main.py for MCP Message Bus
import atexit
import time
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.observability import get_logger

app = FastAPI()

ready = False

# Configure centralized logging
logger = get_logger(__name__)

# Register common shutdown endpoint (after logger is configured)
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for mcp_bus")

# Register reload endpoint if available
try:
    from agents.common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for mcp_bus")

agents = {}
cb_state = {}
CB_FAIL_THRESHOLD = 3
CB_COOLDOWN_SEC = 10

class Agent(BaseModel):
    name: str
    address: str

class ToolCall(BaseModel):
    agent: str
    tool: str
    args: list
    kwargs: dict

@app.post("/register")
def register_agent(agent: Agent):
    logger.info(f"Registering agent: {agent.name} at {agent.address}")
    agents[agent.name] = agent.address
    # Reset circuit breaker on registration
    cb_state[agent.name] = {"fails": 0, "open_until": 0}
    return {"status": "ok"}

@app.post("/call")
def call_tool(call: ToolCall):
    if call.agent not in agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {call.agent}")

    agent_name = call.agent
    agent_address = agents[agent_name]

    # Circuit breaker check
    state = cb_state.get(agent_name, {"fails": 0, "open_until": 0})
    now = time.time()
    if state.get("open_until", 0) > now:
        raise HTTPException(status_code=503, detail=f"Circuit open for agent {agent_name}")

    payload = {"args": call.args, "kwargs": call.kwargs}
    url = f"{agent_address}/{call.tool}"
    timeout = (3, 10)  # (connect, read)

    # Simple retry with backoff
    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            # Success: reset failures
            cb_state[agent_name] = {"fails": 0, "open_until": 0}
            return response.json()
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            time.sleep(0.2 * (2 ** attempt))

    # Failure after retries: increment failure count
    fails = state.get("fails", 0) + 1
    if fails >= CB_FAIL_THRESHOLD:
        cb_state[agent_name] = {"fails": 0, "open_until": now + CB_COOLDOWN_SEC}
        logger.warning(f"Circuit opened for {agent_name} for {CB_COOLDOWN_SEC}s after failures")
    else:
        cb_state[agent_name] = {"fails": fails, "open_until": 0}

    raise HTTPException(status_code=502, detail=f"Tool call failed: {last_error}")

@app.get("/agents")
def get_agents():
    return agents


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready_endpoint():
    return {"ready": ready}

@asynccontextmanager
async def lifespan(app):
    logger.info("MCP_Bus is starting up.")
    try:
        response = requests.get("http://localhost:8000/register", timeout=(2, 5))
        response.raise_for_status()
        logger.info("MCP Bus connected successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to MCP Bus: {e}")
    global ready
    ready = True
    yield
    logger.info("MCP_Bus is shutting down.")

atexit.register(lambda: logger.info("MCP_Bus has exited."))

# Attach lifespan context if available (defined above). Use router.lifespan_context to avoid
# referencing lifespan before it is declared earlier in the module.
try:
    # starlette exposes router.lifespan_context to set an asynccontextmanager
    app.router.lifespan_context = lifespan  # type: ignore[attr-defined]
except Exception:
    # If assigning fails, the module will still run with a default no-op lifespan
    logger.debug("Could not attach custom lifespan to MCP Bus router; continuing without it.")
