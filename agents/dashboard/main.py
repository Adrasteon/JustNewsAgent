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
import json
import time
from typing import Dict, List, Optional

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

class GPUMonitor:
    """GPU monitoring and management class for the dashboard."""

    def __init__(self):
        self.gpu_history = []
        self.max_history_size = 100

    def get_gpu_info(self) -> Dict:
        """Get current GPU information using nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,power.limit',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpus = []
                for line in lines:
                    if line.strip():
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 10:
                            gpu_info = {
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_used_mb': int(parts[2]),
                                'memory_total_mb': int(parts[3]),
                                'memory_free_mb': int(parts[4]),
                                'gpu_utilization_percent': int(parts[5]),
                                'memory_utilization_percent': int(parts[6]),
                                'temperature_celsius': int(parts[7]),
                                'fan_speed_percent': int(parts[8]) if parts[8] != '[Not Supported]' else 0,
                                'power_draw_watts': float(parts[9]) if parts[9] != '[Not Supported]' else 0.0,
                                'power_limit_watts': float(parts[10]) if len(parts) > 10 and parts[10] != '[Not Supported]' else 0.0,
                                'timestamp': time.time()
                            }
                            gpus.append(gpu_info)

                # Store in history
                current_data = {
                    'timestamp': time.time(),
                    'gpus': gpus
                }
                self.gpu_history.append(current_data)
                if len(self.gpu_history) > self.max_history_size:
                    self.gpu_history.pop(0)

                return {
                    'status': 'success',
                    'gpu_count': len(gpus),
                    'gpus': gpus,
                    'timestamp': time.time()
                }
            else:
                return {
                    'status': 'error',
                    'message': f'nvidia-smi command failed: {result.stderr}',
                    'timestamp': time.time()
                }
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def get_gpu_history(self, hours: int = 1) -> List[Dict]:
        """Get GPU history for the specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [entry for entry in self.gpu_history if entry['timestamp'] >= cutoff_time]

    def get_agent_gpu_usage(self) -> Dict:
        """Get GPU usage statistics per agent."""
        try:
            # This would integrate with the GPU manager to get per-agent usage
            # For now, return mock data based on agent activity
            agent_ports = {
                'scout': 8002,
                'fact_checker': 8003,
                'analyst': 8004,
                'synthesizer': 8005,
                'critic': 8006,
                'memory': 8007,
                'newsreader': 8009
            }

            agent_usage = {}
            for agent_name, port in agent_ports.items():
                try:
                    # Check if agent is active
                    response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    is_active = response.status_code == 200
                except:
                    is_active = False

                # Mock GPU usage based on agent type and activity
                base_usage = {
                    'scout': {'memory_mb': 800, 'utilization_percent': 15},
                    'fact_checker': {'memory_mb': 600, 'utilization_percent': 12},
                    'analyst': {'memory_mb': 400, 'utilization_percent': 8},
                    'synthesizer': {'memory_mb': 1000, 'utilization_percent': 20},
                    'critic': {'memory_mb': 500, 'utilization_percent': 10},
                    'memory': {'memory_mb': 300, 'utilization_percent': 6},
                    'newsreader': {'memory_mb': 900, 'utilization_percent': 18}
                }.get(agent_name, {'memory_mb': 0, 'utilization_percent': 0})

                agent_usage[agent_name] = {
                    'active': is_active,
                    'memory_used_mb': base_usage['memory_mb'] if is_active else 0,
                    'gpu_utilization_percent': base_usage['utilization_percent'] if is_active else 0,
                    'last_updated': time.time()
                }

            return {
                'status': 'success',
                'agents': agent_usage,
                'total_memory_used_mb': sum(agent['memory_used_mb'] for agent in agent_usage.values()),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting agent GPU usage: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

# Global GPU monitor instance
gpu_monitor = GPUMonitor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="dashboard",
            agent_address=f"http://localhost:{DASHBOARD_AGENT_PORT}",
            tools=["get_status", "send_command", "receive_logs", "get_gpu_info", "get_gpu_history", "get_agent_gpu_usage", "get_gpu_config", "update_gpu_config"],
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

# Register reload endpoint if available
try:
    from agents.common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for dashboard")

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

# GPU Monitoring Endpoints

@app.get("/gpu/info")
def get_gpu_info():
    """Get current GPU information and status."""
    try:
        gpu_info = gpu_monitor.get_gpu_info()
        return gpu_info
    except Exception as e:
        logger.error(f"Error in get_gpu_info endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/history")
def get_gpu_history(hours: int = 1):
    """Get GPU usage history for the specified number of hours."""
    try:
        history = gpu_monitor.get_gpu_history(hours)
        return {
            "status": "success",
            "hours": hours,
            "data_points": len(history),
            "history": history,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error in get_gpu_history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/agents")
def get_agent_gpu_usage():
    """Get GPU usage statistics per agent."""
    try:
        agent_usage = gpu_monitor.get_agent_gpu_usage()
        return agent_usage
    except Exception as e:
        logger.error(f"Error in get_agent_gpu_usage endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/config")
def get_gpu_config():
    """Get current GPU configuration from the GPU manager."""
    try:
        # Try to get config from GPU manager
        response = requests.get("http://localhost:8000/gpu/config", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback to local config
            return {
                "status": "success",
                "source": "dashboard_fallback",
                "config": config.get("gpu_config", {}),
                "timestamp": time.time()
            }
    except Exception as e:
        logger.error(f"Error getting GPU config: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        }

@app.post("/gpu/config")
def update_gpu_config(new_config: Dict):
    """Update GPU configuration."""
    try:
        # Try to update via GPU manager
        response = requests.post("http://localhost:8000/gpu/config", json=new_config, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback to local update
            if "gpu_config" not in config:
                config["gpu_config"] = {}
            config["gpu_config"].update(new_config)
            save_config(config)
            return {
                "status": "success",
                "source": "dashboard_fallback",
                "message": "Configuration updated locally",
                "config": config["gpu_config"],
                "timestamp": time.time()
            }
    except Exception as e:
        logger.error(f"Error updating GPU config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/dashboard")
def get_gpu_dashboard_data():
    """Get comprehensive GPU dashboard data."""
    try:
        gpu_info = gpu_monitor.get_gpu_info()
        agent_usage = gpu_monitor.get_agent_gpu_usage()
        gpu_config = get_gpu_config()

        return {
            "status": "success",
            "timestamp": time.time(),
            "gpu_info": gpu_info,
            "agent_usage": agent_usage,
            "gpu_config": gpu_config,
            "summary": {
                "total_gpus": gpu_info.get("gpu_count", 0),
                "total_memory_used_mb": agent_usage.get("total_memory_used_mb", 0),
                "active_agents": sum(1 for agent in agent_usage.get("agents", {}).values() if agent.get("active", False)),
                "gpu_utilization_avg": sum(gpu.get("gpu_utilization_percent", 0) for gpu in gpu_info.get("gpus", [])) / max(1, len(gpu_info.get("gpus", [])))
            }
        }
    except Exception as e:
        logger.error(f"Error in get_gpu_dashboard_data endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
