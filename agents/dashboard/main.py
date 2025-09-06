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
import time
from typing import Dict, List, Optional

# Ensure the current package directory is on sys.path so sibling modules can be imported
# This makes `from config import load_config` work when running the FastAPI app directly.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config, save_config

# Import production GPU manager
try:
    from agents.common.gpu_manager_production import get_gpu_manager
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False
    get_gpu_manager = None

# Import storage module
from .storage import get_storage

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

class EnhancedGPUMonitor:
    """Enhanced GPU monitoring class integrated with production GPU manager."""

    def __init__(self):
        self.gpu_history = []
        self.max_history_size = 100
        self.gpu_manager = get_gpu_manager() if GPU_MANAGER_AVAILABLE else None

    def get_gpu_info(self) -> Dict:
        """Get current GPU information using production GPU manager."""
        try:
            if self.gpu_manager:
                # Use production GPU manager for comprehensive data
                system_status = self.gpu_manager.get_system_status()
                gpu_statuses = system_status.get('gpu_statuses', [])

                if gpu_statuses:
                    gpus = []
                    for status in gpu_statuses:
                        gpu_info = {
                            'index': status['device_id'],
                            'name': f"GPU {status['device_id']}",  # Could be enhanced to get actual name
                            'memory_used_mb': int(status['used_memory_gb'] * 1024),
                            'memory_total_mb': int(status['total_memory_gb'] * 1024),
                            'memory_free_mb': int(status['free_memory_gb'] * 1024),
                            'gpu_utilization_percent': int(status['utilization_percent']),
                            'memory_utilization_percent': int((status['used_memory_gb'] / status['total_memory_gb']) * 100) if status['total_memory_gb'] > 0 else 0,
                            'temperature_celsius': int(status['temperature_c']),
                            'fan_speed_percent': 0,  # Not available from manager
                            'power_draw_watts': int(status['power_draw_w']),
                            'power_limit_watts': 0,  # Not available from manager
                            'is_healthy': status['is_healthy'],
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
                        'message': 'No GPU status data available from manager',
                        'timestamp': time.time()
                    }
            else:
                # Fallback to nvidia-smi if manager not available
                return self._get_nvidia_smi_fallback()

        except Exception as e:
            logger.error(f"Error getting GPU info from manager: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def _get_nvidia_smi_fallback(self) -> Dict:
        """Fallback GPU monitoring using nvidia-smi."""
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
                                'is_healthy': True,  # Assume healthy if nvidia-smi works
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

                # Store data in historical storage
                try:
                    storage.store_gpu_metrics(current_data)
                except Exception as e:
                    logger.warning(f"Failed to store GPU metrics: {e}")

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
        """Get GPU usage statistics per agent using production GPU manager."""
        try:
            if self.gpu_manager:
                # Get allocation data from production manager
                system_status = self.gpu_manager.get_system_status()
                allocations = system_status.get('active_allocations', 0)

                # Get per-agent allocation details
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
                total_memory_used = 0

                for agent_name, port in agent_ports.items():
                    try:
                        # Check if agent is active
                        response = requests.get(f"http://localhost:{port}/health", timeout=2)
                        is_active = response.status_code == 200
                    except (requests.RequestException, ConnectionError, TimeoutError) as e:
                        logger.debug(f"Failed to check agent {agent_name} health: {e}")
                        is_active = False

                    # Get allocation status from GPU manager
                    allocation_status = None
                    if self.gpu_manager:
                        allocation_status = self.gpu_manager.get_allocation_status(agent_name)

                    if allocation_status:
                        memory_used = allocation_status.get('allocated_memory_gb', 0) * 1024  # Convert to MB
                        gpu_util = 0  # Could be enhanced with actual utilization tracking
                    else:
                        # Mock data based on agent type and activity
                        base_usage = {
                            'scout': {'memory_mb': 800, 'utilization_percent': 15},
                            'fact_checker': {'memory_mb': 600, 'utilization_percent': 12},
                            'analyst': {'memory_mb': 400, 'utilization_percent': 8},
                            'synthesizer': {'memory_mb': 1000, 'utilization_percent': 20},
                            'critic': {'memory_mb': 500, 'utilization_percent': 10},
                            'memory': {'memory_mb': 300, 'utilization_percent': 6},
                            'newsreader': {'memory_mb': 900, 'utilization_percent': 18}
                        }.get(agent_name, {'memory_mb': 0, 'utilization_percent': 0})

                        memory_used = base_usage['memory_mb'] if is_active else 0
                        gpu_util = base_usage['utilization_percent'] if is_active else 0

                    agent_usage[agent_name] = {
                        'active': is_active,
                        'memory_used_mb': memory_used,
                        'gpu_utilization_percent': gpu_util,
                        'allocation_status': allocation_status,
                        'last_updated': time.time()
                    }

                    if is_active:
                        total_memory_used += memory_used

                return {
                    'status': 'success',
                    'agents': agent_usage,
                    'total_memory_used_mb': total_memory_used,
                    'active_allocations': allocations,
                    'gpu_manager_available': True,
                    'timestamp': time.time()
                }
            else:
                # Fallback to basic agent checking
                return self._get_agent_gpu_usage_fallback()

        except Exception as e:
            logger.error(f"Error getting agent GPU usage: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def _get_agent_gpu_usage_fallback(self) -> Dict:
        """Fallback agent GPU usage when GPU manager not available."""
        try:
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
            total_memory_used = 0

            for agent_name, port in agent_ports.items():
                try:
                    # Check if agent is active
                    response = requests.get(f"http://localhost:{port}/health", timeout=2)
                    is_active = response.status_code == 200
                except (requests.RequestException, ConnectionError, TimeoutError) as e:
                    logger.debug(f"Failed to check agent {agent_name} health (fallback): {e}")
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

                memory_used = base_usage['memory_mb'] if is_active else 0
                gpu_util = base_usage['utilization_percent'] if is_active else 0

                agent_usage[agent_name] = {
                    'active': is_active,
                    'memory_used_mb': memory_used,
                    'gpu_utilization_percent': gpu_util,
                    'last_updated': time.time()
                }

                if is_active:
                    total_memory_used += memory_used

            return {
                'status': 'success',
                'agents': agent_usage,
                'total_memory_used_mb': total_memory_used,
                'gpu_manager_available': False,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error getting agent GPU usage fallback: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

# Global enhanced GPU monitor instance
gpu_monitor = EnhancedGPUMonitor()

# Global storage instance
storage = get_storage()

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

@app.get("/gpu/manager/status")
def get_gpu_manager_status():
    """Get comprehensive GPU manager system status."""
    try:
        if gpu_monitor.gpu_manager:
            system_status = gpu_monitor.gpu_manager.get_system_status()
            return {
                'status': 'success',
                'gpu_manager_available': True,
                'system_status': system_status,
                'timestamp': time.time()
            }
        else:
            return {
                'status': 'success',
                'gpu_manager_available': False,
                'message': 'GPU manager not available, using fallback monitoring',
                'timestamp': time.time()
            }
    except Exception as e:
        logger.error(f"Error getting GPU manager status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/allocations")
def get_gpu_allocations():
    """Get all current GPU allocations."""
    try:
        if gpu_monitor.gpu_manager:
            # Get allocation data from GPU manager
            allocations = []
            agent_names = ['scout', 'fact_checker', 'analyst', 'synthesizer', 'critic', 'memory', 'newsreader']

            for agent_name in agent_names:
                allocation_status = gpu_monitor.gpu_manager.get_allocation_status(agent_name)
                if allocation_status:
                    allocations.append(allocation_status)

            return {
                'status': 'success',
                'allocations': allocations,
                'total_allocations': len(allocations),
                'timestamp': time.time()
            }
        else:
            return {
                'status': 'success',
                'gpu_manager_available': False,
                'allocations': [],
                'message': 'GPU manager not available',
                'timestamp': time.time()
            }
    except Exception as e:
        logger.error(f"Error getting GPU allocations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/metrics")
def get_gpu_metrics():
    """Get GPU performance metrics from the manager."""
    try:
        if gpu_monitor.gpu_manager:
            system_status = gpu_monitor.gpu_manager.get_system_status()
            metrics = system_status.get('metrics', {})

            return {
                'status': 'success',
                'gpu_manager_available': True,
                'metrics': metrics,
                'timestamp': time.time()
            }
        else:
            return {
                'status': 'success',
                'gpu_manager_available': False,
                'metrics': {},
                'message': 'GPU manager not available',
                'timestamp': time.time()
            }
    except Exception as e:
        logger.error(f"Error getting GPU metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/dashboard")
def get_gpu_dashboard_data():
    """Get comprehensive GPU dashboard data including manager integration."""
    try:
        gpu_info = gpu_monitor.get_gpu_info()
        agent_usage = gpu_monitor.get_agent_gpu_usage()
        gpu_config = get_gpu_config()

        # Get additional data from GPU manager if available
        manager_status = None
        allocations = []
        metrics = {}

        if gpu_monitor.gpu_manager:
            manager_status = gpu_monitor.gpu_manager.get_system_status()

            # Get allocations
            agent_names = ['scout', 'fact_checker', 'analyst', 'synthesizer', 'critic', 'memory', 'newsreader']
            for agent_name in agent_names:
                allocation_status = gpu_monitor.gpu_manager.get_allocation_status(agent_name)
                if allocation_status:
                    allocations.append(allocation_status)

            metrics = manager_status.get('metrics', {})

        # Calculate enhanced summary
        summary = {
            "total_gpus": gpu_info.get("gpu_count", 0),
            "total_memory_used_mb": agent_usage.get("total_memory_used_mb", 0),
            "active_agents": sum(1 for agent in agent_usage.get("agents", {}).values() if agent.get("active", False)),
            "gpu_utilization_avg": sum(gpu.get("gpu_utilization_percent", 0) for gpu in gpu_info.get("gpus", [])) / max(1, len(gpu_info.get("gpus", []))),
            "gpu_manager_available": gpu_monitor.gpu_manager is not None,
            "active_allocations": len(allocations),
            "total_allocation_requests": metrics.get('total_allocations', 0),
            "successful_allocations": metrics.get('successful_allocations', 0),
            "failed_allocations": metrics.get('failed_allocations', 0),
            "cpu_fallbacks": metrics.get('cpu_fallbacks', 0),
            "gpu_recoveries": metrics.get('gpu_recoveries', 0)
        }

        return {
            "status": "success",
            "timestamp": time.time(),
            "gpu_info": gpu_info,
            "agent_usage": agent_usage,
            "gpu_config": gpu_config,
            "gpu_manager": {
                "available": gpu_monitor.gpu_manager is not None,
                "status": manager_status,
                "allocations": allocations,
                "metrics": metrics
            },
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error in get_gpu_dashboard_data endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/history/db")
def get_gpu_history_from_db(hours: int = 24, gpu_index: Optional[int] = None, metric: str = "utilization"):
    """Get GPU metrics history from database."""
    try:
        if metric == "utilization":
            history = storage.get_gpu_metrics_history(hours, gpu_index, metric_type="utilization")
        elif metric == "memory":
            history = storage.get_gpu_metrics_history(hours, gpu_index, metric_type="memory_used_mb")
        elif metric == "temperature":
            history = storage.get_gpu_metrics_history(hours, gpu_index, metric_type="temperature_celsius")
        elif metric == "performance":
            # For performance, we'll use processing time or similar metrics
            history = storage.get_performance_trends(hours)
        else:
            history = storage.get_gpu_metrics_history(hours, gpu_index)

        # Format data for Chart.js
        formatted_data = []
        for entry in history:
            if metric == "utilization":
                value = entry.get('gpu_utilization_percent', 0)
            elif metric == "memory":
                value = entry.get('memory_used_mb', 0) / 1024  # Convert to GB
            elif metric == "temperature":
                value = entry.get('temperature_celsius', 0)
            elif metric == "performance":
                value = entry.get('processing_time_ms', 0)
            else:
                value = entry.get('gpu_utilization_percent', 0)

            formatted_data.append({
                'timestamp': entry.get('timestamp', 0),
                'value': value
            })

        return {
            'status': 'success',
            'hours': hours,
            'gpu_index': gpu_index,
            'metric': metric,
            'data_points': len(formatted_data),
            'data': formatted_data,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting GPU history from DB: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/allocations/history")
def get_allocation_history(hours: int = 24, agent_name: Optional[str] = None):
    """Get agent allocation history from database."""
    try:
        history = storage.get_agent_allocation_history(hours, agent_name)
        return {
            'status': 'success',
            'hours': hours,
            'agent_name': agent_name,
            'data_points': len(history),
            'history': history,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting allocation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/trends")
def get_performance_trends(hours: int = 24):
    """Get performance trends data."""
    try:
        trends = storage.get_performance_trends(hours)
        return {
            'status': 'success',
            'hours': hours,
            'trends': trends,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/alerts")
def get_recent_alerts(limit: int = 50):
    """Get recent alerts from database."""
    try:
        alerts = storage.get_recent_alerts(limit)
        return {
            'status': 'success',
            'limit': limit,
            'alerts': alerts,
            'total_alerts': len(alerts),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/stats")
def get_storage_stats():
    """Get database storage statistics."""
    try:
        stats = storage.get_storage_stats()
        return {
            'status': 'success',
            'storage_stats': stats,
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_fallback_dashboard_html():
    """Fallback HTML dashboard if template file is not available."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JustNewsAgent GPU Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .metric { background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🚀 JustNewsAgent GPU Dashboard</h1>
        <div id="dashboard">
            <div class="metric">
                <h3>GPU Status</h3>
                <p>Loading...</p>
            </div>
        </div>
        <button onclick="loadData()">Refresh</button>

        <script>
            async function loadData() {
                try {
                    const response = await fetch('/gpu/dashboard');
                    const data = await response.json();
                    updateDashboard(data);
                } catch (error) {
                    console.error('Error:', error);
                }
            }

            function updateDashboard(data) {
                const dashboard = document.getElementById('dashboard');
                if (data.status === 'success') {
                    dashboard.innerHTML = `
                        <div class="metric">
                            <h3>GPU Summary</h3>
                            <p>Total GPUs: ${data.summary.total_gpus}</p>
                            <p>Active Agents: ${data.summary.active_agents}</p>
                            <p>Avg Utilization: ${data.summary.gpu_utilization_avg.toFixed(1)}%</p>
                        </div>
                    `;
                }
            }

            // Auto-refresh every 5 seconds
            setInterval(loadData, 5000);
            loadData();
        </script>
    </body>
    </html>
    """
