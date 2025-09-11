"""
GPU Orchestrator Service

Centralized, conservative GPU coordination service with SAFE_MODE awareness.
Provides read-only GPU telemetry and placeholder allocation/policy endpoints
to be expanded in later phases. Designed to run under systemd via the
standard justnews-start-agent.sh launcher.
"""

import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.observability import get_logger


logger = get_logger(__name__)

# Environment configuration
GPU_ORCHESTRATOR_PORT = int(os.environ.get("GPU_ORCHESTRATOR_PORT", 8014))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
SAFE_MODE = os.environ.get("SAFE_MODE", "false").lower() == "true"


# In-memory state (intentionally simple/minimal for safety)
READINESS: bool = False
POLICY: Dict[str, Any] = {
	"max_memory_per_agent_mb": 2048,
	"allow_fractional_shares": True,
	"kill_on_oom": False,
	"safe_mode_read_only": SAFE_MODE,
}
ALLOCATIONS: Dict[str, Dict[str, Any]] = {}


class MCPBusClient:
	def __init__(self, base_url: str = MCP_BUS_URL):
		self.base_url = base_url

	def register_agent(self, agent_name: str, agent_address: str, tools: List[str]):
		import requests

		registration_data = {
			"name": agent_name,
			"address": agent_address,
		}
		try:
			response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(2, 5))
			response.raise_for_status()
			logger.info(f"Registered {agent_name} with MCP Bus")
		except requests.exceptions.RequestException as e:
			logger.warning(f"MCP Bus unavailable for registration: {e}")


def _run_nvidia_smi() -> Optional[str]:
	"""Run nvidia-smi and return raw XML or CSV output, or None if unavailable."""
	# Prefer CSV for simpler parsing at this stage
	cmd = [
		"nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
		"--format=csv,noheader,nounits",
	]
	try:
		output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=3)
		return output
	except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
		logger.debug(f"nvidia-smi unavailable or failed: {e}")
		return None


def _parse_nvidia_smi_csv(csv_text: str) -> List[Dict[str, Any]]:
	gpus: List[Dict[str, Any]] = []
	for line in csv_text.strip().splitlines():
		parts = [p.strip() for p in line.split(",")]
		if len(parts) < 7:
			continue
		try:
			gpus.append(
				{
					"index": int(parts[0]),
					"name": parts[1],
					"memory_total_mb": float(parts[2]),
					"memory_used_mb": float(parts[3]),
					"utilization_gpu_pct": float(parts[4]),
					"temperature_c": float(parts[5]),
					"power_draw_w": float(parts[6]),
					"memory_utilization_pct": (
						(float(parts[3]) / float(parts[2]) * 100.0) if float(parts[2]) > 0 else 0.0
					),
				}
			)
		except ValueError:
			# Skip malformed rows
			continue
	return gpus


def get_gpu_snapshot() -> Dict[str, Any]:
	"""Return a conservative, read-only snapshot of GPU state."""
	smi = _run_nvidia_smi()
	if smi is None:
		return {"gpus": [], "available": False, "message": "nvidia-smi not available"}
	gpus = _parse_nvidia_smi_csv(smi)
	return {"gpus": gpus, "available": True}


class PolicyUpdate(BaseModel):
	max_memory_per_agent_mb: Optional[int] = Field(None, ge=256, description="Per-agent memory cap in MB")
	allow_fractional_shares: Optional[bool] = None
	kill_on_oom: Optional[bool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
	global READINESS
	logger.info("GPU Orchestrator starting up")

	# Try registering to MCP Bus (best-effort)
	try:
		client = MCPBusClient()
		client.register_agent(
			agent_name="gpu_orchestrator",
			agent_address=f"http://localhost:{GPU_ORCHESTRATOR_PORT}",
			tools=["health", "gpu_info", "get_policy", "set_policy", "get_allocations"],
		)
	except Exception:
		pass

	READINESS = True
	yield
	logger.info("GPU Orchestrator shutting down")


app = FastAPI(title="GPU Orchestrator", lifespan=lifespan)


# Optional shared endpoints if available
try:
	from agents.common.shutdown import register_shutdown_endpoint

	register_shutdown_endpoint(app)
except Exception:
	logger.debug("shutdown endpoint not registered for gpu_orchestrator")

try:
	from agents.common.reload import register_reload_endpoint

	register_reload_endpoint(app)
except Exception:
	logger.debug("reload endpoint not registered for gpu_orchestrator")


@app.get("/health")
def health():
	return {"status": "ok", "safe_mode": SAFE_MODE}


@app.get("/ready")
def ready():
	return {"ready": READINESS}


@app.get("/gpu/info")
def gpu_info():
	"""Return current GPU telemetry (read-only)."""
	try:
		return get_gpu_snapshot()
	except Exception as e:
		logger.error(f"Failed to get GPU snapshot: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/policy")
def get_policy():
	return POLICY


@app.post("/policy")
def set_policy(update: PolicyUpdate):
	if SAFE_MODE:
		# In SAFE_MODE, accept but do not enact changes (read-only posture)
		return {
			**POLICY,
			"note": "SAFE_MODE enabled: policy updates accepted but not enacted",
		}

	changed = False
	if update.max_memory_per_agent_mb is not None:
		POLICY["max_memory_per_agent_mb"] = int(update.max_memory_per_agent_mb)
		changed = True
	if update.allow_fractional_shares is not None:
		POLICY["allow_fractional_shares"] = bool(update.allow_fractional_shares)
		changed = True
	if update.kill_on_oom is not None:
		POLICY["kill_on_oom"] = bool(update.kill_on_oom)
		changed = True

	if changed:
		logger.info(f"Updated GPU policy: {POLICY}")
	return POLICY


@app.get("/allocations")
def get_allocations():
	"""Return current agent→GPU allocation view (placeholder)."""
	return {"allocations": ALLOCATIONS}


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=GPU_ORCHESTRATOR_PORT)

