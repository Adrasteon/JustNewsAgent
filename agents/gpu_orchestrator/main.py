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
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from common.observability import get_logger


logger = get_logger(__name__)

# Environment configuration
GPU_ORCHESTRATOR_PORT = int(os.environ.get("GPU_ORCHESTRATOR_PORT", 8014))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")
SAFE_MODE = os.environ.get("SAFE_MODE", "false").lower() == "true"
ENABLE_NVML = os.environ.get("ENABLE_NVML", "false").lower() == "true"
LEASE_TTL_SECONDS = int(os.environ.get("GPU_ORCHESTRATOR_LEASE_TTL", "3600"))  # 1 hour default; 0 disables TTL


# In-memory state (intentionally simple/minimal for safety)
READINESS: bool = False
POLICY: Dict[str, Any] = {
	"max_memory_per_agent_mb": 2048,
	"allow_fractional_shares": True,
	"kill_on_oom": False,
	"safe_mode_read_only": SAFE_MODE,
}
ALLOCATIONS: Dict[str, Dict[str, Any]] = {}

# Simple in-process metrics (avoid external deps). Prometheus exposition via /metrics.
_METRICS_COUNTERS: Dict[str, int] = {
	"requests_total": 0,
	"gpu_info_requests_total": 0,
	"policy_get_requests_total": 0,
	"policy_post_requests_total": 0,
	"lease_requests_total": 0,
	"release_requests_total": 0,
	"lease_expired_total": 0,
}
_START_TIME = time.time()
_NVML_SUPPORTED = False
_NVML_HANDLE_CACHE: Dict[int, Any] = {}
_NVML_INIT_ERROR: Optional[str] = None

# Model preload state (single global job)
_MODEL_PRELOAD_STATE: Dict[str, Any] = {
	"started_at": None,
	"completed_at": None,
	"in_progress": False,
	"summary": {"total": 0, "done": 0, "failed": 0},
	"per_agent": {},  # { agent: { model_id: {status, error, duration_s} } }
}
_MODEL_PRELOAD_THREAD: Optional[object] = None


def _inc(metric: str) -> None:
	_METRICS_COUNTERS[metric] = _METRICS_COUNTERS.get(metric, 0) + 1
	_METRICS_COUNTERS["requests_total"] = _METRICS_COUNTERS.get("requests_total", 0) + 1


def _purge_expired_leases() -> None:
	"""Remove expired leases based on LEASE_TTL_SECONDS.

	Executed opportunistically at read/write endpoints to avoid background threads.
	"""
	if LEASE_TTL_SECONDS <= 0 or not ALLOCATIONS:
		return
	now = time.time()
	expired: List[str] = []
	for token, alloc in list(ALLOCATIONS.items()):
		try:
			started = float(alloc.get("timestamp", now))
		except Exception:  # noqa: BLE001
			started = now
		if now - started > LEASE_TTL_SECONDS:
			expired.append(token)
	for token in expired:
		ALLOCATIONS.pop(token, None)
	if expired:
		_METRICS_COUNTERS["lease_expired_total"] = _METRICS_COUNTERS.get("lease_expired_total", 0) + len(expired)


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
	# If NVML enrichment enabled and not SAFE_MODE, merge fields
	if ENABLE_NVML and not SAFE_MODE and _NVML_SUPPORTED:
		for g in gpus:
			idx = g.get("index")
			if idx in _NVML_HANDLE_CACHE:
				try:  # pragma: no cover - environment dependent
					import pynvml  # type: ignore
					util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE_CACHE[idx])
					mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE_CACHE[idx])
					g["nvml_gpu_util_pct"] = getattr(util, "gpu", None)
					g["nvml_mem_used_mb"] = round(mem.used / 1024**2, 2)
					g["nvml_mem_total_mb"] = round(mem.total / 1024**2, 2)
					g["nvml_mem_util_pct"] = round((mem.used / mem.total * 100.0) if mem.total else 0.0, 2)
				except Exception as e:  # noqa: BLE001
					g["nvml_error"] = str(e)
	return {"gpus": gpus, "available": True, "nvml_enriched": bool(ENABLE_NVML and not SAFE_MODE and _NVML_SUPPORTED), "nvml_supported": _NVML_SUPPORTED}


def _detect_mps() -> Dict[str, Any]:
	"""Best-effort NVIDIA MPS detection.

	Returns a dict with:
	- enabled: bool
	- pipe_dir: str | None
	- control_process: bool (whether nvidia-cuda-mps-control appears active)
	"""
	pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
	control_process = False
	enabled = False
	try:
		out = subprocess.run([
			"pgrep", "-x", "nvidia-cuda-mps-control"
		], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
		control_process = (out.returncode == 0)
	except Exception:
		control_process = False
	try:
		if pipe_dir and os.path.exists(pipe_dir):
			enabled = True
	except Exception:
		enabled = False
	enabled = enabled or control_process
	return {"enabled": bool(enabled), "pipe_dir": pipe_dir, "control_process": bool(control_process)}


class PolicyUpdate(BaseModel):
	max_memory_per_agent_mb: Optional[int] = Field(None, ge=256, description="Per-agent memory cap in MB")
	allow_fractional_shares: Optional[bool] = None
	kill_on_oom: Optional[bool] = None


class LeaseRequest(BaseModel):
	agent: str
	min_memory_mb: Optional[int] = Field(0, ge=0)


class ReleaseRequest(BaseModel):
	token: str


class PreloadRequest(BaseModel):
	agents: Optional[List[str]] = Field(default=None, description="Subset of agents to preload; default all from AGENT_MODEL_MAP.json")
	refresh: bool = Field(default=False, description="Restart preloading even if a job already completed")
	strict: Optional[bool] = Field(default=None, description="Override STRICT_MODEL_STORE env for this preload run")


@asynccontextmanager
async def lifespan(app: FastAPI):
	global READINESS
	global _NVML_SUPPORTED, _NVML_HANDLE_CACHE, _NVML_INIT_ERROR
	logger.info("GPU Orchestrator starting up")
	# Attempt NVML initialization (best-effort, gated by ENABLE_NVML & SAFE_MODE)
	if ENABLE_NVML and not SAFE_MODE:
		try:  # pragma: no cover - environment dependent
			import pynvml  # type: ignore
			pynvml.nvmlInit()
			count = pynvml.nvmlDeviceGetCount()
			for i in range(count):
				_NVML_HANDLE_CACHE[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
			_NVML_SUPPORTED = True
			logger.info(f"NVML initialized for {count} device(s)")
		except Exception as e:  # noqa: BLE001
			_NVML_INIT_ERROR = str(e)
			logger.warning(f"NVML initialization failed: {e}")

	# Try registering to MCP Bus (best-effort)
	try:
		client = MCPBusClient()
		client.register_agent(
			agent_name="gpu_orchestrator",
			agent_address=f"http://localhost:{GPU_ORCHESTRATOR_PORT}",
			tools=[
				"health",
				"gpu_info",
				"get_policy",
				"set_policy",
				"get_allocations",
				"lease",
				"release",
				"models_preload",
				"models_status",
			],
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
	_inc("policy_get_requests_total")  # reuse counter group for simplicity
	return {"status": "ok", "safe_mode": SAFE_MODE}


@app.get("/ready")
def ready():
	_inc("policy_get_requests_total")
	return {"ready": READINESS}


@app.get("/gpu/info")
def gpu_info():
	"""Return current GPU telemetry (read-only)."""
	try:
		_inc("gpu_info_requests_total")
		data = get_gpu_snapshot()
		if ENABLE_NVML and not SAFE_MODE and not _NVML_SUPPORTED:
			data["nvml_init_error"] = _NVML_INIT_ERROR or "unsupported"
		# Include MPS status (best-effort)
		mps = _detect_mps()
		data["mps_enabled"] = bool(mps.get("enabled", False))
		data["mps"] = mps
		return data
	except Exception as e:
		logger.error(f"Failed to get GPU snapshot: {e}")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/policy")
def get_policy():
	_inc("policy_get_requests_total")
	return POLICY


@app.post("/policy")
def set_policy(update: PolicyUpdate):
	if SAFE_MODE:
		# In SAFE_MODE, accept but do not enact changes (read-only posture)
		_inc("policy_post_requests_total")
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
	_inc("policy_post_requests_total")
	return POLICY


@app.get("/allocations")
def get_allocations():
	"""Return current agent→GPU allocation view (placeholder)."""
	_inc("policy_get_requests_total")
	_purge_expired_leases()
	return {"allocations": ALLOCATIONS}


@app.post("/lease")
def lease(req: LeaseRequest):
	"""Obtain a simple ephemeral GPU lease. SAFE_MODE returns note only.

	Strategy: first-fit; if no GPU info or none available, return cpu fallback.
	NOT a hard guarantee—placeholder for future sophisticated allocator.
	"""
	_inc("lease_requests_total")
	_purge_expired_leases()
	if SAFE_MODE:
		return {"granted": False, "note": "SAFE_MODE", "agent": req.agent}

	snapshot = get_gpu_snapshot()
	gpu_index: Optional[int] = None
	if snapshot.get("available") and snapshot.get("gpus"):
		# naive: choose lowest used memory GPU meeting minimum
		candidates = []
		for g in snapshot["gpus"]:
			if req.min_memory_mb and (g["memory_total_mb"] - g["memory_used_mb"]) < req.min_memory_mb:
				continue
			candidates.append(g)
		if candidates:
			gpu_index = sorted(candidates, key=lambda x: x["memory_used_mb"])[0]["index"]

	token = str(uuid.uuid4())
	allocation = {
		"agent": req.agent,
		"gpu": gpu_index if gpu_index is not None else "cpu",
		"token": token,
		"timestamp": time.time(),
	}
	ALLOCATIONS[token] = allocation
	return {"granted": True, **allocation}


@app.post("/release")
def release(req: ReleaseRequest):
	_inc("release_requests_total")
	_purge_expired_leases()
	alloc = ALLOCATIONS.pop(req.token, None)
	if not alloc:
		raise HTTPException(status_code=404, detail="unknown_token")
	return {"released": True, "token": req.token}


@app.get("/metrics")
def metrics():  # pragma: no cover - simple string builder
	# Prometheus exposition format
	uptime = time.time() - _START_TIME
	_purge_expired_leases()
	lines = [
		"# HELP gpu_orchestrator_uptime_seconds Process uptime in seconds",
		"# TYPE gpu_orchestrator_uptime_seconds counter",
		f"gpu_orchestrator_uptime_seconds {uptime:.0f}",
		"# HELP gpu_orchestrator_active_leases Current active GPU (or CPU placeholder) leases",
		"# TYPE gpu_orchestrator_active_leases gauge",
		f"gpu_orchestrator_active_leases {len(ALLOCATIONS)}",
	]
	for k, v in sorted(_METRICS_COUNTERS.items()):
		name = f"gpu_orchestrator_{k}"
		lines.append(f"# HELP {name} {k.replace('_', ' ')}")
		lines.append(f"# TYPE {name} counter")
		lines.append(f"{name} {v}")
	# NVML status gauges
	if ENABLE_NVML and not SAFE_MODE:
		lines.append("# HELP gpu_orchestrator_nvml_supported NVML initialization success flag")
		lines.append("# TYPE gpu_orchestrator_nvml_supported gauge")
		lines.append(f"gpu_orchestrator_nvml_supported {1 if _NVML_SUPPORTED else 0}")
		if _NVML_INIT_ERROR and not _NVML_SUPPORTED:
			lines.append("# HELP gpu_orchestrator_nvml_error_info Last NVML initialization error (label)")
			lines.append("# TYPE gpu_orchestrator_nvml_error_info gauge")
			# Represent error presence as gauge; message is not standard but kept minimal
			lines.append("gpu_orchestrator_nvml_error_info 1")
	# MPS status gauge
	mps = _detect_mps()
	lines.append("# HELP gpu_orchestrator_mps_enabled NVIDIA MPS enabled flag")
	lines.append("# TYPE gpu_orchestrator_mps_enabled gauge")
	lines.append(f"gpu_orchestrator_mps_enabled {1 if mps.get('enabled') else 0}")
	return PlainTextResponse("\n".join(lines) + "\n")


# ---------------------------
# Model preload functionality
# ---------------------------

def _project_root() -> str:
	try:
		import pathlib
		return str(pathlib.Path(__file__).resolve().parents[2])
	except Exception:
		return os.getcwd()


def _load_agent_model(agent: str, model_id: str, strict: bool) -> Tuple[bool, Optional[str]]:
	"""Load a model on CPU to warm caches and validate availability.

	Returns (ok, error_msg). Does not keep model in memory; frees immediately.
	"""
	start = time.time()
	try:
		# Ensure STRICT_MODEL_STORE respected for this call
		prev_strict = os.environ.get("STRICT_MODEL_STORE")
		if strict:
			os.environ["STRICT_MODEL_STORE"] = "1"
		elif strict is False:
			os.environ["STRICT_MODEL_STORE"] = "0"

		ok = False
		err: Optional[str] = None

		# Prefer sentence-transformers path when obvious
		if model_id.startswith("sentence-transformers/"):
			try:
				# Use model-store aware loader with explicit agent to avoid caller detection issues
				from agents.common.model_loader import load_sentence_transformer

				m = load_sentence_transformer(model_id, agent=agent)
				# Do a tiny encode to ensure all modules initialize
				try:
					_ = m.encode(["warmup"])  # type: ignore[attr-defined]
				except Exception:
					# Not all wrappers support encode at this point; ignore
					pass
				ok = True
				del m
			except Exception as e:  # noqa: BLE001
				err = f"SentenceTransformer load failed: {e}"
		else:
			# Fallback to transformers
			try:
				from agents.common.model_loader import load_transformers_model
				model, tokenizer = load_transformers_model(model_id, agent=agent, cache_dir=None, model_class=None, tokenizer_class=None)
				# Touch a minimal tokenization to ensure files are present
				try:
					_ = tokenizer("warmup")  # type: ignore[operator]
				except Exception:
					pass
				ok = True
				# Release quickly
				del model
				del tokenizer
			except Exception as e:  # noqa: BLE001
				err = f"Transformers load failed: {e}"

		# Restore STRICT_MODEL_STORE env
		if prev_strict is None:
			os.environ.pop("STRICT_MODEL_STORE", None)
		else:
			os.environ["STRICT_MODEL_STORE"] = prev_strict

		# Force GC to reduce working set post-load
		try:
			import gc
			gc.collect()
		except Exception:
			pass

		duration = time.time() - start
		if ok:
			logger.info(f"Preloaded model for agent={agent} id={model_id} in {duration:.2f}s")
			return True, None
		else:
			logger.warning(f"Failed to preload model for agent={agent} id={model_id}: {err}")
			return False, err or "unknown_error"
	except Exception as e:  # noqa: BLE001
		return False, str(e)


def _read_agent_model_map() -> Dict[str, List[str]]:
	import json
	from pathlib import Path

	root = Path(_project_root())
	map_path = root / "markdown_docs" / "agent_documentation" / "AGENT_MODEL_MAP.json"
	if not map_path.exists():
		raise FileNotFoundError(f"AGENT_MODEL_MAP.json not found at {map_path}")
	with open(map_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _start_preload_job(selected_agents: Optional[List[str]], strict_override: Optional[bool]) -> Dict[str, Any]:
	global _MODEL_PRELOAD_STATE, _MODEL_PRELOAD_THREAD

	if _MODEL_PRELOAD_STATE.get("in_progress"):
		return _MODEL_PRELOAD_STATE

	# Initialize state
	_MODEL_PRELOAD_STATE = {
		"started_at": time.time(),
		"completed_at": None,
		"in_progress": True,
		"summary": {"total": 0, "done": 0, "failed": 0},
		"per_agent": {},
	}

	def _worker():
		global _MODEL_PRELOAD_STATE
		try:
			model_map = _read_agent_model_map()
			agents = selected_agents or list(model_map.keys())
			# Prepare status entries
			for a in agents:
				models = model_map.get(a, [])
				_MODEL_PRELOAD_STATE["per_agent"][a] = {}
				for mid in models:
					_MODEL_PRELOAD_STATE["per_agent"][a][mid] = {"status": "pending", "error": None, "duration_s": None}
			total = sum(len(model_map.get(a, [])) for a in agents)
			_MODEL_PRELOAD_STATE["summary"]["total"] = total

			strict_env = os.environ.get("STRICT_MODEL_STORE", "0").lower() in ("1", "true", "yes")
			strict = strict_override if strict_override is not None else strict_env

			# Iterate and preload each model on CPU
			for a in agents:
				for mid in model_map.get(a, []):
					st = _MODEL_PRELOAD_STATE["per_agent"][a][mid]
					st["status"] = "loading"
					t0 = time.time()
					ok, err = _load_agent_model(a, mid, strict)
					if ok:
						st["status"] = "ok"
						st["duration_s"] = time.time() - t0
						_MODEL_PRELOAD_STATE["summary"]["done"] += 1
					else:
						st["status"] = "error"
						st["error"] = err
						st["duration_s"] = time.time() - t0
						_MODEL_PRELOAD_STATE["summary"]["failed"] += 1
		except Exception as e:  # noqa: BLE001
			logger.error(f"Model preload job crashed: {e}")
		finally:
			_MODEL_PRELOAD_STATE["in_progress"] = False
			_MODEL_PRELOAD_STATE["completed_at"] = time.time()

	import threading

	_MODEL_PRELOAD_THREAD = threading.Thread(target=_worker, name="model-preloader", daemon=True)
	_MODEL_PRELOAD_THREAD.start()
	return _MODEL_PRELOAD_STATE


@app.post("/models/preload")
def models_preload(req: PreloadRequest):
	"""Start a background model preload job (CPU warming) using AGENT_MODEL_MAP.json.

	Returns current job state. If a job is already completed and refresh=false, returns that state.
	"""
	# If job completed and not refreshing, return existing state.
	# If there were failures, return 503 to enforce a hard failure with clear reasons.
	if _MODEL_PRELOAD_STATE.get("started_at") and not _MODEL_PRELOAD_STATE.get("in_progress") and not req.refresh:
		failed = int(_MODEL_PRELOAD_STATE.get("summary", {}).get("failed", 0) or 0)
		all_ready = (failed == 0 and _MODEL_PRELOAD_STATE["summary"].get("done", 0) == _MODEL_PRELOAD_STATE["summary"].get("total", 0))
		state = {**_MODEL_PRELOAD_STATE, "all_ready": all_ready}
		# Build a concise error list for clarity
		errors: List[Dict[str, Any]] = []
		for a, models in _MODEL_PRELOAD_STATE.get("per_agent", {}).items():
			for mid, st in models.items():
				if st.get("status") == "error":
					errors.append({"agent": a, "model": mid, "error": st.get("error")})
		state["errors"] = errors
		if failed > 0:
			raise HTTPException(status_code=503, detail=state)
		return state

	state = _start_preload_job(req.agents, req.strict)
	return {
		**state,
		"all_ready": False,
	}


@app.get("/models/status")
def models_status():
	"""Return the current model preload status and readiness summary."""
	state = _MODEL_PRELOAD_STATE
	total = state.get("summary", {}).get("total", 0)
	done = state.get("summary", {}).get("done", 0)
	failed = state.get("summary", {}).get("failed", 0)
	all_ready = (total > 0 and failed == 0 and done == total) and not state.get("in_progress", False)
	# Build a concise error list for clarity
	errors: List[Dict[str, Any]] = []
	for a, models in state.get("per_agent", {}).items():
		for mid, st in models.items():
			if st.get("status") == "error":
				errors.append({"agent": a, "model": mid, "error": st.get("error")})
	return {**state, "all_ready": all_ready, "errors": errors}


if __name__ == "__main__":
	import uvicorn

	# Place the runner at the very end so all endpoints above are registered
	uvicorn.run(app, host="0.0.0.0", port=GPU_ORCHESTRATOR_PORT)

