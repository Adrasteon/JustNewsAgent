"""
GPU Orchestrator Service
"""

from fastapi import FastAPI

# Global state for model preloading
_MODEL_PRELOAD_STATE = {
    "started_at": None,
    "completed_at": None,
    "in_progress": False,
    "summary": {"total": 0, "done": 0, "failed": 0},
    "per_agent": {},
}

app = FastAPI()

def _project_root():
    return "/tmp"  # Mock implementation

def _load_agent_model(agent, model_id, strict=True):
    return True, "Mock success"  # Mock implementation
