from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agents.gpu_orchestrator.utils import _validate_and_load_model

logger = logging.getLogger(__name__)


_MODEL_PRELOAD_STATE: dict[str, Any] = {
    "started_at": None,
    "completed_at": None,
    "in_progress": False,
    "summary": {"total": 0, "done": 0, "failed": 0},
    "per_agent": {},
}


def _resolve_project_root() -> Path:
    """Resolve the repository root, honoring test overrides when available."""
    try:
        from agents.gpu_orchestrator import main as orchestrator_main  # noqa: WPS433

        candidate = Path(orchestrator_main._project_root())
        if candidate.exists():
            return candidate
    except Exception:  # pragma: no cover - defensive
        logger.debug("Falling back to local project root resolution", exc_info=True)
    return Path(__file__).resolve().parents[2]


def load_agent_model_map() -> dict[str, list[str]]:
    """Load the agent->model map from the documentation directory."""
    project_root = _resolve_project_root()
    candidates = [
        project_root / "markdown_docs" / "agent_documentation" / "AGENT_MODEL_MAP.json",
        project_root / "AGENT_MODEL_MAP.json",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw_map = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read agent model map at %s: %s", path, exc)
            return {}

        if not isinstance(raw_map, dict):
            logger.warning("Agent model map at %s is not a dictionary; ignoring", path)
            return {}

        cleaned: dict[str, list[str]] = {}
        for agent, models in raw_map.items():
            if not isinstance(models, list):
                logger.warning(
                    "Skipping agent %s in model map because value is not a list",
                    agent,
                )
                continue
            cleaned[str(agent)] = [str(model_id) for model_id in models]

        logger.debug("Loaded agent model map from %s", path)
        return cleaned

    logger.warning("No AGENT_MODEL_MAP.json found; continuing with empty map")
    return {}


def _deduplicate_agents(values: Iterable[str]) -> list[str]:
    """Return a list with duplicate agent identifiers removed."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def start_preload_job(
    agents: Iterable[str] | None = None,
    strict: bool | None = None,
) -> dict[str, Any]:
    """Start a preload job and synchronously warm models for the requested agents."""
    model_map = load_agent_model_map()
    selected_agents = (
        _deduplicate_agents(list(agents))
        if agents is not None
        else list(model_map.keys())
    )

    if agents is not None:
        selected_agents = [agent for agent in selected_agents if agent in model_map]
    if not selected_agents and model_map:
        selected_agents = list(model_map.keys())

    _MODEL_PRELOAD_STATE["started_at"] = time.time()
    _MODEL_PRELOAD_STATE["completed_at"] = None
    _MODEL_PRELOAD_STATE["in_progress"] = True
    _MODEL_PRELOAD_STATE["per_agent"] = {}

    previous_summary = _MODEL_PRELOAD_STATE.get(
        "summary",
        {"total": 0, "done": 0, "failed": 0},
    )
    summary = {
        "total": 0,
        "done": 0,
        "failed": int(previous_summary.get("failed", 0) or 0),
    }

    strict_env = os.environ.get("STRICT_MODEL_STORE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    effective_strict = strict if strict is not None else strict_env

    for agent in selected_agents:
        agent_models = model_map.get(agent, [])
        agent_state: dict[str, dict[str, Any]] = {}
        _MODEL_PRELOAD_STATE["per_agent"][agent] = agent_state

        for model_id in agent_models:
            entry = {"status": "pending", "error": None, "duration_s": None}
            agent_state[model_id] = entry
            summary["total"] += 1

            entry["status"] = "loading"
            start_ts = time.time()
            try:
                success, error = _validate_and_load_model(
                    agent,
                    model_id,
                    effective_strict,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Unhandled exception preloading %s/%s: %s",
                    agent,
                    model_id,
                    exc,
                )
                success, error = False, str(exc)

            entry["duration_s"] = time.time() - start_ts
            if success:
                entry["status"] = "ok"
                summary["done"] += 1
            else:
                entry["status"] = "error"
                entry["error"] = error
                summary["failed"] += 1

    _MODEL_PRELOAD_STATE["summary"] = summary
    _MODEL_PRELOAD_STATE["in_progress"] = False
    _MODEL_PRELOAD_STATE["completed_at"] = time.time()

    return _MODEL_PRELOAD_STATE


def get_preload_status() -> dict[str, Any]:
    """Return the recorded status of the most recent preload job."""
    return _MODEL_PRELOAD_STATE
