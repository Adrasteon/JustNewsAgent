"""Simple GPU manager shim used by agents during testing and lightweight runs.

This module provides a minimal, dependency-free GPU allocation API so
modules that import it (e.g. agent gpu_tools) can operate without
requiring the full production GPU manager. The implementation is
intentionally conservative: it simulates a single-GPU environment and
provides no-op register/release operations.

The real project contains a more advanced GPUModelManager used in
production; this shim keeps the codebase importable for linting and
unit tests.
"""
from __future__ import annotations

from threading import Lock
from typing import Any

__all__ = [
    "request_agent_gpu",
    "release_agent_gpu",
    "get_gpu_manager",
    "GPUModelManager",
]


class GPUModelManager:
    """Lightweight in-process GPU model registry.

    This class does not manage real GPU devices. It offers a simple
    registry and context-manager API that mirrors the production
    manager enough for tests and linting.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._registry: dict[str, Any] = {}

    def register_model(self, name: str, model: Any) -> None:
        """Register a model object under a name.

        Args:
            name: Unique name for the model.
            model: The model or pipeline object to store.
        """
        with self._lock:
            self._registry[name] = model

    def get(self, name: str) -> Any | None:
        """Return a registered model or None if not present."""
        with self._lock:
            return self._registry.get(name)

    def __enter__(self) -> GPUModelManager:
        # No-op resource acquisition for the shim
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # No-op cleanup
        return None


# Global, shared manager instance used by get_gpu_manager()
_GLOBAL_MANAGER: GPUModelManager | None = None
_GLOBAL_LOCK = Lock()


def get_gpu_manager() -> GPUModelManager:
    """Return a shared GPUModelManager instance.

    The function lazily creates a singleton to avoid import-time side
    effects.
    """
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = GPUModelManager()
        return _GLOBAL_MANAGER


def request_agent_gpu(agent_name: str, memory_gb: float = 2.0) -> int | None:
    """Request allocation of a GPU for an agent.

    This shim always returns GPU index 0 to indicate a (simulated)
    allocation. Callers should handle None if no allocation is
    possible; production code will attempt real allocation.

    Args:
        agent_name: Human-readable agent name (used for logging in
            production manager).
        memory_gb: Requested memory in GB (ignored by shim).

    Returns:
        An integer GPU index or None if allocation failed.
    """
    # In the shim we optimistically return 0. Tests and linting code
    # should not rely on multiple GPU indices.
    return 0


def release_agent_gpu(agent_name: str) -> None:
    """Release a previously requested GPU for an agent.

    The shim performs no action; it exists to keep call sites simple.
    """
    return None
