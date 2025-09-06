"""
GPU Manager - Production Implementation Wrapper

This module now serves as a wrapper around the production MultiAgentGPUManager,
maintaining backward compatibility while providing production-ready features.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

# Import production GPU manager
try:
    from agents.common.gpu_manager_production import (
        get_gpu_manager,
        request_agent_gpu as _request_agent_gpu,
        release_agent_gpu as _release_agent_gpu
    )
    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False

# Fallback to original shim if production manager unavailable
if not PRODUCTION_AVAILABLE:
    from threading import Lock

    class GPUModelManager:
        """Lightweight in-process GPU model registry (fallback)"""
        def __init__(self) -> None:
            self._lock = Lock()
            self._registry: Dict[str, Any] = {}

        def register_model(self, name: str, model: Any) -> None:
            with self._lock:
                self._registry[name] = model

        def get(self, name: str) -> Optional[Any]:
            with self._lock:
                return self._registry.get(name)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    _GLOBAL_MANAGER: Optional[GPUModelManager] = None
    _GLOBAL_LOCK = Lock()

    def get_gpu_manager() -> GPUModelManager:
        global _GLOBAL_MANAGER
        with _GLOBAL_LOCK:
            if _GLOBAL_MANAGER is None:
                _GLOBAL_MANAGER = GPUModelManager()
            return _GLOBAL_MANAGER

    def _request_agent_gpu(agent_name: str, memory_gb: float = 2.0) -> Optional[int]:
        return 0  # Always return GPU 0 for compatibility

    def _release_agent_gpu(agent_name: str) -> None:
        return None


__all__ = [
    "request_agent_gpu",
    "release_agent_gpu",
    "get_gpu_manager",
    "GPUModelManager",
]


class GPUModelManager:
    """Wrapper around production GPU manager for backward compatibility"""

    def __init__(self) -> None:
        if PRODUCTION_AVAILABLE:
            self._manager = get_gpu_manager()
        else:
            # Fallback to simple registry
            import threading
            self._lock = threading.Lock()
            self._registry: Dict[str, Any] = {}

    def register_model(self, name: str, model: Any) -> None:
        """Register a model object under a name."""
        if PRODUCTION_AVAILABLE:
            # Use production manager's model registry if available
            self._manager._model_registry = getattr(self._manager, '_model_registry', {})
            self._manager._model_registry[name] = model
        else:
            with self._lock:
                self._registry[name] = model

    def get(self, name: str) -> Optional[Any]:
        """Return a registered model or None if not present."""
        if PRODUCTION_AVAILABLE:
            registry = getattr(self._manager, '_model_registry', {})
            return registry.get(name)
        else:
            with self._lock:
                return self._registry.get(name)

    def __enter__(self):
        if PRODUCTION_AVAILABLE:
            return self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if PRODUCTION_AVAILABLE:
            return self._manager.__exit__(exc_type, exc, tb)
        return None


# Global, shared manager instance
_GLOBAL_MANAGER: Optional[GPUModelManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_gpu_manager() -> GPUModelManager:
    """Return a shared GPUModelManager instance."""
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = GPUModelManager()
        return _GLOBAL_MANAGER


def request_agent_gpu(agent_name: str, memory_gb: float = 2.0) -> Optional[int]:
    """Request allocation of a GPU for an agent."""
    if PRODUCTION_AVAILABLE:
        result = _request_agent_gpu(agent_name, memory_gb)
        if isinstance(result, dict):
            return result.get('gpu_device', 0) if result.get('status') == 'allocated' else None
        return result
    else:
        return _request_agent_gpu(agent_name, memory_gb)


def release_agent_gpu(agent_name: str) -> None:
    """Release a previously requested GPU for an agent."""
    if PRODUCTION_AVAILABLE:
        _release_agent_gpu(agent_name)
    else:
        _release_agent_gpu(agent_name)
