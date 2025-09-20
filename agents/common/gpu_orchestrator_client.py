"""
GPU Orchestrator Client

Client for communicating with the GPU Orchestrator service.
"""

import requests
from typing import Dict, Any, Optional


class GPUOrchestratorClient:
    """Client for GPU Orchestrator service communication"""

    def __init__(self, base_url: str = "http://localhost:8014"):
        self.base_url = base_url.rstrip('/')
        self._session = requests.Session()

    def get_policy(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get current GPU policy from orchestrator"""
        try:
            url = f"{self.base_url}/policy"
            if force_refresh:
                url += "?force_refresh=true"
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Return safe defaults if service unavailable
            return {"safe_mode_read_only": True, "error": str(e)}

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information from orchestrator"""
        try:
            response = self._session.get(f"{self.base_url}/gpu/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Return safe defaults if service unavailable
            return {"available": False, "error": str(e)}

    def cpu_fallback_decision(self) -> Dict[str, Any]:
        """Get CPU fallback decision from orchestrator"""
        try:
            response = self._session.get(f"{self.base_url}/cpu/fallback", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Return safe defaults if service unavailable
            return {"use_gpu": False, "reason": f"Service unavailable: {str(e)}"}