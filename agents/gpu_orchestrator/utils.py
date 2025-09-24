# Utility functions for GPU Orchestrator
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def _load_agent_model(agent: str, model_id: str, strict: bool) -> Tuple[bool, Optional[str]]:
    """Simulate model loading for testing purposes."""
    if "fail" in model_id:
        return False, "simulated load failure"
    logger.info(f"Simulating model load: agent={agent}, model_id={model_id}, strict={strict}")
    return True, None