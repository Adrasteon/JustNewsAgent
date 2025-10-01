# Utility functions for GPU Orchestrator
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_spacy_model(
    agent: str, model_id: str, strict: bool
) -> tuple[bool, str | None]:
    """Load a spaCy model with ModelStore awareness."""
    try:
        import spacy
    except Exception as exc:  # pragma: no cover - spaCy should be installed
        logger.error("spaCy import failed: %s", exc)
        return False, str(exc)

    root = os.environ.get("MODEL_STORE_ROOT")
    candidate_error: str | None = None
    if root:
        try:
            from agents.common.model_store import ModelStore

            store = ModelStore(Path(root))
            current = store.get_current(agent)
            if current:
                dir_name = f"models--{model_id.replace('/', '--')}"
                candidate = current / dir_name
                if candidate.exists():
                    spacy.load(str(candidate))
                    logger.info(
                        "Loaded spaCy model %s for agent %s from ModelStore",
                        model_id,
                        agent,
                    )
                    return True, None
                candidate_error = f"Model not found in ModelStore: {candidate}"
        except Exception as exc:  # pragma: no cover - defensive
            candidate_error = str(exc)

    if candidate_error and strict:
        logger.error(candidate_error)
        return False, candidate_error

    try:
        spacy.load(model_id)
        logger.info("Loaded spaCy model %s via default registry", model_id)
        return True, None
    except Exception as exc:  # pragma: no cover - spaCy should handle downloads
        logger.error("spaCy load failed for %s: %s", model_id, exc)
        if candidate_error:
            combined = f"{candidate_error}; fallback load failed: {exc}"
            return False, combined
        return False, str(exc)


def _load_agent_model(
    agent: str, model_id: str, strict: bool
) -> tuple[bool, str | None]:
    """Attempt to load the model for an agent, returning success status."""
    logger.debug(
        "_load_agent_model called with agent=%s, model_id=%s, strict=%s",
        agent,
        model_id,
        strict,
    )
    if model_id == "en_core_web_sm":
        return _load_spacy_model(agent, model_id, strict)
    try:
        from agents.common import (
            model_loader,
        )  # noqa: WPS433 (local import for test patching)

        model_loader.load_sentence_transformer(model_id, agent=agent)
    except Exception as exc:  # pragma: no cover - exercised via tests
        logger.error(
            "Model preload failed for agent=%s model=%s: %s", agent, model_id, exc
        )
        return False, str(exc)

    logger.info(
        "Successfully validated model %s for agent %s (strict=%s)",
        model_id,
        agent,
        strict,
    )
    return True, None


def _validate_and_load_model(agent, model_id, strict):
    logger.debug(
        "Invoked _validate_and_load_model with agent=%s, model_id=%s, strict=%s",
        agent,
        model_id,
        strict,
    )
    return _load_agent_model(agent, model_id, strict)
