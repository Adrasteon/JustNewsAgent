"""Centralized model loading helpers.

Provides small wrappers around transformers and sentence-transformers loading to
prefer canonical ModelStore paths when `MODEL_STORE_ROOT` is configured, and to
fall back to per-agent cache_dir behavior otherwise.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _resolve_model_store_path(agent: Optional[str]) -> Optional[Path]:
    root = os.environ.get("MODEL_STORE_ROOT")
    if not root or not agent:
        return None
    try:
        from agents.common.model_store import ModelStore
        ms = ModelStore(Path(root))
        cur = ms.get_current(agent)
        if cur and cur.exists():
            return cur
    except Exception:
        logger.debug("ModelStore not available or current not found for agent=%s", agent)
    return None


def load_transformers_model(
    model_id_or_path: str,
    agent: Optional[str] = None,
    cache_dir: Optional[str] = None,
    model_class: Optional[object] = None,
    tokenizer_class: Optional[object] = None,
) -> Tuple[object, object]:
    """Load a transformers model + tokenizer with safe ModelStore support.

    Parameters:
    - model_id_or_path: HF id or filesystem path
    - agent: optional agent name to resolve ModelStore current path
    - cache_dir: optional cache_dir passed to from_pretrained
    - model_class: optional class to instantiate the model (e.g. AutoModelForCausalLM)
    - tokenizer_class: optional class to instantiate tokenizer/processor (e.g. AutoTokenizer)

    Returns (model, tokenizer/processor). If `MODEL_STORE_ROOT` is configured and a
    current model exists for `agent`, that path is used. Otherwise, cache_dir
    (if provided) or model_id_or_path is used with from_pretrained.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise ImportError("transformers is required to load models") from e

    ModelClass = model_class or AutoModel
    TokenizerClass = tokenizer_class or AutoTokenizer

    # Prefer model store canonical path when configured
    ms_path = _resolve_model_store_path(agent)
    strict = os.environ.get("STRICT_MODEL_STORE") == "1"
    if ms_path:
        try:
            model = ModelClass.from_pretrained(str(ms_path))
            tokenizer = TokenizerClass.from_pretrained(str(ms_path))
            return model, tokenizer
        except Exception:
            logger.warning("Failed to load model from ModelStore path %s, falling back", ms_path)
            if strict:
                raise RuntimeError(f"STRICT_MODEL_STORE=1 but failed to load model for agent={agent} from {ms_path}")

    # Fallback to supplied cache_dir or model_id_or_path
    load_kwargs = {}
    if cache_dir:
        load_kwargs['cache_dir'] = cache_dir

    # If model_id_or_path is a filesystem path, transformers will load from there.
    model = ModelClass.from_pretrained(model_id_or_path, **load_kwargs)
    tokenizer = TokenizerClass.from_pretrained(model_id_or_path, **load_kwargs)
    return model, tokenizer


def load_sentence_transformer(model_name: str, agent: Optional[str] = None, cache_folder: Optional[str] = None):
    """Load a SentenceTransformer instance preferring ModelStore when configured."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError("sentence-transformers is required") from e

    ms_path = _resolve_model_store_path(agent)
    strict = os.environ.get("STRICT_MODEL_STORE") == "1"
    if ms_path:
        try:
            return SentenceTransformer(str(ms_path))
        except Exception:
            logger.warning("Failed to load SentenceTransformer from ModelStore %s", ms_path)
            if strict:
                raise RuntimeError(f"STRICT_MODEL_STORE=1 but failed to load SentenceTransformer for agent={agent} from {ms_path}")

    if cache_folder:
        return SentenceTransformer(model_name, cache_folder=cache_folder)
    return SentenceTransformer(model_name)
