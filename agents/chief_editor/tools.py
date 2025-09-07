# Optimized Chief Editor Configuration
# Phase 1 Memory Optimization: Context and batch size optimization for orchestration

import os
from common.observability import get_logger

import requests
from datetime import datetime, timezone

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

# PHASE 1 OPTIMIZATIONS APPLIED
MODEL_NAME = "distilgpt2"
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/distilgpt2")
OPTIMIZED_MAX_LENGTH = 1024  # Reduced from 2048 (orchestration tasks are brief)
OPTIMIZED_BATCH_SIZE = 4     # Small batches for orchestration efficiency

MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://mcp_bus:8000")
FEEDBACK_LOG = os.environ.get("CHIEF_EDITOR_FEEDBACK_LOG", "./feedback_chief_editor.log")


logger = get_logger(__name__)

def get_llama_model():
    """Load optimized DialoGPT (deprecated)-medium model for orchestration tasks."""
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("transformers library is not installed.")

    try:
        from agents.common.model_loader import load_transformers_model
        model, tokenizer = load_transformers_model(MODEL_NAME, agent='chief_editor', cache_dir=MODEL_PATH)
        return model, tokenizer
    except Exception:
        # Fallback to original behavior
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            print(f"Downloading {MODEL_NAME} to {MODEL_PATH}...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_PATH)
        else:
            print(f"Loading {MODEL_NAME} from local cache {MODEL_PATH}...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer

def log_feedback(event: str, details: dict):
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\t{event}\t{details}\n")

def request_story_brief(topic: str, scope: str):
    """Generate a story brief based on the given topic and scope."""
    logger.info(f"Requesting story brief for topic: {topic}, scope: {scope}")
    brief = f"Story brief for topic '{topic}' within scope '{scope}'."
    log_feedback("request_story_brief", {"topic": topic, "scope": scope, "brief": brief})
    return brief

def publish_story(story_id: str):
    """Publishes a story via MCP bus with optimized memory usage."""
    logger.info(f"[ChiefEditor] Publishing story with ID: {story_id}")
    payload = {
        "agent": "librarian",
        "tool": "update_story_timeline",
        "args": [story_id],
        "kwargs": {}
    }
    try:
        resp = requests.post(f"{MCP_BUS_URL}/call", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        log_feedback("publish_story", {"story_id": story_id, "result": result})
        return {
            "status": "published",
            "story_id": story_id,
            "mcp_result": result,
            "message": "Librarian Agent notified and story status updated via MCP bus."
        }
    except Exception as e:
        logger.error(f"Error calling MCP bus for publish_story: {e}")
        log_feedback("publish_story_error", {"story_id": story_id, "error": str(e)})
        return {
            "status": "error",
            "story_id": story_id,
            "error": str(e)
        }