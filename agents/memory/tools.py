from pathlib import Path
"""
Tools for the Memory Agent.
"""

import logging
import os
from datetime import datetime
import json

import psycopg2
import requests
try:
    import torch
except Exception:
    torch = None

try:
    # Prefer the shared model helper to avoid repeated loads
    SentenceTransformer = None
except Exception:
    # Fallback: try to import SentenceTransformer directly
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

# Environment variables
FEEDBACK_LOG = os.environ.get("MEMORY_FEEDBACK_LOG", "./feedback_memory.log")
EMBEDDING_MODEL_NAME = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
MEMORY_AGENT_PORT = int(os.environ.get("MEMORY_AGENT_PORT", 8007))
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
# Canonical cache folder for shared embedding model (keep consistent with engines)
# Canonical cache folder for memory agent: prefer agent-local models directory
DEFAULT_MODEL_CACHE = os.environ.get("MEMORY_V2_CACHE") or os.environ.get('MEMORY_MODEL_CACHE') or str(Path('./agents/memory/models').resolve())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory.tools")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        return None

def log_feedback(event: str, details: dict):
    """Logs feedback to a file."""
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.utcnow().isoformat()}\t{event}\t{details}\n")

def get_embedding_model():
    """Return a SentenceTransformer instance, using the shared helper when available."""
    # If shared helper is available, use it (it will import sentence_transformers under the hood)
    try:
        from agents.common.embedding import get_shared_embedding_model
        # Use a canonical cache folder and device so cached instances are reused
        device = None
        if torch is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return get_shared_embedding_model(EMBEDDING_MODEL_NAME, cache_folder=DEFAULT_MODEL_CACHE, device=device)
    except Exception:
        # Fallback to direct construction
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
    from agents.common.embedding import get_shared_embedding_model
    agent_cache = os.environ.get('MEMORY_MODEL_CACHE') or str(Path('./agents/memory/models').resolve())
    return get_shared_embedding_model(EMBEDDING_MODEL_NAME, cache_folder=agent_cache)

def save_article(content: str, metadata: dict, embedding_model=None) -> dict:
    """Saves an article to the database and generates an embedding for the content.

    Args:
        content: Article text to embed and store.
        metadata: Arbitrary metadata dict to store alongside the article.
        embedding_model: Optional pre-initialized SentenceTransformer instance.
            If not provided, a new model will be created via get_embedding_model().
    """
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}
    try:
        with conn.cursor() as cur:
            # Use provided model if available to avoid re-loading model per-call
            if embedding_model is None:
                embedding_model = get_embedding_model()
            # encode may return numpy array; convert later to list of floats
            embedding = embedding_model.encode(content)

            # Ensure metadata is a JSON-serializable string for safe insertion
            try:
                metadata_payload = json.dumps(metadata) if metadata is not None else json.dumps({})
            except Exception:
                # Fallback: coerce to string
                metadata_payload = json.dumps({"raw": str(metadata)})

            # Get the next available ID (simple approach without sequence)
            cur.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM articles")
            next_id = cur.fetchone()[0]

            # Insert with explicit ID - metadata as JSON string (Postgres will cast)
            cur.execute(
                "INSERT INTO articles (id, content, metadata, embedding) VALUES (%s, %s, %s::jsonb, %s)",
                (next_id, content, metadata_payload, list(map(float, embedding))),
            )
            conn.commit()
            log_feedback("save_article", {"status": "success", "article_id": next_id})
            return {"status": "success", "article_id": next_id}
    except Exception as e:
        logger.error(f"Error saving article: {e}")
        conn.rollback()
        return {"error": str(e)}
    finally:
        conn.close()

def get_article(article_id: int) -> dict:
    """Retrieves an article from the memory agent."""
    response = requests.get(f"http://localhost:{MEMORY_AGENT_PORT}/get_article/{article_id}")
    response.raise_for_status()
    return response.json()

def vector_search_articles(query: str, top_k: int = 5) -> list:
    """Performs a vector search for articles using the memory agent."""
    response = requests.post(
        f"http://localhost:{MEMORY_AGENT_PORT}/vector_search_articles",
        json={"query": query, "top_k": top_k},
    )
    response.raise_for_status()
    return response.json()

def log_training_example(task: str, input: dict, output: dict, critique: str) -> dict:
    """Logs a training example using the memory agent."""
    response = requests.post(
        f"http://localhost:{MEMORY_AGENT_PORT}/log_training_example",
        json={"task": task, "input": input, "output": output, "critique": critique},
    )
    response.raise_for_status()
    return response.json()