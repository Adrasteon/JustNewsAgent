"""
Main file for the Memory Agent.
"""
import logging
import os
import requests
from contextlib import asynccontextmanager

try:
    # Optional import for Hugging Face hub login and snapshot_download
    import huggingface_hub
except Exception:
    huggingface_hub = None

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents.memory.tools import (
    get_embedding_model,
    log_training_example,
    save_article,
    vector_search_articles,
    vector_search_articles_local,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Readiness flag
ready = False

# Shared pre-warmed embedding model (initialized at startup)
embedding_model = None

# Async storage queue and background worker
storage_queue: "asyncio.Queue[dict]" = None
storage_executor: ThreadPoolExecutor = None

# Environment variables
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
MEMORY_AGENT_PORT = int(os.environ.get("MEMORY_AGENT_PORT", 8007))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Pydantic models
class Article(BaseModel):
    content: str
    metadata: dict

class TrainingExample(BaseModel):
    task: str
    input: dict
    output: dict
    critique: str

class VectorSearch(BaseModel):
    query: str
    top_k: int = 5

class ToolCall(BaseModel):
    args: list
    kwargs: dict

class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(2, 5))
            response.raise_for_status()
            logger.info(f"Successfully registered {agent_name} with MCP Bus.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register {agent_name} with MCP Bus: {e}")
            raise

# Use connection pooling for database connections
def get_db_connection():
    """Establishes a connection to the PostgreSQL database with pooling."""
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            options='-c search_path=public'
        )
        logger.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")


async def _storage_consumer():
    """Background consumer to process storage tasks from the queue."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            task = await storage_queue.get()
            # task is dict with content and metadata
            # Run save_article in thread pool to avoid blocking event loop
            # Pass pre-warmed embedding_model to avoid repeated loads
            await loop.run_in_executor(
                storage_executor,
                lambda: save_article(task.get('content'), task.get('metadata'), embedding_model=embedding_model),
            )
            try:
                storage_queue.task_done()
            except Exception:
                pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in storage consumer: {e}")
            try:
                storage_queue.task_done()
            except Exception:
                pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("Memory agent is starting up.")
    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="memory",
            agent_address=f"http://localhost:{MEMORY_AGENT_PORT}",
            tools=[
                "save_article",
                "get_article",
                "vector_search_articles",
                "log_training_example",
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    # Mark ready after startup and optional registration
    global ready
    ready = True
    # Initialize and pre-warm embedding model once
    global embedding_model
    # If operator provided HF token or cache path, configure environment for huggingface_hub
    hf_token = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        # Prefer explicit HF_HOME to guide huggingface_hub cache location
        os.environ["HF_HOME"] = hf_cache
        logger.info(f"Set HF_HOME to {hf_cache} for model caching")
    if hf_token and huggingface_hub is not None:
        try:
            # login() sets token for subsequent hub downloads
            huggingface_hub.login(token=hf_token)
            logger.info("Authenticated with Hugging Face hub using HF_HUB_TOKEN")
        except Exception as e:
            logger.warning(f"Could not login to Hugging Face hub: {e}")
    elif hf_token and huggingface_hub is None:
        logger.warning("HF_HUB_TOKEN provided but huggingface_hub package is not available")
    try:
        embedding_model = get_embedding_model()
        logger.info("Pre-warmed embedding model successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm embedding model at startup: {e}")
    # Initialize async queue and background worker
    global storage_queue, storage_executor
    storage_queue = asyncio.Queue()
    storage_executor = ThreadPoolExecutor(max_workers=2)

    # Start background consumer
    app.state.storage_consumer = asyncio.create_task(_storage_consumer())
    yield
    logger.info("Memory agent is shutting down.")
    # Shutdown background consumer
    try:
        # Cancel consumer task and drain remaining items
        app.state.storage_consumer.cancel()
    except Exception:
        pass
    # Drain the queue (wait briefly for pending tasks to complete)
    try:
        async def _drain():
            # Allow a short grace period for in-flight items
            timeout = 10
            start = asyncio.get_event_loop().time()
            while not storage_queue.empty() and (asyncio.get_event_loop().time() - start) < timeout:
                logger.info(f"Draining storage queue; remaining={storage_queue.qsize()}")
                await asyncio.sleep(0.5)

        # We're already in an async context here; await the drain coroutine
        await _drain()
    except Exception:
        pass
    try:
        storage_executor.shutdown(wait=False)
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)

# Register common shutdown endpoint
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for memory")

# Register reload endpoint and handler for runtime reloads (embedding etc.)
try:
    from agents.common.reload import register_reload_endpoint, register_reload_handler

    def _reload_embedding_model():
        """Reload the embedding model from ModelStore / cache and return a small status."""
        global embedding_model
        try:
            embedding_model = get_embedding_model()
            return {"reloaded": True}
        except Exception as e:
            # preserve previous model on failure
            return {"reloaded": False, "error": str(e)}

    register_reload_handler('embedding_model', _reload_embedding_model)
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for memory")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Simple metrics endpoint exposing queue size and embedding model readiness."""
    qsize = storage_queue.qsize() if storage_queue is not None else -1
    model_ready = embedding_model is not None
    return {"queue_size": qsize, "model_ready": model_ready}

@app.get("/ready")
def ready_endpoint():
    """Readiness endpoint for startup gating."""
    return {"ready": ready}

@app.post("/save_article")
def save_article_endpoint(request: dict):
    """Saves an article to the database. Handles both direct calls and MCP Bus format."""
    try:
        # Handle MCP Bus format: {"args": [...], "kwargs": {...}}
        if "args" in request and "kwargs" in request:
            if len(request["args"]) > 0:
                article_data = request["args"][0]
            else:
                article_data = request["kwargs"]
        else:
            # Direct call format
            article_data = request

        # Create Article object from the data
        article = Article(**article_data)
        # Use pre-warmed embedding_model if available to avoid re-loading the model
        return save_article(article.content, article.metadata, embedding_model=embedding_model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error saving article: {str(e)}")

@app.get("/get_article/{article_id}")
def get_article_endpoint(article_id: int):
    """Retrieves an article from the database."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
            article = cur.fetchone()
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            return article
    finally:
        conn.close()

@app.post("/vector_search_articles")
def vector_search_articles_endpoint(request: dict):
    """Performs a vector search for articles. Handles both direct calls and MCP Bus format."""
    try:
        # Handle MCP Bus format: {"args": [...], "kwargs": {...}}
        if "args" in request and "kwargs" in request:
            if len(request["args"]) > 0:
                search_data = request["args"][0]
            else:
                search_data = request["kwargs"]
        else:
            # Direct call format
            search_data = request

        # Create VectorSearch object from the data
        search = VectorSearch(**search_data)
        # Use local in-process search implementation to avoid making HTTP
        # requests to ourselves which can cause recursive blocking behavior.
        return vector_search_articles_local(search.query, search.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error searching articles: {str(e)}")

@app.post("/log_training_example")
def log_training_example_endpoint(example: TrainingExample):
    """Logs a training example to the database."""
    return log_training_example(
        example.task, example.input, example.output, example.critique
    )

# Improved error handling and logging
@app.post("/store_article")
def store_article_endpoint(call: ToolCall):
    """Stores an article in the database."""
    try:
        article_data = call.kwargs
        if not article_data.get("content"):
            raise ValueError("Missing required field: 'content'")
        # Provide default metadata if missing to avoid 400s from crawlers that don't include metadata
        if not article_data.get("metadata"):
            article_data["metadata"] = {"source": "unknown"}

        # Enqueue for async storage
        try:
            storage_queue.put_nowait({"content": article_data["content"], "metadata": article_data["metadata"]})
            logger.info("Article enqueued for async storage")
            return {"status": "enqueued"}
        except Exception as e:
            logger.error(f"Failed to enqueue article: {e}")
            raise
    except ValueError as ve:
        logger.warning(f"Validation error in store_article: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"An error occurred in store_article: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
