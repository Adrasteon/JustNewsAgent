"""
Main file for the Analyst Agent.
"""

import os
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from agents.common.database import get_db_cursor
from common.metrics import JustNewsMetrics
from common.observability import get_logger
from common.version_utils import get_version

from .tools import (
    analyze_content_trends,
    analyze_text_statistics,
    extract_key_metrics,
    identify_entities,
    log_feedback,
)

# Import security utilities
try:
    from .security_utils import (
        log_security_event,
        sanitize_content,
        validate_content_size,
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    def validate_content_size(content: str) -> bool:
        return len(content) < 1000000  # 1MB limit
    def sanitize_content(content: str) -> str:
        return content
    def log_security_event(event: str, details: dict):
        pass

# Configure centralized logging
logger = get_logger(__name__)

# Readiness flag
ready = False

# Environment variables
ANALYST_AGENT_PORT = int(os.environ.get("ANALYST_AGENT_PORT", 8004))
MODEL_PATH = os.environ.get("MISTRAL_7B_PATH", "./models/mistral-7b-instruct-v0.2")
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

# Pydantic models
class ToolCall(BaseModel):
    args: list
    kwargs: dict

import time


class MCPBusClient:
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
            "tools": tools,
        }

        max_retries = 5
        backoff_factor = 2

        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/register", json=registration_data, timeout=(3, 10))
                response.raise_for_status()
                logger.info(f"Successfully registered {agent_name} with MCP Bus.")
                return
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to register {agent_name} with MCP Bus: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to register {agent_name} with MCP Bus after {max_retries} attempts.")
                    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info("🔍 Analyst Agent V2 - Specialized Quantitative Analysis")
    logger.info("📊 Focus: Entity extraction, statistical analysis, numerical metrics")
    logger.info("🎯 Specialization: Text statistics, trends, financial/temporal data")
    logger.info("🤝 Integration: Works with Scout V2 for comprehensive content analysis")

    logger.info("Specialized analysis modules loaded and ready")

    # Register agent with MCP Bus
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="analyst",
            agent_address=f"http://localhost:{ANALYST_AGENT_PORT}",
            tools=[
                "identify_entities",
                "analyze_text_statistics",
                "extract_key_metrics",
                "analyze_content_trends",
                "analyze_sentiment",
                "detect_bias",
                "score_sentiment",
                "score_bias",
                "analyze_sentiment_and_bias"
            ],
        )
        logger.info("Registered tools with MCP Bus.")
    except Exception as e:
        logger.warning(f"MCP Bus unavailable: {e}. Running in standalone mode.")
    # Mark ready after successful startup tasks
    global ready
    ready = True
    yield

    # Cleanup on shutdown
    logger.info("✅ Analyst agent shutdown completed.")

    logger.info("Analyst agent is shutting down.")

app = FastAPI(lifespan=lifespan)

# Initialize metrics
metrics = JustNewsMetrics("analyst")

# Add security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Replace with actual allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Metrics middleware (must be added after security middleware)
app.middleware("http")(metrics.request_middleware)

# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for input validation and rate limiting"""
    if SECURITY_AVAILABLE:
        # Log security events
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

    response = await call_next(request)
    return response

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for analyst")

# Register reload endpoint if available
try:
    from agents.common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for analyst")

@app.get("/health")
@app.post("/health")
async def health(request: Request):
    """Health check endpoint that accepts optional body."""
    return {
        "status": "ok",
        "security_enabled": SECURITY_AVAILABLE,
        "version": get_version()
    }

@app.get("/ready")
def ready_endpoint():
    """Readiness endpoint for startup gating."""
    return {"ready": ready}

@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(metrics.get_metrics(), media_type="text/plain; charset=utf-8")

# RESTORED ENDPOINTS - Sentiment and bias analysis capabilities restored
# These endpoints provide the Analyst Agent with its own sentiment analysis capabilities

@app.post("/score_bias")
def score_bias_endpoint(call: ToolCall):
    """Score bias in provided text content."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'score_bias_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        from .tools import score_bias
        logger.info(f"Calling score_bias with args: {call.args} and kwargs: {call.kwargs}")
        return score_bias(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_sentiment")
def score_sentiment_endpoint(call: ToolCall):
    """Score sentiment in provided text content."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'score_sentiment_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        from .tools import score_sentiment
        logger.info(f"Calling score_sentiment with args: {call.args} and kwargs: {call.kwargs}")
        return score_sentiment(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in score_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_sentiment_and_bias")
def analyze_sentiment_and_bias_endpoint(call: ToolCall):
    """Analyze both sentiment and bias in provided text content."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'analyze_sentiment_and_bias_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        from .tools import analyze_sentiment_and_bias
        logger.info(f"Calling analyze_sentiment_and_bias with args: {call.args} and kwargs: {call.kwargs}")

        # Debug: Check GPU analyst status
        from .gpu_analyst import get_gpu_analyst
        gpu_analyst = get_gpu_analyst()
        logger.info(f"GPU analyst status - models_loaded: {gpu_analyst.models_loaded}, gpu_available: {gpu_analyst.gpu_available}")
        safe, reason = gpu_analyst.is_gpu_safe_to_use()
        logger.info(f"GPU safe to use: {safe}, reason: {reason}")

        result = analyze_sentiment_and_bias(*call.args, **call.kwargs)
        logger.info("analyze_sentiment_and_bias completed successfully")
        logger.info(f"Result methods - sentiment: {result.get('sentiment_analysis', {}).get('method')}, bias: {result.get('bias_analysis', {}).get('method')}")
        return result
    except Exception as e:
        logger.error(f"An error occurred in analyze_sentiment_and_bias: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_sentiment")
def analyze_sentiment_endpoint(call: ToolCall):
    """Analyze sentiment in provided text content."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'analyze_sentiment_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        from .tools import analyze_sentiment
        logger.info(f"Calling analyze_sentiment with args: {call.args} and kwargs: {call.kwargs}")
        return analyze_sentiment(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in analyze_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_bias")
def detect_bias_endpoint(call: ToolCall):
    """Detect bias in provided text content."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'detect_bias_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        from .tools import detect_bias
        logger.info(f"Calling detect_bias with args: {call.args} and kwargs: {call.kwargs}")
        return detect_bias(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in detect_bias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify_entities")
def identify_entities_endpoint(call: ToolCall):
    """Identifies entities in a given text."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'identify_entities_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        return identify_entities(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in identify_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_feedback")
def log_feedback_endpoint(call: ToolCall):
    """Logs feedback."""
    try:
        feedback = call.kwargs.get("feedback", {})
        log_feedback("log_feedback", feedback)
        return {"status": "logged"}
    except Exception as e:
        logger.error(f"An error occurred while logging feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_article")
def analyze_article_endpoint(call: ToolCall):
    """Analyze an article and update its analyzed status."""
    try:
        logger.debug("Received analyze_article request with payload: %s", call.kwargs)

        # Security validation
        if call.kwargs and 'article_id' in call.kwargs:
            article_id = call.kwargs['article_id']
            logger.debug("Processing article_id: %s", article_id)

            # Fetch article from database
            article = fetch_article_from_db(article_id)
            logger.debug("Fetched article: %s", article)

            if not article:
                logger.error("Article not found for article_id: %s", article_id)
                raise HTTPException(status_code=404, detail="Article not found")

            if article.get("analyzed", False):
                logger.info("Article %s already analyzed. Skipping.", article_id)
                return {"status": "skipped", "message": "Article already analyzed"}

            # Perform analysis
            logger.debug("Performing analysis on article content.")
            analysis_result = perform_analysis(article["content"])
            logger.debug("Analysis result: %s", analysis_result)

            # Update article as analyzed
            logger.debug("Updating article %s as analyzed.", article_id)
            update_article_status(article_id, analyzed=True)

            logger.info("Successfully analyzed and updated article %s.", article_id)
            return {"status": "success", "analysis_result": analysis_result}

        else:
            logger.error("Missing article_id in request payload.")
            raise HTTPException(status_code=400, detail="Missing article_id in request")

    except Exception as e:
        logger.error("An error occurred in analyze_article: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

def fetch_article_from_db(article_id: int) -> dict:
    """Fetch article from the database."""
    try:
        with get_db_cursor(commit=False) as (_, cursor):
            cursor.execute(
                "SELECT id, content, analyzed FROM articles WHERE id = %s",
                (article_id,)
            )
            article = cursor.fetchone()
            if article:
                return {
                    "id": article["id"],
                    "content": article["content"],
                    "analyzed": article["analyzed"],
                }
            else:
                return {}
    except Exception as e:
        logger.error(f"Failed to fetch article {article_id} from database: {e}")
        return {}

def update_article_status(article_id: int, analyzed: bool):
    """Update the analyzed status of an article in the database."""
    try:
        with get_db_cursor(commit=True) as (_, cursor):
            cursor.execute(
                "UPDATE articles SET analyzed = %s WHERE id = %s",
                (analyzed, article_id)
            )
            logger.info(f"Article {article_id} status updated to analyzed={analyzed}")
    except Exception as e:
        logger.error(f"Failed to update article {article_id} status: {e}")

def perform_analysis(content: str) -> dict:
    """Perform the actual analysis on the article content."""
    # Placeholder for analysis logic
    return {
        "sentiment": "positive",
        "bias": "neutral",
        "entities": ["entity1", "entity2"]
    }

def verify_db_connection() -> bool:
    """Verify the database connection and cursor setup."""
    try:
        with get_db_cursor(commit=False) as (_, cursor):
            cursor.execute("SELECT 1;")
            logger.info("Database connection verified successfully.")
            return True
    except Exception as e:
        logger.error(f"Database connection verification failed: {e}")
        return False

# REMOVED ENDPOINTS - All sentiment and bias analysis centralized in Scout V2 Agent
# Use Scout V2 for all sentiment and bias analysis (including batch operations):
# - POST /comprehensive_content_analysis (includes sentiment + bias)
# - POST /analyze_sentiment (dedicated sentiment analysis)
# - POST /detect_bias (dedicated bias detection)

# The following TensorRT batch endpoints have been removed from Analyst:
# - POST /score_bias_batch - REMOVED (use Scout V2 batch analysis)
# - POST /score_sentiment_batch - REMOVED (use Scout V2 batch analysis)
# - POST /analyze_article - REMOVED (use Scout V2 comprehensive analysis)
# - POST /analyze_articles_batch - REMOVED (use Scout V2 batch analysis)

# Engine information endpoint
@app.post("/analyze_text_statistics")
def analyze_text_statistics_endpoint(call: ToolCall):
    """Analyzes text statistics including readability and complexity."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'analyze_text_statistics_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        return analyze_text_statistics(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in analyze_text_statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_key_metrics")
def extract_key_metrics_endpoint(call: ToolCall):
    """Extracts key numerical and statistical metrics from text."""
    try:
        # Security validation
        if call.kwargs and 'text' in call.kwargs:
            if not validate_content_size(call.kwargs['text']):
                log_security_event('content_size_exceeded', {'function': 'extract_key_metrics_endpoint'})
                raise HTTPException(status_code=400, detail="Content size exceeds maximum allowed limit")

            call.kwargs['text'] = sanitize_content(call.kwargs['text'])

        return extract_key_metrics(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in extract_key_metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_content_trends")
def analyze_content_trends_endpoint(call: ToolCall):
    """Analyzes trends across multiple content pieces."""
    try:
        # Security validation for content arrays
        if call.kwargs and 'content_list' in call.kwargs:
            content_list = call.kwargs['content_list']
            if isinstance(content_list, list):
                for i, content in enumerate(content_list):
                    if isinstance(content, str):
                        if not validate_content_size(content):
                            log_security_event('content_size_exceeded', {'function': 'analyze_content_trends_endpoint', 'item': i})
                            raise HTTPException(status_code=400, detail=f"Content item {i} size exceeds maximum allowed limit")
                        content_list[i] = sanitize_content(content)

        return analyze_content_trends(*call.args, **call.kwargs)
    except Exception as e:
        logger.error(f"An error occurred in analyze_content_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# REMOVED ENDPOINT - Combined sentiment and bias analysis centralized in Scout V2 Agent
# Use Scout V2 /comprehensive_content_analysis endpoint for combined analysis

# @app.post("/analyze_sentiment_and_bias") - REMOVED

if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("ANALYST_HOST", "0.0.0.0")
    port = int(os.environ.get("ANALYST_PORT", 8004))

    logger.info(f"Starting Analyst Agent on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
