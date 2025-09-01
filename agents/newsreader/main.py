"""
NewsReader Agent - Unified Implementation
Combines LLaVA-based processing with V2 multi-modal capabilities

Features:
- Screenshot-based webpage processing with LLaVA
- Multi-modal content analysis (text, images, PDF, web)
- GPU acceleration with CPU fallbacks
- Comprehensive error handling and logging
- MCP Bus integration for inter-agent communication
- Security utilities and rate limiting
- Legacy compatibility with existing endpoints
"""

import logging
from typing import Dict, List, Optional, Any, Union
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import torch

# Import V2 tools for core processing
from .tools import (
    process_article_content,
    analyze_content_structure,
    extract_multimedia_content,
    extract_news_from_url,
    capture_webpage_screenshot,
    analyze_image_with_llava,
    health_check,
    get_engine,
    clear_engine
)

# Import security utilities (following Scout Agent pattern)
try:
    from ..scout.security_utils import validate_url, sanitize_content, rate_limit, security_wrapper
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    def validate_url(url: str) -> bool:
        return True
    def sanitize_content(content: str) -> str:
        return content
    def rate_limit(request: Request) -> bool:
        return True
    def security_wrapper(func):
        return func

# Import MCP Bus client
try:
    from ..common.mcp_bus import MCPBusClient
    MCP_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from common.mcp_bus import MCPBusClient
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False
        class MCPBusClient:
            def __init__(self, base_url: str = "http://localhost:8000"):
                self.base_url = base_url

            def register_agent(self, agent_name: str, agent_address: str, tools: list):
                pass  # Placeholder

            async def send_message(self, message_type: str, payload: dict):
                pass  # Placeholder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("newsreader.unified")

# Modern datetime utility
def utc_now() -> datetime:
    """Get current UTC datetime using timezone-aware approach"""
    return datetime.now(timezone.utc)

# Pydantic models for API
class ContentRequest(BaseModel):
    content: Union[str, Dict[str, Any]]
    content_type: str = Field(default="article", description="Type of content (article, image, pdf, webpage)")
    processing_mode: str = Field(default="comprehensive", description="Processing mode (comprehensive, fast, basic)")
    include_visual_analysis: bool = Field(default=True, description="Include visual analysis")
    include_layout_analysis: bool = Field(default=True, description="Include layout analysis")

class URLRequest(BaseModel):
    url: str = Field(..., description="URL to process")
    screenshot_path: Optional[str] = Field(default=None, description="Path to save screenshot")

class ImageRequest(BaseModel):
    image_path: str = Field(..., description="Path to image file")

class StructureRequest(BaseModel):
    content: str = Field(..., description="Content to analyze")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth")

class MultimediaRequest(BaseModel):
    content: Union[str, bytes, Dict[str, Any]]
    extraction_types: List[str] = Field(default=["images", "text", "layout", "metadata"])

class ToolCall(BaseModel):
    args: List[Any]
    kwargs: Dict[str, Any]

# Global variables
mcp_client: Optional[MCPBusClient] = None
ready = False

# Environment variables
NEWSREADER_AGENT_PORT = int(os.environ.get("NEWSREADER_AGENT_PORT", 8009))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global mcp_client, ready

    # Startup
    logger.info("🚀 Starting NewsReader Agent (Unified)")

    # Initialize MCP Bus client
    if MCP_AVAILABLE:
        try:
            mcp_client = MCPBusClient(base_url=MCP_BUS_URL)
            await mcp_client.register_agent(
                agent_name="newsreader",
                agent_address=f"http://localhost:{NEWSREADER_AGENT_PORT}",
                tools=[
                    "extract_news_content",
                    "capture_screenshot",
                    "analyze_screenshot",
                    "analyze_content",
                    "extract_structure",
                    "extract_multimedia"
                ]
            )
            logger.info("✅ MCP Bus registration successful")
        except Exception as e:
            logger.warning(f"⚠️ MCP Bus registration failed: {e}")
    else:
        logger.warning("⚠️ MCP Bus not available - running in standalone mode")

    # Initialize engine
    engine = get_engine()
    if engine:
        logger.info("✅ NewsReader V2 engine initialized")
    else:
        logger.warning("⚠️ NewsReader V2 engine not available - using fallback processing")

    ready = True
    yield

    # Shutdown
    logger.info("🛑 Shutting down NewsReader Agent")
    clear_engine()

app = FastAPI(
    title="NewsReader Agent - Unified",
    description="Unified news content extraction and analysis agent combining LLaVA and multi-modal V2 capabilities",
    version="3.0.0",
    lifespan=lifespan
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for input validation and rate limiting"""
    if SECURITY_AVAILABLE:
        # Rate limiting
        if not rate_limit(request):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Log security events
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

    response = await call_next(request)
    return response

# Register common shutdown endpoint
try:
    from ..common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for newsreader")

# Register reload endpoint if available
try:
    from ..common.reload import register_reload_endpoint
    register_reload_endpoint(app)
except Exception:
    logger.debug("reload endpoint not registered for newsreader")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with agent information"""
    return {
        "agent": "NewsReader (Unified)",
        "version": "3.0.0",
        "capabilities": [
            "news_extraction",
            "visual_analysis",
            "multi_modal_processing",
            "structure_analysis",
            "multimedia_extraction",
            "screenshot_capture",
            "llava_analysis"
        ],
        "endpoints": [
            "/extract_news",
            "/analyze_content",
            "/extract_structure",
            "/extract_multimedia",
            "/capture_screenshot",
            "/analyze_image",
            "/health",
            "/ready"
        ]
    }

@app.get("/health")
async def get_health():
    """Comprehensive health check"""
    try:
        health_data = health_check()

        # Add unified-specific information
        health_data.update({
            "unified_version": "3.0.0",
            "security_enabled": SECURITY_AVAILABLE,
            "mcp_bus_connected": MCP_AVAILABLE and mcp_client is not None,
            "gpu_available": torch.cuda.is_available(),
            "engine_initialized": get_engine() is not None,
            "ready": ready
        })

        return health_data

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "unified_version": "3.0.0",
            "ready": ready
        }

@app.get("/ready")
async def get_ready():
    """Ready check endpoint"""
    return {"ready": ready}

@app.post("/extract_news")
@security_wrapper
async def extract_news(request: URLRequest):
    """Extract news content from URL with LLaVA screenshot analysis"""
    try:
        # Validate URL if security is available
        if SECURITY_AVAILABLE and not validate_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL")

        logger.info(f"🔍 Extracting news from URL: {request.url}")

        # Use V2 tools for processing
        result = await extract_news_from_url(
            url=request.url,
            screenshot_path=request.screenshot_path
        )

        # Send MCP Bus notification if available
        if mcp_client:
            await mcp_client.send_message(
                message_type="news_extracted",
                payload={
                    "url": request.url,
                    "success": result["success"],
                    "processing_time": result["processing_time"],
                    "method": result["method"]
                }
            )

        return result

    except Exception as e:
        logger.error(f"News extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_content")
@security_wrapper
async def analyze_content(request: ContentRequest):
    """Analyze content with multi-modal processing"""
    try:
        # Sanitize content if security is available
        if SECURITY_AVAILABLE and isinstance(request.content, str):
            request.content = sanitize_content(request.content)

        logger.info(f"🔍 Analyzing content (type: {request.content_type}, mode: {request.processing_mode})")

        # Use V2 tools for processing
        result = await process_article_content(
            content=request.content,
            content_type=request.content_type,
            processing_mode=request.processing_mode,
            include_visual_analysis=request.include_visual_analysis,
            include_layout_analysis=request.include_layout_analysis
        )

        # Send MCP Bus notification if available
        if mcp_client:
            await mcp_client.send_message(
                message_type="content_analyzed",
                payload={
                    "content_type": request.content_type,
                    "processing_mode": request.processing_mode,
                    "success": result["status"] == "success",
                    "processing_time": result["processing_time"]
                }
            )

        return result

    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_structure")
@security_wrapper
async def extract_structure(request: StructureRequest):
    """Extract and analyze content structure"""
    try:
        # Sanitize content if security is available
        if SECURITY_AVAILABLE:
            request.content = sanitize_content(request.content)

        logger.info(f"🔍 Analyzing content structure (depth: {request.analysis_depth})")

        # Use V2 tools for processing
        result = await analyze_content_structure(
            content=request.content,
            analysis_depth=request.analysis_depth
        )

        return result

    except Exception as e:
        logger.error(f"Structure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract_multimedia")
@security_wrapper
async def extract_multimedia(request: MultimediaRequest):
    """Extract multimedia content from various sources"""
    try:
        logger.info(f"🔍 Extracting multimedia content (types: {request.extraction_types})")

        # Use V2 tools for processing
        result = await extract_multimedia_content(
            content=request.content,
            extraction_types=request.extraction_types
        )

        return result

    except Exception as e:
        logger.error(f"Multimedia extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/capture_screenshot")
@security_wrapper
async def capture_screenshot(request: URLRequest):
    """Capture webpage screenshot"""
    try:
        # Validate URL if security is available
        if SECURITY_AVAILABLE and not validate_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL")

        logger.info(f"📸 Capturing screenshot for URL: {request.url}")

        # Use V2 tools for processing
        result = await capture_webpage_screenshot(
            url=request.url,
            output_path=request.screenshot_path or "screenshot.png"
        )

        return result

    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_image")
@security_wrapper
async def analyze_image(request: ImageRequest):
    """Analyze image with LLaVA model"""
    try:
        # Validate file path
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=400, detail="Image file not found")

        logger.info(f"🖼️ Analyzing image: {request.image_path}")

        # Use V2 tools for processing
        result = analyze_image_with_llava(image_path=request.image_path)

        return result

    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MCP Bus compatible endpoints (legacy compatibility)
@app.post("/extract_news_content")
async def extract_news_content_endpoint(call: ToolCall):
    """Extract news content from URL - MCP Bus compatible"""
    try:
        url = call.args[0] if call.args else call.kwargs.get("url")
        screenshot_path = call.args[1] if len(call.args) > 1 else call.kwargs.get("screenshot_path")

        if not url:
            return {"error": "URL is required"}

        result = await extract_news_from_url(url=url, screenshot_path=screenshot_path)
        return result
    except Exception as e:
        return {"error": str(e), "success": False}

@app.post("/analyze_screenshot")
async def analyze_screenshot(call: ToolCall):
    """Analyze screenshot with LLaVA - MCP Bus compatible"""
    try:
        image_path = call.args[0] if call.args else call.kwargs.get("image_path")

        if not image_path:
            return {"error": "Image path is required"}

        result = analyze_image_with_llava(image_path=image_path)
        return result
    except Exception as e:
        return {"error": str(e), "success": False}

# Background task management
@app.post("/clear_cache")
async def clear_cache(background_tasks: BackgroundTasks):
    """Clear processing cache and free GPU memory"""
    try:
        background_tasks.add_task(clear_engine)
        return {"status": "cache_clearing_scheduled"}
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Tool functions for direct import (legacy compatibility)
async def extract_news_content(url: str, screenshot_path: str = None) -> Dict[str, Any]:
    """Extract news content from URL"""
    result = await extract_news_from_url(url=url, screenshot_path=screenshot_path)
    return result

def analyze_image_content(image_path: str) -> Dict[str, str]:
    """Analyze image content with LLaVA"""
    return analyze_image_with_llava(image_path=image_path)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=NEWSREADER_AGENT_PORT,
        reload=True,
        log_level="info"
    )
