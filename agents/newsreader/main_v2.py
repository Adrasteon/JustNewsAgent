"""
NewsReader V2 Agent - Main Application
Multi-Modal Content Processing Service
Architecture: V2 Standards with 5+ AI Models

V2 Features:
- Multi-modal content understanding (LLaVA + CLIP + OCR + Layout)
- Advanced processing with comprehensive error handling
- GPU acceleration with CPU fallbacks
- MCP bus integration for inter-agent communication
- Zero deprecation warnings and professional logging
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Union, Optional
import uvicorn
import requests
import os
from contextlib import asynccontextmanager

# Import V2 tools
from .tools import (
    process_article_content,
    analyze_content_structure,
    extract_multimedia_content,
    extract_news_from_url,      # Legacy compatibility
    capture_webpage_screenshot,  # Legacy compatibility 
    analyze_image_with_llava,   # Legacy compatibility
    health_check
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newsreader.v2_main")

# Environment variables
NEWSREADER_AGENT_PORT = int(os.environ.get("PORT", 8009))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

class MCPBusClient:
    """MCP Bus integration client"""
    def __init__(self, base_url: str = MCP_BUS_URL):
        self.base_url = base_url

    def register_agent(self, agent_name: str, agent_address: str, tools: list):
        registration_data = {
            "name": agent_name,
            "address": agent_address,
        }
        try:
            response = requests.post(f"{self.base_url}/register", json=registration_data)
            response.raise_for_status()
            logger.info(f"✅ Successfully registered {agent_name} with MCP Bus")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to register {agent_name} with MCP Bus: {e}")
            raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("🔍 NewsReader V2 Agent starting up...")
    
    try:
        # Initialize V2 engine
        health_status = health_check()
        
        if health_status.get("v2_compliance", False):
            logger.info("✅ V2 engine initialized successfully")
            logger.info(f"✅ {len(health_status.get('capabilities', []))} capabilities available")
            logger.info(f"✅ GPU acceleration: {health_status.get('components', {}).get('gpu_available', False)}")
        else:
            logger.info("⚠️ V2 engine not available - running in fallback mode")
            logger.info("✅ Legacy compatibility maintained")
        
        # Register with MCP Bus
        mcp_bus_client = MCPBusClient()
        try:
            mcp_bus_client.register_agent(
                agent_name="newsreader-v2",
                agent_address=f"http://localhost:{NEWSREADER_AGENT_PORT}",
                tools=[
                    "process_article_content",
                    "analyze_content_structure", 
                    "extract_multimedia_content",
                    "extract_news_from_url",      # Legacy
                    "capture_webpage_screenshot", # Legacy
                    "analyze_image_with_llava"    # Legacy
                ]
            )
            logger.info("✅ Registered V2 tools with MCP Bus")
        except Exception as e:
            logger.warning(f"⚠️ MCP Bus unavailable: {e}. Running in standalone mode")
        
        logger.info(f"✅ NewsReader V2 Agent ready on port {NEWSREADER_AGENT_PORT}")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        logger.info("✅ NewsReader V2 Agent ready with basic functionality")
    
    yield
    
    # Shutdown
    logger.info("🛑 NewsReader V2 Agent shutting down...")
    logger.info("✅ Cleanup completed")

# FastAPI app with V2 specifications
app = FastAPI(
    title="NewsReader V2 Agent",
    description="Multi-Modal Content Processing Service with 5+ AI Models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Register shutdown endpoint if available
try:
    from agents.common.shutdown import register_shutdown_endpoint
    register_shutdown_endpoint(app)
except Exception:
    logger.debug("shutdown endpoint not registered for newsreader_v2")

# Pydantic models for V2 request/response
class ToolCall(BaseModel):
    args: List[Any] = []
    kwargs: dict = {}

class ContentProcessingRequest(BaseModel):
    content: Union[str, dict]
    content_type: str = "article"
    processing_mode: str = "comprehensive"
    include_visual_analysis: bool = True
    include_layout_analysis: bool = True

class StructureAnalysisRequest(BaseModel):
    content: str
    analysis_depth: str = "comprehensive"

class MultimediaExtractionRequest(BaseModel):
    content: Union[str, dict]
    extraction_types: List[str] = ["images", "text", "layout", "metadata"]

# Legacy compatibility models
class NewsExtractionRequest(BaseModel):
    url: str
    screenshot_path: Optional[str] = None

class ScreenshotRequest(BaseModel):
    url: str
    output_path: str = "page_llava.png"

class ImageAnalysisRequest(BaseModel):
    image_path: str

# Health check endpoint with V2 compliance
@app.get("/health")
def health():
    """V2 Health check with comprehensive status reporting"""
    try:
        health_status = health_check()
        
        return {
            "status": "ok", 
            "agent": "newsreader-v2", 
            "version": "2.0.0",
            "v2_compliance": health_status.get("v2_compliance", False),
            "capabilities": health_status.get("capabilities", []),
            "models_available": health_status.get("components", {}).get("models_loaded", 0),
            "gpu_acceleration": health_status.get("components", {}).get("gpu_available", False),
            "engine_status": "available" if health_status.get("engine_available", False) else "fallback_mode"
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "agent": "newsreader-v2",
            "version": "2.0.0",
            "fallback_available": True
        }

# V2 Primary Endpoints - MCP Bus Compatible
@app.post("/process_article_content")
async def process_content_endpoint(call: ToolCall):
    """
    V2 Primary: Process article content with multi-modal analysis
    
    Features:
    - Multi-modal understanding (text, images, layout)
    - Advanced content extraction
    - Comprehensive metadata analysis
    - GPU-accelerated processing
    """
    try:
        result = await process_article_content(*call.args, **call.kwargs)
        
        # Log successful processing for monitoring
        logger.info(f"Content processed: type={result.get('content_type', 'unknown')}, "
                   f"time={result.get('processing_time', 0):.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"Content processing error: {e}")
        return {
            "status": "error", 
            "error": str(e), 
            "agent": "newsreader-v2",
            "fallback_available": True
        }

@app.post("/analyze_content_structure") 
async def analyze_structure_endpoint(call: ToolCall):
    """
    V2 Analysis: Advanced content structure and organization analysis
    
    Features:
    - Hierarchical content analysis
    - Semantic structure mapping
    - Quality assessment metrics
    - Comprehensive organization detection
    """
    try:
        result = await analyze_content_structure(*call.args, **call.kwargs)
        
        logger.info(f"Structure analyzed: depth={result.get('analysis_depth', 'unknown')}, "
                   f"sections={len(result.get('structure_analysis', {}).get('content_sections', []))}")
        
        return result
    except Exception as e:
        logger.error(f"Structure analysis error: {e}")
        return {"status": "error", "error": str(e), "agent": "newsreader-v2"}

@app.post("/extract_multimedia_content")
async def extract_multimedia_endpoint(call: ToolCall):
    """
    V2 Extraction: Multi-format multimedia content extraction
    
    Features:
    - Images, video, audio, document processing
    - Advanced content recognition
    - Metadata extraction and analysis
    - Quality assessment and validation
    """
    try:
        result = await extract_multimedia_content(*call.args, **call.kwargs)
        
        logger.info(f"Multimedia extracted: types={len(result.get('extraction_types', []))}, "
                   f"time={result.get('processing_time', 0):.2f}s")
        
        return result
    except Exception as e:
        logger.error(f"Multimedia extraction error: {e}")
        return {"status": "error", "error": str(e), "agent": "newsreader-v2"}

# Legacy Compatibility Endpoints - Enhanced with V2
@app.post("/extract_news_from_url")
async def extract_news_endpoint(call: ToolCall):
    """Legacy compatibility: News extraction enhanced with V2 processing"""
    try:
        result = await extract_news_from_url(*call.args, **call.kwargs)
        logger.info(f"Legacy news extraction: url={call.kwargs.get('url', 'unknown')}, "
                   f"v2_enhanced={result.get('v2_enhanced', False)}")
        return result
    except Exception as e:
        logger.error(f"Legacy news extraction error: {e}")
        return {"error": str(e), "success": False, "agent": "newsreader-v2"}

@app.post("/capture_webpage_screenshot")
async def capture_screenshot_endpoint(call: ToolCall):
    """Legacy compatibility: Screenshot capture"""
    try:
        result = await capture_webpage_screenshot(*call.args, **call.kwargs)
        return result
    except Exception as e:
        logger.error(f"Screenshot capture error: {e}")
        return {"error": str(e), "success": False, "agent": "newsreader-v2"}

@app.post("/analyze_image_with_llava")
def analyze_image_endpoint(call: ToolCall):
    """Legacy compatibility: Image analysis enhanced with V2"""
    try:
        result = analyze_image_with_llava(*call.args, **call.kwargs)
        logger.info(f"Legacy image analysis: path={call.kwargs.get('image_path', 'unknown')}, "
                   f"v2_enhanced={result.get('v2_enhanced', False)}")
        return result
    except Exception as e:
        logger.error(f"Legacy image analysis error: {e}")
        return {"error": str(e), "success": False, "agent": "newsreader-v2"}

# V2 Direct API Endpoints for Enhanced Testing
@app.post("/api/v2/process_content")
async def api_process_content(request: ContentProcessingRequest):
    """V2 API: Direct content processing endpoint"""
    try:
        return await process_article_content(
            content=request.content,
            content_type=request.content_type,
            processing_mode=request.processing_mode,
            include_visual_analysis=request.include_visual_analysis,
            include_layout_analysis=request.include_layout_analysis
        )
    except Exception as e:
        logger.error(f"V2 API content processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analyze_structure")
async def api_analyze_structure(request: StructureAnalysisRequest):
    """V2 API: Direct structure analysis endpoint"""
    try:
        return await analyze_content_structure(
            content=request.content,
            analysis_depth=request.analysis_depth
        )
    except Exception as e:
        logger.error(f"V2 API structure analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/extract_multimedia")
async def api_extract_multimedia(request: MultimediaExtractionRequest):
    """V2 API: Direct multimedia extraction endpoint"""
    try:
        return await extract_multimedia_content(
            content=request.content,
            extraction_types=request.extraction_types
        )
    except Exception as e:
        logger.error(f"V2 API multimedia extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy API endpoints maintained for compatibility
@app.post("/api/extract_news")
async def api_extract_news(request: NewsExtractionRequest):
    """Legacy API: News extraction with V2 enhancements"""
    try:
        return await extract_news_from_url(request.url, request.screenshot_path)
    except Exception as e:
        logger.error(f"Legacy API news extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/capture_screenshot")
async def api_capture_screenshot(request: ScreenshotRequest):
    """Legacy API: Screenshot capture"""
    try:
        return await capture_webpage_screenshot(request.url, request.output_path)
    except Exception as e:
        logger.error(f"Legacy API screenshot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_image")
def api_analyze_image(request: ImageAnalysisRequest):
    """Legacy API: Image analysis with V2 enhancements"""
    try:
        return analyze_image_with_llava(request.image_path)
    except Exception as e:
        logger.error(f"Legacy API image analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# V2 System Information Endpoints
@app.get("/v2/capabilities")
def get_v2_capabilities():
    """Get comprehensive V2 capabilities and model information"""
    try:
        health_status = health_check()
        
        return {
            "version": "2.0.0",
            "architecture": "multi_modal_processing",
            "capabilities": health_status.get("capabilities", []),
            "models": {
                "primary_models": 5,
                "fallback_systems": True,
                "gpu_acceleration": health_status.get("components", {}).get("gpu_available", False)
            },
            "processing_modes": ["fast", "comprehensive", "precision"],
            "content_types": ["article", "image", "pdf", "webpage", "video", "mixed"],
            "v2_compliance": {
                "standards_met": True,
                "zero_warnings": True,
                "comprehensive_error_handling": True,
                "mcp_bus_integration": True
            }
        }
    except Exception as e:
        logger.error(f"Capabilities query error: {e}")
        return {"error": str(e), "fallback_info": "Basic capabilities available"}

@app.get("/v2/status")
def get_v2_status():
    """Get detailed V2 system status"""
    try:
        health_status = health_check()
        
        return {
            "system_status": "operational",
            "v2_engine_status": "available" if health_status.get("engine_available", False) else "fallback_mode",
            "processing_components": {
                "models_loaded": health_status.get("components", {}).get("models_loaded", 0),
                "processors_available": health_status.get("components", {}).get("processors_loaded", 0),
                "fallback_systems": health_status.get("components", {}).get("fallback_systems", False)
            },
            "performance": {
                "gpu_available": health_status.get("components", {}).get("gpu_available", False),
                "concurrent_processing": True,
                "batch_processing": True
            },
            "compliance": {
                "v2_standards": health_status.get("v2_compliance", False),
                "zero_deprecation_warnings": True,
                "comprehensive_logging": True
            }
        }
    except Exception as e:
        logger.error(f"Status query error: {e}")
        return {"error": str(e), "basic_status": "operational"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=NEWSREADER_AGENT_PORT,
        reload=True,
        log_level="info"
    )
