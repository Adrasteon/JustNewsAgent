"""FastAPI application for the crawler agent."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from common.metrics import JustNewsMetrics
from common.observability import get_logger

logger = get_logger(__name__)

ready: bool = False
crawl_jobs: dict[str, Any] = {}

CRAWLER_AGENT_PORT = int(os.environ.get("CRAWLER_AGENT_PORT", 8015))
MCP_BUS_URL = os.environ.get("MCP_BUS_URL", "http://localhost:8000")

ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8000"
).split(",")


class MCPBusClient:
    """Utility for registering the crawler agent with the MCP bus."""

    def __init__(self, base_url: str = MCP_BUS_URL) -> None:
        self.base_url = base_url

    def register_agent(
        self,
        agent_name: str,
        agent_address: str,
    tools: list[str],
    ) -> None:
        """Register the crawler agent and its exposed tools with the MCP bus."""

        registration_data = {
            "name": agent_name,
            "address": agent_address,
            "tools": tools,
        }
        try:
            response = requests.post(
                f"{self.base_url}/register", json=registration_data, timeout=(1, 2)
            )
            response.raise_for_status()
            logger.info("Crawler agent registered with MCP bus.")
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to register crawler agent with MCP bus: %s", exc)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""

    logger.info("Crawler agent is starting up.")
    mcp_bus_client = MCPBusClient()
    try:
        mcp_bus_client.register_agent(
            agent_name="crawler",
            agent_address=f"http://localhost:{CRAWLER_AGENT_PORT}",
            tools=[
                "unified_production_crawl",
                "get_crawler_info",
                "get_performance_metrics",
                "get_job_status",
                "list_jobs",
                "clear_jobs",
                "reset_crawler",
            ],
        )
        logger.info("Registered crawler tools with MCP bus.")
    except Exception as exc:  # noqa: BLE001 - prevent crash on bus failure
        logger.warning("MCP bus unavailable: %s. Running in standalone mode.", exc)

    global ready
    ready = True

    try:
        yield
    finally:
        logger.info("Crawler agent is shutting down.")
        ready = False


app = FastAPI(
    lifespan=lifespan,
    title="Crawler Agent",
    description="Unified production crawling agent",
)

metrics = JustNewsMetrics("crawler")

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.middleware("http")(metrics.request_middleware)


class ToolCall(BaseModel):
    """Standard MCP tool call payload."""

    args: list[Any]
    kwargs: dict[str, Any]


def _get_job_status(job_id: str) -> dict[str, Any]:
    """Return job status data or raise a 404 error."""

    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return crawl_jobs[job_id]


def _list_job_statuses() -> dict[str, str]:
    """Return mapping of job IDs to statuses."""

    return {
        job_id: status.get("status", "unknown")
        for job_id, status in crawl_jobs.items()
    }


def _clear_all_jobs() -> dict[str, Any]:
    """Remove all jobs from memory and report cleared IDs."""

    cleared_jobs = list(crawl_jobs.keys())
    crawl_jobs.clear()
    return {
        "cleared_jobs": cleared_jobs,
        "message": f"Cleared {len(cleared_jobs)} jobs from memory",
    }


def _reset_crawler_state() -> dict[str, Any]:
    """Reset job state and attempt to reset performance metrics."""

    response = _clear_all_jobs()
    try:
        from agents.crawler.performance_monitoring import reset_performance_metrics

        reset_performance_metrics()
        response["metrics_reset"] = True
    except ImportError:
        response["metrics_reset"] = False
    response["message"] = (
        f"Completely reset crawler: cleared {len(response['cleared_jobs'])} "
        "jobs and reset metrics"
    )
    return response


@app.post("/unified_production_crawl")
async def unified_production_crawl_endpoint(
    call: ToolCall,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Enqueue a background unified production crawl job and return job ID."""

    job_id = uuid.uuid4().hex
    crawl_jobs[job_id] = {"status": "pending"}

    domains: list[str] = call.args[0] if call.args else call.kwargs.get("domains", [])
    max_articles: int = call.kwargs.get("max_articles_per_site", 25)
    concurrent: int = call.kwargs.get("concurrent_sites", 3)

    logger.info(
        "Enqueueing background crawl job %s for %s domains",
        job_id,
        len(domains),
    )

    async def _crawl_task(
    crawl_domains: list[str],
        max_articles_per_site: int,
        concurrent_sites: int,
        crawl_job_id: str,
    ) -> None:
        from agents.crawler.unified_production_crawler import UnifiedProductionCrawler

        try:
            crawl_jobs[crawl_job_id]["status"] = "running"
            async with UnifiedProductionCrawler() as crawler:
                await crawler._load_ai_models()
                result = await crawler.run_unified_crawl(
                    crawl_domains,
                    max_articles_per_site,
                    concurrent_sites,
                )
            crawl_jobs[crawl_job_id] = {"status": "completed", "result": result}
            logger.info(
                "Background crawl %s complete. Articles: %s",
                crawl_job_id,
                len(result.get("articles", [])),
            )
        except Exception as exc:  # noqa: BLE001 - capture all errors for job state
            crawl_jobs[crawl_job_id] = {"status": "failed", "error": str(exc)}
            logger.error("Background crawl %s failed: %s", crawl_job_id, exc)

    background_tasks.add_task(
        _crawl_task,
        domains,
        max_articles,
        concurrent,
        job_id,
    )
    return JSONResponse(
        status_code=202,
        content={"status": "accepted", "job_id": job_id},
    )


@app.get("/job_status/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    """Retrieve the status for a specific crawl job."""

    return _get_job_status(job_id)


@app.post("/job_status")
def job_status_tool(call: ToolCall) -> dict[str, Any]:
    """MCP tool wrapper for retrieving job status."""

    job_id = call.kwargs.get("job_id") or (call.args[0] if call.args else None)
    if not isinstance(job_id, str):
        raise HTTPException(status_code=400, detail="Missing job_id parameter")
    return _get_job_status(job_id)


@app.get("/jobs")
def list_jobs() -> dict[str, str]:
    """List all current crawl job IDs with their status."""

    return _list_job_statuses()


@app.post("/list_jobs")
def list_jobs_tool(_: ToolCall) -> dict[str, str]:
    """MCP tool wrapper for listing crawl job statuses."""

    return _list_job_statuses()


@app.post("/clear_jobs")
def clear_jobs(call: ToolCall | None = None) -> dict[str, Any]:
    """Clear completed and failed jobs from memory."""

    if call and (call.args or call.kwargs):
        logger.warning("clear_jobs called with unexpected parameters; ignoring")
    return _clear_all_jobs()


@app.post("/reset_crawler")
def reset_crawler(call: ToolCall | None = None) -> dict[str, Any]:
    """Completely reset crawler state and metrics where available."""

    if call and (call.args or call.kwargs):
        logger.warning("reset_crawler called with unexpected parameters; ignoring")
    return _reset_crawler_state()


@app.post("/get_crawler_info")
def get_crawler_info_endpoint(call: ToolCall) -> dict[str, Any]:
    """Fetch crawler configuration details via the production crawler module."""

    try:
        from agents.crawler.unified_production_crawler import get_crawler_info

        logger.info(
            "Calling get_crawler_info with args: %s and kwargs: %s",
            call.args,
            call.kwargs,
        )
        return get_crawler_info(*call.args, **call.kwargs)
    except Exception as exc:  # noqa: BLE001 - surface errors to MCP clients
        logger.error("An error occurred in get_crawler_info: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/get_performance_metrics")
def get_performance_metrics_endpoint(call: ToolCall) -> dict[str, Any]:
    """Return current crawler performance metrics."""

    try:
        from agents.crawler.performance_monitoring import get_performance_monitor

        monitor = get_performance_monitor()
        logger.info(
            "Calling get_performance_metrics with args: %s and kwargs: %s",
            call.args,
            call.kwargs,
        )
        return monitor.get_current_metrics()
    except Exception as exc:  # noqa: BLE001 - surface errors to MCP clients
        logger.error("An error occurred in get_performance_metrics: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Return basic health information."""

    return {"status": "ok"}


@app.get("/ready")
def ready_endpoint() -> dict[str, bool]:
    """Expose readiness state for load balancers and health checks."""

    return {"ready": ready}


@app.get("/metrics")
def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""

    return Response(metrics.get_metrics(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Crawler Agent on port %s", CRAWLER_AGENT_PORT)
    uvicorn.run(
        "agents.crawler.main:app",
        host="0.0.0.0",
        port=CRAWLER_AGENT_PORT,
        reload=False,
        log_level="info",
    )
