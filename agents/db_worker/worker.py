"""Minimal db_worker FastAPI app used as a development stub.

This module provides a very small FastAPI application exposing the
endpoints the orchestrator expects from the db_worker agent. It is
intentionally lightweight: it returns deterministic success values and
does not execute any database operations. Replace with the full
implementation when ready.
"""
from typing import Any, Dict
import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class IngestPayload(BaseModel):
    """Payload shape accepted by /handle_ingest.

    This is a flexible placeholder type so the full project's ingest
    contract can be simulated during development.
    """

    statements: list | None = None
    metadata: Dict[str, Any] | None = None


app = FastAPI()


@app.get("/health")
async def health() -> Dict[str, str]:
    """Simple health endpoint used by the startup orchestrator.

    Returns a compact status payload so the startup script can assert
    the agent is alive.
    """

    return {"status": "ok"}


@app.post("/handle_ingest")
async def handle_ingest(payload: IngestPayload, request: Request) -> Dict[str, Any]:
    """Accepts an ingest payload and returns a deterministic response.

    This stub accepts the same endpoint name used throughout the
    repository and returns a predictable response so other agents can
    exercise the call/response contract during local development.
    """

    try:
        # Log received payload at debug level to aid local troubleshooting
        logger.debug("Received ingest payload: %s", payload.json())
        # Return a lightweight success structure that mirrors the
        # minimal behaviour tests and stubs expect.
        return {"status": "ok", "chosen_source_id": None}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error handling ingest: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
