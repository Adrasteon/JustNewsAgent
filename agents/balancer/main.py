from typing import Any, Dict

try:
    # Optional runtime dependency for running the HTTP server
    from fastapi import FastAPI, HTTPException  # type: ignore
    from pydantic import BaseModel  # type: ignore
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

from . import tools


class ToolCall:  # lightweight Pydantic-like fallback for tests
    def __init__(self, args=None, kwargs=None):
        self.args = args or []
        self.kwargs = kwargs or {}


def health() -> Dict[str, Any]:
    """Health handler (callable directly from tests)."""
    return {"status": "ok", "agent": "balancer"}


def call_tool(name: str, call: ToolCall) -> Dict[str, Any]:
    """Call a tool implemented in agents.balancer.tools.<name>.

    When FastAPI is available, the same function is mounted as a route.
    """
    try:
        func = getattr(tools, name)
    except AttributeError:
        if HAS_FASTAPI:
            raise HTTPException(status_code=404, detail=f"tool {name} not found")
        raise

    try:
        result = func(*call.args, **call.kwargs)
        return {"status": "success", "data": result}
    except Exception as e:
        if HAS_FASTAPI:
            raise HTTPException(status_code=500, detail=str(e))
        raise


# If FastAPI is installed, create the ASGI app and mount routes.
if HAS_FASTAPI:
    app = FastAPI(title="Balancer Agent")

    # Register shutdown endpoint if available
    try:
        from agents.common.shutdown import register_shutdown_endpoint
        register_shutdown_endpoint(app)
    except Exception:
        # No logger variable in this module's top-level scope before structlog resolution; safe no-op
        pass


    class _ToolCallModel(BaseModel):
        args: list = []
        kwargs: dict = {}


    @app.get("/health")
    def _health():
        return health()


    @app.post("/call")
    def _call_tool(name: str, call: _ToolCallModel):
        return call_tool(name=name, call=ToolCall(call.args, call.kwargs))

