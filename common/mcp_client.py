"""Utilities for interacting with the MCP bus.

These helpers provide a thin wrapper around the MCP bus `/call` endpoint
with structured retries and consistent response handling so that callers do
not need to duplicate boilerplate request logic.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Iterable, Mapping, MutableMapping

import httpx
import requests

from common.observability import get_logger


class MCPCallError(RuntimeError):
    """Raised when a call to the MCP bus cannot be completed successfully."""


class MCPClient:
    """Simple HTTP client for the MCP bus with retry and parsing helpers."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        default_timeout: tuple[float, float] | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("MCP_BUS_URL", "http://localhost:8000")
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        connect_timeout = float(os.getenv("MCP_CALL_CONNECT_TIMEOUT", "3"))
        read_timeout = float(os.getenv("MCP_CALL_READ_TIMEOUT", "300"))
        if default_timeout is not None:
            connect_timeout, read_timeout = default_timeout
        self.timeout = (connect_timeout, read_timeout)
        self._session = requests.Session()
        self.logger = get_logger(__name__)

    def call(
        self,
        agent: str,
        tool: str,
        *,
        args: Iterable[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        timeout: tuple[float, float] | None = None,
    ) -> Any:
        """Invoke an MCP tool and return the unwrapped response payload."""
        payload: MutableMapping[str, Any] = {
            "agent": agent,
            "tool": tool,
            "args": list(args or []),
            "kwargs": dict(kwargs or {}),
        }

        last_error: Exception | None = None
        effective_timeout = timeout or self.timeout

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/call",
                    json=payload,
                    timeout=effective_timeout,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict) and "data" in body:
                    return body["data"]
                return body
            except requests.RequestException as exc:  # pragma: no cover - networking
                last_error = exc
                self.logger.warning(
                    "mcp_call_failed",
                    extra={
                        "agent": agent,
                        "tool": tool,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        raise MCPCallError(
            f"Failed to call {agent}.{tool} after {self.max_retries} attempts"
        ) from last_error

    def list_agents(self) -> dict[str, str]:
        """Return the registered agents from the MCP bus registry."""
        try:
            response = self._session.get(
                f"{self.base_url}/agents", timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover - networking
            raise MCPCallError("Failed to retrieve agents from MCP bus") from exc

        if not isinstance(payload, dict):
            raise MCPCallError("Unexpected payload when listing MCP agents")
        return {str(key): str(value) for key, value in payload.items()}

    def health_check(self) -> bool:
        """Return True when the MCP bus reports a healthy status."""
        try:
            response = self._session.get(
                f"{self.base_url}/health", timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"


class AsyncMCPClient:
    """Asynchronous HTTP client for the MCP bus using httpx.AsyncClient."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        default_timeout: tuple[float, float] | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("MCP_BUS_URL", "http://localhost:8000")
        self.max_retries = max(1, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        connect_timeout = float(os.getenv("MCP_CALL_CONNECT_TIMEOUT", "3"))
        read_timeout = float(os.getenv("MCP_CALL_READ_TIMEOUT", "300"))
        if default_timeout is not None:
            connect_timeout, read_timeout = default_timeout
        self._default_timeout = (connect_timeout, read_timeout)
        timeout = httpx.Timeout(
            read=read_timeout,
            connect=connect_timeout,
            write=connect_timeout,
            pool=None,
        )
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self.logger = get_logger(__name__)

    async def aclose(self) -> None:
        """Close the underlying httpx client."""

        await self._client.aclose()

    async def call(
        self,
        agent: str,
        tool: str,
        *,
        args: Iterable[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        timeout: tuple[float, float] | None = None,
    ) -> Any:
        """Invoke an MCP tool asynchronously and return the response payload."""

        payload: MutableMapping[str, Any] = {
            "agent": agent,
            "tool": tool,
            "args": list(args or []),
            "kwargs": dict(kwargs or {}),
        }

        effective_timeout = timeout or self._default_timeout
        request_timeout = httpx.Timeout(
            read=effective_timeout[1],
            connect=effective_timeout[0],
            write=effective_timeout[0],
            pool=None,
        )

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._client.post(
                    "/call",
                    json=payload,
                    timeout=request_timeout,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict) and "data" in body:
                    return body["data"]
                return body
            except httpx.HTTPError as exc:
                last_error = exc
                self.logger.warning(
                    "async_mcp_call_failed",
                    extra={
                        "agent": agent,
                        "tool": tool,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_seconds * attempt)

        raise MCPCallError(
            f"Failed to call {agent}.{tool} after {self.max_retries} attempts"
        ) from last_error

    async def list_agents(self) -> dict[str, str]:
        """Return registered agents asynchronously."""

        try:
            response = await self._client.get("/agents")
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise MCPCallError("Failed to retrieve agents from MCP bus") from exc

        if not isinstance(payload, dict):
            raise MCPCallError("Unexpected payload when listing MCP agents")
        return {str(key): str(value) for key, value in payload.items()}

    async def health_check(self) -> bool:
        """Return True when the MCP bus reports a healthy status."""

        try:
            response = await self._client.get("/health")
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError:
            return False
        return isinstance(payload, dict) and payload.get("status") == "ok"
