"""SeaweedFS HTTP client for uploading artifacts to a SeaweedFS filer or S3-compatible gateway.

This client uses the configured OBJECT_STORE_ENDPOINT and performs a
simple HTTP PUT to upload bytes to a deterministic object path. This is a
lightweight implementation for the scaffold; production deployments may
wish to use a more complete S3-compatible SDK or direct SeaweedFS
filer API.
"""
from __future__ import annotations

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class SeaweedFSHttpClient:
    def __init__(self, endpoint: Optional[str] = None, bucket: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("OBJECT_STORE_ENDPOINT", "http://localhost:8333")
        # The scaffold uses a simple key namespace; in production this
        # should be replaced with proper bucket/namespace handling.
        self.bucket = bucket or os.getenv("OBJECT_STORE_BUCKET", "justnews")

    def upload_bytes(self, data: bytes, key_hint: str = "obj") -> str:
        """Upload bytes to SeaweedFS by performing an HTTP PUT to the
        endpoint. Returns a deterministic object path used by the
        scaffold as the content identifier.

        Note: SeaweedFS offers multiple HTTP endpoints and an S3 API.
        This method uses a minimal approach for the dev scaffold. For
        production use, switch to an S3 SDK or the official SeaweedFS
        API with error handling and retries.
        """
        import hashlib
        digest = hashlib.sha256(data).hexdigest()[:16]
        filename = f"{key_hint}-{digest}.bin"
        url = f"{self.endpoint.rstrip('/')}/{self.bucket}/{filename}"
        logger.debug("Uploading to SeaweedFS URL=%s", url)
        try:
            resp = requests.put(url, data=data)
            resp.raise_for_status()
        except Exception as exc:
            logger.error(
                "SeaweedFS upload failed: status=%s body=%s error=%s",
                getattr(resp, "status_code", None),
                getattr(resp, "text", None),
                exc,
            )
            raise
        # Return a canonical object path that other components can store
        return f"seaweed://{self.bucket}/{filename}"

    def object_url(self, object_path: str) -> str:
        # Convert our object path into an HTTP-accessible URL for preview
        # (very basic mapping used by scaffold dev tools)
        # object_path example: seaweed://justnews/key
        if not object_path.startswith("seaweed://"):
            return object_path
        _, rest = object_path.split("//", 1)
        return f"{self.endpoint.rstrip('/')}/{rest}"
