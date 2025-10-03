"""S3 adapter that uses boto3 to interact with S3-compatible endpoints.

This lightweight adapter provides an upload_bytes method and a method to
construct object URLs. It is intended for use in the kafka scaffold where
SeaweedFS exposes an S3-compatible API. Why boto3: it is widely used and
lets us rely on standard S3 semantics.
"""
from __future__ import annotations

from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config
except Exception:  # pragma: no cover - boto3 may not be installed in test env
    boto3 = None
    Config = None


raise RuntimeError("S3Adapter removed from kafka scaffold - use SeaweedFS or IPFS instead")
